"""SimpleTES evaluator for SyCCL scheme 1: compact sketch -> direct events."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

try:
  from simpletes.construction import capture_construction_if_requested
except Exception:  # pragma: no cover - evaluator also works outside SimpleTES
  def capture_construction_if_requested(value: Any) -> bool:
    return False


AGENT_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BASE_CONFIG = REPO_ROOT / "config" / "a100-8gpu-4nic-clos-ag-4k.json"
FALLBACK_BASE_CONFIG = REPO_ROOT / "config" / "a100-8gpu-4nic-clos-ag.json"
DEFAULT_FLOW_SIM_BIN = (
  Path("/home/antl/wzd/llm-ccl/Flow-Simulator/flow-sim-rs/target/release/flow-sim-rs")
)
FLOW_SIM_BIN = Path(os.environ.get("FLOW_SIM_BIN", str(DEFAULT_FLOW_SIM_BIN)))
SCHEME_NAME = "scheme1_direct_events"
DEFAULT_ARTIFACT_ROOT = AGENT_ROOT / "eval_artifacts"

TIMEOUT_SECONDS = int(os.environ.get("SYCCL_EVALUATOR_TIMEOUT_SECONDS", "600"))
LOG_PREVIEW_CHARS = 4000
FAILURE_SCORE = -1_000_000_000_000.0


def _resolve_base_config() -> Path:
  raw_path = os.environ.get("SYCCL_BASE_CONFIG")
  if not raw_path:
    return DEFAULT_BASE_CONFIG if DEFAULT_BASE_CONFIG.exists() else FALLBACK_BASE_CONFIG
  path = Path(raw_path)
  if not path.is_absolute():
    path = REPO_ROOT / path
  return path


BASE_CONFIG = _resolve_base_config()

def _load_config(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as f:
    return json.load(f)


try:
  BASE_CONFIG_DATA = _load_config(BASE_CONFIG)
except OSError:
  BASE_CONFIG_DATA = {
    "hosts": {"host_num": 0, "host_gpu_num": 0, "host_nic_num": 0},
    "coll": {"name": "unknown", "byte": 0},
    "topo": [],
  }

TASK_HOST_NUM = int(BASE_CONFIG_DATA.get("hosts", {}).get("host_num", 4))
TASK_HOST_GPU_NUM = int(BASE_CONFIG_DATA.get("hosts", {}).get("host_gpu_num", 8))
TASK_HOST_NIC_NUM = int(BASE_CONFIG_DATA.get("hosts", {}).get("host_nic_num", 4))
NGPUS = TASK_HOST_NUM * TASK_HOST_GPU_NUM
COLL_BYTE = int(BASE_CONFIG_DATA.get("coll", {}).get("byte", 0))


class FlowSimOutputError(RuntimeError):
  """Raised when flow-sim-rs output is missing required evaluator fields."""


class FatalFlowSimError(RuntimeError):
  """Raised for flow-sim-rs environment/output errors that must stop the run."""

  simpletes_fatal = True


class FlowSimCaseResult:
  def __init__(self, *, case: dict[str, Any], output: dict[str, Any]) -> None:
    self.case = case
    self.output = output


def _resolve_artifact_root() -> Path:
  raw_path = os.environ.get("SYCCL_EVAL_ARTIFACT_DIR")
  if not raw_path:
    return DEFAULT_ARTIFACT_ROOT
  path = Path(raw_path)
  if not path.is_absolute():
    path = AGENT_ROOT / path
  return path


def _make_eval_workdir() -> Path:
  base = _resolve_artifact_root() / SCHEME_NAME
  base.mkdir(parents=True, exist_ok=True)
  stamp = time.strftime("%Y%m%d-%H%M%S")
  return Path(tempfile.mkdtemp(prefix=f"eval_{stamp}_{os.getpid()}_", dir=str(base)))


def _copy_program(program_path: str, workdir: Path) -> None:
  try:
    source = Path(program_path)
    (workdir / "program.py").write_text(
        source.read_text(encoding="utf-8", errors="replace"),
        encoding="utf-8",
    )
  except OSError:
    pass


def _write_metrics(workdir: Path | None, metrics: dict[str, Any]) -> dict[str, Any]:
  if workdir is None:
    return metrics
  try:
    (workdir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
  except (OSError, TypeError, ValueError):
    pass
  return metrics


def _failure_details(message: str) -> dict[str, str]:
  lower = message.lower()
  details = {
    "failure_category": "unknown_failure",
    "failure_feedback": _preview(message, 1200),
  }

  if "missing flow-sim-rs binary" in lower or "missing base config" in lower:
    details["failure_category"] = "flow_sim_config_error"
    if "missing base config" in lower:
      details["failure_feedback"] = (
        "The configured SyCCL base config does not exist. Set SYCCL_BASE_CONFIG "
        "to a valid config path or restore the default config file."
      )
    else:
      details["failure_feedback"] = (
        "The configured flow-sim-rs binary does not exist. Build flow-sim-rs "
        "or set FLOW_SIM_BIN to the release binary path."
      )
    return details

  if "flow-sim config" in lower or "configerror:" in lower:
    details["failure_category"] = "flow_sim_config_error"
    details["failure_feedback"] = (
      "flow-sim-rs could not parse or support the active SyCCL config. "
      "Use a config with supported collective, topology, layers, and link specs."
    )
    return details

  if "flow-sim-rs batch-sketch timed out" in lower or "timed out" in lower:
    details["failure_category"] = "flow_sim_timeout"
    details["failure_feedback"] = (
      "flow-sim-rs timed out evaluating this sketch batch. Prefer fewer "
      "candidate sketches and simpler dependency structure."
    )
    return details

  if "flow-sim-rs batch-sketch exited" in lower:
    details["failure_category"] = "flow_sim_sketch_error"
    error_match = re.search(r"Error:\s*(.+)", message, re.DOTALL)
    if error_match:
      details["failure_feedback"] = _preview(error_match.group(1).strip(), 1200)
    else:
      details["failure_feedback"] = _preview(message, 1200)
    return details

  if "flow-sim" in lower and (
      "output" in lower
      or "manifest" in lower
      or "bottleneck_profile" in lower
      or "time_us" in lower
  ):
    details["failure_category"] = "flow_sim_output_error"
    details["failure_feedback"] = (
      "flow-sim-rs completed but did not produce the required JSON fields. "
      "The evaluator requires positive time_us and a bottleneck_profile object."
    )
    return details

  match = re.search(r"source GPU (\d+).*step (\d+)", message)
  if match:
    gpu, step = match.groups()
    details["failure_category"] = "dependency_error"
    details["failure_location"] = f"GPU {gpu}, step {step}"
    reached_match = re.search(r"first reached at step (\d+)", message)
    if reached_match:
      reached_step = reached_match.group(1)
      details["failure_feedback"] = (
        f"GPU {gpu} is used as a source at step {step}, but it is first "
        f"reached at step {reached_step}. A GPU may fan out only in a "
        "strictly later step than the step that first delivers it."
      )
    else:
      details["failure_feedback"] = (
        f"GPU {gpu} is used as a source at step {step} before it is reachable. "
        "Deliver this GPU in an earlier step, or move its fanout to a later step."
      )
    return details

  match = re.search(r"destination GPU (\d+) is reached more than once", message)
  if match:
    gpu = match.group(1)
    details["failure_category"] = "duplicate_destination"
    details["failure_location"] = f"GPU {gpu}"
    details["failure_feedback"] = (
      f"GPU {gpu} appears as a destination in more than one transmission. "
      "Each GPU should be reached exactly once in the single-root broadcast tree."
    )
    return details

  match = re.search(r"does not cover all GPUs; missing (.+)$", message)
  if match:
    missing = match.group(1)
    details["failure_category"] = "incomplete_coverage"
    details["failure_feedback"] = (
      f"The sketch leaves these GPUs unreachable: {missing}. Add transmissions "
      "from already-reached sources so every GPU 0..31 is covered exactly once."
    )
    return details

  match = re.search(r"layer (\d+) group (\d+) cannot connect GPUs (.+)$", message)
  if match:
    layer, group, gpus = match.groups()
    details["failure_category"] = "topology_group_error"
    details["failure_location"] = f"layer {layer}, group {group}"
    details["failure_feedback"] = (
      f"Layer {layer} group {group} cannot contain GPUs {gpus}. Use only "
      "the layer/group memberships described in the task prompt generated "
      "from the active SyCCL config."
    )
    return details

  if "must not be empty" in lower:
    details["failure_category"] = "dsl_schema_error"
    details["failure_feedback"] = (
      "A transmission has an empty source or destination set. Remove that "
      "transmission or provide at least one integer GPU id."
    )
    return details

  if "flat list" in lower or "must be a gpu id" in lower or "must be an integer" in lower:
    details["failure_category"] = "dsl_schema_error"
    details["failure_feedback"] = (
      "Use either one integer GPU id or a flat list of integer GPU ids for "
      "srcs/dsts. Nested lists such as [[10, 11], [12, 13]] are not valid."
    )
    return details

  return details


def _error_result(message: str, **extra: Any) -> dict[str, Any]:
  result = {
    "combined_score": FAILURE_SCORE,
    "validity": 0.0,
    "bottleneck_profile": None,
    "error": message,
  }
  result.update(_failure_details(message))
  if result.get("failure_category") in {"flow_sim_config_error", "flow_sim_output_error"}:
    result["simpletes_fatal"] = True
  result.update(extra)
  return result


def _fatal_flow_sim_error(message: str) -> FatalFlowSimError:
  metrics = _error_result(message)
  return FatalFlowSimError(json.dumps(metrics, ensure_ascii=True, sort_keys=True))


def _preview(text: str, limit: int = LOG_PREVIEW_CHARS) -> str:
  if len(text) <= limit:
    return text
  return text[-limit:]


def _load_program(program_path: str):
  spec = importlib.util.spec_from_file_location("candidate_syccl_sketch", program_path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load program: {program_path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _write_flow_sim_inputs(
    raw_sketches: Any,
    workdir: Path,
) -> tuple[Path, Path, Path, Path, Path]:
  config_path = workdir / "candidate-config.json"
  input_dir = workdir / "flow-sim-inputs"
  output_dir = workdir / "flow-sim-runs"
  manifest_path = workdir / "flow-sim-manifest.json"
  summary_path = workdir / "flow-sim-summary.csv"

  with BASE_CONFIG.open("r", encoding="utf-8") as f:
    config = json.load(f)

  host_gpu_num = int(config.get("hosts", {}).get("host_gpu_num", 0))
  if host_gpu_num <= 0 or NGPUS % host_gpu_num != 0:
    raise ValueError(
        f"base config host_gpu_num={host_gpu_num} is incompatible with {NGPUS} GPUs"
    )
  config["hosts"]["host_num"] = NGPUS // host_gpu_num

  config.setdefault("sketch", {})
  config["sketch"]["customize_sketch"] = False
  config["sketch"]["use_sketch_input"] = True
  config["sketch"]["save_sketch"] = False
  config["sketch"]["sketch_path"] = str(input_dir)

  input_dir.mkdir(parents=True, exist_ok=True)
  output_dir.mkdir(parents=True, exist_ok=True)
  candidate_dir = input_dir / "candidate-000"
  candidate_dir.mkdir(parents=True, exist_ok=True)
  (candidate_dir / "candidate-sketch.json").write_text(
      json.dumps(raw_sketches, indent=2),
      encoding="utf-8",
  )
  config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
  return config_path, input_dir, output_dir, manifest_path, summary_path


def _run_flow_sim_batch_sketch(
    config_path: Path,
    input_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    summary_path: Path,
) -> tuple[int, float, str]:
  env = os.environ.copy()
  start = time.time()
  proc = subprocess.run(
      [
        str(FLOW_SIM_BIN),
        "batch-sketch",
        "--config",
        str(config_path),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--manifest",
        str(manifest_path),
        "--summary",
        str(summary_path),
      ],
      cwd=str(REPO_ROOT),
      env=env,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      timeout=TIMEOUT_SECONDS,
      check=False,
  )
  wall = time.time() - start
  return proc.returncode, wall, proc.stdout


def _float_field(value: Any) -> float | None:
  if isinstance(value, bool):
    return None
  if isinstance(value, (int, float)):
    return float(value)
  if isinstance(value, str):
    try:
      return float(value)
    except ValueError:
      return None
  return None


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
  if not path.exists():
    raise FlowSimOutputError(f"flow-sim {label} does not exist: {path}")
  try:
    payload = json.loads(path.read_text(encoding="utf-8"))
  except json.JSONDecodeError as exc:
    raise FlowSimOutputError(f"flow-sim {label} is not valid JSON at {path}: {exc}") from exc
  if not isinstance(payload, dict):
    raise FlowSimOutputError(f"flow-sim {label} must be a JSON object: {path}")
  return payload


def _validate_flow_sim_output(output: dict[str, Any], output_path: Path) -> None:
  time_us = _float_field(output.get("time_us"))
  if time_us is None or time_us <= 0:
    raise FlowSimOutputError(
        f"flow-sim output {output_path} has no positive time_us"
    )
  output["time_us"] = time_us

  if not isinstance(output.get("bottleneck_profile"), dict):
    raise FlowSimOutputError(
        f"flow-sim output {output_path} missing required bottleneck_profile object"
    )


def _read_best_flow_sim_case(manifest_path: Path) -> FlowSimCaseResult:
  manifest = _read_json_object(manifest_path, "manifest")
  cases = manifest.get("cases")
  if not isinstance(cases, list) or not cases:
    raise FlowSimOutputError(f"flow-sim manifest has no cases: {manifest_path}")

  best: FlowSimCaseResult | None = None
  for case in cases:
    if not isinstance(case, dict):
      raise FlowSimOutputError(f"flow-sim manifest contains a non-object case: {case!r}")
    output_raw = case.get("rust_output")
    if not isinstance(output_raw, str) or not output_raw:
      raise FlowSimOutputError(f"flow-sim manifest case missing rust_output: {case!r}")
    output_path = Path(output_raw)
    output = _read_json_object(output_path, "case output")
    _validate_flow_sim_output(output, output_path)
    selected = FlowSimCaseResult(case=case, output=output)
    if best is None or output["time_us"] < best.output["time_us"]:
      best = selected

  if best is None:
    raise FlowSimOutputError(f"flow-sim manifest produced no valid cases: {manifest_path}")
  return best


def _success_metrics(
    *,
    best: FlowSimCaseResult,
) -> dict[str, Any]:
  output = best.output
  time_us = float(output["time_us"])
  print(
      f"the best result is from sketch {best.case.get('name', '')} "
      f"with sketch index {best.case.get('sketch_index', 0)} and output file {best.output}"
  )
  return {
    "combined_score": COLL_BYTE / time_us,
    "validity": 1.0,

    "bottleneck_profile": output["bottleneck_profile"],
  }


def evaluate(program_path: str) -> dict[str, Any]:
  """Evaluate a generated SimpleTES program via flow-sim-rs batch-sketch."""
  workdir: Path | None = None
  try:
    if not BASE_CONFIG.exists():
      raise _fatal_flow_sim_error(f"missing base config: {BASE_CONFIG}")
    if not FLOW_SIM_BIN.exists():
      raise _fatal_flow_sim_error(f"missing flow-sim-rs binary: {FLOW_SIM_BIN}")

    workdir = _make_eval_workdir()
    _copy_program(program_path, workdir)

    module = _load_program(program_path)
    if not hasattr(module, "run_code"):
      return _write_metrics(workdir, _error_result("program must define run_code()"))

    raw = module.run_code()

    config_path, input_dir, output_dir, manifest_path, summary_path = _write_flow_sim_inputs(
        raw,
        workdir,
    )
    try:
      returncode, wall, log = _run_flow_sim_batch_sketch(
          config_path,
          input_dir,
          output_dir,
          manifest_path,
          summary_path,
      )
    except subprocess.TimeoutExpired:
      return _write_metrics(workdir, _error_result(
          f"flow-sim-rs batch-sketch timed out after {TIMEOUT_SECONDS}s",
      ))

    if returncode != 0:
      return _write_metrics(workdir, _error_result(
          f"flow-sim-rs batch-sketch exited with code {returncode}: {_preview(log)}",
      ))

    try:
      best = _read_best_flow_sim_case(manifest_path)
    except FlowSimOutputError as exc:
      raise _fatal_flow_sim_error(str(exc))

    capture_construction_if_requested(raw)
    metrics = _success_metrics(
        best=best,
    )
    return _write_metrics(workdir, metrics)

  except FatalFlowSimError as exc:
    if workdir is not None:
      try:
        payload = json.loads(str(exc))
      except json.JSONDecodeError:
        payload = _error_result(str(exc), simpletes_fatal=True)
      _write_metrics(workdir, payload)
    raise
  except Exception as exc:
    return _write_metrics(
        workdir,
        _error_result(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"),
    )
