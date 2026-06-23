from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any


class FlowSimOutputError(RuntimeError):
    pass


class FlowSimRunner:
    def __init__(self, binary: str | Path, timeout_s: float | None = None) -> None:
        self.binary = str(binary)
        self.timeout_s = timeout_s

    def simulate_sketch(
        self,
        *,
        config_path: str | Path,
        sketch_path: str | Path,
        output_path: str | Path,
    ) -> dict[str, Any]:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.binary,
            "simulate-sketch",
            "--config",
            str(config_path),
            "--sketch",
            str(sketch_path),
            "--output",
            str(output),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
        )
        # 这里应该对flow-sim-rs的错误进行细分，必要的时候修改flow-sim-rs 源码
        if completed.returncode != 0:
            raise RuntimeError(
                f"flow-sim-rs simulate-sketch failed with {completed.returncode}: {completed.stderr or completed.stdout} with command: {' '.join(command)}"
            )
        if not output.exists():
            raise FlowSimOutputError(f"flow-sim-rs did not produce output file: {output}")
        try:
            result = json.loads(output.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise FlowSimOutputError(f"flow-sim-rs output is not valid JSON at {output}: {exc}") from exc
        return validate_flow_sim_result(result, output_path=output)


def validate_flow_sim_result(result: dict[str, Any], *, output_path: str | Path | None = None) -> dict[str, Any]:
    context = f" at {output_path}" if output_path is not None else ""
    if not isinstance(result, dict):
        raise FlowSimOutputError(f"flow-sim result{context} must be a JSON object")
    if "time_us" not in result:
        raise FlowSimOutputError(f"flow-sim result{context} missing required time_us")
    try:
        time_us = float(result["time_us"])
    except (TypeError, ValueError) as exc:
        raise FlowSimOutputError(f"flow-sim result{context} has non-numeric time_us: {result['time_us']!r}") from exc
    if not math.isfinite(time_us) or time_us <= 0:
        raise FlowSimOutputError(f"flow-sim result{context} has invalid time_us: {result['time_us']!r}")
    result["time_us"] = time_us

    flow_count = result.get("flow_count", 0)
    try:
        flow_count_int = int(flow_count)
    except (TypeError, ValueError) as exc:
        raise FlowSimOutputError(f"flow-sim result{context} has non-integer flow_count: {flow_count!r}") from exc
    if flow_count_int < 0:
        raise FlowSimOutputError(f"flow-sim result{context} has negative flow_count: {flow_count!r}")
    result["flow_count"] = flow_count_int

    profile = result.get("bottleneck_profile")
    if not isinstance(profile, dict):
        links = result.get("links")
        link_note = "; legacy links were present but cannot be mapped to sketch transmissions" if links else ""
        raise FlowSimOutputError(f"flow-sim result{context} missing required bottleneck_profile{link_note}")
    return result
