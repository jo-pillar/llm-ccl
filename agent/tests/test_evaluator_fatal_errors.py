import asyncio
import importlib.util
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_MISSING = object()
_PATCHED_MODULE_NAMES = [
    "simpletes",
    "simpletes.construction",
    "simpletes.evaluator",
    "simpletes.generator",
    "simpletes.node",
    "simpletes.utils",
    "simpletes.utils.text",
    "simpletes.engine.runtime",
]
_SAVED_MODULES = {
    name: sys.modules.get(name, _MISSING)
    for name in _PATCHED_MODULE_NAMES
}

simpletes_pkg = types.ModuleType("simpletes")
simpletes_pkg.__path__ = [str(ROOT / "simpletes")]
sys.modules.setdefault("simpletes", simpletes_pkg)

SPEC = importlib.util.spec_from_file_location(
    "simpletes.evaluator",
    ROOT / "simpletes" / "evaluator.py",
)
assert SPEC is not None and SPEC.loader is not None
EVALUATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EVALUATOR
SPEC.loader.exec_module(EVALUATOR)
EvaluatorWorker = EVALUATOR.EvaluatorWorker

generator_stub = types.ModuleType("simpletes.generator")


@dataclass
class GenerationResult:
    code: str | None = None
    llm_input: str | None = None
    llm_output: str | None = None
    token_usage: dict | None = None


generator_stub.GenerationResult = GenerationResult
sys.modules["simpletes.generator"] = generator_stub

node_stub = types.ModuleType("simpletes.node")


@dataclass
class Node:
    id: str
    code: str | None = None
    parent_ids: list | None = None
    gen_id: int | None = None
    chain_idx: int | None = None
    shared_construction_id: str | None = None
    status: object | None = None
    llm_input: str | None = None
    llm_output: str | None = None
    token_usage: dict | None = None


class Status:
    EVAL_PENDING = object()


node_stub.Node = Node
node_stub.Status = Status
sys.modules["simpletes.node"] = node_stub

RUNTIME_SPEC = importlib.util.spec_from_file_location(
    "simpletes.engine.runtime",
    ROOT / "simpletes" / "engine" / "runtime.py",
)
assert RUNTIME_SPEC is not None and RUNTIME_SPEC.loader is not None
RUNTIME = importlib.util.module_from_spec(RUNTIME_SPEC)
sys.modules[RUNTIME_SPEC.name] = RUNTIME
RUNTIME_SPEC.loader.exec_module(RUNTIME)

for _name, _module in _SAVED_MODULES.items():
    if _module is _MISSING:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _module


def _clean_subprocess_env(_capture_path: str, _shared_construction_path: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return env


def test_evaluator_worker_raises_on_fatal_metrics(tmp_path: Path):
    evaluator_path = tmp_path / "fatal_evaluator.py"
    evaluator_path.write_text(
        "\n".join([
            "def evaluate(_path):",
            "    return {",
            "        'combined_score': -1.0,",
            "        'error': 'flow-sim output missing time_us',",
            "        'simpletes_fatal': True,",
            "    }",
        ]),
        encoding="utf-8",
    )
    worker = EvaluatorWorker(str(evaluator_path))

    with patch.object(worker, "_subprocess_env", _clean_subprocess_env):
        try:
            asyncio.run(worker.evaluate("print('candidate')"))
        except EVALUATOR.FatalEvaluatorError as exc:
            message = str(exc)
        else:
            raise AssertionError("expected FatalEvaluatorError")

    assert "Fatal evaluator error" in message
    assert "flow-sim output missing time_us" in message


def test_evaluator_runner_marks_fatal_exceptions_fatal(tmp_path: Path):
    evaluator_path = tmp_path / "raises_evaluator.py"
    evaluator_path.write_text(
        "\n".join([
            "class FatalFlowSimError(RuntimeError):",
            "    simpletes_fatal = True",
            "",
            "def evaluate(_path):",
            "    raise FatalFlowSimError('flow-sim output missing time_us')",
        ]),
        encoding="utf-8",
    )
    worker = EvaluatorWorker(str(evaluator_path))

    with patch.object(worker, "_subprocess_env", _clean_subprocess_env):
        try:
            asyncio.run(worker.evaluate("print('candidate')"))
        except EVALUATOR.FatalEvaluatorError as exc:
            message = str(exc)
        else:
            raise AssertionError("expected FatalEvaluatorError")

    assert "Fatal evaluator error" in message
    assert "flow-sim output missing time_us" in message


def test_evaluator_worker_keeps_nonfatal_metrics(tmp_path: Path):
    evaluator_path = tmp_path / "nonfatal_evaluator.py"
    expected = {
        "combined_score": -1.0,
        "error": "invalid sketch",
        "simpletes_fatal": False,
    }
    evaluator_path.write_text(
        "\n".join([
            "def evaluate(_path):",
            f"    return {expected!r}",
        ]),
        encoding="utf-8",
    )
    worker = EvaluatorWorker(str(evaluator_path))

    with patch.object(worker, "_subprocess_env", _clean_subprocess_env):
        outcome = asyncio.run(worker.evaluate("print('candidate')"))

    assert outcome.metrics == expected


def test_local_runtime_surfaces_fatal_worker_errors():
    fatal = EVALUATOR.FatalEvaluatorError("fatal flow-sim output")

    assert RUNTIME._first_fatal_worker_error([asyncio.CancelledError(), fatal]) is fatal
