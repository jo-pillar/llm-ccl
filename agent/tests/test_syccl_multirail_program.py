import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROGRAM_PATH = (
    ROOT
    / "datasets"
    / "syccl"
    / "scheme1_direct_events"
    / "multirail_program.py.py"
)


def load_program():
    spec = importlib.util.spec_from_file_location("multirail_program", PROGRAM_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def as_list(value):
    return value if isinstance(value, list) else [value]


def test_multirail_program_uses_layer1_fanout_then_per_rail_double_rings():
    module = load_program()

    sketch = module.construct_sketches(ngpus=512, root_gpu=0, gpus_per_host=8)

    assert sketch[0] == (0, 1, 0, 0, [1, 2, 3, 4, 5, 6, 7])
    assert {tx[1] for tx in sketch} == {1, 3}
    assert all(tx[1] in (1, 3) for tx in sketch)

    first_rail_step = [tx for tx in sketch if tx[0] == 1 and tx[1] == 3]
    assert (1, 3, 0, 0, 8) in first_rail_step
    assert (1, 3, 0, 0, 504) in first_rail_step
    assert (1, 3, 7, 7, 15) in first_rail_step
    assert (1, 3, 7, 7, 511) in first_rail_step

    reached_at = {0: -1}
    for step, _layer, _group, srcs, dsts in sketch:
        for src in as_list(srcs):
            assert src in reached_at
            assert reached_at[src] < step
        for dst in as_list(dsts):
            assert dst not in reached_at
            reached_at[dst] = step

    assert set(reached_at) == set(range(512))
