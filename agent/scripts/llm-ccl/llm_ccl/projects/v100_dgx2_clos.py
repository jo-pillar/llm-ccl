from pathlib import Path

from llm_ccl.models import ExperimentSpec, LayerShape, ScaleSpec, case_matrix


AGENT_ROOT = Path(__file__).resolve().parents[4]
TEMPLATE_ROOT = (
    AGENT_ROOT
    / "datasets"
    / "syccl"
    / "scheme1_direct_events"
    / "templates"
    / "v100_dgx2_clos"
)

SCALE_4 = ScaleSpec(
    "4hosts-64gpu",
    64,
    (
        LayerShape(1, 4, 16),
        LayerShape(2, 4, 1),
        LayerShape(3, 4, 1),
        LayerShape(4, 1, 4),
    ),
)
SIZES = (
    1024, 4096, 16384,
    65536, 262144, 1048576, 
    4194304, 16777216, 67108864,
    268435456, 1073741824, 4294967296,
)

PROJECT = ExperimentSpec(
    name="v100_dgx2_clos",
    topology_template=TEMPLATE_ROOT / "clos_topo.py",
    instruction_template=TEMPLATE_ROOT / "prompt_template.txt",
    initial_program=TEMPLATE_ROOT / "clos_program.py",
    cases=case_matrix((SCALE_4,), ("allgather",), SIZES),
)

