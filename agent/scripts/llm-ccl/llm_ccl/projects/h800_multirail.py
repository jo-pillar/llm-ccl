from pathlib import Path

from llm_ccl.models import ExperimentSpec, LayerShape, ScaleSpec, case_matrix


AGENT_ROOT = Path(__file__).resolve().parents[4]
DATASET_ROOT = AGENT_ROOT / "datasets" / "syccl" / "scheme1_direct_events"
TEMPLATE_ROOT = DATASET_ROOT / "templates" / "H800_multirail"

SCALE_32 = ScaleSpec(
    "32hosts-256gpu",
    256,
    (LayerShape(1, 32, 8), LayerShape(2, 256, 1), LayerShape(3, 8, 32)),
)
SCALE_64 = ScaleSpec(
    "64hosts-512gpu",
    512,
    (LayerShape(1, 64, 8), LayerShape(2, 512, 1), LayerShape(3, 8, 64)),
)
SCALE_128 = ScaleSpec(
    "128hosts-1024gpu",
    1024,
    (LayerShape(1, 128, 8), LayerShape(2, 1024, 1), LayerShape(3, 8, 128)),
)
SCALE_256 = ScaleSpec(
    "256hosts-2048gpu",
    2048,
    (LayerShape(1, 256, 8), LayerShape(2, 2048, 1), LayerShape(3, 8, 256)),
)

SIZES_32 = (
    1024, 4096, 16384,
    65536, 262144, 1048576, 
    4194304, 16777216, 67108864,
    268435456, 1073741824, 4294967296,
)

SIZES_SCALE= ( 16777216,)


PROJECT = ExperimentSpec(
    name="h800_multirail",
    topology_template=TEMPLATE_ROOT / "multirail_topo.py",
    instruction_template=DATASET_ROOT / "prompt_templete.txt",
    initial_program=TEMPLATE_ROOT / "multirail_program.py",
    cases=(
        *case_matrix((SCALE_32,), ("allgather", "alltoall"), SIZES_32),
        *case_matrix((SCALE_128,), ("allgather",), SIZES_SCALE),
        *case_matrix((SCALE_256,), ("allgather",), SIZES_SCALE),
    ),
)

