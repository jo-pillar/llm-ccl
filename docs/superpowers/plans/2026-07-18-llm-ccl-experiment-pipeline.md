# LLM-CCL Experiment Pipeline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new, resumable `agent/scripts/llm-ccl/` workflow that prepares topology-scaled cases, runs one `all`-strategy LLM-CCL search per case, selects the fastest valid candidate, exports its translated schedule with FlowSim, runs SyCCL resim, and reports results without invoking SyCCL solve.

**Architecture:** Keep the four legacy preparation scripts unchanged. Put the new user-facing CLI and an importable `llm_ccl` package under `agent/scripts/llm-ccl/`. Project modules declare only templates, scale shapes, collectives, and total message sizes; shared modules own topology rendering, config/instruction/init generation, immutable search attempts, selection, resim, manifests, and reports.

**Tech Stack:** Python 3.12 standard library, existing `syccl_agents.topodsl` and `syccl_agents.config_render`, SimpleTES/LLM-CCL `main.py`, dataset-local `flow-sim-rs`, SyCCL `synthesize resim`, `unittest`.

**Design spec:** `docs/superpowers/specs/2026-07-18-llm-ccl-experiment-pipeline-design.md`

---

## File Structure

### New implementation

- `agent/scripts/llm-ccl/run.py`: the only user-facing CLI; parses subcommands and delegates to package modules.
- `agent/scripts/llm-ccl/README.md`: explains the workflow and how to add a project.
- `agent/scripts/llm-ccl/llm_ccl/models.py`: immutable project, scale, layer, and case declarations plus validation and matrix helpers.
- `agent/scripts/llm-ccl/llm_ccl/project_loader.py`: automatic project discovery with duplicate-name validation.
- `agent/scripts/llm-ccl/llm_ccl/topology.py`: AST-guided topology-instantiation rendering and TopoDSL/config validation.
- `agent/scripts/llm-ccl/llm_ccl/preparation.py`: bundle creation and case input materialization.
- `agent/scripts/llm-ccl/llm_ccl/manifest.py`: schema-v1 manifest loading, atomic writes, and stage/attempt transitions.
- `agent/scripts/llm-ccl/llm_ccl/search.py`: `all`-strategy command construction and immutable attempt execution.
- `agent/scripts/llm-ccl/llm_ccl/selection.py`: same-attempt artifact scanning and deterministic best-candidate selection.
- `agent/scripts/llm-ccl/llm_ccl/resim.py`: FlowSim translated-schedule export and SyCCL resim.
- `agent/scripts/llm-ccl/llm_ccl/reporting.py`: full-bundle JSON/CSV summaries and status views.
- `agent/scripts/llm-ccl/llm_ccl/projects/h800_multirail.py`: migrated H800 case declarations.
- `agent/scripts/llm-ccl/llm_ccl/projects/v100_dgx2_clos.py`: migrated V100 case declarations.
- `agent/scripts/llm-ccl/tests/`: focused unit and fake-executable integration tests.

### Existing files changed

- `agent/datasets/syccl/scheme1_direct_events/templates/H800_multirail/multirail_topo.py`: capture the user's corrected flat-ID NVSwitch-star topology in the implementation history.
- `agent/datasets/syccl/scheme1_direct_events/templates/v100_dgx2_clos/clos_topo.py`: normalize host-local GPU identifiers to flat global IDs.
- `agent/datasets/syccl/scheme1_direct_events/templates/v100_dgx2_clos/prompt_template.txt`: remove duplicated stale numeric link descriptions; refer to the rendered topology as authority.
- `agent/examples/topologies/clos_topo.py`: keep the shared example on the same flat-ID NVSwitch-star semantics.
- `agent/tests/test_syccl_v100_dgx2_clos_experiment.py`: replace full-mesh connection invariants with star/NVSwitch invariants while preserving the legacy script's own generated link values.
- `agent/tests/test_topology_examples.py`: replace full-mesh example invariants with star/NVSwitch invariants.

The four legacy preparation scripts are not modified.

---

### Task 1: Lock the corrected star-topology template semantics

**Files:**
- Create: `agent/scripts/llm-ccl/tests/__init__.py`
- Create: `agent/scripts/llm-ccl/tests/test_topology_templates.py`
- Modify: `agent/datasets/syccl/scheme1_direct_events/templates/H800_multirail/multirail_topo.py:21-31`
- Modify: `agent/datasets/syccl/scheme1_direct_events/templates/v100_dgx2_clos/clos_topo.py:24-35`
- Modify: `agent/datasets/syccl/scheme1_direct_events/templates/v100_dgx2_clos/prompt_template.txt:14-23`
- Modify: `agent/examples/topologies/clos_topo.py:25-50`
- Modify: `agent/tests/test_syccl_v100_dgx2_clos_experiment.py:57-82`
- Modify: `agent/tests/test_topology_examples.py:13-25`

- [ ] **Step 1: Write topology-template regression tests**

```python
from __future__ import annotations

import re
import runpy
import unittest
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parents[3]
H800_TEMPLATE_DIR = AGENT_ROOT / "datasets" / "syccl" / "scheme1_direct_events" / "templates" / "H800_multirail"
TEMPLATE_DIR = AGENT_ROOT / "datasets" / "syccl" / "scheme1_direct_events" / "templates" / "v100_dgx2_clos"


class TopologyTemplateTest(unittest.TestCase):
  def test_h800_host_local_links_form_a_flat_id_nvswitch_star(self):
    topology = runpy.run_path(H800_TEMPLATE_DIR / "multirail_topo.py")["topology"]
    edges = {(src.node_id, dst.node_id) for src, dst in topology.connections}

    self.assertIn(("gpu[0]", "nvswitch[0]"), edges)
    self.assertIn(("gpu[511]", "nvswitch[63]"), edges)
    self.assertFalse(any(src.startswith("gpu[") and dst.startswith("gpu[") for src, dst in edges))
    self.assertFalse(any(re.fullmatch(r"gpu\[\d+\]\[\d+\]", endpoint) for edge in edges for endpoint in edge))

  def test_v100_host_local_gpu_ids_are_flat_and_links_are_authoritative(self):
    topology = runpy.run_path(TEMPLATE_DIR / "clos_topo.py")["topology"]
    links = {
        (src.node_id, dst.node_id): (spec.bandwidth, spec.latency)
        for (src, dst), spec in topology.connections.items()
    }

    self.assertEqual(("150GB/s", "3us"), links[("gpu[0]", "nvswitch[0]")])
    self.assertEqual(("150GB/s", "3us"), links[("gpu[63]", "nvswitch[3]")])
    self.assertEqual(("12.5GB/s", "0us"), links[("gpu[63]", "nic[3]")])
    self.assertFalse(any(re.fullmatch(r"gpu\[\d+\]\[\d+\]", endpoint) for edge in links for endpoint in edge))

  def test_v100_prompt_does_not_duplicate_numeric_link_constants(self):
    prompt = (TEMPLATE_DIR / "prompt_template.txt").read_text(encoding="utf-8")
    self.assertNotIn("125 GB/s", prompt)
    self.assertNotIn("400 GB/s", prompt)
    self.assertIn("Topology and link costs", prompt)


if __name__ == "__main__":
  unittest.main()
```

- [ ] **Step 2: Run the new tests and verify the flat-ID test fails**

Run from `agent/`:

```bash
.venv/bin/python -m unittest -v scripts/llm-ccl/tests/test_topology_templates.py
```

Expected: FAIL because the current V100 NVSwitch edge uses `gpu[0][0]`.

- [ ] **Step 3: Capture the corrected templates and update the V100 prompt**

Keep the user's H800 `gpu[global_id] -> nvswitch[host_id]` implementation and include it in the Task 1 commit so a clean checkout does not revert to the old full mesh.

Change the host-local source node to:

```python
src_node = Node(
    node_id=f"gpu[{host_id * gpu_per_host + gpu_i}]",
    node_type=NodeType.GPU,
)
```

Replace the prompt's numeric interpretation bullets with topology-driven wording:

```text
- Layer 1 is the DGX-2 host-local NVSwitch fabric described in the topology above.
- Layers 2 and 3 describe the host NIC and leaf path; use their rendered link costs.
- Layer 4 crosses the Clos spine; use its rendered bandwidth and latency.
```

In `agent/examples/topologies/clos_topo.py`, keep the star structure but normalize both host-local and host-attachment GPU nodes to `gpu[{host_id * gpu_per_host + gpu_i}]` so the example does not retain a second identifier convention.

- [ ] **Step 4: Update legacy regression tests to assert star structure**

For `test_dgx2_topology_scales_to_four_and_eight_host_clos`, change expected connection counts to `(136, 272)` and assert the generated legacy topology contains:

```python
self.assertEqual(
    ("125GB/s", "3us"),
    links[(f"gpu[{last_host_first_gpu}]", f"nvswitch[{scale.hosts - 1}]")],
)
```

Keep the legacy script's `12.5GB/s`, `25us`, and `400GB/s`, `25us` assertions because that script remains unchanged.

For the example topology, assert:

```python
assert len(connections) == 138
assert _count_edges(connections, "gpu[", "nvswitch[") == 64
assert _count_edges(connections, "gpu[", "gpu[") == 0
assert not any(re.fullmatch(r"gpu\[\d+\]\[\d+\]", node.node_id) for edge in connections for node in edge)
```

- [ ] **Step 5: Run targeted topology tests**

```bash
.venv/bin/python -m unittest -v scripts/llm-ccl/tests/test_topology_templates.py
.venv/bin/python -m unittest -v \
  tests.test_syccl_v100_dgx2_clos_experiment.SycclV100ClosExperimentTest.test_dgx2_topology_scales_to_four_and_eight_host_clos
.venv/bin/python -c 'from tests.test_topology_examples import test_clos_topo_example_builds_expected_connections; test_clos_topo_example_builds_expected_connections()'
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add \
  agent/datasets/syccl/scheme1_direct_events/templates/H800_multirail/multirail_topo.py \
  agent/datasets/syccl/scheme1_direct_events/templates/v100_dgx2_clos/clos_topo.py \
  agent/datasets/syccl/scheme1_direct_events/templates/v100_dgx2_clos/prompt_template.txt \
  agent/examples/topologies/clos_topo.py \
  agent/tests/test_syccl_v100_dgx2_clos_experiment.py \
  agent/tests/test_topology_examples.py \
  agent/scripts/llm-ccl/tests
git commit -m "fix: model host-local links through nvswitch"
```

---

### Task 2: Add validated project models and case-matrix helpers

**Files:**
- Create: `agent/scripts/llm-ccl/llm_ccl/__init__.py`
- Create: `agent/scripts/llm-ccl/llm_ccl/models.py`
- Create: `agent/scripts/llm-ccl/tests/test_models.py`

- [ ] **Step 1: Write failing model tests**

```python
from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT))

from llm_ccl.models import CaseSpec, ExperimentSpec, LayerShape, ScaleSpec, case_matrix


class ModelTest(unittest.TestCase):
  def test_case_matrix_builds_path_safe_divisible_cases(self):
    scale = ScaleSpec(
        name="4hosts-64gpu",
        gpu_count=64,
        layers=(LayerShape(1, 4, 16), LayerShape(2, 4, 1)),
    )
    cases = case_matrix(
        scales=(scale,),
        collectives=("allgather",),
        total_message_sizes=(65536, 262144),
    )
    self.assertEqual(
        ["4hosts-64gpu-allgather-65536B", "4hosts-64gpu-allgather-262144B"],
        [case.case_id for case in cases],
    )
    self.assertEqual(1024, cases[0].coll_byte)

  def test_scale_rejects_duplicate_layer_ids(self):
    with self.assertRaisesRegex(ValueError, "duplicate layer_id"):
      ScaleSpec("bad", 8, (LayerShape(1, 1, 8), LayerShape(1, 1, 8)))

  def test_layer_zero_is_supported_for_existing_topodsl_templates(self):
    self.assertEqual(0, LayerShape(0, 1, 8).layer_id)

  def test_case_rejects_non_divisible_total_message_size(self):
    scale = ScaleSpec("8gpu", 8, (LayerShape(1, 1, 8),))
    with self.assertRaisesRegex(ValueError, "divisible"):
      CaseSpec("bad", scale, "allgather", 10)

  def test_experiment_rejects_duplicate_case_ids(self):
    scale = ScaleSpec("8gpu", 8, (LayerShape(1, 1, 8),))
    case = CaseSpec("same", scale, "allgather", 64)
    with self.assertRaisesRegex(ValueError, "duplicate case_id"):
      ExperimentSpec("demo", Path("topo.py"), Path("prompt.txt"), Path("init.py"), (case, case))
```

- [ ] **Step 2: Run tests and verify import failure**

```bash
.venv/bin/python -m unittest -v scripts/llm-ccl/tests/test_models.py
```

Expected: ERROR because `llm_ccl.models` does not exist.

- [ ] **Step 3: Implement the immutable models**

`models.py` must provide:

```python
_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._=-]*$")
_COLLECTIVES = {"allgather", "alltoall", "allreduce", "broadcast"}


@dataclass(frozen=True)
class LayerShape:
  layer_id: int
  group_num: int
  node_num: int

  def __post_init__(self) -> None:
    if self.layer_id < 0 or self.group_num <= 0 or self.node_num <= 0:
      raise ValueError("layer_id must be non-negative; group_num and node_num must be positive")


@dataclass(frozen=True)
class ScaleSpec:
  name: str
  gpu_count: int
  layers: tuple[LayerShape, ...]

  def __post_init__(self) -> None:
    _validate_slug(self.name, "scale name")
    if self.gpu_count <= 0:
      raise ValueError("gpu_count must be positive")
    ids = [layer.layer_id for layer in self.layers]
    if len(ids) != len(set(ids)):
      raise ValueError("duplicate layer_id in scale")


@dataclass(frozen=True)
class CaseSpec:
  case_id: str
  scale: ScaleSpec
  collective: str
  total_message_size: int

  def __post_init__(self) -> None:
    _validate_slug(self.case_id, "case_id")
    if self.collective not in _COLLECTIVES:
      raise ValueError(f"unsupported collective: {self.collective}")
    if self.total_message_size <= 0 or self.total_message_size % self.scale.gpu_count != 0:
      raise ValueError("total_message_size must be positive and divisible by gpu_count")

  @property
  def coll_byte(self) -> int:
    return self.total_message_size // self.scale.gpu_count
```

Implement `ExperimentSpec` path normalization and duplicate-case validation. Implement `case_matrix()` with deterministic scale/collective/size iteration.

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m unittest -v scripts/llm-ccl/tests/test_models.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/scripts/llm-ccl/llm_ccl agent/scripts/llm-ccl/tests/test_models.py
git commit -m "feat: add llm-ccl experiment models"
```

---

### Task 3: Add automatic project discovery and initial project declarations

**Files:**
- Create: `agent/scripts/llm-ccl/llm_ccl/project_loader.py`
- Create: `agent/scripts/llm-ccl/llm_ccl/projects/__init__.py`
- Create: `agent/scripts/llm-ccl/llm_ccl/projects/h800_multirail.py`
- Create: `agent/scripts/llm-ccl/llm_ccl/projects/v100_dgx2_clos.py`
- Create: `agent/scripts/llm-ccl/tests/test_projects.py`

- [ ] **Step 1: Write failing discovery and matrix tests**

```python
class ProjectTest(unittest.TestCase):
  def test_discovers_h800_and_v100_projects(self):
    projects = discover_projects()
    self.assertEqual({"h800_multirail", "v100_dgx2_clos"}, set(projects))

  def test_h800_project_preserves_existing_case_matrix(self):
    project = discover_projects()["h800_multirail"]
    self.assertEqual(35, len(project.cases))
    self.assertEqual(
        {"32hosts-256gpu", "64hosts-512gpu", "512hosts-4096gpu"},
        {case.scale.name for case in project.cases},
    )
    self.assertEqual({"allgather", "alltoall"}, {case.collective for case in project.cases})

  def test_v100_project_has_twenty_four_allgather_cases(self):
    project = discover_projects()["v100_dgx2_clos"]
    self.assertEqual(24, len(project.cases))
    self.assertEqual({"allgather"}, {case.collective for case in project.cases})
```

The H800 total is `24 + 8 + 3 = 35` cases. V100 is `2 scales * 12 sizes = 24` cases.

- [ ] **Step 2: Run tests and verify discovery fails**

```bash
.venv/bin/python -m unittest -v scripts/llm-ccl/tests/test_projects.py
```

Expected: ERROR because project discovery does not exist.

- [ ] **Step 3: Implement automatic discovery**

```python
def discover_projects(package_name: str = "llm_ccl.projects") -> dict[str, ExperimentSpec]:
  package = importlib.import_module(package_name)
  projects: dict[str, ExperimentSpec] = {}
  for module_info in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
    module = importlib.import_module(module_info.name)
    project = getattr(module, "PROJECT", None)
    if not isinstance(project, ExperimentSpec):
      raise ValueError(f"{module_info.name} must export PROJECT: ExperimentSpec")
    if project.name in projects:
      raise ValueError(f"duplicate project name: {project.name}")
    projects[project.name] = project
  return dict(sorted(projects.items()))
```

- [ ] **Step 4: Implement the H800 and V100 project modules**

Use `AGENT_ROOT = Path(__file__).resolve().parents[4]`. Declare the exact scales and message-size lists from the design spec. Construct H800 cases by concatenating three `case_matrix()` calls so the 32-, 64-, and 512-host sweeps remain explicit. Export exactly one `PROJECT` per module.

- [ ] **Step 5: Run tests**

```bash
.venv/bin/python -m unittest -v scripts/llm-ccl/tests/test_projects.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add agent/scripts/llm-ccl/llm_ccl/project_loader.py \
  agent/scripts/llm-ccl/llm_ccl/projects \
  agent/scripts/llm-ccl/tests/test_projects.py
git commit -m "feat: declare h800 and v100 llm-ccl projects"
```

---

### Task 4: Render scaled TopoDSL sources without changing link constants

**Files:**
- Create: `agent/scripts/llm-ccl/llm_ccl/topology.py`
- Create: `agent/scripts/llm-ccl/tests/test_topology_rendering.py`

- [ ] **Step 1: Write failing renderer tests**

Tests must render the first V100 8-host case and assert:

```python
rendered = render_case_topology(case, project.topology_template)
self.assertIn("128,", rendered.source)
self.assertIn("group_num=8, node_num=16", rendered.source)
self.assertIn('LinkSpec("150GB/s", "3us")', rendered.source)
self.assertIn('LinkSpec("12.5GB/s", "3us")', rendered.source)
self.assertIn('LinkSpec("100GB/s", "0.5us")', rendered.source)
self.assertEqual(128, rendered.gpu_count)
self.assertEqual(case.coll_byte, rendered.topo.params.message_size)
```

Add tests that reject a missing scale layer and prove a comment before the topology assignment remains byte-for-byte present.

- [ ] **Step 2: Run tests and verify import failure**

```bash
.venv/bin/python -m unittest -v scripts/llm-ccl/tests/test_topology_rendering.py
```

Expected: ERROR because `llm_ccl.topology` does not exist.

- [ ] **Step 3: Implement AST-guided assignment splicing**

Provide:

```python
@dataclass(frozen=True)
class RenderedTopology:
  source: str
  topo: TopoDSLSpec
  gpu_count: int
  total_message_size: int
  coll_byte: int


def render_case_topology(case: CaseSpec, template_path: Path) -> RenderedTopology:
  ...
```

Implementation requirements:

1. Parse with `ast.parse`.
2. Find one `topology = Constructor(...)` assignment.
3. Read each `LayerSpec` keyword's layer ID, `LinkSpec` source segment, and `node_type` source segment.
4. Require template layer IDs to equal the scale's layer IDs.
5. Rebuild only the assignment using original constructor, keyword names, layer IDs, link expressions, and node types plus new dimensions.
6. Splice by AST line/column offsets; do not unparse the whole module.
7. Write a temporary source, load it through `load_topodsl`, and delete the temporary file.
8. Load the unscaled template once and validate that the rendered topology preserves its family and link values.
9. Validate `params.message_size == case.coll_byte`, collective, GPU count, and every layer-derived dimension.

Use an explicit collective mapping:

```python
_COLLECTIVE_ENUM = {
    "allgather": "CollectiveType.ALLGATHER",
    "alltoall": "CollectiveType.ALLTOALL",
    "allreduce": "CollectiveType.ALLREDUCE",
    "broadcast": "CollectiveType.BROADCAST",
}
```

- [ ] **Step 4: Run renderer tests**

```bash
.venv/bin/python -m unittest -v scripts/llm-ccl/tests/test_topology_rendering.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/scripts/llm-ccl/llm_ccl/topology.py \
  agent/scripts/llm-ccl/tests/test_topology_rendering.py
git commit -m "feat: render scaled topology templates"
```

---

### Task 5: Add schema-v1 manifests and bundle preparation

**Files:**
- Create: `agent/scripts/llm-ccl/llm_ccl/manifest.py`
- Create: `agent/scripts/llm-ccl/llm_ccl/preparation.py`
- Create: `agent/scripts/llm-ccl/tests/test_preparation.py`

- [ ] **Step 1: Write failing preparation tests**

Use a temporary bundle root and prepare one filtered V100 case. Assert:

```python
bundle = prepare_bundle(
    project,
    bundle_root=tmp_path,
    launch_id="unit",
    case_ids={"4hosts-64gpu-allgather-65536B"},
)
manifest = json.loads((bundle / "manifest.json").read_text())
self.assertEqual(1, manifest["schema_version"])
self.assertEqual("v100_dgx2_clos", manifest["project"])
self.assertEqual(65536, manifest["cases"][case_id]["total_message_size"])
self.assertEqual(1024, manifest["cases"][case_id]["coll_byte"])
self.assertEqual("pending", manifest["cases"][case_id]["stages"]["search"]["status"])
```

Assert the case directory contains `topodsl.py`, `config.json`, `instruction.txt`, and `init_program.py`; the config contains `coll.byte == 1024` and the V100 link values `0.15/3`, `0.0125/3`, `0.1/0.5`. Assert preparing into an existing launch directory raises `FileExistsError`.

Add a manifest-locking regression test with two worker processes, each using its own `ManifestStore` instance to increment a manifest counter repeatedly; the final value must equal the sum of both workers, proving the file lock prevents lost updates beyond one Python object.

Assert the schema-v1 manifest includes creation/update timestamps, project source paths and SHA256 hashes, case input paths and hashes, both message-size representations, redacted resolved configuration, and complete pending stage records with timestamp/error/provenance fields initialized to `None`.

- [ ] **Step 2: Run tests and verify failure**

```bash
.venv/bin/python -m unittest -v scripts/llm-ccl/tests/test_preparation.py
```

Expected: ERROR because preparation modules do not exist.

- [ ] **Step 3: Implement atomic manifest storage**

`ManifestStore` must expose:

```python
class ManifestStore:
  def __init__(self, path: Path): ...
  def load(self) -> dict[str, Any]: ...
  def create(self, payload: dict[str, Any]) -> None: ...
  def update(self, mutator: Callable[[dict[str, Any]], None]) -> dict[str, Any]: ...
```

Write JSON to `manifest.json.tmp`, flush and `os.fsync`, then `os.replace`. Reject a missing schema or unsupported schema version on load.
Guard every read-modify-write operation with one process-local `threading.RLock` shared by the stage coordinator and an OS file lock on `manifest.lock`; atomic replacement alone does not prevent lost updates when `--jobs > 1` or two recovery commands overlap.

Use this schema boundary:

```text
manifest
  schema_version, project, launch_id, created_at, updated_at
  sources.{topology,instruction,initial_program}.{path,sha256}
  resolved_config                         # no secrets
  cases.CASE_ID
    scale, collective, total_message_size, coll_byte
    inputs.{topodsl,config,instruction,init_program}.{path,sha256}
    stages.search
      status, latest_attempt, latest_successful_attempt
      attempts[]                          # number/status/times/wall/exit/error/paths/redacted config
    stages.selection
      status, selected_attempt, score, times, error
    stages.resim
      status, times, wall_time_s, flow_exit_status, syccl_exit_status
      flow_time_us, syccl_time_us, binaries, error
```

Every stage/attempt record uses UTC ISO-8601 timestamps, `error_category`, and a bounded `error_message`. Binary records contain resolved path and SHA256 when readable. Attempt records contain the redacted resolved model/API configuration and command-record path; API keys are represented only by `api_key_present: bool`.

- [ ] **Step 4: Implement instruction and initial-program rendering**

Instruction values must include:

```python
{
  "GPU_NUM": str(case.scale.gpu_count),
  "Collective": case.collective,
  "COLLECTIVE": case.collective,
  "TOPOLOGY": rendered.topo.prompt_source,
  "TOPOLOGY_FAMILY": rendered.topo.params.family,
  "MESSAGE_SIZE": str(case.coll_byte),
  "TOTAL_MESSAGE_SIZE": str(case.total_message_size),
  "HOST_NUM": str(rendered.topo.params.hosts),
  "HOST_GPU_NUM": str(rendered.topo.params.gpus_per_host),
  "NIC_NUM": str(rendered.topo.params.nics_per_host),
  "TOPODSL": rendered.topo.prompt_source,
}
```

Extract only the `EVOLVE-BLOCK` from the initial program and generate:

```python
GPU_NUM = <case GPU count>

# EVOLVE-BLOCK-START
<block>
# EVOLVE-BLOCK-END

def run_code():
  return construct_sketches(GPU_NUM)
```

Document and validate that project initial programs accept GPU count as their first parameter.

- [ ] **Step 5: Implement `prepare_bundle`**

Create the destination `bundle_root / project.name / launch_id`, reject an existing destination, validate requested case IDs, render all cases before exposing runnable paths, write files, hashes, and pending stage records, then atomically create the manifest.

- [ ] **Step 6: Run preparation tests**

```bash
.venv/bin/python -m unittest -v scripts/llm-ccl/tests/test_preparation.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add agent/scripts/llm-ccl/llm_ccl/manifest.py \
  agent/scripts/llm-ccl/llm_ccl/preparation.py \
  agent/scripts/llm-ccl/tests/test_preparation.py
git commit -m "feat: prepare resumable llm-ccl bundles"
```

---

### Task 6: Run one `all`-strategy search per case with immutable attempts

**Files:**
- Create: `agent/scripts/llm-ccl/llm_ccl/search.py`
- Create: `agent/scripts/llm-ccl/tests/test_search.py`

- [ ] **Step 1: Write failing search-attempt tests**

Test a prepared one-case bundle with an injected fake command runner. Assert the built command contains:

```python
self.assertIn("--selector", command)
self.assertIn("llm_elite", command)
self.assertIn("--elite-selection-strategy", command)
self.assertIn("all", command)
self.assertNotIn("linear_rank", command)
self.assertNotIn("balance", command)
```

Assert the first execution uses `search/attempt-0001`, a failed retry uses `attempt-0002`, a successful attempt is skipped without force, and `force=True` creates the next attempt without deleting earlier artifacts. Seed a successful selection/resim state before a forced successful search and assert both downstream stages become pending while the new attempt becomes `latest_successful_attempt`.

Add an out-of-order completion test: allocate attempts 1 and 2 for the same case, finish attempt 2 successfully first, then finish attempt 1. Assert `latest_attempt == 2`, `latest_successful_attempt == 2`, search remains succeeded, and the older completion neither invalidates the attempt-2 selection nor overwrites attempt-2 status/error fields.

- [ ] **Step 2: Run tests and verify failure**

```bash
.venv/bin/python -m unittest -v scripts/llm-ccl/tests/test_search.py
```

Expected: ERROR because `llm_ccl.search` does not exist.

- [ ] **Step 3: Implement model config, preflight, and command construction**

Provide `SearchOptions` with model, API base/key, max tokens, generation counts, candidate count, evaluator/generation concurrency, LLM policy pool size, jobs, timeout, FlowSim path, and force.

Resolve values in this exact order:

```text
model:       --model > MODEL_NAME > env.toml:model                 # required
api_base:    --api-base > API_BASE > env.toml:api_base             # optional
api_key:     --api-key > API_KEY > OPENAI_API_KEY > env.toml:api_key
max_tokens:  --max-tokens > MAX_TOKENS > 32768
flow_sim:    --flow-sim-bin > FLOW_SIM_BIN >
             AGENT_ROOT/datasets/syccl/scheme1_direct_events/flow-sim-rs/target/release/flow-sim-rs
```

`--env-toml` defaults to `AGENT_ROOT / "env.toml"`; a missing file is allowed only when the required model resolves elsewhere. Parse environment numeric values with positive-integer validation. Explicit CLI values always win. Resolve relative CLI/environment paths against the invocation working directory; repository defaults are constructed from `AGENT_ROOT`/`REPO_ROOT` and therefore do not depend on the caller's directory.

Before allocating an attempt, require a non-empty model name, readable case-local inputs, the repository `main.py` and evaluator, an executable `uv`, and an executable FlowSim binary. Build this exact SimpleTES command with case-local inputs and outputs, adding `--api-base`, `--api-key`, and `--max-tokens` only when configured:

```python
command = [
    "uv", "run", "python", str(agent_root / "main.py"),
    "--init-program", str(case_dir / "init_program.py"),
    "--evaluator", str(agent_root / "datasets/syccl/scheme1_direct_events/evaluator.py"),
    "--instruction", str(case_dir / "instruction.txt"),
    "--selector", "llm_elite",
    "--elite-selection-strategy", "all",
    "--num-chains", "1",
    "--k-candidates", str(options.k_candidates),
    "--stream-k-candidates",
    "--max-generations", str(options.max_generations),
    "--eval-concurrency", str(options.eval_concurrency),
    "--gen-concurrency", str(options.gen_concurrency),
    "--init-eval-repeats", "1",
    "--llm-policy-pool-size", str(options.llm_policy_pool_size),
    "--output-path", str(attempt_dir / "checkpoints"),
    "--model", options.model,
    "--save-llm-io",
    "--disable-reflection",
    "--skip-preflight",
]
```

Run it with `cwd=agent_root`, the configured subprocess timeout, and these environment values:

```python
env["SYCCL_BASE_CONFIG"] = str(case_dir / "config.json")
env["SYCCL_EVAL_ARTIFACT_DIR"] = str(attempt_dir / "eval_artifacts")
env["FLOW_SIM_BIN"] = str(flow_sim_bin)
```

Write a redacted `command.json`; never persist API keys in the manifest, command record, or log.

- [ ] **Step 4: Implement immutable attempt state**

Before execution, atomically allocate `attempt_number = latest_attempt + 1`, append its `running` record, and update `latest_attempt` under the same manifest lock. On completion, update only that numbered attempt.

Recompute aggregate search state from attempt records instead of using completion order:

```python
succeeded = [a["number"] for a in attempts if a["status"] == "succeeded"]
case_search["latest_successful_attempt"] = max(succeeded, default=None)
case_search["status"] = (
    "succeeded" if succeeded
    else "running" if any(a["status"] == "running" for a in attempts)
    else "failed"
)
```

Invalidate selection/resim only when a success increases `latest_successful_attempt`; an older attempt finishing later cannot overwrite the pointer or invalidate artifacts selected from a newer attempt. The result of the current invocation still reports its own attempt failure even if an older success keeps aggregate search state usable. Use a `ThreadPoolExecutor(max_workers=jobs)` for independent selected cases and the locked shared `ManifestStore` for all transitions.

- [ ] **Step 5: Run search tests**

```bash
.venv/bin/python -m unittest -v scripts/llm-ccl/tests/test_search.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add agent/scripts/llm-ccl/llm_ccl/search.py \
  agent/scripts/llm-ccl/tests/test_search.py
git commit -m "feat: run isolated llm-ccl search attempts"
```

---

### Task 7: Select the fastest valid candidate from exactly one attempt

**Files:**
- Create: `agent/scripts/llm-ccl/llm_ccl/selection.py`
- Create: `agent/scripts/llm-ccl/tests/test_selection.py`

- [ ] **Step 1: Write failing selection tests**

Create two successful attempt trees with FlowSim manifests and candidate artifacts. Give attempt 1 a faster stale candidate and attempt 2 two candidates with times `12.0` and `9.0`. Assert default selection chooses `9.0` from attempt 2, not the stale `5.0` from attempt 1. Add invalid zero/NaN/malformed-JSON/missing-output candidates and a deterministic path tie. Assert a successful selection is skipped by default, `force=True` replaces the mutable `best/` result, and an explicit different successful attempt requires force. After force-selecting attempt 1, assert the manifest records attempt 1 as authoritative even though attempt 2 remains `latest_successful_attempt`; Task 8 verifies resim accepts it.

- [ ] **Step 2: Run tests and verify failure**

```bash
.venv/bin/python -m unittest -v scripts/llm-ccl/tests/test_selection.py
```

Expected: ERROR because `llm_ccl.selection` does not exist.

- [ ] **Step 3: Implement FlowSim time extraction and artifact scanning**

Scan only:

```text
search/attempt-NNNN/eval_artifacts/scheme1_direct_events/*/flow-sim-manifest.json
```

For each manifest case, resolve `rust_output`, require a positive finite `time_us` or `solutions[*].rust_time_us`, and require:

```text
candidate-config.json
flow-sim-inputs/candidate-000/candidate-sketch.json
```

Sort eligible candidates by `(time_us, source_path)`.

- [ ] **Step 4: Implement selection output and invalidation**

Copy the selected config/sketch to `best/`, write `selection.json` with attempt number, source paths, and score, set selection state to succeeded, and reset resim state to pending. Reject an explicit attempt that is not recorded as successful. Skip an already successful selection unless `force=True`; forced selection may atomically replace `best/` but never modifies a search attempt. The selected attempt is authoritative for resim and may be older than `latest_successful_attempt`; any later successful search invalidates it again.

- [ ] **Step 5: Run selection tests**

```bash
.venv/bin/python -m unittest -v scripts/llm-ccl/tests/test_selection.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add agent/scripts/llm-ccl/llm_ccl/selection.py \
  agent/scripts/llm-ccl/tests/test_selection.py
git commit -m "feat: select best llm-ccl candidate"
```

---

### Task 8: Export the translated schedule and run SyCCL resim

**Files:**
- Create: `agent/scripts/llm-ccl/llm_ccl/resim.py`
- Create: `agent/scripts/llm-ccl/tests/test_resim.py`

- [ ] **Step 1: Write failing preflight and resim tests**

Use fake executable scripts under a temporary directory:

- fake FlowSim prints help containing `--dump-translated`, then writes `flow-sim.json` and `translated.json`;
- fake SyCCL writes a large-ish JSON file with an earlier nested `"Time"` and a top-level `"Time": 42.5`;
- an unsupported FlowSim help output omits `--dump-translated`.

Assert command order, output paths, parsed times, manifest success, skip behavior, force behavior, unsupported-binary failure, and acceptance of a valid explicitly selected older successful attempt.

- [ ] **Step 2: Run tests and verify failure**

```bash
.venv/bin/python -m unittest -v scripts/llm-ccl/tests/test_resim.py
```

Expected: ERROR because `llm_ccl.resim` does not exist.

- [ ] **Step 3: Implement preflight**

Provide `ResimOptions` with FlowSim path, synthesize path, timeout seconds, and force.

Run:

```python
[flow_sim_bin, "simulate-sketch", "--help"]
```

Require successful exit and `--dump-translated` in combined output. Require `synthesize_bin.is_file()` and executable access. Record binary path and SHA256 when readable.
Require selection stage status `succeeded`, verify `selection.json` matches the manifest's selected attempt, and verify that attempt is recorded as successful. Do not require it to equal `latest_successful_attempt`, because `select --attempt` intentionally supports replaying an older successful search.

Resolve binaries in this exact order:

```text
flow_sim:   --flow-sim-bin > FLOW_SIM_BIN >
            AGENT_ROOT/datasets/syccl/scheme1_direct_events/flow-sim-rs/target/release/flow-sim-rs
synthesize: --synthesize-bin > SYNTHESIZE_BIN > REPO_ROOT/syccl/build/synthesize
timeout:    --resim-timeout-seconds > SYCCL_RESIM_TIMEOUT_SECONDS > 3600
```

Relative CLI/environment paths are resolved against the invocation working directory before being stored; repository defaults are rooted explicitly. No inaccessible `/root/...` default is used.

- [ ] **Step 4: Implement the two commands**

```python
flow_cmd = [
    str(flow_sim_bin), "simulate-sketch",
    "--config", str(best_config),
    "--sketch", str(best_sketch),
    "--output", str(resim_dir / "flow-sim.json"),
    "--dump-translated", str(resim_dir / "translated.json"),
]
syccl_cmd = [
    str(synthesize_bin), "-f", str(best_config),
    "resim", "-i", str(resim_dir / "translated.json"),
    "-o", str(resim_dir / "resim.json"),
]
```

Log both commands and outputs. Run SyCCL with `cwd=synthesize_bin.parent.parent`.

- [ ] **Step 5: Parse outputs without loading huge SyCCL JSON**

Reuse selection's FlowSim time parser. Scan `resim.json` incrementally without loading the full file, tracking object depth plus JSON string/escape state across chunks. Accept the numeric value for key `"Time"` only at top-level object depth 1, so an earlier nested `Time` cannot be reported accidentally. Require positive finite values and record both FlowSim and SyCCL time in the manifest.

- [ ] **Step 6: Run resim tests**

```bash
.venv/bin/python -m unittest -v scripts/llm-ccl/tests/test_resim.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add agent/scripts/llm-ccl/llm_ccl/resim.py \
  agent/scripts/llm-ccl/tests/test_resim.py
git commit -m "feat: run syccl resim for selected candidates"
```

---

### Task 9: Add full-bundle reports and status views

**Files:**
- Create: `agent/scripts/llm-ccl/llm_ccl/reporting.py`
- Create: `agent/scripts/llm-ccl/tests/test_reporting.py`

- [ ] **Step 1: Write failing report tests**

Create a manifest with one successful case, one search failure, and one pending case. Assert `summary.json` and `summary.csv` contain all three cases, selected FlowSim time, resim FlowSim time, SyCCL time, attempt number, stage statuses, and error category. Assert `summary.json` also contains aggregate counts for prepared, searched, selected, resimulated, and failed cases. Assert status filtering affects display data only and does not rewrite reports.

- [ ] **Step 2: Run tests and verify failure**

```bash
.venv/bin/python -m unittest -v scripts/llm-ccl/tests/test_reporting.py
```

Expected: ERROR because `llm_ccl.reporting` does not exist.

- [ ] **Step 3: Implement reporting**

Provide:

```python
def build_summary(manifest: dict[str, Any]) -> dict[str, Any]: ...
def write_report(bundle: Path) -> tuple[Path, Path]: ...
def status_rows(bundle: Path, case_ids: set[str] | None = None) -> list[dict[str, Any]]: ...
```

Write reports atomically. Always include the full bundle in persisted reports. Keep CSV field order stable. Write `summary.json` as an object containing project/bundle metadata, `counts`, and `cases`; define counts as: prepared = all manifest cases, searched/selected/resimulated = cases whose respective stage succeeded, and failed = cases with any current stage in failed state.

- [ ] **Step 4: Run report tests**

```bash
.venv/bin/python -m unittest -v scripts/llm-ccl/tests/test_reporting.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/scripts/llm-ccl/llm_ccl/reporting.py \
  agent/scripts/llm-ccl/tests/test_reporting.py
git commit -m "feat: report llm-ccl pipeline results"
```

---

### Task 10: Add the CLI and end-to-end orchestration

**Files:**
- Create: `agent/scripts/llm-ccl/run.py`
- Create: `agent/scripts/llm-ccl/tests/test_cli.py`
- Create: `agent/scripts/llm-ccl/tests/test_pipeline.py`

- [ ] **Step 1: Write failing CLI parser tests**

Assert the parser exposes `list`, `prepare`, `search`, `select`, `resim`, `report`, `status`, and `run`; `prepare`/`run` accept project, bundle root, launch ID, and repeated case filters; stage commands accept `--bundle`; `select` accepts `--attempt`; search, select, and resim accept `--force`.

Assert these option groups and defaults:

```text
search:
  --env-toml agent/env.toml
  --model/--api-base/--api-key/--max-tokens
  --max-generations 10000, --k-candidates 4
  --eval-concurrency 1, --gen-concurrency 1
  --llm-policy-pool-size 100, --jobs 1
  --search-timeout-seconds 7200, --flow-sim-bin, --force
resim:
  --flow-sim-bin, --synthesize-bin
  --resim-timeout-seconds 3600, --force
run:
  every search and resim option above except stage-only --force
```

- [ ] **Step 2: Write a failing fake-executable pipeline test**

Build a one-case temporary project package and fake runner adapters so:

1. `prepare` writes inputs;
2. fake search creates two candidate artifacts in attempt 1;
3. `select` chooses the faster candidate;
4. fake FlowSim dumps a translated schedule;
5. fake SyCCL writes `Time`;
6. `report` writes one successful row;
7. rerunning `search`, `select`, and `resim` against the existing bundle skips successful stages;
8. rerunning `run` with the same project/launch destination returns non-zero, reports that the destination exists, and leaves it unchanged instead of silently resuming or overwriting.

Add a two-case orchestration test where one fake search fails: the eligible case still reaches selection/resim, the report contains both cases, and the final command status is non-zero. Add a forced-rerun assertion proving search creates `attempt-0002`, selection consumes only that attempt, and resim reruns without deleting the earlier search attempt.

- [ ] **Step 3: Run tests and verify CLI import failure**

```bash
.venv/bin/python -m unittest -v \
  scripts/llm-ccl/tests/test_cli.py \
  scripts/llm-ccl/tests/test_pipeline.py
```

Expected: ERROR because `run.py` does not exist.

- [ ] **Step 4: Implement `run.py`**

At startup, insert `agent/` and `agent/scripts/llm-ccl/` into `sys.path`. Build subparsers and delegate without embedding pipeline logic. `list` prints project name and case count. Resolve `--case` values against the project or bundle before launching stages. Map every CLI option into `SearchOptions` or `ResimOptions`; modules perform the environment/TOML/default resolution described above. Return non-zero when any requested case fails.

- [ ] **Step 5: Implement `run` orchestration**

`run` calls prepare, search, select, resim, and report in order using the newly created bundle. It forwards the same `--flow-sim-bin` override to search and resim (both modules use the identical environment/default fallback), forwards all model/search options only to search, and forwards the synthesize path plus resim timeout only to resim. If search or selection fails for some cases, later stages continue only for eligible cases, reports are still written, and the final exit status is non-zero.

- [ ] **Step 6: Run CLI and pipeline tests**

```bash
.venv/bin/python -m unittest -v \
  scripts/llm-ccl/tests/test_cli.py \
  scripts/llm-ccl/tests/test_pipeline.py
```

Expected: PASS.

- [ ] **Step 7: Run a real preparation-only smoke test**

```bash
.venv/bin/python scripts/llm-ccl/run.py list
.venv/bin/python scripts/llm-ccl/run.py prepare \
  --project v100_dgx2_clos \
  --bundle-root /tmp/llm-ccl-plan-smoke \
  --launch-id smoke \
  --case 4hosts-64gpu-allgather-65536B
```

Expected: project list shows H800/V100; the prepared bundle contains one valid case and does not launch a model.

- [ ] **Step 8: Commit**

```bash
git add agent/scripts/llm-ccl/run.py \
  agent/scripts/llm-ccl/tests/test_cli.py \
  agent/scripts/llm-ccl/tests/test_pipeline.py
git commit -m "feat: add llm-ccl experiment pipeline cli"
```

---

### Task 11: Document extension workflow and run final verification

**Files:**
- Create: `agent/scripts/llm-ccl/README.md`
- Modify: any new implementation files only if verification exposes defects

- [ ] **Step 1: Write the extension guide**

README sections:

1. Pipeline overview and explicit statement that SyCCL solve is not run.
2. CLI examples for each stage.
3. Bundle layout and resume semantics.
4. Required initial-program interface: `construct_sketches(gpu_count, ...)` with project-compatible defaults.
5. Minimal `PROJECT = ExperimentSpec(...)` example.
6. How to declare `LayerShape`, `ScaleSpec`, and `case_matrix()`.
7. Rule that link bandwidth/latency belongs only in the topology template.
8. Preflight checklist for model, FlowSim, and SyCCL binaries.
9. How to add tests for a new topology and project.

- [ ] **Step 2: Run all new tests**

From `agent/`:

```bash
.venv/bin/python -m unittest discover -v -s scripts/llm-ccl/tests -p 'test_*.py'
```

Expected: PASS.

- [ ] **Step 3: Run targeted existing regressions**

```bash
.venv/bin/python -m unittest -v \
  tests.test_syccl_topodsl \
  tests.test_prepare_v100_dgx2_flow_sim_compare.PrepareV100Dgx2FlowSimCompareTest.test_case_specs_match_dgx2_v100_allgather_sweep \
  tests.test_syccl_v100_dgx2_clos_experiment.SycclV100ClosExperimentTest.test_dgx2_topology_scales_to_four_and_eight_host_clos
.venv/bin/python -c 'from tests.test_topology_examples import test_clos_topo_example_builds_expected_connections; test_clos_topo_example_builds_expected_connections()'
```

Expected: PASS. Do not run the known environment-sensitive legacy bundle tests as topology regressions.

- [ ] **Step 4: Check syntax and legacy-script preservation**

```bash
.venv/bin/python -m compileall -q scripts/llm-ccl
git diff --check
git diff --exit-code HEAD~11 -- \
  scripts/prepare_h800_flow_sim_compare.py \
  scripts/prepare_syccl_h80064_llmelite_experiment.py \
  scripts/prepare_syccl_v100_dgx2_clos_experiment.py \
  scripts/prepare_v100_dgx2_flow_sim_compare.py
```

Expected: compile succeeds, diff check is clean, and all four legacy scripts are unchanged across the implementation commit series. If the number of implementation commits differs, compare against the pre-implementation commit captured before Task 1 instead of `HEAD~11`.

- [ ] **Step 5: Inspect the final worktree**

```bash
git status --short
git diff --stat <pre-implementation-commit>..HEAD
```

Confirm only intended new pipeline files, the captured H800/V100/example topology corrections, the V100 prompt template, and topology regression tests changed. Preserve all unrelated user changes.

- [ ] **Step 6: Commit documentation**

```bash
git add agent/scripts/llm-ccl/README.md
git commit -m "docs: explain adding llm-ccl experiments"
```

---

## Final Acceptance Checklist

- [ ] The four legacy preparation scripts are unchanged.
- [ ] H800 and V100 projects are auto-discovered.
- [ ] V100 uses flat GPU IDs and authoritative `150GB/s,3us`, `12.5GB/s,3us`, `100GB/s,0.5us` template values.
- [ ] Each case has exactly one `all`-strategy search task.
- [ ] Search retries create immutable attempt directories.
- [ ] Concurrent stage updates cannot lose manifest transitions.
- [ ] Selection reads exactly one successful attempt.
- [ ] FlowSim exports `translated.json` for the selected candidate.
- [ ] SyCCL `resim` runs without SyCCL `solve`.
- [ ] Reports cover every bundle case.
- [ ] Reports include prepared, searched, selected, resimulated, and failed aggregate counts.
- [ ] Existing successful stages resume without rerunning.
- [ ] README documents how to add a new project without a central registry edit.
