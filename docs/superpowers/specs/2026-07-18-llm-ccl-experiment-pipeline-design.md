# LLM-CCL Experiment Pipeline Design

**Date:** 2026-07-18

**Status:** Approved in conversation; ready for implementation planning after document review.

## Context

The repository currently has four experiment-preparation scripts for H800 multirail and V100 DGX-2 Clos experiments. They duplicate topology rendering, configuration generation, instruction rendering, initial-program rendering, manifest writing, and launch-script generation. Three of them also prepare Origin-SyCCL solver comparisons, even though the desired main workflow no longer includes SyCCL solving.

The desired workflow is:

1. Instantiate a topology template at one or more cluster scales.
2. Generate LLM-CCL search cases for the requested collective and total message sizes.
3. Run LLM-CCL with FlowSim as the search evaluator, using only the `all` elite-selection strategy.
4. Select the fastest valid candidate within that single search run.
5. Export the selected candidate as a translated schedule with `flow-sim-rs simulate-sketch --dump-translated`.
6. Run SyCCL `resim` on the translated schedule.
7. Produce resumable manifests and machine-readable reports.

The existing experiment scripts and historical experiment outputs must remain intact. The new implementation will live under `agent/scripts/llm-ccl/`.

## Goals

- Provide one topology-independent workflow for `prepare -> search -> select -> resim -> report`.
- Treat each topology DSL template as the single source of truth for topology structure, bandwidth, and latency.
- Support multiple cluster scales, collectives, and total message sizes without duplicating preparation code.
- Keep each pipeline stage separately runnable and resumable.
- Make adding a new experiment project require only templates and a small declarative project module.
- Give every case deterministic artifact paths and explicit stage status.
- Use SyCCL `resim`; do not invoke SyCCL `solve`.

## Non-goals

- Do not delete, rename, or refactor the four existing preparation scripts.
- Do not modify historical artifacts under `experiments/` or `agent/result/`.
- Do not implement Origin-SyCCL solver comparisons.
- Do not redesign the LLM-CCL evaluator, elite policy, FlowSim, or SyCCL internals.
- Do not incorporate the existing ablation experiments into the main pipeline.
- Do not support multiple elite-selection strategies in the new workflow. The strategy is fixed to `all`.
- Do not generate MSCCL XML as part of this pipeline.

## Terminology

- **Project:** A named experiment family with shared topology, prompt, initial program, and case matrix.
- **Scale:** One concrete cluster or mesh size, including the total GPU count and per-layer group dimensions.
- **Case:** One scale, collective, and total-message-size combination.
- **Search artifact:** A valid LLM-CCL candidate configuration, sketch, and FlowSim result.
- **Selected artifact:** The fastest valid search artifact in the single `all`-strategy run for one case.
- **Bundle:** The immutable prepared inputs plus mutable stage outputs for one project launch.

## Directory Structure

```text
agent/scripts/llm-ccl/
├── run.py
├── README.md
├── llm_ccl/
│   ├── __init__.py
│   ├── models.py
│   ├── project_loader.py
│   ├── topology.py
│   ├── preparation.py
│   ├── search.py
│   ├── selection.py
│   ├── resim.py
│   ├── manifest.py
│   ├── reporting.py
│   └── projects/
│       ├── __init__.py
│       ├── h800_multirail.py
│       └── v100_dgx2_clos.py
└── tests/
    ├── test_topology.py
    ├── test_preparation.py
    ├── test_selection.py
    ├── test_resim.py
    └── test_pipeline.py
```

`run.py` is the only user-facing executable. The `llm_ccl` package is importable for tests and keeps the implementation behind a small command interface.

## Project Interface

The external project interface consists of four immutable models.

```python
@dataclass(frozen=True)
class LayerShape:
  layer_id: int
  group_num: int
  node_num: int


@dataclass(frozen=True)
class ScaleSpec:
  name: str
  gpu_count: int
  layers: tuple[LayerShape, ...]


@dataclass(frozen=True)
class CaseSpec:
  case_id: str
  scale: ScaleSpec
  collective: str
  total_message_size: int


@dataclass(frozen=True)
class ExperimentSpec:
  name: str
  topology_template: Path
  instruction_template: Path
  initial_program: Path
  cases: tuple[CaseSpec, ...]
```

The project interface deliberately does not expose bandwidth or latency. Those values come only from the topology template. Project modules may use a shared `case_matrix()` helper to generate `CaseSpec` values, but the stored interface remains an explicit tuple of cases.

Each file in `llm_ccl/projects/` exports exactly one `PROJECT: ExperimentSpec`. Project discovery imports modules in that directory and rejects duplicate project names. Adding a project does not require editing a central registry.

## Initial Projects

The new implementation starts with two projects while leaving their old scripts untouched.

### H800 multirail

- Topology template: `agent/datasets/syccl/scheme1_direct_events/templates/H800_multirail/multirail_topo.py`
- Instruction template: `agent/datasets/syccl/scheme1_direct_events/prompt_templete.txt`
- Initial program: `agent/datasets/syccl/scheme1_direct_events/templates/H800_multirail/multirail_program.py`
- Existing experiment matrices are represented as project cases:
  - 32 hosts / 256 GPUs: AllGather and Alltoall.
  - 64 hosts / 512 GPUs: AllGather.
  - 512 hosts / 4096 GPUs: AllGather.
- Message-size lists:
  - 32-host AllGather and Alltoall: `65536`, `262144`, `1048576`, `4194304`, `16777216`, `67108864`, `268435456`, `1073741824`, `4294967296`, `17179869184`, `68719476736`, and `274877906944` total bytes.
  - 64-host AllGather: `65536`, `262144`, `1048576`, `4194304`, `16777216`, `67108864`, `268435456`, and `1073741824` total bytes.
  - 512-host AllGather: `262144`, `1048576`, and `4194304` total bytes.

### V100 DGX-2 Clos

- Topology template: `agent/datasets/syccl/scheme1_direct_events/templates/v100_dgx2_clos/clos_topo.py`
- Instruction template: `agent/datasets/syccl/scheme1_direct_events/templates/v100_dgx2_clos/prompt_template.txt`
- Initial program: `agent/datasets/syccl/scheme1_direct_events/templates/v100_dgx2_clos/clos_program.py`
- Scales:
  - 4 hosts / 64 GPUs.
  - 8 hosts / 128 GPUs.
- Collective: AllGather.
- Total message sizes: `65536`, `262144`, `1048576`, `4194304`, `16777216`, `67108864`, `268435456`, `1073741824`, `4294967296`, `17179869184`, `68719476736`, and `274877906944` bytes.
- The V100 template uses flat global GPU node identifiers consistently across the NVSwitch and NIC layers.
- The implementation is explicitly authorized to update
  `agent/datasets/syccl/scheme1_direct_events/templates/v100_dgx2_clos/clos_topo.py`
  so its host-local NVSwitch edges use the same flat global GPU identifiers as
  its GPU-to-NIC edges. This is the only existing topology-template source edit
  required by the new pipeline.
- The V100 template values are authoritative:
  - Layer 1: `150GB/s`, `3us`.
  - Layer 3: `12.5GB/s`, `3us`.
  - Layer 4: `100GB/s`, `0.5us`.

## Topology Rendering

`topology.py` renders one case-specific TopoDSL from a project template.

1. Parse the Python template with `ast`.
2. Find exactly one assignment whose target is `topology` and whose value is a constructor call.
3. Replace only:
   - total GPU count;
   - total message size;
   - collective enum;
   - each declared layer's `group_num` and `node_num`.
4. Preserve the topology class implementation and every `LinkSpec` bandwidth and latency literal.
   The renderer uses AST source locations to splice only the topology assignment;
   it does not run `ast.unparse` over the whole template or rewrite comments and formatting.
5. Reject missing layers, duplicate layer IDs, unexpected layers, non-positive dimensions, or multiple topology assignments.
6. Write the rendered source and reload it through `syccl_agents.topodsl.load_topodsl`.
7. Verify the parsed family, total GPU count, collective, and layer-derived dimensions.
8. Generate the FlowSim/SyCCL config through the existing `syccl_agents.config_render` module.

The topology template is therefore the single source for physical structure and link values. The project supplies only scale dimensions and workload parameters.

Message-size semantics have exactly one conversion:

- `CaseSpec.total_message_size`, the rendered topology constructor's
  `coll_bytes`, and the manifest's `total_message_size` are total bytes across
  the collective.
- `total_message_size` must be positive and divisible by `gpu_count`.
- `syccl_agents.topodsl.load_topodsl` normalizes the rendered topology object to
  `TopologyParams.message_size = total_message_size // gpu_count`. This is
  existing behavior in `_normalize_params`; the new pipeline does not change
  `syccl_agents.topodsl` to implement this conversion.
- `syccl_agents.config_render` copies that normalized value into
  `config["coll"]["byte"]`; the pipeline must not divide it a second time.
- Instruction rendering exposes `MESSAGE_SIZE` as the per-rank `coll.byte` and
  additionally exposes `TOTAL_MESSAGE_SIZE` for future templates. Existing
  templates continue to work without using the new placeholder.
- The manifest records both `total_message_size` and `coll_byte` and validates
  `coll_byte * gpu_count == total_message_size`.

## Pipeline Stages

### Prepare

`prepare` creates a bundle and materializes each case:

- rendered `topodsl.py`;
- generated `config.json`;
- rendered `instruction.txt`;
- rendered `init_program.py`;
- immutable case metadata in the bundle manifest.

Preparation validates all cases before writing runnable task files. A partially invalid project does not produce a launchable bundle.

### Search

Each case has one LLM-CCL search task. The command uses the existing SimpleTES entrypoint and evaluator with:

```text
--selector llm_elite
--elite-selection-strategy all
```

The new pipeline does not create `linear_rank` or `balance` tasks. Search concurrency is controlled by `--jobs`; tool-specific evaluator and generation concurrency remain explicit CLI options or project-independent defaults.

Every search invocation writes to a new immutable attempt directory:

```text
search/attempt-0001/
search/attempt-0002/
```

Each attempt contains its own checkpoints, evaluator artifacts, log, command,
and result metadata. The manifest records the current attempt and all prior
attempts. A successful prior search is skipped unless `--force` is supplied.
Rerunning a failed search or forcing a successful search creates the next
attempt directory; it never writes new artifacts into an older attempt.

### Select

Selection scans only one search attempt: by default the latest successful
attempt recorded in the manifest, or an explicit `--attempt` when provided.
It never merges candidates across attempts. A candidate is eligible when:

- its candidate configuration exists and parses;
- its candidate sketch exists and parses;
- its FlowSim output exists;
- the FlowSim time is finite and positive.

The candidate with the lowest FlowSim time is selected. Ties are broken by a
stable lexical artifact path so reruns are deterministic. The selected
configuration and sketch are copied into the case's `best/` directory together
with `selection.json` recording the source search attempt, source artifact, and
score. A new successful search attempt invalidates the previous selection and
resim state until selection is run again.

### Resim

Resim uses the validated repository conversion path without invoking SyCCL solving:

```text
flow-sim-rs simulate-sketch
  --config candidate-config.json
  --sketch candidate-sketch.json
  --output flow-sim.json
  --dump-translated translated.json

synthesize -f candidate-config.json
  resim
  -i translated.json
  -o resim.json
```

The stage records command lines, binary paths, binary hashes when available, wall time, exit status, FlowSim time, and SyCCL `Time`. A successful prior resim is skipped unless `--force` is supplied.

### Report

`report` reads only manifests and stage outputs. It produces:

- `reports/summary.json` with complete structured data;
- `reports/summary.csv` with one row per case;
- counts of prepared, searched, selected, resimulated, and failed cases.

Reports never trigger search or resim work.

## CLI

```text
python agent/scripts/llm-ccl/run.py list
python agent/scripts/llm-ccl/run.py prepare --project PROJECT [options]
python agent/scripts/llm-ccl/run.py search --bundle BUNDLE [options]
python agent/scripts/llm-ccl/run.py select --bundle BUNDLE [options]
python agent/scripts/llm-ccl/run.py resim --bundle BUNDLE [options]
python agent/scripts/llm-ccl/run.py report --bundle BUNDLE
python agent/scripts/llm-ccl/run.py status --bundle BUNDLE
python agent/scripts/llm-ccl/run.py run --project PROJECT [options]
```

`run` executes `prepare`, `search`, `select`, `resim`, and `report` in order. The separate commands are the primary recovery and cluster-operation interface.

The default bundle destination is
`experiments/llm-ccl/PROJECT/YYYYMMDD-HHMMSS/`. `prepare` and `run` accept
`--bundle-root` and `--launch-id` to override it. They reject an already
existing destination rather than merging with or overwriting it; resuming an
existing launch always uses a stage command with `--bundle`.

`prepare` and `run` accept a repeatable `--case CASE_ID` to create and run a
project subset. `search`, `select`, and `resim` accept the same filter for an
existing bundle; with no filter they operate on every case. `status` may filter
displayed cases. `report` always reports the full bundle and does not accept a
case filter, so a partial report cannot silently overwrite the bundle-wide
summary with a subset. `select` additionally accepts `--attempt N` when a
specific successful search attempt must be selected.

Binary and model configuration is supplied through explicit flags or environment/config files. New code must not probe inaccessible hard-coded `/root/...` paths at import time.

## Bundle Layout

```text
bundle/
├── manifest.json
├── cases/
│   └── CASE_ID/
│       ├── topodsl.py
│       ├── config.json
│       ├── instruction.txt
│       ├── init_program.py
│       ├── search/
│       │   ├── attempt-0001/
│       │   │   ├── checkpoints/
│       │   │   ├── eval_artifacts/
│       │   │   ├── command.json
│       │   │   └── run.log
│       │   └── attempt-0002/
│       │       └── ...
│       ├── best/
│       │   ├── candidate-config.json
│       │   ├── candidate-sketch.json
│       │   └── selection.json
│       └── resim/
│           ├── flow-sim.json
│           ├── translated.json
│           ├── resim.json
│           └── run.log
└── reports/
    ├── summary.json
    └── summary.csv
```

Case IDs are validated path-safe slugs and must be unique within a project.

## Manifest and State

The bundle manifest contains:

- `schema_version`;
- project name;
- creation timestamp;
- topology, prompt, and initial-program paths and SHA256 hashes;
- case metadata;
- resolved command configuration without secrets;
- per-case stage records;
- binary provenance gathered when a stage runs.

Stage states are:

```text
pending -> running -> succeeded
                   -> failed
```

Each stage also records attempt count, timestamps, error category, and a short
error message. Search keeps an append-only list of attempt records and a
`latest_successful_attempt` pointer. Selection records the exact attempt it
consumed. Manifest writes use a temporary file followed by atomic replacement.
Secrets are never written to the manifest or logs by the orchestration layer.

## Error Handling

- Template and project errors fail the entire `prepare` command before search begins.
- Search, selection, and resim failures are isolated per case.
- Commands return non-zero when any requested case fails.
- Existing successful stages are skipped by default.
- `--force` is required to rerun a successful stage. Search force-runs create a
  new attempt instead of overwriting artifacts.
- Missing or corrupt selected artifacts invalidate selection and prevent resim for that case.
- Search preflight checks model configuration, evaluator path, topology inputs, and FlowSim availability.
- Resim preflight checks `flow-sim-rs simulate-sketch --help` for `--dump-translated` support and verifies the SyCCL `synthesize` binary.
- Large SyCCL output files are scanned incrementally for the top-level `Time` field instead of loaded fully into memory.

## Testing

Tests use `unittest` so they run in the existing agent virtual environment without requiring pytest.

### Unit tests

- Model validation and case ID uniqueness.
- Automatic project discovery and duplicate-name rejection.
- AST topology rendering changes scale fields while preserving `LinkSpec` values.
- Missing or malformed topology assignments fail clearly.
- Total-message-size to per-rank-byte conversion.
- Instruction and initial-program rendering.
- Candidate selection ignores invalid results and deterministically chooses the lowest positive FlowSim time.
- Manifest transitions, atomic writes, resume behavior, and `--force` behavior.
- Search-attempt isolation and selection from exactly one attempt.
- Incremental SyCCL `Time` extraction.

### Topology regression tests

- H800 host-local edges are `GPU -> nvswitch`, not pairwise GPU edges.
- V100 host-local edges are `GPU -> nvswitch`, not pairwise GPU edges.
- V100 GPU identifiers are flat and consistent across NVSwitch and NIC edges.
- V100 rendered and generated config values match `150GB/s, 3us`, `12.5GB/s, 3us`, and `100GB/s, 0.5us`.

### Integration tests

Fake executable adapters stand in for LLM-CCL, FlowSim, and SyCCL. A full test runs:

```text
prepare -> search -> select -> resim -> report
```

The integration tests cover success, a failed case among successful cases, resume, forced rerun, missing artifacts, and unsupported `--dump-translated` preflight.

## Adding a New Experiment Project

`agent/scripts/llm-ccl/README.md` documents this workflow:

1. Add or choose a topology DSL template whose `topology = ...` instantiation contains the default `LinkSpec` values.
2. Add or choose an instruction template and initial program.
3. Create `llm_ccl/projects/PROJECT_NAME.py`.
4. Define `ScaleSpec` values with layer dimensions only.
5. Build explicit cases, normally with `case_matrix()`.
6. Export `PROJECT = ExperimentSpec(...)`.
7. Run `run.py list` to confirm discovery.
8. Run `prepare` into a temporary bundle and inspect the generated TopoDSL and config.
9. Run the project tests before launching a real search.

The README includes one minimal example project and a checklist for topology, configuration, binary, and artifact validation.

## Compatibility

- The four existing preparation scripts remain byte-for-byte unchanged by this
  implementation. The separately authorized V100 topology-template identifier
  correction is not a change to those scripts.
- Existing experiment outputs are read-only and are not migrated automatically.
- The new bundle schema begins at version 1 and is independent of legacy manifests.
- The new pipeline may read the same model configuration and evaluator implementation as existing scripts, but does not depend on their directory layouts.
- Existing conversion and ablation scripts remain separate tools.

## Success Criteria

- `run.py list` discovers the H800 and V100 projects.
- Preparation renders valid star/NVSwitch TopoDSL files and configs for every declared scale.
- No new workflow command invokes SyCCL `solve`.
- Each case launches exactly one `all`-strategy search.
- The fastest valid candidate within that run is selected deterministically.
- FlowSim exports a translated schedule for the selected candidate.
- SyCCL `resim` completes from that translated schedule and its `Time` is reported.
- Interrupted runs resume without repeating successful work.
- A new example project can be added without modifying the orchestration implementation or a central registry.
- Existing preparation scripts and historical outputs remain untouched.
