# AGENT.md

## Purpose

This repository is a working area for LLM-assisted collective-communication scheduling research. It combines:

- `agent/`: Python SimpleTES plus SyCCL-specific LLM search workflows.
- `syccl/`: C++ SyCCL synthesizer, sketch expansion, solver, and performance model code.
- `syccl-sketch-search/`: Rust extraction of SyCCL sketch search that emits SyCCL-compatible sketch JSON.

Treat this as a multi-project repo. Do not assume a command from one subtree applies globally.

## First Files To Read

Before editing, read the local guide for the subtree you are touching:

- Root coordination: this file.
- Python agent work: `agent/README.md`.
- C++ SyCCL work: `syccl/AGENTS.md` and `syccl/README.md`.
- Rust sketch search: `syccl-sketch-search/README.md`.
- Paper work: `cclpaper/AGENTS.md`, `cclpaper/writingrules.md`, and the active CoPaper state if needed.

Note: `agent/AGNETS.md` is misspelled and is a task note about TopoDSL input flow, not a general agent guide. It is still useful context for that specific feature.

## Current Mental Model

The active engineering direction is to connect user-provided TopoDSL descriptions to SyCCL evaluation:

1. Parse a TopoDSL file describing topology family, collective type, and communication size.
2. Support at least `clos` and `multirail` topology families.
3. Render a SyCCL/flow-sim config from normalized topology parameters.
4. Build an initial sketch from a user-provided init program, using the parsed GPU count.
5. Fill instruction templates with parsed topology values.
6. Evaluate candidate sketches through the flow simulator or SyCCL-compatible resimulation.

Important design pressure: TopoDSL should stay topology-oriented. Avoid leaking internal SyCCL layer/group prompt constraints into the public TopoDSL unless the task explicitly asks for that. Existing TODO notes also call out chunk/chunk-split support as future design space.

## Subproject Commands

### Python Agent

Run from `agent/`:

```bash
uv sync
uv run pytest
uv run python main_wizard.py
uv run python main.py --list-policies
```

Targeted SyCCL tests:

```bash
cd agent
uv run pytest tests/test_syccl_topodsl.py tests/test_syccl_simpletes_runner.py
```

This selection currently includes four known baseline failures: two stale
multirail fixture references and two message-size expectation mismatches. See
`docs/experiments/syccl-retired-flow-cleanup.md` for the exact failures and the
pre-removal comparison.

The Python SyCCL entry point expects `TOPODSL` in the environment:

```bash
cd agent
TOPODSL=datasets/syccl/scheme1_direct_events/templates/v100_dgx2_clos/clos_topo.py \
uv run python scripts/run_syccl_simpletes.py \
  --init-program datasets/syccl/scheme1_direct_events/templates/v100_dgx2_clos/clos_program.py \
  --instruction datasets/syccl/scheme1_direct_events/templates/v100_dgx2_clos/prompt_template.txt \
  --dry-run
```

### C++ SyCCL

Run from `syccl/`. Building requires local SCIP and SCIPpp installations:

```bash
mkdir -p build
cd build
cmake .. -DSCIP_SUITE_DIR=/path/to/SCIP -DSCIP_PP_DIR=/path/to/SCIPpp
make -j
./synthesize -f ../config/a100-8gpu-4nic-clos-ag.json solve
```

Python helper tests live under `syccl/scripts/`:

```bash
cd syccl
python -m pytest scripts
```

### Rust Sketch Search

Run from `syccl-sketch-search/`:

```bash
cargo test
cargo run -- --config ../syccl/config/single-host-8gpu.json --output /tmp/syccl-sketches.json --limit 8 --pretty
```


## Editing Rules
prefer uv for python environment management

## SyCCL Agent Notes

- `agent/syccl_agents/topodsl.py` normalizes TopoDSL into `TopologyParams`.
- `agent/syccl_agents/config_render.py` renders JSON configs from `TopologyParams`.
- Config templates live under `agent/syccl_agents/config_templates/`.
- `agent/scripts/run_syccl_simpletes.py` is the Python SyCCL entry point.

The legacy environment variables named `SYCCL_TASK_*` are intentionally removed by `run_syccl_simpletes.py`. New topology-dependent behavior should flow through TopoDSL-derived config and explicit generated files instead.
