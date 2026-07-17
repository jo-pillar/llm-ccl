# SyCCL Two-Agent Removal Design

## Goal

Remove the retired SyCCL two-agent workflow from the active repository while preserving the current SimpleTES/SyCCL mainline experiment path, then prove the retained path with one real dataset-backed end-to-end search.

The removed workflow is the proposal-agent/record-agent runtime entered through `agent/scripts/run_syccl_two_agent.py`. The retained mainline is the standard SimpleTES flow, including `agent/scripts/run_syccl_simpletes.py` and the experiment preparation scripts that consume TopoDSL-derived topology configuration.

## Motivation

The two-agent runtime is no longer part of the mainline test or experiment path. It is already excluded explicitly by the current H800 comparison plan, and its interfaces have drifted from the shared TopoDSL/config-rendering modules enough that its test suite no longer imports successfully. Keeping the dead path creates misleading documentation, duplicate evaluation logic, and maintenance burden.

The retained runner also currently exits immediately after logging its generated command, so it cannot launch a non-dry-run search. This defect must be repaired before running the required end-to-end validation.

## Removal Boundary

### Delete executable workflow code

- `agent/simpletes/engine/syccl_two_agent.py`
- `agent/scripts/run_syccl_two_agent.py`
- `agent/prompt/syccl_two_agent/`
- `.vscode/launch.json`, because every tracked launch configuration in the file targets the deleted CLI or its test suite

### Delete two-agent-only support modules

- `agent/syccl_agents/flow_sim.py`
- `agent/syccl_agents/llm.py`
- `agent/syccl_agents/prompts.py`
- `agent/syccl_agents/record_agent.py`
- `agent/syccl_agents/records.py`
- `agent/syccl_agents/runner.py`
- `agent/syccl_agents/sketch_dsl.py`
- `agent/syccl_agents/README.md`

Repository-wide import inspection found no consumers of these modules outside the two-agent runtime, its CLI, and its tests.

### Delete two-agent tests and tracked generated fixtures

- `agent/tests/test_syccl_two_agent.py`
- `agent/tests/integration/test_syccl_two_agent_runtime.py`
- `agent/tests/integration/__init__.py` if the integration directory becomes empty
- Tracked files under:
  - `agent/test_result/syccl_two_agent_demo/`
  - `agent/test_result/syccl_two_agent_5round_scoring/`
  - `agent/test_result/syccl_two_agent_5round_scoring_retry/`

Ignored checkpoints, caches, and result directories are historical experiment data and will not be deleted.

### Delete obsolete design records

- `docs/superpowers/plans/2026-06-09-python-syccl-two-agent.md`
- `docs/superpowers/specs/2026-06-09-rust-agents-flow-design.md`
- `docs/superpowers/plans/2026-06-09-rust-agents-flow.md`

These documents describe the retired proposal-agent/record-agent direction and have no live references.

## Shared Code That Must Remain

The following modules stay in `agent/syccl_agents/` because the active SimpleTES/SyCCL runner and experiment preparation scripts still import them:

- `__init__.py`
- `base_topology.py`
- `topodsl.py`
- `config_render.py`
- `config_templates/clos.json`
- `config_templates/multirail.json`

The following consumers must continue working unchanged:

- `agent/scripts/run_syccl_simpletes.py`
- `agent/scripts/prepare_syccl_h80064_llmelite_experiment.py`
- `agent/scripts/prepare_syccl_v100_dgx2_clos_experiment.py`
- topology examples and dataset TopoDSL files
- shared TopoDSL and mainline SimpleTES regression tests

Renaming the surviving `syccl_agents` package is outside this change. Although the package name is now broader than its remaining contents, migrating every mainline import would add risk without improving removal of the executable two-agent workflow.

## Reference Cleanup

Update live guidance and experiment metadata so they describe only the retained mainline:

- Remove the two-agent command, files, and targeted test references from `AGENT.md`.
- Update `docs/experiments/h800-llm-ccl-vs-syccl-flow-sim.md` to state the selected SimpleTES `llm_elite` path directly instead of warning against deleted files.
- Update the provenance text in `agent/scripts/prepare_h800_flow_sim_compare.py` so it no longer names the deleted workflow.

Paper sources under `cclpaper/` are outside this engineering cleanup. Conceptual discussion in papers is not an executable repository path and will not be changed.

## Required Mainline Runner Repair

Make the minimum changes needed for `agent/scripts/run_syccl_simpletes.py` to execute the retained path safely:

- Remove the unconditional `exit(0)` between command logging and the `--dry-run` branch.
- Preserve the existing behavior in which `--dry-run` prints generated paths and returns without launching SimpleTES.
- Preserve the existing non-dry-run behavior of invoking `subprocess.run(..., check=True)` with the generated command, working directory, and environment.
- Add regression tests proving that dry-run still does not launch a subprocess and non-dry-run does launch the generated SimpleTES command.
- Do not change API-key argument construction, logging, or handling in repository code as part of this work.

No selector, search-policy, evaluator, TopoDSL, or experiment-configuration behavior should be redesigned as part of this repair.

## Resulting Architecture

After removal, the Python SyCCL-related mainline is:

```text
TopoDSL
  -> syccl_agents.topodsl
  -> syccl_agents.config_render
  -> run_syccl_simpletes.py / experiment preparation scripts
  -> standard SimpleTES runtime and evaluator
```

There will be no alternative SyCCL-specific runtime, proposal/record loop, direct runner, prompt set, or two-agent CLI.

## Verification

1. Search tracked source, hidden tooling files, and live documentation for remaining `syccl_two_agent`, `SycclTwoAgent`, and SyCCL `two-agent` references. Exclude this removal design itself from the search. Other remaining matches are allowed only in explicitly excluded paper sources or ignored historical result data.
2. Confirm no retained Python file imports a deleted `syccl_agents` module.
3. Run the shared topology and mainline SyCCL tests, including:
   - `tests/test_syccl_topodsl.py`
   - `tests/test_topology_examples.py`
   - `tests/test_syccl_simpletes_runner.py`
   - active experiment-preparation tests that use TopoDSL/config rendering
4. Compile the affected retained Python entry points.
5. Run one real end-to-end search from `agent/` using the smallest tracked dataset case that exercises the current Clos path:

   ```bash
   set -o pipefail
   TOPODSL=datasets/syccl/scheme1_direct_events/templates/v100_dgx2_clos/clos_topo.py \
   FLOW_SIM_BIN=$PWD/datasets/syccl/scheme1_direct_events/flow-sim-rs/target/release/flow-sim-rs \
   uv run python scripts/run_syccl_simpletes.py \
     --init-program datasets/syccl/scheme1_direct_events/templates/v100_dgx2_clos/clos_program.py \
     --instruction datasets/syccl/scheme1_direct_events/templates/v100_dgx2_clos/prompt_template.txt \
     --max-generations 1 \
     --output-root /tmp/syccl-dataset-e2e \
     --env-toml env.toml \
     --save-llm-io 2>&1 \
     | python3 -c 'import sys, tomllib; from pathlib import Path; secret = str(tomllib.loads(Path("env.toml").read_text()).get("api_key", "")); sys.stdout.writelines((line.replace(secret, "***") if secret else line) for line in sys.stdin)'
   ```

   Do not pass `--skip-preflight`: the validation must reach the configured real LLM endpoint. The existing dataset-local release `flow-sim-rs` binary is the simulator boundary for this check. The output-only Python filter replaces every occurrence of the configured key, including exception-rendered command lists, before output reaches the validation transcript; it does not change repository behavior.

   E2E success requires all of the following evidence:

   - LLM preflight succeeds and at least one model generation is attempted.
   - The dataset TopoDSL is parsed and generated config, instruction, and init-program files are created under the temporary output root.
   - The initial dataset program receives a finite flow-sim score.
   - At least one generated, non-root SimpleTES node is evaluated and committed to the checkpoint.
   - The process exits successfully and saved LLM/checkpoint artifacts exist.

   If the configured external LLM endpoint is unavailable after the command is run with required network permission, report the external blocker with command evidence and do not claim the E2E validation passed.

6. Run `git diff --check` and inspect the final deletion list to ensure unrelated untracked files and ignored experiment results were not touched.

## Success Criteria

- No runnable SyCCL two-agent entry point or implementation remains.
- No two-agent-only support module or regression test remains.
- Live repository guidance does not advertise or warn about the deleted path.
- The current SimpleTES/SyCCL mainline retains its TopoDSL and config-rendering dependencies.
- The retained `run_syccl_simpletes.py` executes non-dry-run searches.
- Targeted mainline tests pass, or any pre-existing unrelated failures are recorded with evidence.
- One real `datasets/syccl` search satisfies the E2E evidence requirements above.
- Untracked user files and ignored historical experiment outputs remain untouched.
