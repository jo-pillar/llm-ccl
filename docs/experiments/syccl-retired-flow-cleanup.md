# SyCCL Retired Two-Agent Workflow Cleanup

## Status

The retired SyCCL proposal-agent/record-agent workflow has been removed from
the active repository. The standard SimpleTES path, TopoDSL parser, and SyCCL
config renderer remain the supported Python path.

This cleanup intentionally does not change API-key construction, forwarding,
logging, or handling.

## Removal Boundary

The cleanup removes the alternate SyCCL runtime and everything that existed
only to support it:

- `agent/scripts/run_syccl_two_agent.py` and
  `agent/simpletes/engine/syccl_two_agent.py`;
- the `agent/prompt/syccl_two_agent/` prompt set;
- the proposal/record runtime modules formerly under `agent/syccl_agents/`;
- the two-agent unit and integration tests;
- tracked two-agent result fixtures and the VS Code launch configurations that
  invoked the deleted path;
- obsolete 2026-06-09 plans and specifications for that runtime.

The exhaustive path list is recorded in
`docs/superpowers/plans/2026-07-17-syccl-two-agent-removal.md`.

The cleanup retains these shared modules because active SimpleTES scripts use
them:

- `agent/syccl_agents/base_topology.py`;
- `agent/syccl_agents/topodsl.py`;
- `agent/syccl_agents/config_render.py`;
- `agent/syccl_agents/config_templates/`.

Live engineering guidance and H800 experiment provenance now refer only to the
standard SimpleTES `llm_elite` path.

## Mainline Runner Repair

`agent/scripts/run_syccl_simpletes.py` previously exited immediately after
logging the generated command. The unconditional `exit(0)` was removed. Its
dispatch tests now verify both branches:

- dry-run writes the generated inputs and does not launch SimpleTES;
- non-dry-run invokes the generated command with its working directory,
  environment, and `check=True`.

No selector, evaluator, TopoDSL, message-size, model, or API-key behavior was
changed as part of this repair.

## Verification Results

### Removal and retained-path checks

- The three runner dispatch tests pass.
- Tracked-reference searches find no live two-agent names outside the current
  removal design and plan.
- Retained Python files do not import any deleted two-agent module.
- The retained runner, experiment-preparation script, TopoDSL parser, topology
  base classes, and config renderer compile successfully.
- `git diff --check` reports no whitespace errors.
- The implementation worktree contains no uncommitted tracked changes.

### Regression comparison

The retained SyCCL test selection was run on both the removal result and its
pre-removal base commit `2b416e3`. Both runs produced exactly the same result:

```text
33 passed, 10 failed
```

The identical failure set demonstrates that the removal introduced no new
failure in the retained selection. The ten existing failures are:

1. `test_syccl_h80064_experiment.py` expects a missing top-level
   `multirail_program.py`.
2. `test_syccl_multirail_program.py` references
   `multirail_program.py.py`.
3. `test_syccl_scheme1_failure_feedback.py` expects a base config that is not
   present in a clean checkout.
4. Two `test_syccl_simpletes_runner.py` assertions expect total collective
   bytes while the renderer emits per-GPU bytes.
5. Two `test_syccl_topodsl.py` cases expect a missing top-level
   `multirail_topo.py`.
6. `test_syccl_v100_dgx2_clos_experiment.py` expects an ignored release
   `flow-sim-rs` binary inside every clean worktree.
7. Two `test_topology_examples.py` cases use stale layer definitions.

These failures were recorded rather than repaired because they are outside the
two-agent removal boundary.

## Dataset E2E Attempt

A real search was started with the tracked V100 DGX-2 Clos dataset, the
configured DeepSeek endpoint, and the dataset-local `flow-sim-rs` binary. The
generated case was:

```text
/tmp/syccl-dataset-e2e/6be06794fd2876fe/clos/allgather/1k/FULL
```

The `1k` label is the current per-GPU byte value derived from the topology's
64 KiB total collective size across 64 GPUs.

The run proved the following parts of the retained path:

- TopoDSL parsing and generated config, instruction, and init-program output;
- real LLM preflight and generation dispatch;
- local flow-sim evaluation of the initial program;
- a finite root score of `2.1383941996057336`;
- finalized checkpoint and evaluator artifacts.

The process exited successfully, but the checkpoint did not satisfy the strict
generated-candidate success criteria:

```text
generation_attempts: 1
generation_cancellations: 1
checkpoint_nodes: 1
non_root_done_nodes: 0
```

The sole in-flight generation was cancelled when the scheduler treated an
empty queue as a completed run. Therefore this attempt is partial E2E evidence,
not a successful one-generation search result. Terminal output was filtered
externally to redact the configured key; repository key handling was not
modified.

## TODO

- Wire the existing `_restart_every_n()` calculation into the command emitted
  by `run_syccl_simpletes.py`, with coverage for very small generation budgets.
- Update the SimpleTES completion check to include in-flight generation and
  evaluation work before setting the stop event. Add a regression test proving
  that `--max-generations 1` waits for its only generation instead of
  cancelling it.
- Repeat the real dataset E2E validation after those fixes. Require a finite
  root score, at least one non-root `DONE` node, saved LLM input/output, and
  evaluator artifacts before declaring success.
- Resolve the ten retained-suite baseline failures listed above in a separate
  cleanup, without coupling that work to removal of the retired runtime.
