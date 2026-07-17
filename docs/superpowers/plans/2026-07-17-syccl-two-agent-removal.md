# SyCCL Two-Agent Removal Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the retired SyCCL proposal/record two-agent workflow, restore execution of the retained SimpleTES SyCCL runner, and prove the retained path with one real `datasets/syccl` search.

**Architecture:** Delete the alternate runtime, CLI, prompts, exclusive helpers, tests, fixtures, launch configuration, and obsolete plans while retaining the shared TopoDSL/config-rendering package used by the mainline. Make one behavior-only repair to `run_syccl_simpletes.py`: remove its unconditional early exit, with a focused regression test. Verify the remaining mainline first with targeted tests and then with a real one-generation V100 DGX-2 Clos search through the configured LLM and dataset-local flow simulator.

**Tech Stack:** Python 3.12, unittest/pytest, SimpleTES, LiteLLM, TopoDSL, `flow-sim-rs`, Git.

---

## File Structure

**Modify:**

- `agent/scripts/run_syccl_simpletes.py`: remove the accidental unconditional process exit; do not change API-key behavior.
- `agent/tests/test_syccl_simpletes_runner.py`: lock dry-run and non-dry-run dispatch behavior.
- `AGENT.md`: remove retired commands and point SyCCL guidance at the retained SimpleTES path.
- `docs/experiments/h800-llm-ccl-vs-syccl-flow-sim.md`: describe the selected `llm_elite` path without referring to deleted alternatives.
- `agent/scripts/prepare_h800_flow_sim_compare.py`: remove the stale two-agent wording from provenance notes.

**Delete:**

- `.vscode/launch.json`
- `agent/simpletes/engine/syccl_two_agent.py`
- `agent/scripts/run_syccl_two_agent.py`
- `agent/prompt/syccl_two_agent/`
- `agent/syccl_agents/README.md`
- `agent/syccl_agents/flow_sim.py`
- `agent/syccl_agents/llm.py`
- `agent/syccl_agents/prompts.py`
- `agent/syccl_agents/record_agent.py`
- `agent/syccl_agents/records.py`
- `agent/syccl_agents/runner.py`
- `agent/syccl_agents/sketch_dsl.py`
- `agent/tests/test_syccl_two_agent.py`
- `agent/tests/integration/`
- tracked files under `agent/test_result/syccl_two_agent_demo/`
- tracked files under `agent/test_result/syccl_two_agent_5round_scoring/`
- tracked files under `agent/test_result/syccl_two_agent_5round_scoring_retry/`
- `docs/superpowers/plans/2026-06-09-python-syccl-two-agent.md`
- `docs/superpowers/plans/2026-06-09-rust-agents-flow.md`
- `docs/superpowers/specs/2026-06-09-rust-agents-flow-design.md`

**Retain unchanged:**

- `agent/syccl_agents/__init__.py`
- `agent/syccl_agents/base_topology.py`
- `agent/syccl_agents/topodsl.py`
- `agent/syccl_agents/config_render.py`
- `agent/syccl_agents/config_templates/`
- ignored checkpoints and result data
- untracked user files shown by `git status`

---

### Task 1: Restore Mainline Runner Dispatch With TDD

**Files:**

- Modify: `agent/tests/test_syccl_simpletes_runner.py:201-234`
- Modify: `agent/scripts/run_syccl_simpletes.py:329-357`

- [ ] **Step 1: Strengthen the dry-run test and add a failing non-dry-run test**

Update the existing dry-run test to patch `runner.subprocess.run` and assert it is not called. Add this focused test:

```python
def test_main_non_dry_run_launches_generated_simpletes_command(self):
  runner = load_runner()
  fake_spec = runner.RunSpec(
      command=["uv", "run", "python", "main.py", "--max-generations", "1"],
      env={"SYCCL_BASE_CONFIG": "/tmp/config.json"},
      cwd=ROOT / "agent",
      instruction_path=Path("/tmp/instruction.txt"),
      output_path=Path("/tmp/output"),
      config_path=Path("/tmp/config.json"),
      init_program_path=Path("/tmp/init_program.py"),
  )

  with (
      patch.object(runner, "load_required_topodsl_from_env", return_value=object()),
      patch.object(runner, "build_command", return_value=fake_spec),
      patch.object(runner.subprocess, "run") as run,
  ):
    result = runner.main([
        "--init-program", "/tmp/input.py",
        "--instruction", "/tmp/prompt.txt",
    ])

  self.assertEqual(0, result)
  run.assert_called_once_with(
      fake_spec.command,
      cwd=fake_spec.cwd,
      env=fake_spec.env,
      check=True,
  )
```

- [ ] **Step 2: Run the runner tests and verify the regression is red**

Run from `agent/`:

```bash
uv run --with pytest python -m pytest \
  tests/test_syccl_simpletes_runner.py::SycclSimpletesRunnerTest::test_main_dry_run_uses_topodsl_env_and_new_cli_inputs \
  tests/test_syccl_simpletes_runner.py::SycclSimpletesRunnerTest::test_main_non_dry_run_launches_generated_simpletes_command \
  -q
```

Expected: failure with `SystemExit: 0`; the mocked subprocess is never reached.

- [ ] **Step 3: Apply the minimal runner fix**

Delete only the unconditional exit:

```python
LOGGER.info("SimpleTES command: %s", " ".join(spec.command))
if args.dry_run:
  ...
subprocess.run(spec.command, cwd=spec.cwd, env=spec.env, check=True)
return 0
```

Do not modify the model flags, `api_key`, logging format, selector, evaluator, or generated command.

- [ ] **Step 4: Run the runner tests and verify green**

Run:

```bash
uv run --with pytest python -m pytest tests/test_syccl_simpletes_runner.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the runner repair**

```bash
git add agent/scripts/run_syccl_simpletes.py agent/tests/test_syccl_simpletes_runner.py
git commit -m "fix: let SyCCL SimpleTES runner execute searches"
```

---

### Task 2: Remove the Retired Two-Agent Workflow

**Files:**

- Delete every path listed in the plan's **Delete** section.
- Modify: `AGENT.md:13-23,51-78,112-122`
- Modify: `docs/experiments/h800-llm-ccl-vs-syccl-flow-sim.md:17-22,103-110`
- Modify: `agent/scripts/prepare_h800_flow_sim_compare.py:1474-1477`

- [ ] **Step 1: Delete the tracked two-agent implementation and artifacts**

Use one `apply_patch` deletion patch covering the exact tracked files. Do not remove ignored checkpoint directories, `result/syccl_two_agent_10r/`, or any unrelated untracked path.

- [ ] **Step 2: Update the root engineering guide**

Apply these content changes to `AGENT.md`:

- Make `agent/README.md` the Python guide; remove the deleted two-agent README reference.
- Change targeted SyCCL tests to `tests/test_syccl_topodsl.py tests/test_syccl_simpletes_runner.py`.
- Remove the two-agent example entirely.
- Update the retained runner example to use the tracked V100 DGX-2 dataset files:

```bash
cd agent
TOPODSL=datasets/syccl/scheme1_direct_events/templates/v100_dgx2_clos/clos_topo.py \
uv run python scripts/run_syccl_simpletes.py \
  --init-program datasets/syccl/scheme1_direct_events/templates/v100_dgx2_clos/clos_program.py \
  --instruction datasets/syccl/scheme1_direct_events/templates/v100_dgx2_clos/prompt_template.txt \
  --dry-run
```

- Retain notes for `topodsl.py`, `config_render.py`, and config templates; replace deleted-module notes with a statement that `run_syccl_simpletes.py` is the SyCCL Python entry point.

- [ ] **Step 3: Remove stale experiment references**

In `docs/experiments/h800-llm-ccl-vs-syccl-flow-sim.md`:

- Remove the non-goal that forbids the now-deleted path.
- Replace the `Use SimpleTES with llm_elite. Do not use:` block and deleted-file list with:

```markdown
Use the standard SimpleTES path with `llm_elite`.
```

In `agent/scripts/prepare_h800_flow_sim_compare.py`, replace the stale note with:

```python
"llm-ccl run scripts use the SimpleTES llm_elite path.",
```

- [ ] **Step 4: Verify reference and import cleanup**

Run from the repository root against Git-tracked files, including tracked hidden files, experiment files, and tracked test fixtures:

```bash
git grep -n -I -i -E \
  'syccl[_ -]?two[_ -]?agent|two-agent|SycclTwoAgent' \
  -- . \
  ':(exclude)cclpaper/**' \
  ':(exclude)docs/superpowers/specs/2026-07-17-syccl-two-agent-removal-design.md' \
  ':(exclude)docs/superpowers/plans/2026-07-17-syccl-two-agent-removal.md'
```

Expected: no matches and exit code 1 from `git grep`. The exclusions are limited to paper sources and the current removal documents; ignored historical results are absent automatically because `git grep` searches tracked files.

Run:

```bash
rg -n 'syccl_agents\.(flow_sim|llm|prompts|record_agent|records|runner|sketch_dsl)|simpletes\.engine\.syccl_two_agent' \
  agent --glob '*.py' --glob '!test_result/**'
```

Expected: no matches.

- [ ] **Step 5: Inspect the deletion boundary**

Run:

```bash
git status --short
git diff --stat
git diff --name-status
```

Expected: only the planned tracked deletions and three live reference edits, plus the runner/test changes already committed. Existing unrelated untracked files remain unmodified.

- [ ] **Step 6: Commit the workflow removal**

Stage only the planned tracked paths, then commit:

```bash
git commit -m "refactor: remove retired SyCCL two-agent flow"
```

---

### Task 3: Verify the Retained SyCCL Mainline

**Files:**

- Test only; no expected source changes.

- [ ] **Step 1: Run the SyCCL regression suite**

Run from `agent/`:

```bash
uv run --with pytest python -m pytest \
  tests/test_syccl_*.py \
  tests/test_topology_examples.py \
  tests/test_prepare_v100_dgx2_flow_sim_compare.py \
  -q
```

Expected: all collected retained SyCCL tests pass. Tests requiring unavailable optional external infrastructure may skip explicitly; failures must be investigated rather than ignored.

- [ ] **Step 2: Compile retained entry points and shared modules**

Run from the repository root:

```bash
python -m py_compile \
  agent/scripts/run_syccl_simpletes.py \
  agent/scripts/prepare_h800_flow_sim_compare.py \
  agent/syccl_agents/base_topology.py \
  agent/syccl_agents/topodsl.py \
  agent/syccl_agents/config_render.py
```

Expected: exit code 0 and no output.

- [ ] **Step 3: Run repository hygiene checks**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; unrelated untracked files remain unchanged.

---

### Task 4: Run the Real Dataset SyCCL End-to-End Search

**Files:**

- Runtime output only under `/tmp/syccl-dataset-e2e`; no repository files should change.

- [ ] **Step 1: Check E2E prerequisites without printing credentials**

Run from `agent/`:

```bash
test -x datasets/syccl/scheme1_direct_events/flow-sim-rs/target/release/flow-sim-rs
uv run python -c 'import tomllib; from pathlib import Path; d=tomllib.loads(Path("env.toml").read_text()); assert d.get("model"); assert d.get("api_base"); assert d.get("api_key"); print("model configuration present")'
```

Expected: binary check succeeds and only `model configuration present` is printed.

- [ ] **Step 2: Execute one real search with output-only secret filtering**

Run from `agent/` with network permission. Do not add `--skip-preflight`:

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

Expected: LLM preflight succeeds, initial evaluation completes with a finite score, one generation is attempted, a checkpoint is finalized, and the pipeline exits 0.

Because `set -o pipefail` is active, a preflight, network, evaluator, or SimpleTES failure makes the pipeline non-zero even when the output filter succeeds. If the sanitized output shows the configured external LLM endpoint is unavailable, stop the success path, preserve the sanitized failure evidence, and report the external blocker. Do not mark the E2E step or overall task complete.

- [ ] **Step 3: Verify generated inputs and checkpoint evidence**

Run this read-only checkpoint assertion from `agent/`:

```bash
python3 - <<'PY'
import gzip
import json
import math
from pathlib import Path

roots = list(Path("/tmp/syccl-dataset-e2e").glob("*/clos/allgather/64k/FULL"))
assert len(roots) == 1, roots
case = roots[0]

for relative in (
    "generated/flow-sim-config.json",
    "generated/init_program.py",
    "instructions/syccl_instruction.txt",
):
  assert (case / relative).is_file(), relative

instances = list((case / "checkpoints").glob("*/instance-*"))
assert instances, "missing SimpleTES instance directory"
instance = max(instances, key=lambda path: path.stat().st_mtime_ns)
assert (instance / "run.log").is_file()

states = list(instance.glob("db_state_*"))
assert states, "missing finalized db_state checkpoint"
state = max(states, key=lambda path: path.stat().st_mtime_ns)
metadata = json.loads((state / "metadata.json").read_text(encoding="utf-8"))

nodes_path = state / "nodes.json"
if nodes_path.is_file():
  nodes = json.loads(nodes_path.read_text(encoding="utf-8"))
else:
  with gzip.open(state / "nodes.json.gz", "rt", encoding="utf-8") as handle:
    nodes = json.load(handle)

assert metadata["generation_attempts"] >= 1, metadata
assert len(nodes) >= 2, len(nodes)
roots = [node for node in nodes if not node.get("parent_ids")]
assert roots and math.isfinite(float(roots[0]["score"])), roots
generated = [
    node
    for node in nodes
    if node.get("parent_ids")
    and isinstance(node.get("gen_id"), int)
    and node["gen_id"] >= 0
    and node.get("status") == "DONE"
]
assert generated, f"no DONE generated nodes; node_count={len(nodes)}"
assert any(node.get("llm_input") and node.get("llm_output") for node in generated)
artifacts = [path for path in (case / "eval_artifacts").rglob("*") if path.is_file()]
assert artifacts, "missing evaluator artifacts"

print({
    "generation_attempts": metadata["generation_attempts"],
    "node_count": len(nodes),
    "done_generated_nodes": len(generated),
    "root_score": roots[0]["score"],
    "eval_artifact_files": len(artifacts),
})
PY
```

Expected: the script exits 0 and prints only the non-sensitive evidence summary. It verifies these artifacts and conditions:

```text
generated/flow-sim-config.json
generated/init_program.py
instructions/syccl_instruction.txt
checkpoints/<date>/instance-*/run.log
checkpoints/<date>/instance-*/db_state_*/metadata.json
checkpoints/<date>/instance-*/db_state_*/nodes.json or nodes.json.gz
eval_artifacts/
```

- `metadata.json["generation_attempts"] >= 1`;
- the node list contains at least two nodes (initial root plus one generated node);
- the root node has a finite numeric score;
- at least one non-root node has `gen_id >= 0` and `status == "DONE"`;
- saved LLM I/O fields or artifacts are present for the generated node.

Do not print prompts, completions, environment files, or credentials.

- [ ] **Step 4: Record final repository state**

Run from the repository root:

```bash
git status --short
git log -3 --oneline
```

Expected: implementation commits are present; only the user's pre-existing unrelated untracked files remain.
