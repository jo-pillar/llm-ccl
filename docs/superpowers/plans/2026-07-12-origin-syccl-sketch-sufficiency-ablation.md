# Origin-SyCCL Sketch Sufficiency Ablation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the agreed H800 multirail ablation over existing Origin-SyCCL results without invoking the solver, comparing FlowSim-rs deterministic root-sketch expansion against each archived `algorithms[0]` MILP schedule under one H800 SyCCL resimulator.

**Architecture:** A standard-library Python driver discovers and semantically deduplicates existing result/config pairs, streams the first algorithm and root sketch out of very large JSON files, invokes FlowSim-rs only to export a translated deterministic schedule, and invokes the fixed H800 `synthesize resim` binary for both arms. Per-case JSON plus aggregate CSV/JSON preserve hashes, provenance, timings, ratios, classifications, and failures; large temporary schedules are deleted after successful summarization.

**Tech Stack:** Python 3 standard library, `unittest`, FlowSim-rs debug binary, Origin-SyCCL H800 `synthesize`, JSON/CSV.

---

### Task 1: Discover And Deduplicate Existing H800 Cases

**Files:**
- Create: `experiments/h800_sketch_sufficiency/__init__.py`
- Create: `experiments/h800_sketch_sufficiency/ablation.py`
- Create: `tests/test_h800_sketch_sufficiency.py`

- [ ] Write a failing unit test that builds temporary 32-host AG/A2A and 64-host AG trees, including integer/`.0B` semantic duplicates, malformed results, and missing results.
- [ ] Run `python -m unittest discover -s tests -p 'test_h800_sketch_sufficiency.py' -v` and confirm failure because discovery APIs do not exist.
- [ ] Implement `canonicalize_config`, `discover_cases`, and `select_primary_artifact` so output paths and integral float encodings do not create extra workloads, integer-named artifacts are primary, and duplicates are tagged as historical replicates.
- [ ] Re-run the focused test and confirm it passes.

### Task 2: Stream Algorithm Zero And Its Root Sketch

**Files:**
- Modify: `experiments/h800_sketch_sufficiency/ablation.py`
- Modify: `tests/test_h800_sketch_sufficiency.py`

- [ ] Write failing tests with a synthetic result containing two algorithms and verify extraction selects only `algorithms[0]`, writes `[sketches[0].src_graph]`, creates a one-algorithm resim input, and retains `alg_times[0]` as the archived solve-time consistency check.
- [ ] Run the focused extraction tests and confirm failure for missing APIs.
- [ ] Implement a bounded-memory JSON object scanner that locates and copies the first `algorithms` element and first `src_graph` object without loading the full result file.
- [ ] Build the MILP resim wrapper from the exact first algorithm plus config-derived `coll_name`, GPU count, and `chunk_size_byte`; stream-parse and retain `alg_times[0]` without loading the full result.
- [ ] Re-run extraction tests and confirm they pass.

### Task 3: Execute A Case Without Any Solve Path

**Files:**
- Modify: `experiments/h800_sketch_sufficiency/ablation.py`
- Modify: `tests/test_h800_sketch_sufficiency.py`

- [ ] Write failing tests for exact subprocess command construction: FlowSim must use `simulate-sketch --dump-translated`; both SyCCL arms must invoke `synthesize -f <that-case-config> resim`; no generated command may contain `solve`.
- [ ] Write failing tests for parsing one resim `Time`, computing `R = Time_MILP / Time_det`, and classifying `<0.95`, `[0.95,1.05]`, and `>1.05`.
- [ ] Run focused tests and confirm expected failures.
- [ ] Implement sequential case execution, command logging, SHA-256 provenance, per-case status JSON, cleanup of large temporary files after hashes/times are recorded, and resume-by-existing-successful-case-summary. Record `translate_failed` and deterministic `resim_failed` separately, suppress numeric ratios for either status, and make either status fail the domain-wide gate. Validate source/config/binary hashes before reuse and invalidate stale resumed summaries.
- [ ] Re-run focused tests and the full unit-test module.

### Task 4: Aggregate Primary And Replicate Results

**Files:**
- Modify: `experiments/h800_sketch_sufficiency/ablation.py`
- Create: `experiments/h800_sketch_sufficiency/README.md`
- Modify: `tests/test_h800_sketch_sufficiency.py`

- [ ] Write failing tests for deterministic `summary.json` and `summary.csv`, primary workload count, replicate separation, worst-case gate, geometric mean, exact scenario-level superiority count, and absolute algorithm bandwidth computed as case-name total bytes divided by resim time in decimal GB/s.
- [ ] Run the focused summary tests and confirm failure.
- [ ] Implement aggregate output with primary and replicate sections. Report deterministic and MILP algorithm bandwidth from total case-name bytes, never `coll.byte`. Domain-wide non-inferiority passes only if every primary case is successful and `R >= 0.95`; domain-wide superiority additionally requires geometric mean `R > 1.05`.
- [ ] Document fixed source directories, binary paths/hashes, run command, output fields, and the prohibition on solver reruns.
- [ ] Run the complete unit-test module.

### Task 5: Run The Existing-Result Ablation And Verify Artifacts

**Files:**
- Create: `experiments/h800_sketch_sufficiency/results/20260712-h800-build-existing/**`
- Modify: `docs/adr/0002-evaluate-sketch-sufficiency-with-origin-syccl-ablation.md` only if the observed artifact inventory differs from the agreed 27 primary plus 5 replicate runs.

- [ ] Run the driver against `/home/antl/wzd/origin-syccl/h800_build/syn_res` with the fixed FlowSim and H800 SyCCL binaries.
- [ ] Keep the process running to completion; poll long-running commands and report concrete progress.
- [ ] Run `python -m unittest discover -s tests -p 'test_h800_sketch_sufficiency.py' -v` fresh after the experiment.
- [ ] Verify `summary.json`, `summary.csv`, all per-case summaries, binary hashes, source hashes, and that no command log contains a `solve` invocation.
- [ ] Report measured results and any observed failures without changing the preregistered thresholds.

No git commit is created because the user requested execution and artifacts, not repository history changes.
