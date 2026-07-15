# Visible-Chinese Rewrite Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite all PDF-visible Chinese in `cclpaper/hpca/main.tex` as fluent academic English while preserving comments, technical meaning, evidence, and LaTeX structure.

**Architecture:** Treat each paper section as an isolated editorial unit. Capture a read-only baseline, patch one unit at a time, run a CJK and scope check, and pause for user review before moving to the next unit. Finish with a whole-file scan and the paper's existing XeLaTeX build.

**Tech Stack:** LaTeX, IEEEtran, XeLaTeX/BibTeX via `make`, ripgrep, unified diff, `apply_patch`

---

## File Map

- Modify: `cclpaper/hpca/main.tex` — the only paper source in scope.
- Reference: `docs/superpowers/specs/2026-07-15-main-tex-visible-chinese-rewrite-design.md` — approved rewrite policy and protected-content rules.
- Temporary: `/tmp/llm-ccl-main.tex.before-visible-chinese-rewrite` — baseline used only for scope review; never committed.

Do not modify `references.bib`, figure files, or any other paper source. Do not manually edit generated LaTeX artifacts; verification commands may refresh them. Because `cclpaper/` is currently untracked user content, do not add or commit `main.tex` unless the user explicitly requests it.

### Task 1: Capture the Baseline and Protected Comments

**Files:**
- Read: `cclpaper/hpca/main.tex`
- Create temporarily: `/tmp/llm-ccl-main.tex.before-visible-chinese-rewrite`

- [ ] **Step 1: Capture the exact pre-edit source**

Run: `cp cclpaper/hpca/main.tex /tmp/llm-ccl-main.tex.before-visible-chinese-rewrite`

Expected: the command exits with status 0 and leaves the working source unchanged.

- [ ] **Step 2: Record protected Chinese comments**

Run: `sed -n '13p;326p;1087p' cclpaper/hpca/main.tex`

Expected: output includes the `%todo` comment, the inline comment near the end of the System Overview workflow, and the trailing Related Work drafting comment.

- [ ] **Step 3: Record the initial visible-Chinese inventory**

Run: `rg -n '\p{Han}|[\x{3000}-\x{303F}\x{FF01}-\x{FF65}]' cclpaper/hpca/main.tex`

Expected: matches cover the protected comments plus all visible rewrite targets described by the spec.

## Required Per-Section Verification

After every rewrite task, run all three checks before asking for user review:

1. Run `rg -n '\p{Han}|[\x{3000}-\x{303F}\x{FF01}-\x{FF65}]' cclpaper/hpca/main.tex` and inspect the active section. Expected: the active section has no visible Han text or CJK/full-width punctuation; matches in later, not-yet-edited sections and protected comments are expected.
2. Run `xelatex -interaction=nonstopmode -halt-on-error main.tex` from `cclpaper/hpca`. Expected: exit status 0, establishing that commands, citations, references, labels, and environments remain syntactically valid after the section edit.
3. Run `diff -u /tmp/llm-ccl-main.tex.before-visible-chinese-rewrite cclpaper/hpca/main.tex`. Expected: every new hunk is confined to approved units, citation keys and labels are unchanged, and all prior approved changes remain intact.

### Task 2: Rewrite the Title and Abstract

**Files:**
- Modify: `cclpaper/hpca/main.tex:45`
- Modify: `cclpaper/hpca/main.tex:59-60`

- [ ] **Step 1: Inspect the complete title and abstract environment**

Run: `sed -n '40,63p' cclpaper/hpca/main.tex`

Expected: the Chinese title subtitle and the mixed-language abstract are visible in full.

- [ ] **Step 2: Apply the contextual English rewrite**

Use `apply_patch` to:

- translate the title as an efficient LLM-driven collective communication schedule generator;
- retain the abstract's opening sentence while fixing its agreement if it belongs to the same mixed paragraph;
- express the motivation, scalability limitation of constraint-based synthesis, structural insight, restricted single-root sketch, deterministic compiler, three-stage scale transfer, and preliminary findings without adding or strengthening claims;
- remove the malformed split `仿真性ls`/`能` by reconstructing the intended statement about simulated performance.

- [ ] **Step 3: Check the active unit for residual Chinese**

Run: `sed -n '40,63p' cclpaper/hpca/main.tex`

Expected: no visible Chinese or Chinese punctuation remains in the title or abstract; the line-13 comment is outside the unit and unchanged.

- [ ] **Step 4: Run all required per-section verification checks**

Expected: the title and abstract contain no visible Chinese, XeLaTeX exits successfully, and the diff is confined to the title and abstract.

- [ ] **Step 5: Pause for user review**

Report the rewrite and wait for approval before Task 3.

### Task 3: Rewrite the Introduction

**Files:**
- Modify: `cclpaper/hpca/main.tex` from `\section{引言}` through the contribution list ending before `\section{背景与动机}`

- [ ] **Step 1: Inspect the full Introduction and its citations**

Run: `sed -n '63,85p' cclpaper/hpca/main.tex`

Expected: all mixed prose, numerical comparisons, visible drafting marker, overview, and three contribution items are present.

- [ ] **Step 2: Rewrite the section heading and body**

Use `apply_patch` to produce `\section{Introduction}` and coherent academic-English paragraphs. Preserve the reported 10-hour, 512-GPU, $17286\times$, 4.3-second, 14146-second, $32\times$, and $3289.8\times$ facts; preserve every citation key and macro. Translate the visible example marker as a concise English placeholder instead of inventing an example. Keep the three challenges, three design components, workflow, and three contributions semantically distinct.

- [ ] **Step 3: Check the section and protected comments**

Run: `rg -n '\p{Han}|[\x{3000}-\x{303F}\x{FF01}-\x{FF65}]' cclpaper/hpca/main.tex`

Expected: no match remains between `\section{Introduction}` and the next section; the line-13 comment still matches.

- [ ] **Step 4: Run all required per-section verification checks**

Expected: XeLaTeX exits successfully; only Tasks 2 and 3 differ from the baseline, and their commands, citations, references, labels, and environments remain valid.

- [ ] **Step 5: Pause for user review**

Report Task 3 and wait for user approval.

### Task 4: Rewrite Background and Motivation

**Files:**
- Modify: `cclpaper/hpca/main.tex` from `\section{背景与动机}` through the two Chinese subsection headings and taxonomy caption

- [ ] **Step 1: Inspect the section**

Run: `sed -n '85,131p' cclpaper/hpca/main.tex`

- [ ] **Step 2: Apply the English rewrite**

Translate the section heading, motivation roadmap, taxonomy caption, and the two question-form subsection headings. Normalize `Fourth，` to English punctuation. Preserve the existing four design properties and all subsequent pure-English paragraphs.

- [ ] **Step 3: Run all required per-section verification checks**

Expected: the Background and Motivation range has no visible CJK, XeLaTeX exits successfully, comments remain, and no pure-English paragraph in this section changed.

- [ ] **Step 4: Pause for user review**

Report and await approval.

### Task 5: Rewrite the Two System Overview Residues

**Files:**
- Modify: the Context Agent sentence near original line 183
- Modify: the figure caption near original line 195

- [ ] **Step 1: Inspect both contexts**

Run: `sed -n '175,201p' cclpaper/hpca/main.tex`

- [ ] **Step 2: Rewrite only the mixed sentence and caption**

Complete the Context Agent sentence with the purpose of preventing an excessively long historical context. Translate the taxonomy caption consistently with Task 4. Do not edit the embedded comment near original line 326.

- [ ] **Step 3: Run all required per-section verification checks**

Expected: no visible Chinese remains in System Overview, XeLaTeX exits successfully, and the inline comment containing `调整润色表达` is unchanged.

- [ ] **Step 4: Pause for user review**

Report and await approval.

### Task 6: Rewrite Evaluation Placeholders

**Files:**
- Modify: `Evaluation` setup fragment near original lines 1066-1067
- Modify: search-time/cost fragment near original lines 1074-1076

- [ ] **Step 1: Inspect the complete Evaluation skeleton**

Run: `sed -n '1064,1084p' cclpaper/hpca/main.tex`

- [ ] **Step 2: Translate only stated facts and placeholders**

Express that the testbed uses four DGX-2 systems in a Clos topology with 16 GPUs per system. Translate the NCCL/SyCCL comparison placeholder without inventing measured values, directions, or conclusions. Normalize heading capitalization only where the heading itself contains a rewrite target or is inseparable from the placeholder.

- [ ] **Step 3: Run all required per-section verification checks**

Expected: no visible Chinese remains in Evaluation, XeLaTeX exits successfully, and no numerical result has been invented.

- [ ] **Step 4: Pause for user review**

Report and await approval.

### Task 7: Rewrite Discussion and Limitations

**Files:**
- Modify: Chinese section heading and one-line drafting instruction near original lines 1084-1085

- [ ] **Step 1: Translate the heading and visible placeholder**

Use `\section{Discussion and Limitations}` and an English drafting placeholder that still asks for a contribution summary, limitations, and next steps. Do not write the missing discussion.

- [ ] **Step 2: Run all required per-section verification checks**

Expected: no visible Chinese remains in Discussion and Limitations, and XeLaTeX exits successfully.

- [ ] **Step 3: Pause for user review**

Report and await approval.

### Task 8: Rewrite Related Work

**Files:**
- Modify: `cclpaper/hpca/main.tex` from the Chinese Related Work heading through the paragraph before Future Work

- [ ] **Step 1: Inspect the complete section and trailing comment**

Run: `sed -n '1086,1111p' cclpaper/hpca/main.tex`

- [ ] **Step 2: Rewrite the section introduction and headings**

Translate `Related Work`, `Collective Communication Schedule Optimization`, `LLMs for Systems`, and `Evaluation-Driven LLM Search and Optimization`. Preserve the trailing Chinese LaTeX comment on the section-introduction line exactly.

- [ ] **Step 3: Rewrite the two Chinese-heavy subsections**

Translate the LLM-for-systems and evaluation-driven-search discussions as cohesive academic English. Preserve every work name and citation group, the distinctions from generic code agents and evolutionary search, the two collective-specific challenges, and the three ways `\mytitle{}` adapts evaluation-driven search. Normalize the full-width comma near ForestColl/SyCCL without revising the surrounding English paragraph.

- [ ] **Step 4: Run all required per-section verification checks**

Expected: the only CJK in the Related Work range is the protected trailing comment, XeLaTeX exits successfully, and citations and labels remain valid.

- [ ] **Step 5: Pause for user review**

Report and await approval.

### Task 9: Rewrite Future Work

**Files:**
- Modify: Future Work heading near original line 1111

- [ ] **Step 1: Translate Future Work without adding content**

Change only the heading to `\section{Future Work}`; leave the section empty.

- [ ] **Step 2: Run all required per-section verification checks**

Expected: the Future Work heading is English, the section remains empty, and XeLaTeX exits successfully.

- [ ] **Step 3: Pause for user review**

Report and await approval.

### Task 10: Rewrite TopoDSL Examples

**Files:**
- Modify: TopoDSL example headings and prose near original lines 1116, 1119, 1121, 1123, 1186, 1188, and 1233

- [ ] **Step 1: Translate the TopoDSL example unit**

Use `TopoDSL Examples`, `Multi-Rail Topology`, and `Clos Topology` headings. Translate the explanatory prose while preserving `\texttt{add\_link}`, the unified `\texttt{lane}` coordinate, host/leaf/spine terminology, and the Alibaba HPN dual-ToR/dual-plane description. Do not edit code listings.

- [ ] **Step 2: Run all required per-section verification checks**

Expected: no visible CJK remains in the TopoDSL examples, code listings are unchanged, and XeLaTeX exits successfully.

- [ ] **Step 3: Pause for user review**

Report and await approval.

### Task 11: Whole-File Verification

**Files:**
- Verify: `cclpaper/hpca/main.tex`
- Build-generated files: existing artifacts under `cclpaper/hpca/` may be refreshed by `make`; do not include them in source edits.

- [ ] **Step 1: Confirm no PDF-visible Han text or CJK/full-width punctuation remains**

Run: `perl -pe 's/(?<!\\)%.*$//' cclpaper/hpca/main.tex | rg -n '\p{Han}|[\x{3000}-\x{303F}\x{FF01}-\x{FF65}]'`

Expected: ripgrep exits with status 1 and prints no matches.

- [ ] **Step 2: Compare protected comments with the baseline**

Run: `diff -u <(rg -o --no-filename '%[^\r\n]*\p{Han}[^\r\n]*' /tmp/llm-ccl-main.tex.before-visible-chinese-rewrite) <(rg -o --no-filename '%[^\r\n]*\p{Han}[^\r\n]*' cclpaper/hpca/main.tex)`

Expected: the command exits with status 0 and produces no diff, proving that every Chinese-bearing comment remains byte-identical regardless of shifted line numbers.

- [ ] **Step 3: Check the complete source diff**

Run: `diff -u /tmp/llm-ccl-main.tex.before-visible-chinese-rewrite cclpaper/hpca/main.tex`

Expected: every change corresponds to a listed rewrite target; pure-English method prose, commands, citations, labels, and listings remain unchanged.

- [ ] **Step 4: Build the paper**

Run: `make` in `cclpaper/hpca`

Expected: XeLaTeX and BibTeX complete successfully and produce `main.pdf` without a new fatal error.

- [ ] **Step 5: Apply verification-before-completion**

Use `@superpowers:verification-before-completion` to review the fresh search, diff, and build evidence before claiming completion.
