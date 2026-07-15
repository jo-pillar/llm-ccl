# Visible-Chinese Rewrite Design for `cclpaper/hpca/main.tex`

## Goal

Rewrite every Chinese fragment that is visible in the compiled PDF as fluent academic English. The rewrite must preserve the paper's technical meaning, evidence, LaTeX structure, and citations.

## Scope

The only source file to edit is `cclpaper/hpca/main.tex`. The work covers 52 lines that currently contain PDF-visible Chinese text or Chinese punctuation. Chinese text inside LaTeX comments is explicitly out of scope and must remain byte-for-byte unchanged.

External content, including figure source files and `references.bib`, is out of scope. The method section titled `LLM-Friendly Collective Schedule Synthesis` contains no visible Chinese and therefore requires no edits.

## Rewrite Policy

Use a contextual sentence or paragraph rewrite rather than word-for-word substitution:

- Rewrite a complete sentence when Chinese appears inside mixed Chinese-English prose.
- Rewrite the complete semantic paragraph when the source is fragmented across lines or is too ungrammatical for sentence-local replacement.
- Leave already-English sentences and paragraphs unchanged unless they are inseparable parts of the same mixed-language sentence or paragraph.
- Preserve all claims, numbers, qualifiers, citations, cross-references, macros, mathematical notation, itemization, and figure placement.
- Use consistent technical terms already established in the draft, including `Propagation Plan`, `TopoDSL`, `root-level sketch`, `symmetry-driven expansion`, `Assessment Pipeline`, and the named agents.
- Do not add evidence, citations, experimental results, limitations, contributions, or future-work claims.
- Translate visible drafting placeholders into concise English placeholders; do not fill them with invented content or hide them as comments.
- Normalize Chinese punctuation that appears in otherwise English text.

## Protected Content

All LaTeX comments must remain unchanged, including:

- the `%todo` comment near line 13;
- the inline comment near line 326;
- the trailing Chinese drafting comment in the Related Work introduction near line 1087.

Do not edit pure-English prose merely to harmonize style. Do not modify data, citation keys, labels, commands, code listings, or topology examples except for their Chinese headings and explanatory prose.

## Section Order and Interaction

Apply and verify changes one section at a time in this order:

1. Title and Abstract
2. Introduction
3. Background and Motivation
4. System Overview
5. Evaluation
6. Discussion and Limitations
7. Related Work
8. Future Work
9. TopoDSL Examples

After each unit, report what was rewritten and pause for user review before proceeding to the next unit.

## Special Cases

- The malformed abstract ending split across two lines must be reconstructed as a coherent statement about simulated performance without strengthening the claim.
- The visible `Example needed` drafting marker in the Introduction remains a visible English placeholder.
- Evaluation fragments are translated only from the facts present in the source. No missing setup details or results are inferred.
- The Discussion and Limitations body is currently a drafting instruction; it remains an English drafting placeholder.
- Future Work currently has only a Chinese heading; translate the heading without generating body text.

## Verification

For each completed section:

1. Inspect the diff and confirm that only the active section changed.
2. Search for remaining CJK characters in that section and confirm that any matches occur only inside protected comments.
3. Check that LaTeX commands, citations, references, labels, and environments remain balanced.

After all sections are complete:

1. Search the entire `main.tex` for CJK characters and confirm that the only remaining matches are protected comments.
2. Build the paper with `make` in `cclpaper/hpca`.
3. Review the build log for LaTeX errors introduced by the rewrite.
4. Review the final diff to confirm that no pure-English section or external file was changed.

## Success Criteria

- The compiled PDF contains no Chinese text originating from `main.tex`.
- Every original LaTeX comment remains unchanged.
- All rewritten passages read as coherent academic English.
- No technical claims or experimental facts are invented or strengthened.
- The paper compiles successfully using its existing Makefile.
