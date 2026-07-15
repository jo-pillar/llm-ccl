# Symmetry-Expansion Evidence Figure Design

## Goal

Turn the final results in `cclpaper/eval_result/result5.md` into one compact,
publication-ready figure and one evidence sentence for the `Why Exploit
Symmetry?` subsection of `cclpaper/hpca/main.tex`. The evidence must show that,
in the tested regular multirail configurations, symmetry-driven deterministic
expansion matches or exceeds the paired solver-generated schedules.

## Evidence Scope

Use only the final deterministic expansion results from 27 simulator-replayed
message-size settings:

- 10 AllGather settings on 32 hosts / 256 GPUs;
- 9 AllGather settings on 64 hosts / 512 GPUs;
- 8 AlltoAll settings on 32 hosts / 256 GPUs using host-pair round-robin
  expansion.

The figure and paper sentence must not show or discuss the original AlltoAll
expansion or its degradation. They must also not imply hardware measurements,
multiple topology families, end-to-end LLM search quality, or a guarantee for
arbitrary symmetric topologies.

The baselines must be named accurately: SyCCL uses MILP for AllGather and LP
for AlltoAll. A claim covering all 27 settings must therefore say
`solver-generated schedules (MILP for AllGather and LP for AlltoAll)`, not use
`MILP` as a blanket label.

## Data Sources and Metric

Read the plotted values from the archived CSV summaries rather than copying
rounded values from Markdown:

- AllGather: `experiments/h800_sketch_sufficiency/results/20260712-h800-build-existing/summary.csv`, restricted to primary, successful AllGather rows;
- AlltoAll: `experiments/h800_sketch_sufficiency/results/20260712-alltoall-round-robin/summary.csv`, restricted to primary, successful AlltoAll rows.

For each setting, plot

\[
R = \frac{T_{\mathrm{solver}}}{T_{\mathrm{sym}}}
  = \frac{B_{\mathrm{sym}}}{B_{\mathrm{solver}}}.
\]

Thus, `R = 1` is parity and larger values favor symmetry-driven expansion. The
expected group sizes and geometric means are:

| Group | Count | Solver | Geometric mean R | Minimum R |
|---|---:|---|---:|---:|
| 32-host / 256-GPU AllGather | 10 | MILP | 1.0269 | 1.0000 |
| 64-host / 512-GPU AllGather | 9 | MILP | 1.0380 | 1.0000 |
| 32-host / 256-GPU AlltoAll | 8 | LP | 1.0266 | 1.0000 |
| Combined | 27 | MILP / LP | 1.0305 | 1.0000 |

## Figure Design

Create a single-column grouped strip plot that fits the background subsection:

- x-axis groups: `AG\n256 GPUs`, `AG\n512 GPUs`, and `A2A\n256 GPUs`;
- y-axis: `Speedup over solver-generated schedule` with a displayed
  mathematical definition of `R` in the caption;
- one lightly jittered point per final message-size setting;
- one prominent diamond per group for the geometric mean, annotated with its
  value;
- a horizontal parity line at `R = 1`;
- a light shaded band for the predeclared equivalent region
  `0.95 <= R <= 1.05`;
- y-range approximately `0.99` to `1.22`, so the 1.2011 maximum remains
  visible;
- colorblind-safe colors, distinct marker shapes, and grayscale-safe outlines;
- no title inside the plot because the LaTeX caption supplies the context.

The plotting program must be reproducible and export:

- a vector PDF for LaTeX at
  `cclpaper/hpca/figures/symmetry-expansion-vs-solver.pdf`;
- a high-resolution PNG for Markdown at
  `cclpaper/eval_result/figures/symmetry_expansion_vs_solver.png`.

## Document Integration

At the top of `cclpaper/eval_result/result5.md`, add a curated paper-facing
summary containing the PNG, metric definition, the four-row aggregate table,
the exact evidence scope, and the simulator/baseline caveats. Preserve the
existing diagnostic material below it as supplementary notes, but mark it as
not used by the paper figure. Correct blanket `MILP` labels where they refer to
AlltoAll so the document does not misidentify an LP baseline.

In `cclpaper/hpca/main.tex`, place the figure and sentence in the `Why Exploit
Symmetry?` subsection after the paragraph that motivates representative
structures and reuse across roots and source--destination pairs. Use this
sentence:

> Across 27 simulated settings on the regular H800-style multirail topology,
> symmetry-driven deterministic expansion matched or outperformed SyCCL's
> solver-generated schedules (MILP for AllGather and LP for AlltoAll), yielding
> a geometric-mean speedup of 1.031x.

The LaTeX version must use `\(1.031\times\)` and reference the new figure.

The caption must state that results are simulator-predicted, identify the three
tested scale/collective groups, define `R`, explain points and diamonds, and
name the AllGather/AlltoAll solver types. It must not claim synthesis-cost
reductions; `result5.md` contains schedule-quality evidence only.

## Verification

1. Assert that the plotting program loads exactly 10, 9, and 8 rows for the
   three groups and recomputes the geometric means above within floating-point
   tolerance.
2. Generate both outputs and inspect the PNG for legibility at one-column size.
3. Confirm that the PDF is a valid vector figure and both outputs are nonempty.
4. Search the new paper text and caption for inaccurate blanket `MILP` wording,
   hardware-performance language, and claims beyond the tested topology.
5. Build `cclpaper/hpca/main.tex` with the existing Makefile and confirm that no
   new LaTeX errors, missing references, or overfull boxes are introduced.
6. Review the final diff to ensure the original-expansion degradation does not
   appear in the new figure or paper-facing sentence.

## Success Criteria

- The figure contains exactly the 27 final settings and no original-expansion
  series.
- Every plotted point has `R >= 1`, and the displayed aggregate values match
  the archived CSVs.
- The background sentence and caption use scientifically accurate solver and
  simulator terminology.
- `result5.md` provides a concise, traceable paper-facing summary.
- The HPCA paper compiles successfully with the new figure and reference.
