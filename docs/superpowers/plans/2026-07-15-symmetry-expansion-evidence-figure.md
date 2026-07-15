# Symmetry-Expansion Evidence Figure Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a reproducible one-column evidence figure from the 27 final symmetry-expansion replays and integrate one accurately qualified result sentence and figure into the HPCA background section without modifying `result5.md`.

**Architecture:** A focused Python plotting module reads the two archived CSV summaries, validates the three approved groups, and renders vector PDF plus high-resolution PNG outputs. Unit tests lock down source selection, counts, solver labels, geometric means, and output formats before the plotting implementation is written. LaTeX then consumes the generated PDF and replaces the existing overbroad result sentence; `result5.md` is protected by a byte-for-byte comparison.

**Tech Stack:** Python 3 standard library, Matplotlib 3.9, pytest 8.4, XeLaTeX/BibTeX, Poppler inspection tools.

**Workspace note:** Do not create a worktree for this plan. The target `cclpaper/` tree is user-owned and currently untracked, so a new Git worktree would not contain the paper. Do not stage or commit implementation files from `cclpaper/`; preserve the user's untracked tree and work only in the current workspace.

---

## File Map

- Create `cclpaper/eval_result/plot_symmetry_expansion_vs_solver.py`: load and validate final CSV rows, calculate geometric means, render the figure, and expose a CLI entry point.
- Create `tests/test_symmetry_expansion_figure.py`: test final-only data selection, aggregate values, solver labels, and PDF/PNG rendering.
- Create `cclpaper/hpca/figures/symmetry-expansion-vs-solver.pdf`: vector figure consumed by LaTeX.
- Create `cclpaper/eval_result/figures/symmetry_expansion_vs_solver.png`: high-resolution inspection/Markdown artifact.
- Modify `cclpaper/hpca/main.tex`: replace the inaccurate existing evidence sentence and insert the single-column figure.
- Read only `cclpaper/eval_result/result5.md`: never modify this file.

### Task 1: Lock Down the Final 27-Setting Dataset

**Files:**
- Create: `tests/test_symmetry_expansion_figure.py`
- Create: `cclpaper/eval_result/plot_symmetry_expansion_vs_solver.py`
- Read: `experiments/h800_sketch_sufficiency/results/20260712-h800-build-existing/summary.csv`
- Read: `experiments/h800_sketch_sufficiency/results/20260712-alltoall-round-robin/summary.csv`

- [ ] **Step 1: Save the protected source artifact**

Run:

```bash
cp cclpaper/eval_result/result5.md /tmp/result5.before.md
sha256sum cclpaper/eval_result/result5.md
```

Expected: the copy succeeds and one SHA-256 digest is printed.

- [ ] **Step 2: Write the failing data-selection test**

Create `tests/test_symmetry_expansion_figure.py` with an import helper and a test that expresses the required public API:

```python
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "cclpaper"
    / "eval_result"
    / "plot_symmetry_expansion_vs_solver.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("symmetry_figure", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_groups_selects_only_final_27_settings():
    module = load_module()
    groups = {group.key: group for group in module.load_groups(REPO_ROOT)}

    assert set(groups) == {"ag256", "ag512", "a2a256"}
    assert {key: len(group.ratios) for key, group in groups.items()} == {
        "ag256": 10,
        "ag512": 9,
        "a2a256": 8,
    }
    assert groups["ag256"].solver == "MILP"
    assert groups["ag512"].solver == "MILP"
    assert groups["a2a256"].solver == "LP"
    assert groups["ag256"].geomean == pytest.approx(1.0269152, abs=5e-7)
    assert groups["ag512"].geomean == pytest.approx(1.0379771, abs=5e-7)
    assert groups["a2a256"].geomean == pytest.approx(1.0266348, abs=5e-7)
    assert all(ratio >= 1.0 for group in groups.values() for ratio in group.ratios)

    all_ratios = [ratio for group in groups.values() for ratio in group.ratios]
    combined = math.exp(sum(math.log(value) for value in all_ratios) / len(all_ratios))
    assert combined == pytest.approx(1.0305, abs=5e-5)
```

- [ ] **Step 3: Run the test and verify RED**

Run:

```bash
MPLCONFIGDIR=/tmp/llm-ccl-matplotlib pytest -q tests/test_symmetry_expansion_figure.py::test_load_groups_selects_only_final_27_settings
```

Expected: FAIL because `plot_symmetry_expansion_vs_solver.py` does not exist.

- [ ] **Step 4: Implement the minimal validated loader**

Create `cclpaper/eval_result/plot_symmetry_expansion_vs_solver.py` with:

```python
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ALLGATHER_CSV = Path(
    "experiments/h800_sketch_sufficiency/results/"
    "20260712-h800-build-existing/summary.csv"
)
ALLTOALL_CSV = Path(
    "experiments/h800_sketch_sufficiency/results/"
    "20260712-alltoall-round-robin/summary.csv"
)


@dataclass(frozen=True)
class Group:
    key: str
    label: str
    solver: str
    ratios: tuple[float, ...]
    geomean: float


GROUP_SPECS = (
    ("ag256", "AG\n256 GPUs", "MILP", ALLGATHER_CSV, "allgather", 32, 256, 10, 1.0269152),
    ("ag512", "AG\n512 GPUs", "MILP", ALLGATHER_CSV, "allgather", 64, 512, 9, 1.0379771),
    ("a2a256", "A2A\n256 GPUs", "LP", ALLTOALL_CSV, "alltoall", 32, 256, 8, 1.0266348),
)


def geometric_mean(values: tuple[float, ...]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def load_groups(repo_root: Path = REPO_ROOT) -> tuple[Group, ...]:
    groups = []
    for key, label, solver, source, collective, hosts, gpus, count, expected_gm in GROUP_SPECS:
        with (repo_root / source).open(newline="", encoding="utf-8") as handle:
            rows = [
                row
                for row in csv.DictReader(handle)
                if row["role"] == "primary"
                and row["status"] == "ok"
                and row["collective"] == collective
                and int(row["host_count"]) == hosts
                and int(row["gpu_count"]) == gpus
            ]
        rows.sort(key=lambda row: int(float(row["total_message_bytes"])))
        ratios = tuple(float(row["ratio"]) for row in rows)
        if len(ratios) != count:
            raise ValueError(f"{key}: expected {count} rows, found {len(ratios)}")
        if min(ratios) < 1.0:
            raise ValueError(f"{key}: final dataset contains a ratio below parity")
        geomean = geometric_mean(ratios)
        if not math.isclose(geomean, expected_gm, abs_tol=5e-7):
            raise ValueError(f"{key}: unexpected geometric mean {geomean}")
        groups.append(Group(key, label, solver, ratios, geomean))
    return tuple(groups)
```

- [ ] **Step 5: Run the test and verify GREEN**

Run:

```bash
MPLCONFIGDIR=/tmp/llm-ccl-matplotlib pytest -q tests/test_symmetry_expansion_figure.py::test_load_groups_selects_only_final_27_settings
```

Expected: `1 passed` with no warnings.

### Task 2: Render the Publication Figure Test-First

**Files:**
- Modify: `tests/test_symmetry_expansion_figure.py`
- Modify: `cclpaper/eval_result/plot_symmetry_expansion_vs_solver.py`
- Create: `cclpaper/hpca/figures/symmetry-expansion-vs-solver.pdf`
- Create: `cclpaper/eval_result/figures/symmetry_expansion_vs_solver.png`

- [ ] **Step 1: Add the failing output-format test**

Append:

```python
def test_render_writes_pdf_and_png(tmp_path: Path):
    module = load_module()
    pdf_path = tmp_path / "figure.pdf"
    png_path = tmp_path / "figure.png"

    module.render(module.load_groups(REPO_ROOT), pdf_path, png_path)

    assert pdf_path.read_bytes().startswith(b"%PDF")
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf_path.stat().st_size > 1_000
    assert png_path.stat().st_size > 10_000
```

- [ ] **Step 2: Run the rendering test and verify RED**

Run:

```bash
MPLCONFIGDIR=/tmp/llm-ccl-matplotlib pytest -q tests/test_symmetry_expansion_figure.py::test_render_writes_pdf_and_png
```

Expected: FAIL with `AttributeError` because `render` is not defined.

- [ ] **Step 3: Implement the minimal renderer and CLI**

Extend the plotting module with an Agg backend, `matplotlib.pyplot`, a
single-column `render()` function, default output paths, and `main()`. The
renderer must implement the approved design exactly:

```python
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PDF_OUTPUT = REPO_ROOT / "cclpaper/hpca/figures/symmetry-expansion-vs-solver.pdf"
PNG_OUTPUT = REPO_ROOT / "cclpaper/eval_result/figures/symmetry_expansion_vs_solver.png"


def render(groups: tuple[Group, ...], pdf_path: Path, png_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    colors = ("#0072B2", "#009E73", "#D55E00")
    fig, ax = plt.subplots(figsize=(3.45, 2.65))
    ax.axhspan(0.95, 1.05, color="#D9D9D9", alpha=0.45, zorder=0)
    ax.axhline(1.0, color="#333333", linewidth=1.0, linestyle="--", zorder=1)

    for x_pos, (group, color) in enumerate(zip(groups, colors, strict=True)):
        count = len(group.ratios)
        offsets = [(index - (count - 1) / 2) * 0.035 for index in range(count)]
        ax.scatter(
            [x_pos + offset for offset in offsets],
            group.ratios,
            s=22,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        ax.scatter(
            x_pos,
            group.geomean,
            marker="D",
            s=42,
            color=color,
            edgecolor="black",
            linewidth=0.7,
            zorder=4,
        )
        ax.annotate(
            f"GM {group.geomean:.3f}x",
            (x_pos, group.geomean),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_xticks(range(len(groups)), [group.label for group in groups])
    ax.set_ylabel("Speedup over solver schedule\n$R=T_{solver}/T_{sym}$", fontsize=8)
    ax.set_ylim(0.99, 1.22)
    ax.set_yticks((1.00, 1.05, 1.10, 1.15, 1.20))
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(axis="y", color="#BDBDBD", linewidth=0.5, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=(
            Line2D([], [], marker="o", linestyle="None", color="#666666", label="Message sizes"),
            Line2D([], [], marker="D", linestyle="None", markeredgecolor="black", color="#666666", label="Geometric mean"),
        ),
        loc="upper left",
        frameon=False,
        fontsize=6.5,
        handletextpad=0.3,
    )
    fig.tight_layout(pad=0.4)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    render(load_groups(), PDF_OUTPUT, PNG_OUTPUT)


if __name__ == "__main__":
    main()
```

Use a proper multiplication sign `\times` in LaTeX prose; the plot annotation
may use the ASCII `x` only if the configured Matplotlib font does not render
`×` reliably.

- [ ] **Step 4: Run both tests and verify GREEN**

Run:

```bash
MPLCONFIGDIR=/tmp/llm-ccl-matplotlib pytest -q tests/test_symmetry_expansion_figure.py
```

Expected: `2 passed` with no warnings.

- [ ] **Step 5: Generate the approved artifacts**

Run:

```bash
MPLCONFIGDIR=/tmp/llm-ccl-matplotlib python cclpaper/eval_result/plot_symmetry_expansion_vs_solver.py
pdfinfo cclpaper/hpca/figures/symmetry-expansion-vs-solver.pdf
pdfimages -list cclpaper/hpca/figures/symmetry-expansion-vs-solver.pdf
```

Expected: both output files exist; `pdfinfo` reports one page; `pdfimages`
lists no embedded raster image, confirming the PDF plot is vector output.

- [ ] **Step 6: Inspect the generated PNG**

Use the local image viewer on
`cclpaper/eval_result/figures/symmetry_expansion_vs_solver.png` at original
detail. Check that all three groups, the parity line, shaded equivalence band,
27 points, and all three geometric-mean labels are readable at one-column
width. If visual defects are found, add a failing assertion where practical,
then make the smallest renderer adjustment and rerun both tests.

### Task 3: Integrate the Evidence into the HPCA Background

**Files:**
- Modify: `cclpaper/hpca/main.tex:139`
- Read only: `cclpaper/eval_result/result5.md`

- [ ] **Step 1: Save the pre-edit LaTeX source for review**

Run:

```bash
cp cclpaper/hpca/main.tex /tmp/main.tex.before-symmetry-figure
```

Expected: the copy succeeds.

- [ ] **Step 2: Replace the existing overbroad result sentence**

Keep the first sentence beginning `The key risk of this design`. Replace its
second sentence with:

```latex
Across 27 simulated settings on the regular H800-style multirail topology, symmetry-driven deterministic expansion matched or outperformed SyCCL's solver-generated schedules (MILP for AllGather and LP for AlltoAll), yielding a geometric-mean speedup of \(1.031\times\) (Figure~\ref{fig:symmetry-expansion-vs-solver}).
```

This replacement removes the unsupported synthesis-cost clause and avoids
mislabeling the AlltoAll LP baseline as MILP.

- [ ] **Step 3: Insert the single-column figure immediately after that paragraph**

Insert:

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/symmetry-expansion-vs-solver.pdf}
  \caption{Simulator-predicted speedup of symmetry-driven deterministic expansion over paired SyCCL solver-generated schedules on the regular H800-style 8-rail multirail model. Points represent message-size settings and diamonds denote geometric means; $R=T_{\mathrm{solver}}/T_{\mathrm{sym}}$, so $R=1$ denotes parity and higher is better. AllGather uses 256- and 512-GPU configurations with MILP baselines; AlltoAll uses the 256-GPU configuration with an LP baseline.}
  \label{fig:symmetry-expansion-vs-solver}
\end{figure}
```

- [ ] **Step 4: Verify the source-level claims and protected file**

Run:

```bash
rg -n -C 3 "27 simulated|symmetry-expansion-vs-solver|MILP for AllGather|LP for AlltoAll" cclpaper/hpca/main.tex
cmp -s cclpaper/eval_result/result5.md /tmp/result5.before.md
sha256sum cclpaper/eval_result/result5.md /tmp/result5.before.md
```

Expected: the new sentence, caption, and label appear once; `cmp` exits 0; the
two SHA-256 digests are identical.

### Task 4: Compile and Visually Verify the Paper

**Files:**
- Verify: `cclpaper/hpca/main.tex`
- Verify: `cclpaper/hpca/main.pdf`
- Verify: `cclpaper/hpca/main.log`
- Verify: `tests/test_symmetry_expansion_figure.py`

- [ ] **Step 1: Run the focused plotting tests**

Run:

```bash
MPLCONFIGDIR=/tmp/llm-ccl-matplotlib pytest -q tests/test_symmetry_expansion_figure.py
```

Expected: `2 passed` with no warnings.

- [ ] **Step 2: Force a full paper build**

Run:

```bash
make -B
```

Working directory: `cclpaper/hpca`.

Expected: XeLaTeX and BibTeX complete successfully and `main.pdf` is rebuilt.

- [ ] **Step 3: Check the build log and reference resolution**

Run:

```bash
rg -n "LaTeX Error|Undefined control sequence|Reference .* undefined|There were undefined references" main.log
```

Working directory: `cclpaper/hpca`.

Expected: no matches and exit status 1 from `rg`.

- [ ] **Step 4: Locate and render the page containing the figure**

Run:

```bash
pdftotext -layout main.pdf /tmp/main-symmetry-figure.txt
awk 'BEGIN { RS="\f" } /Simulator-predicted speedup of symmetry-driven/ { print NR }' /tmp/main-symmetry-figure.txt
```

Working directory: `cclpaper/hpca`.

Expected: exactly one page number is printed. Render that page with:

```bash
awk 'BEGIN { RS="\f" } /Simulator-predicted speedup of symmetry-driven/ { print NR }' /tmp/main-symmetry-figure.txt | xargs -I PAGE pdftoppm -f PAGE -l PAGE -singlefile -png -r 160 main.pdf /tmp/main-symmetry-page
```

Then inspect `/tmp/main-symmetry-page.png` in the local image viewer. Confirm
that the figure, caption, surrounding paragraph, and column layout are legible
and do not overlap.

- [ ] **Step 5: Review the exact source changes**

Run:

```bash
git diff --no-index /tmp/main.tex.before-symmetry-figure cclpaper/hpca/main.tex
cmp -s cclpaper/eval_result/result5.md /tmp/result5.before.md
git status --short -- cclpaper/hpca/main.tex cclpaper/hpca/figures/symmetry-expansion-vs-solver.pdf cclpaper/eval_result/plot_symmetry_expansion_vs_solver.py cclpaper/eval_result/figures/symmetry_expansion_vs_solver.png tests/test_symmetry_expansion_figure.py
```

Expected: the source diff contains only the approved sentence and figure
environment; `result5.md` remains identical; status lists only the intended
implementation artifacts. Do not stage or commit these user-tree changes.

### Task 5: Final Evidence Audit

**Files:**
- Verify all files listed above.

- [ ] **Step 1: Recompute the final aggregate directly from source CSVs**

Run the loader test once more:

```bash
MPLCONFIGDIR=/tmp/llm-ccl-matplotlib pytest -q tests/test_symmetry_expansion_figure.py::test_load_groups_selects_only_final_27_settings
```

Expected: `1 passed`; this confirms the plotted values still comprise exactly
10, 9, and 8 final settings with combined geometric mean `1.0305`.

- [ ] **Step 2: Confirm no original-expansion series entered the deliverable**

Run:

```bash
rg -n -i "old expansion|original expansion|旧展开|0\.9381|0\.7979" cclpaper/eval_result/plot_symmetry_expansion_vs_solver.py tests/test_symmetry_expansion_figure.py cclpaper/hpca/main.tex
```

Expected: no matches and exit status 1 from `rg`.

- [ ] **Step 3: Record final verification evidence**

Report the focused pytest result, successful HPCA build, resolved figure
reference, visual inspection result, exact output paths, and identical
`result5.md` hashes. Do not claim hardware performance or generalize beyond the
tested regular H800-style multirail model.
