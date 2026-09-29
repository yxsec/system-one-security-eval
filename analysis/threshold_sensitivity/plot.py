"""Plot allowance and blocking coverage for the final threshold comparison."""

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "protocol").exists() else HERE.parents[1]
NAMES = {"jev": "Jev", "laya": "Laya", "decider": "Decider", "nimble": "Nimble",
         "gpt41_logprobs": "GPT-4.1", "qwen_native": "Qwen3-8B"}


def draw(summary, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.55), layout="constrained")
    colors = {"allow": "#2375a5", "block": "#d6a35a"}
    for ax, dataset, title in zip(axes, ("wainject", "rjudge"), ("WAInjectBench", "R-Judge")):
        rows = [r for r in summary["policies"] if r["dataset"] == dataset and r["score"] == "shipped" and r["risk_target"] == .01]
        labels, ys = [], []
        for i, model in enumerate(NAMES):
            for j, family in enumerate(("symmetric", "independent")):
                row = next(r for r in rows if r["model"] == model and r["family"] == family)
                y = i * 2.5 + j
                metrics = row["test"]
                allowed = metrics["allow_coverage"] * 100
                blocked = 100 * metrics["automatic_block"] / metrics["n"]
                ax.barh(y, allowed, height=.74, color=colors["allow"])
                ax.barh(y, blocked, left=allowed, height=.74, color=colors["block"])
                marker = "x" if not metrics["budgets_met"] else ("o" if row["confirmation"]["budgets_met"] else "^")
                ax.plot(allowed + blocked, y, marker=marker, markersize=4.5, markerfacecolor="white", markeredgecolor="#252525", markeredgewidth=.9, clip_on=False)
                labels.append(f"{NAMES[model]}  {'S' if j == 0 else 'I'}")
                ys.append(y)
        ax.set_yticks(ys, labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, 14 if dataset == "wainject" else 60)
        ax.set_xlabel("Automatic coverage (%)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles = [Patch(color=colors["allow"], label="Allow"), Patch(color=colors["block"], label="Block"),
               Line2D([], [], marker="o", color="none", markeredgecolor="#252525", label="Confirmation + test pass"),
               Line2D([], [], marker="^", color="none", markeredgecolor="#252525", label="Confirmation fails"),
               Line2D([], [], marker="x", color="#252525", linestyle="none", label="Test fails")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=7)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(Path(str(output) + f".{extension}"), dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Output prefix for PDF, SVG, and PNG")
    args = parser.parse_args()
    summary = json.loads((ROOT / "results/threshold_sensitivity.json").read_text())
    draw(summary, args.output)


if __name__ == "__main__":
    main()
