from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).parent / "images"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "paper": "#9aa0a6",
    "baseline": "#4C78A8",
    "targeted": "#E45756",
    "generic": "#54A24B",
    "guided": "#B279A2",
    "control": "#72B7B2",
    "gold": "#F2CF5B",
}

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 180,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
    }
)


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# Figure 1: headline paper-versus-observed deltas.
labels = ["Runtime: weaker", "Runtime: stronger", "Guided tuning"]
paper = np.array([-0.30, 3.85, 3.89])
observed = np.array([-1.89, -1.89, -1.26])
y = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(8.2, 3.7))
ax.axvline(0, color="#333333", linewidth=1)
ax.scatter(paper, y + 0.13, s=90, color=COLORS["paper"], label="Paper")
ax.scatter(observed, y - 0.13, s=90, color=COLORS["targeted"], label="Reproduction")
for i, (p, o) in enumerate(zip(paper, observed)):
    ax.plot([p, o], [i + 0.13, i - 0.13], color="#c4c7c5", linewidth=2, zorder=0)
    ax.text(p, i + 0.28, f"{p:+.2f}", ha="center", va="bottom", color="#5f6368")
    ax.text(o, i - 0.28, f"{o:+.2f}", ha="center", va="top", color="#a33b3b")
ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.set_xlabel("Executable substep pass-rate change (percentage points)")
ax.set_title("The reported positive effects did not appear in the public reconstruction")
ax.legend(frameon=False, ncols=2, loc="upper right")
ax.set_xlim(-7, 6)
ax.grid(axis="x", alpha=0.2)
save(fig, "headline_claim_deltas.png")


# Figure 2: absolute primary-slice runtime rates.
models = ["7B", "14B", "32B"]
baseline = np.array([5, 15, 11]) / 53 * 100
targeted = np.array([4, 10, 10]) / 53 * 100
generic = np.array([5, 16, 15]) / 53 * 100
x = np.arange(len(models))
w = 0.24
fig, ax = plt.subplots(figsize=(8.2, 4.2))
for offset, values, name in [
    (-w, baseline, "Baseline"),
    (0, targeted, "Targeted procedure"),
    (w, generic, "Generic reminder"),
]:
    bars = ax.bar(x + offset, values, w, label=name, color=COLORS[name.split()[0].lower()])
    ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=9)
ax.set_xticks(x, models)
ax.set_ylabel("Executable substep pass rate (%)")
ax.set_xlabel("Qwen2.5-Coder-Instruct parameter scale")
ax.set_ylim(0, 36)
ax.set_title("Targeted procedures never beat the matched baseline")
ax.legend(frameon=False, ncols=3, loc="upper left")
ax.grid(axis="y", alpha=0.2)
save(fig, "runtime_conditions.png")


# Figure 3: exact eight-example adapter pairs.
seeds = ["42", "7", "123"]
guided = np.array([3, 5, 4]) / 53 * 100
control = np.array([4, 3, 7]) / 53 * 100
fig, ax = plt.subplots(figsize=(8.2, 4.3))
for i, seed in enumerate(seeds):
    ax.plot([0, 1], [control[i], guided[i]], color="#b8b8b8", linewidth=2)
    ax.scatter(0, control[i], s=80, color=COLORS["control"])
    ax.scatter(1, guided[i], s=80, color=COLORS["guided"])
    ax.text(1.04, guided[i], f"seed {seed}", va="center", fontsize=9)
ax.scatter(0, control.mean(), marker="D", s=110, color="#1b7f79", label="Mean")
ax.scatter(1, guided.mean(), marker="D", s=110, color="#7c3f73")
ax.plot([0, 1], [control.mean(), guided.mean()], color="#333333", linewidth=2.5)
ax.set_xticks([0, 1], ["No-procedure teacher", "Procedure-guided teacher"])
ax.set_ylabel("Procedure-free executable substep pass rate (%)")
ax.set_ylim(0, 16)
ax.set_xlim(-0.25, 1.35)
ax.set_title("Adapter effect changed sign across seeds")
ax.text(
    0.5,
    14.7,
    f"mean difference = {guided.mean() - control.mean():+.2f} percentage points",
    ha="center",
    fontsize=10,
)
ax.grid(axis="y", alpha=0.2)
save(fig, "adapter_seed_pairs.png")


# Figure 4: tuning and verification diagnostics.
fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1))
conditions = ["Untuned", "Guided\n2e-4", "Control\n2e-4", "Guided\n5e-5", "Control\n5e-5", "Gold\n(both LR)"]
rates = [5 / 53 * 100, 4 / 53 * 100, (14 / 3) / 53 * 100, 4 / 53 * 100, 4 / 53 * 100, 3 / 53 * 100]
colors = [
    COLORS["baseline"],
    COLORS["guided"],
    COLORS["control"],
    COLORS["guided"],
    COLORS["control"],
    COLORS["gold"],
]
bars = axes[0].bar(np.arange(len(conditions)), rates, color=colors)
axes[0].bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
axes[0].set_xticks(np.arange(len(conditions)), conditions, fontsize=8)
axes[0].set_ylabel("Substep pass rate (%)")
axes[0].set_ylim(0, 12)
axes[0].set_title("Tuning remained fragile")
axes[0].grid(axis="y", alpha=0.2)

yield_names = ["Guided", "No-procedure"]
yield_rates = np.array([8, 27]) / 50 * 100
bars = axes[1].bar(yield_names, yield_rates, color=[COLORS["guided"], COLORS["control"]], width=0.58)
axes[1].bar_label(bars, labels=["8/50", "27/50"], padding=3)
axes[1].set_ylabel("Teacher candidates passing official tests (%)")
axes[1].set_ylim(0, 65)
axes[1].set_title("Procedure guidance reduced verified yield")
axes[1].grid(axis="y", alpha=0.2)
save(fig, "tuning_and_verification.png")
