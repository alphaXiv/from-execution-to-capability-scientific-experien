# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo>=0.14.17",
#   "matplotlib>=3.9",
#   "numpy>=1.26",
# ]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # From execution to capability: a public SciCode reproduction

    Scientific coding models may improve when given a useful procedure at runtime,
    but can a weaker model retain that experience after the procedure is removed?
    This notebook reproduces the two central comparisons from arXiv:2607.24459 using
    public SciCode tests and open Qwen2.5-Coder models. The evidence is embedded:
    opening the notebook does **not** rerun expensive model inference.

    **Verdict: not reproduced under this bounded public reconstruction.**
    Runtime procedures did not help the stronger substitute, and three matched
    procedure-guided adapters averaged slightly below their no-procedure controls.
    """)
    return


@app.cell
def _(np):
    headline_labels = ["Runtime: weaker", "Runtime: stronger", "Guided tuning"]
    paper_deltas = np.array([-0.30, 3.85, 3.89])
    observed_deltas = np.array([-1.89, -1.89, -1.26])
    runtime_models = ["7B", "14B", "32B"]
    runtime_baseline = np.array([5, 15, 11]) / 53 * 100
    runtime_targeted = np.array([4, 10, 10]) / 53 * 100
    runtime_generic = np.array([5, 16, 15]) / 53 * 100
    adapter_seeds = ["42", "7", "123"]
    adapter_guided = np.array([3, 5, 4]) / 53 * 100
    adapter_control = np.array([4, 3, 7]) / 53 * 100
    return (
        adapter_control,
        adapter_guided,
        adapter_seeds,
        headline_labels,
        observed_deltas,
        paper_deltas,
        runtime_baseline,
        runtime_generic,
        runtime_models,
        runtime_targeted,
    )


@app.cell
def _(headline_labels, np, observed_deltas, paper_deltas, plt):
    y_headline = np.arange(len(headline_labels))
    fig_headline, ax_headline = plt.subplots(figsize=(8.2, 3.7))
    ax_headline.axvline(0, color="#333333", linewidth=1)
    ax_headline.scatter(paper_deltas, y_headline + 0.13, s=90, color="#9aa0a6", label="Paper")
    ax_headline.scatter(
        observed_deltas, y_headline - 0.13, s=90, color="#E45756", label="Reproduction"
    )
    for i_headline, (p_headline, o_headline) in enumerate(
        zip(paper_deltas, observed_deltas)
    ):
        ax_headline.plot(
            [p_headline, o_headline],
            [i_headline + 0.13, i_headline - 0.13],
            color="#c4c7c5",
            linewidth=2,
            zorder=0,
        )
    ax_headline.set_yticks(y_headline, headline_labels)
    ax_headline.invert_yaxis()
    ax_headline.set_xlabel("Executable substep pass-rate change (percentage points)")
    ax_headline.set_title("Paper effects versus observed effects")
    ax_headline.legend(frameon=False, ncols=2, loc="upper right")
    ax_headline.grid(axis="x", alpha=0.2)
    fig_headline.tight_layout()
    fig_headline
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Runtime procedure injection

    Twelve stratified public test problems produced 53 executable substeps. Each
    model saw either no extra prompt, a reconstructed cross-task scientific-computing
    procedure, or a length-matched generic code-quality reminder. The official
    SciCode HDF5 tests determined pass/fail.
    """)
    return


@app.cell
def _(
    np,
    plt,
    runtime_baseline,
    runtime_generic,
    runtime_models,
    runtime_targeted,
):
    x_runtime = np.arange(len(runtime_models))
    width_runtime = 0.24
    fig_runtime, ax_runtime = plt.subplots(figsize=(8.2, 4.2))
    for offset_runtime, values_runtime, name_runtime, color_runtime in [
        (-width_runtime, runtime_baseline, "Baseline", "#4C78A8"),
        (0, runtime_targeted, "Targeted procedure", "#E45756"),
        (width_runtime, runtime_generic, "Generic reminder", "#54A24B"),
    ]:
        bars_runtime = ax_runtime.bar(
            x_runtime + offset_runtime,
            values_runtime,
            width_runtime,
            label=name_runtime,
            color=color_runtime,
        )
        ax_runtime.bar_label(bars_runtime, fmt="%.1f", padding=2)
    ax_runtime.set_xticks(x_runtime, runtime_models)
    ax_runtime.set_ylabel("Executable substep pass rate (%)")
    ax_runtime.set_title("Targeted procedures never beat the matched baseline")
    ax_runtime.legend(frameon=False, ncols=3, loc="upper left")
    ax_runtime.grid(axis="y", alpha=0.2)
    fig_runtime.tight_layout()
    fig_runtime
    return


@app.cell
def _(mo):
    mo.md(r"""
    The targeted procedure changed the 7B, 14B, and 32B pass rates by −1.89,
    −9.43, and −1.89 percentage points. Generic reminders were neutral or helpful,
    so prompt length does not explain the targeted loss. The paper's stronger model
    instead gained +3.85 substep points.

    ## 2. Procedure-guided consolidation

    The 32B teacher generated 50 development-step candidates in each condition.
    Official tests gated examples, and both adapters received exactly eight verified
    examples. Separate rank-16, one-epoch LoRA adapters were trained on the 7B model
    and evaluated without procedures on the same held-out test slice.
    """)
    return


@app.cell
def _(adapter_control, adapter_guided, adapter_seeds, plt):
    fig_adapter, ax_adapter = plt.subplots(figsize=(8.2, 4.3))
    for i_adapter, seed_adapter in enumerate(adapter_seeds):
        ax_adapter.plot(
            [0, 1],
            [adapter_control[i_adapter], adapter_guided[i_adapter]],
            color="#b8b8b8",
            linewidth=2,
        )
        ax_adapter.scatter(0, adapter_control[i_adapter], s=80, color="#72B7B2")
        ax_adapter.scatter(1, adapter_guided[i_adapter], s=80, color="#B279A2")
        ax_adapter.text(
            1.04, adapter_guided[i_adapter], f"seed {seed_adapter}", va="center"
        )
    ax_adapter.plot(
        [0, 1],
        [adapter_control.mean(), adapter_guided.mean()],
        color="#333333",
        linewidth=2.5,
    )
    ax_adapter.scatter(0, adapter_control.mean(), marker="D", s=110, color="#1b7f79")
    ax_adapter.scatter(1, adapter_guided.mean(), marker="D", s=110, color="#7c3f73")
    ax_adapter.set_xticks([0, 1], ["No-procedure teacher", "Procedure-guided teacher"])
    ax_adapter.set_ylabel("Procedure-free executable substep pass rate (%)")
    ax_adapter.set_ylim(0, 16)
    ax_adapter.set_xlim(-0.25, 1.35)
    ax_adapter.set_title("Adapter effect changed sign across seeds")
    ax_adapter.grid(axis="y", alpha=0.2)
    fig_adapter.tight_layout()
    fig_adapter
    return


@app.cell
def _(adapter_control, adapter_guided, mo):
    mean_delta = adapter_guided.mean() - adapter_control.mean()
    mo.md(
        f"""
    Guided-minus-control differences were −1/53, +2/53, and −3/53 passes.
    The mean difference was **{mean_delta:+.2f} percentage points**, opposite the
    paper's +3.89-point all-problem result. Lowering the learning rate made the two
    conditions tie at 4/53. Gold-code adapters scored 3/53 at both rates, below the
    untuned 5/53 baseline, so this tiny tuning regime was intrinsically fragile.

    ## Takeaway

    This run is evidence about the public reconstruction, not a refutation of the
    paper. The unavailable original procedures, trajectories, concretized data,
    checkpoints, and full-tuning recipe could materially change the result. The
    experiment campaign used Kubernetes, NVIDIA RTX PRO 6000 Blackwell GPUs,
    16 GPUs allocated concurrently at peak, and 3.01 hours of elapsed wall time.
    """
    )
    return


if __name__ == "__main__":
    app.run()
