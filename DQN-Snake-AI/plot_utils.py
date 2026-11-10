"""
Plotting and Visualization Utilities for DQN Snake AI.
Generates publication-quality charts for scores, moving averages, loss, epsilon decay,
evaluation score distributions, and Phase 2 multi-experiment comparative benchmarks.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

# Set Matplotlib temp directory to avoid sandbox/permission warnings on Windows
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import numpy as np


def compute_moving_average(data: List[float], window: int = 50) -> List[float]:
    """Calculate moving average with expanding window at boundaries."""
    if not data:
        return []
    moving_avg = []
    for i in range(len(data)):
        start_idx = max(0, i - window + 1)
        sub = data[start_idx : i + 1]
        moving_avg.append(float(np.mean(sub)))
    return moving_avg


def plot_training_results(
    scores: List[int],
    moving_avg_scores: Optional[List[float]] = None,
    losses: Optional[List[float]] = None,
    epsilons: Optional[List[float]] = None,
    save_dir: Union[str, Path] = "results",
    legacy_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Generate and save comprehensive training graphs:
        1. Episode score versus training episode
        2. Moving average score (50 episodes)
        3. Training loss curve
        4. Epsilon decay curve
        5. Combined 4-panel dashboard
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    episodes = list(range(1, len(scores) + 1))

    if moving_avg_scores is None or len(moving_avg_scores) != len(scores):
        moving_avg_scores = compute_moving_average(scores, window=50)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = {
        "score": "#1f77b4",
        "avg": "#ff7f0e",
        "loss": "#d62728",
        "epsilon": "#2ca02c",
    }

    # 1. Episode Score Plot
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(episodes, scores, label="Episode Score", color=colors["score"], alpha=0.6, linewidth=1.2)
    ax.plot(episodes, moving_avg_scores, label="50-Episode Moving Avg", color=colors["avg"], linewidth=2.0)
    ax.set_title("DQN Snake Training - Episode Scores", fontsize=14, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.12)
    fig.savefig(save_dir / "training_scores.png")
    plt.close(fig)

    # 2. Moving Average Score Plot
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(episodes, moving_avg_scores, label="50-Episode Moving Avg", color=colors["avg"], linewidth=2.2)
    if moving_avg_scores:
        peak_idx = int(np.argmax(moving_avg_scores))
        peak_val = moving_avg_scores[peak_idx]
        ax.scatter([peak_idx + 1], [peak_val], color="red", zorder=5)
        ax.annotate(
            f"Peak: {peak_val:.2f} (Ep {peak_idx + 1})",
            xy=(peak_idx + 1, peak_val),
            xytext=(peak_idx + 1 + 10, peak_val + 0.5),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontweight="bold",
        )
    ax.set_title("50-Episode Moving Average Score", fontsize=14, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Average Score", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.12)
    fig.savefig(save_dir / "moving_average_score.png")
    plt.close(fig)

    # 3. Training Loss Plot
    if losses and len(losses) > 0:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        loss_x = list(range(1, len(losses) + 1))
        ax.plot(loss_x, losses, label="Training Loss", color=colors["loss"], alpha=0.4, linewidth=0.8)
        smooth_losses = compute_moving_average(losses, window=min(50, len(losses)))
        ax.plot(loss_x, smooth_losses, label="Smoothed Loss", color="#800000", linewidth=2.0)
        ax.set_title("DQN Training Loss (Smooth L1 / Huber)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Update Step / Episode", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.7)
        fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.12)
        fig.savefig(save_dir / "training_loss.png")
        plt.close(fig)

    # 4. Epsilon Decay Plot
    if epsilons and len(epsilons) > 0:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        eps_x = list(range(1, len(epsilons) + 1))
        ax.plot(eps_x, epsilons, label="Epsilon (Exploration Rate)", color=colors["epsilon"], linewidth=2.0)
        ax.set_title("Epsilon-Greedy Exploration Decay", fontsize=14, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Epsilon", fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.7)
        fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.12)
        fig.savefig(save_dir / "epsilon_decay.png")
        plt.close(fig)

    # 5. Combined 4-Panel Training Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    
    # Subplot (0, 0): Scores
    axes[0, 0].plot(episodes, scores, color=colors["score"], alpha=0.5, label="Score")
    axes[0, 0].plot(episodes, moving_avg_scores, color=colors["avg"], linewidth=2.0, label="50-Ep Avg")
    axes[0, 0].set_title("Episode Score & Moving Average", fontweight="bold")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle="--", alpha=0.7)

    # Subplot (0, 1): Moving Average Only
    axes[0, 1].plot(episodes, moving_avg_scores, color=colors["avg"], linewidth=2.2)
    axes[0, 1].set_title("50-Episode Moving Average Score", fontweight="bold")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Moving Avg Score")
    axes[0, 1].grid(True, linestyle="--", alpha=0.7)

    # Subplot (1, 0): Loss
    if losses and len(losses) > 0:
        loss_x = list(range(1, len(losses) + 1))
        smooth_losses = compute_moving_average(losses, window=min(50, len(losses)))
        axes[1, 0].plot(loss_x, losses, color=colors["loss"], alpha=0.3, label="Loss")
        axes[1, 0].plot(loss_x, smooth_losses, color="#800000", linewidth=1.8, label="Smoothed")
        axes[1, 0].set_title("Huber Loss Curve", fontweight="bold")
        axes[1, 0].set_xlabel("Step / Episode")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].legend()
        axes[1, 0].grid(True, linestyle="--", alpha=0.7)
    else:
        axes[1, 0].text(0.5, 0.5, "No Loss Data", ha="center", va="center")

    # Subplot (1, 1): Epsilon
    if epsilons and len(epsilons) > 0:
        eps_x = list(range(1, len(epsilons) + 1))
        axes[1, 1].plot(eps_x, epsilons, color=colors["epsilon"], linewidth=2.0)
        axes[1, 1].set_title("Epsilon Exploration Decay", fontweight="bold")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Epsilon")
        axes[1, 1].set_ylim(-0.05, 1.05)
        axes[1, 1].grid(True, linestyle="--", alpha=0.7)
    else:
        axes[1, 1].text(0.5, 0.5, "No Epsilon Data", ha="center", va="center")

    fig.suptitle("DQN Snake AI - Training Performance Dashboard", fontsize=16, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.95, top=0.92, bottom=0.08, hspace=0.28, wspace=0.22)
    fig.savefig(save_dir / "training_dashboard.png")
    plt.close(fig)

    # Legacy figure compatibility
    if legacy_path is not None:
        legacy_p = Path(legacy_path)
        legacy_p.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(scores, label="Scores")
        ax.plot(moving_avg_scores, label="50-Game Avg", linewidth=2.0)
        ax.set_xlabel("Game")
        ax.set_ylabel("Score")
        ax.set_title("Improved DQN Snake Training")
        ax.legend()
        ax.grid(True)
        fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.12)
        fig.savefig(str(legacy_p))
        plt.close(fig)


def plot_evaluation_distribution(
    scores: List[int],
    save_path: Union[str, Path] = "results/evaluation_distribution.png",
) -> None:
    """
    Generate publication-quality evaluation score distribution graph:
        - Histogram with mean and median indicators
        - Boxplot indicating score spread, quartiles, and outliers
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if not scores:
        return

    scores_arr = np.array(scores)
    mean_val = float(np.mean(scores_arr))
    median_val = float(np.median(scores_arr))
    std_val = float(np.std(scores_arr))
    max_val = int(np.max(scores_arr))
    min_val = int(np.min(scores_arr))

    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, figsize=(10, 8), dpi=150, sharex=True, gridspec_kw={"height_ratios": [0.3, 0.7]}
    )

    # Boxplot
    ax_box.boxplot(scores_arr, vert=False, patch_artist=True,
                   boxprops=dict(facecolor="#a6c8e0", color="#1f77b4"),
                   medianprops=dict(color="red", linewidth=2.0),
                   whiskerprops=dict(color="#1f77b4"),
                   capprops=dict(color="#1f77b4"))
    ax_box.set_yticks([])
    ax_box.grid(True, linestyle="--", alpha=0.7)

    # Histogram
    bins = max(5, min(30, max_val - min_val + 1))
    ax_hist.hist(scores_arr, bins=bins, color="#1f77b4", edgecolor="black", alpha=0.7, density=False)
    ax_hist.axvline(mean_val, color="green", linestyle="--", linewidth=2.0, label=f"Mean: {mean_val:.2f}")
    ax_hist.axvline(median_val, color="red", linestyle="-", linewidth=2.0, label=f"Median: {median_val:.1f}")

    # Stats box
    stats_text = (
        f"Games: {len(scores)}\n"
        f"Mean: {mean_val:.2f}\n"
        f"Std: {std_val:.2f}\n"
        f"Median: {median_val:.1f}\n"
        f"Min: {min_val} | Max: {max_val}"
    )
    ax_hist.text(
        0.95,
        0.95,
        stats_text,
        transform=ax_hist.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9),
        fontsize=10,
    )

    ax_hist.set_title("Evaluation Score Distribution", fontsize=14, fontweight="bold")
    ax_hist.set_xlabel("Score (Food Collected)", fontsize=12)
    ax_hist.set_ylabel("Game Count (Frequency)", fontsize=12)
    ax_hist.legend(loc="upper left")
    ax_hist.grid(True, linestyle="--", alpha=0.7)

    fig.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08, hspace=0.15)
    fig.savefig(str(save_path))
    plt.close(fig)


def plot_experiment_comparisons(
    exp_data: dict,
    save_path: Union[str, Path] = "experiments/experiments_comparison.png",
) -> None:
    """
    Generate comparative multi-panel chart comparing Phase 2 experiment variants:
        Panel 1: Moving average score learning curves across experiments.
        Panel 2: Training loss curves across experiments.
        Panel 3: Evaluation mean scores with standard deviation error bars.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    variant_colors = [
        "#1f77b4",  # Blue (DQN)
        "#ff7f0e",  # Orange (DDQN)
        "#2ca02c",  # Green (Dueling DDQN)
        "#d62728",  # Red (Dueling DDQN + PER)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)

    # 1. Moving Average Curves
    for (name, info), col in zip(exp_data.items(), variant_colors):
        scores = info.get("scores", [])
        if scores:
            ma = compute_moving_average(scores, window=min(50, len(scores)))
            axes[0].plot(range(1, len(ma) + 1), ma, label=name, color=col, linewidth=2.0)
    axes[0].set_title("Moving Average Score Progression", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Average Score")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(True, linestyle="--", alpha=0.7)

    # 2. Smoothed Loss Curves
    for (name, info), col in zip(exp_data.items(), variant_colors):
        losses = info.get("losses", [])
        if losses:
            smooth_loss = compute_moving_average(losses, window=min(20, len(losses)))
            axes[1].plot(range(1, len(smooth_loss) + 1), smooth_loss, label=name, color=col, linewidth=1.8)
    axes[1].set_title("Training Huber Loss", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Loss")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, linestyle="--", alpha=0.7)

    # 3. Evaluation Scores Bar Chart with Error Bars
    names = list(exp_data.keys())
    means = [exp_data[k].get("eval_mean", 0.0) for k in names]
    stds = [exp_data[k].get("eval_std", 0.0) for k in names]
    short_names = [k.replace("Experiment ", "Exp ") for k in names]

    bars = axes[2].bar(
        short_names,
        means,
        yerr=stds,
        capsize=5,
        color=variant_colors[: len(names)],
        alpha=0.85,
        edgecolor="black",
    )
    for bar, mean_v in zip(bars, means):
        height = bar.get_height()
        axes[2].annotate(
            f"{mean_v:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    axes[2].set_title("Independent Greedy Evaluation Score", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("Average Score (+/- Std Dev)")
    axes[2].grid(True, linestyle="--", alpha=0.7)
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=15, ha="right", fontsize=9)

    fig.suptitle("Phase 2 Reinforcement Learning Algorithms Comparison", fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.18, wspace=0.25)
    fig.savefig(str(save_path))
    plt.close(fig)
