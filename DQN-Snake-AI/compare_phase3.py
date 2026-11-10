"""
Phase 3 Multi-Algorithm Experimental Comparison Suite.
Compares 6 distinct AI intelligence configurations under identical environment conditions:
- Experiment A: Original DQN (Phase 1 Baseline, 11 features)
- Experiment B: Best Phase 2 Variant (Double Dueling DQN + PER, 11 features)
- Experiment C: Enhanced State Representation (Double Dueling + PER, 24 features)
- Experiment D: Pure A* Pathfinding Agent (Algorithmic)
- Experiment E: DQN with A* Safety Shield Guidance (Hybrid AI)
- Experiment F: Best Hyperparameter Configuration

Outputs:
- experiments/phase3_comparison/phase3_comparison_results.txt
- experiments/phase3_comparison/phase3_comparison_results.csv
- experiments/phase3_comparison/phase3_comparison_barchart.png
"""

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import Agent, set_seed
from config import BASE_DIR, EXPERIMENTS_DIR, RESULTS_DIR
from model import build_network, load_model
from pathfinding import AStarAgent
from snake_ai import SnakeGameAI

PHASE3_EXP_DIR = EXPERIMENTS_DIR / "phase3_comparison"
PHASE3_EXP_DIR.mkdir(parents=True, exist_ok=True)


def train_enhanced_state_model(
    episodes: int = 60, seed: int = 42, output_path: Optional[Path] = None
) -> Path:
    """Train Double Dueling DQN with 24-feature Enhanced State Representation."""
    output_path = output_path or (PHASE3_EXP_DIR / "exp_c_enhanced_state.pth")
    if output_path.is_file():
        return output_path

    set_seed(seed)
    print(f"\nTraining Enhanced State Model (24 features) for {episodes} episodes...")
    game = SnakeGameAI(display=False, state_mode="enhanced", reward_mode="advanced")
    agent = Agent(
        state_size=24,
        dueling=True,
        use_double_dqn=True,
        use_per=True,
        seed=seed,
    )

    for ep in range(1, episodes + 1):
        game.reset()
        done = False
        while not done:
            s = game.get_state()
            act = agent.get_action(s)
            r, done, score = game.move(act.index(1))
            s_next = game.get_state()
            agent.remember(s, act, r, s_next, done)
            agent.train_replay_batch()

        agent.n_games = ep
        agent.decay_epsilon()
        if ep % 20 == 0 or ep == episodes:
            print(f"  [Enhanced State] Ep {ep}/{episodes} -> Score: {score} | Eps: {agent.epsilon:.3f}")

    agent.save_checkpoint(output_path, episode=episodes)
    return output_path


def train_best_hp_model(
    episodes: int = 60, seed: int = 42, output_path: Optional[Path] = None
) -> Path:
    """Train model using tuned optimal hyperparameters."""
    import json
    output_path = output_path or (PHASE3_EXP_DIR / "exp_f_best_hyperparams.pth")
    if output_path.is_file():
        return output_path

    set_seed(seed)
    hp_path = RESULTS_DIR / "best_hyperparameters.json"
    if hp_path.is_file():
        with open(hp_path, "r", encoding="utf-8") as f:
            hp = json.load(f)
        lr = hp.get("lr", 0.0003)
        gamma = hp.get("gamma", 0.95)
        batch_size = hp.get("batch_size", 128)
        target_sync = hp.get("target_update_freq", 50)
        eps_decay = hp.get("epsilon_decay", 0.992)
        per_alpha = hp.get("per_alpha", 0.6)
        per_beta = hp.get("per_beta", 0.4)
    else:
        lr, gamma, batch_size, target_sync, eps_decay, per_alpha, per_beta = (
            0.0003, 0.95, 128, 50, 0.992, 0.6, 0.4
        )

    print(f"\nTraining Best Hyperparameters Model (lr={lr}, gamma={gamma}, batch={batch_size}) for {episodes} episodes...")
    game = SnakeGameAI(display=False, state_mode="standard", reward_mode="advanced")
    agent = Agent(
        state_size=11,
        lr=lr,
        gamma=gamma,
        epsilon_decay=eps_decay,
        batch_size=batch_size,
        target_update_freq=target_sync,
        dueling=True,
        use_double_dqn=True,
        use_per=True,
        per_alpha=per_alpha,
        per_beta=per_beta,
        seed=seed,
    )

    for ep in range(1, episodes + 1):
        game.reset()
        done = False
        while not done:
            s = game.get_state()
            act = agent.get_action(s)
            r, done, score = game.move(act.index(1))
            s_next = game.get_state()
            agent.remember(s, act, r, s_next, done)
            agent.train_replay_batch()

        agent.n_games = ep
        agent.decay_epsilon()
        if ep % 20 == 0 or ep == episodes:
            print(f"  [Best HP Model] Ep {ep}/{episodes} -> Score: {score} | Eps: {agent.epsilon:.3f}")

    agent.save_checkpoint(output_path, episode=episodes)
    return output_path


def evaluate_experiment(
    exp_id: str,
    name: str,
    model_path: Optional[Path],
    is_pure_astar: bool = False,
    use_shield: bool = False,
    state_mode: str = "standard",
    games: int = 100,
    seed: int = 123,
) -> Dict[str, Any]:
    """Evaluate a single experimental configuration over N independent games."""
    set_seed(seed)
    astar_bot = AStarAgent() if (is_pure_astar or use_shield) else None
    agent = None

    if not is_pure_astar and model_path is not None:
        model, arch, detected_dim = load_model(model_path, map_location="cpu")
        state_mode = "enhanced" if detected_dim == 24 else "standard"
        agent = Agent(state_size=detected_dim, seed=seed)
        agent.model = model
        agent.dueling = (arch == "dueling")
        agent.model.eval()
        agent.training = False
        agent.epsilon = 0.0

    game = SnakeGameAI(display=False, state_mode=state_mode)

    scores = []
    steps = []
    rewards = []

    for _ in range(games):
        game.reset()
        done = False

        while not done:
            if is_pure_astar:
                act_idx = astar_bot.get_action(game)
            else:
                s = game.get_state()
                act_idx = agent.get_action_index(
                    s,
                    deterministic=True,
                    astar_shield=use_shield,
                    game=game if use_shield else None,
                )

            _, done, _ = game.move(act_idx)

        scores.append(game.score)
        steps.append(game.frame_iteration)
        rewards.append(game.total_reward)

    scores_arr = np.array(scores)
    return {
        "exp_id": exp_id,
        "name": name,
        "scores": scores,
        "mean_score": float(np.mean(scores_arr)),
        "max_score": int(np.max(scores_arr)),
        "min_score": int(np.min(scores_arr)),
        "median_score": float(np.median(scores_arr)),
        "std_score": float(np.std(scores_arr)),
        "avg_steps": float(np.mean(steps)),
        "avg_reward": float(np.mean(rewards)),
    }


def plot_phase3_comparison(
    results: List[Dict[str, Any]],
    save_path: Path = PHASE3_EXP_DIR / "phase3_comparison_barchart.png",
) -> None:
    """Generate professional comparative bar chart and distribution boxplot."""
    fig, (ax_bar, ax_box) = plt.subplots(1, 2, figsize=(16, 7), dpi=150)

    names = [r["name"] for r in results]
    short_names = [r["exp_id"] for r in results]
    means = [r["mean_score"] for r in results]
    stds = [r["std_score"] for r in results]
    scores_list = [r["scores"] for r in results]

    colors = ["#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c", "#d62728", "#17becf"]

    # 1. Bar Chart with Error Bars
    bars = ax_bar.bar(
        short_names,
        means,
        yerr=stds,
        capsize=5,
        color=colors[: len(results)],
        alpha=0.85,
        edgecolor="black",
    )
    for bar, m in zip(bars, means):
        ax_bar.annotate(
            f"{m:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    ax_bar.set_title("100-Game Greedy Evaluation Score (Mean +/- Std)", fontsize=13, fontweight="bold")
    ax_bar.set_ylabel("Average Score", fontsize=11)
    ax_bar.grid(True, linestyle="--", alpha=0.7)
    plt.setp(ax_bar.xaxis.get_majorticklabels(), rotation=15, ha="right")

    # 2. Boxplot Distributions
    ax_box.boxplot(
        scores_list,
        tick_labels=short_names,
        patch_artist=True,
        boxprops=dict(facecolor="#c6dbef", color="#1f77b4"),
        medianprops=dict(color="red", linewidth=2.0),
    )
    ax_box.set_title("Score Distribution Across 100 Games", fontsize=13, fontweight="bold")
    ax_box.set_ylabel("Score (Food Collected)", fontsize=11)
    ax_box.grid(True, linestyle="--", alpha=0.7)
    plt.setp(ax_box.xaxis.get_majorticklabels(), rotation=15, ha="right")

    fig.suptitle("Phase 3 AI Intelligence & Performance Comparison", fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.96, top=0.88, bottom=0.15, wspace=0.22)
    fig.savefig(str(save_path))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 Multi-Algorithm Benchmark Comparison")
    parser.add_argument("--games", type=int, default=100, help="Evaluation games per experiment (default: 100)")
    parser.add_argument("--train-episodes", type=int, default=60, help="Episodes to train new variants if needed (default: 60)")
    parser.add_argument("--seed", type=int, default=123, help="Evaluation seed (default: 123)")
    args = parser.parse_args()

    # Model paths
    model_a = EXPERIMENTS_DIR / "original_dqn" / "model.pth"
    if not model_a.is_file():
        model_a = BASE_DIR / "checkpoints" / "best_model.pth"

    model_b = EXPERIMENTS_DIR / "double_dueling_plus_per" / "model.pth"
    if not model_b.is_file():
        model_b = BASE_DIR / "checkpoints" / "best_model.pth"

    # Train / resolve models for Exp C and Exp F
    model_c = train_enhanced_state_model(episodes=args.train_episodes)
    model_f = train_best_hp_model(episodes=args.train_episodes)

    experiments_to_run = [
        {
            "exp_id": "Exp A",
            "name": "Exp A: Original DQN",
            "model_path": model_a,
            "is_pure_astar": False,
            "use_shield": False,
            "state_mode": "standard",
        },
        {
            "exp_id": "Exp B",
            "name": "Exp B: Best Phase 2 (DDQN+Dueling+PER)",
            "model_path": model_b,
            "is_pure_astar": False,
            "use_shield": False,
            "state_mode": "standard",
        },
        {
            "exp_id": "Exp C",
            "name": "Exp C: Enhanced State (24 features)",
            "model_path": model_c,
            "is_pure_astar": False,
            "use_shield": False,
            "state_mode": "enhanced",
        },
        {
            "exp_id": "Exp D",
            "name": "Exp D: Pure A* Agent",
            "model_path": None,
            "is_pure_astar": True,
            "use_shield": False,
            "state_mode": "standard",
        },
        {
            "exp_id": "Exp E",
            "name": "Exp E: DQN + A* Shield Guidance",
            "model_path": model_b,
            "is_pure_astar": False,
            "use_shield": True,
            "state_mode": "standard",
        },
        {
            "exp_id": "Exp F",
            "name": "Exp F: Best Hyperparameters",
            "model_path": model_f,
            "is_pure_astar": False,
            "use_shield": False,
            "state_mode": "standard",
        },
    ]

    print("\n" + "=" * 75)
    print("      PHASE 3 EXPERIMENTAL COMPARISON BENCHMARK (6 VARIANTS)")
    print("=" * 75)
    print(f"Independent Games per Configuration : {args.games}")
    print(f"Evaluation Random Seed              : {args.seed}")
    print(f"Output Directory                    : {PHASE3_EXP_DIR}")
    print("=" * 75)

    all_results = []

    for exp in experiments_to_run:
        print(f"\nEvaluating {exp['name']} over {args.games} games...")
        res = evaluate_experiment(
            exp_id=exp["exp_id"],
            name=exp["name"],
            model_path=exp["model_path"],
            is_pure_astar=exp["is_pure_astar"],
            use_shield=exp["use_shield"],
            state_mode=exp["state_mode"],
            games=args.games,
            seed=args.seed,
        )
        print(f"  -> Mean: {res['mean_score']:.2f} | Max: {res['max_score']} | Median: {res['median_score']:.1f} | Std: {res['std_score']:.2f} | Avg Steps: {res['avg_steps']:.1f}")
        all_results.append(res)

    # Print summary table
    sep_double = "=" * 95
    sep_single = "-" * 95

    table_lines = [
        "",
        sep_double,
        "                    PHASE 3 COMPREHENSIVE BENCHMARK RESULTS",
        sep_double,
        f"{'Experiment':<38} | {'Mean Score':<10} | {'Max':<5} | {'Median':<6} | {'Std Dev':<8} | {'Avg Steps':<10}",
        sep_single,
    ]
    for r in all_results:
        table_lines.append(
            f"{r['name']:<38} | {r['mean_score']:<10.2f} | {r['max_score']:<5d} | {r['median_score']:<6.1f} | {r['std_score']:<8.2f} | {r['avg_steps']:<10.1f}"
        )
    table_lines.append(sep_double)
    table_str = "\n".join(table_lines)
    print(table_str)

    # Save summary TXT
    txt_out = PHASE3_EXP_DIR / "phase3_comparison_results.txt"
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write(table_str)
    print(f"\nReport TXT saved to: {txt_out}")

    # Save summary CSV
    csv_out = PHASE3_EXP_DIR / "phase3_comparison_results.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Experiment_ID", "Name", "Mean_Score", "Max_Score", "Min_Score", "Median_Score", "Std_Dev", "Avg_Steps", "Avg_Reward"])
        for r in all_results:
            writer.writerow([
                r["exp_id"],
                r["name"],
                f"{r['mean_score']:.2f}",
                r["max_score"],
                r["min_score"],
                f"{r['median_score']:.1f}",
                f"{r['std_score']:.2f}",
                f"{r['avg_steps']:.1f}",
                f"{r['avg_reward']:.2f}",
            ])
    print(f"Report CSV saved to: {csv_out}")

    # Plot comparison charts
    plot_out = PHASE3_EXP_DIR / "phase3_comparison_barchart.png"
    plot_phase3_comparison(all_results, save_path=plot_out)
    print(f"Comparison chart saved to: {plot_out}")


if __name__ == "__main__":
    main()

