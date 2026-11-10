"""
Hyperparameter Optimization for DQN Snake AI (Phases 3 & 4).
Performs reproducible Random Search across multiple seeds to identify optimal hyperparameters.
Ranks candidates based on risk-adjusted evaluation score (Mean - 0.5 * Std + 0.1 * Median), avoiding
selection based solely on max score or single fortunate runs.

Outputs:
- results/hyperparameter_tuning_results.json
- results/hyperparameter_tuning_summary.csv
- results/best_hyperparameters.json
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from agent import Agent, set_seed
from config import BASE_DIR, RESULTS_DIR
from snake_ai import SnakeGameAI

SEARCH_SPACE = {
    "lr": [0.0003, 0.0005, 0.001],
    "gamma": [0.95, 0.98, 0.99],
    "epsilon_decay": [0.992, 0.995, 0.997],
    "batch_size": [32, 64, 128],
    "target_update_freq": [50, 100, 200],
    "per_alpha": [0.4, 0.6],
    "per_beta": [0.4, 0.5],
}


def sample_hyperparameters(rng: Optional[random.Random] = None) -> Dict[str, Any]:
    """Sample a random candidate hyperparameter configuration from search space."""
    chooser = rng.choice if rng else random.choice
    return {
        "lr": chooser(SEARCH_SPACE["lr"]),
        "gamma": chooser(SEARCH_SPACE["gamma"]),
        "epsilon_decay": chooser(SEARCH_SPACE["epsilon_decay"]),
        "batch_size": chooser(SEARCH_SPACE["batch_size"]),
        "target_update_freq": chooser(SEARCH_SPACE["target_update_freq"]),
        "per_alpha": chooser(SEARCH_SPACE["per_alpha"]),
        "per_beta": chooser(SEARCH_SPACE["per_beta"]),
        "dueling": True,
        "use_double_dqn": True,
        "use_per": True,
    }


def evaluate_trial_seed(
    params: Dict[str, Any],
    train_episodes: int,
    eval_games: int,
    seed: int,
) -> Tuple[float, float, float, int]:
    """Train and greedily evaluate a trial on a single random seed."""
    set_seed(seed)
    agent = Agent(
        lr=params["lr"],
        gamma=params["gamma"],
        epsilon_decay=params["epsilon_decay"],
        batch_size=params["batch_size"],
        min_replay_size=max(params["batch_size"], 64),
        target_update_freq=params["target_update_freq"],
        use_double_dqn=params["use_double_dqn"],
        dueling=params["dueling"],
        use_per=params["use_per"],
        per_alpha=params["per_alpha"],
        per_beta=params["per_beta"],
        seed=seed,
    )
    game = SnakeGameAI(display=False, reward_mode="phase1")

    # 1. Training Phase
    for ep in range(1, train_episodes + 1):
        game.reset()
        done = False
        while not done:
            s_old = game.get_state()
            action = agent.get_action(s_old)
            r, done, score = game.move(action.index(1))
            s_new = game.get_state()
            agent.remember(s_old, action, r, s_new, done)
            agent.train_replay_batch(params["batch_size"])

        agent.n_games = ep
        agent.decay_epsilon()

    # 2. Greedy Evaluation Phase
    agent.model.eval()
    agent.training = False
    agent.epsilon = 0.0

    eval_scores = []
    for _ in range(eval_games):
        game.reset()
        done = False
        while not done:
            s = game.get_state()
            act = agent.get_action(s, deterministic=True)
            _, done, _ = game.move(act.index(1))
        eval_scores.append(game.score)

    scores_arr = np.array(eval_scores)
    return (
        float(np.mean(scores_arr)),
        float(np.std(scores_arr)),
        float(np.median(scores_arr)),
        int(np.max(scores_arr)),
    )


def run_hyperparameter_optimization(
    trials: int = 5,
    episodes: int = 80,
    eval_games: int = 20,
    seeds: Optional[List[int]] = None,
    output_dir: Path = RESULTS_DIR,
    search_seed: int = 42,
) -> Dict[str, Any]:
    """
    Execute reproducible hyperparameter search across multiple trials and seeds.
    """
    seeds = seeds or [42, 101, 777]
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(search_seed)
    all_trial_results = []

    print("=" * 72)
    print("      DQN SNAKE AI - HYPERPARAMETER OPTIMIZATION (PHASE 4)")
    print("=" * 72)
    print(f"Total Candidate Trials : {trials}")
    print(f"Training Episodes/Seed : {episodes}")
    print(f"Evaluation Games/Seed  : {eval_games}")
    print(f"Seeds Evaluated        : {seeds}")
    print(f"Search Random Seed     : {search_seed}")
    print("=" * 72)

    for trial_idx in range(1, trials + 1):
        params = sample_hyperparameters(rng)
        print(f"\n[Trial {trial_idx}/{trials}] Testing Parameters:")
        print(f"  lr={params['lr']}, gamma={params['gamma']}, batch_size={params['batch_size']}, eps_decay={params['epsilon_decay']}")
        print(f"  target_sync={params['target_update_freq']}, per_alpha={params['per_alpha']}, per_beta={params['per_beta']}")

        seed_means = []
        seed_stds = []
        seed_medians = []
        seed_maxs = []

        for s in seeds:
            mean_s, std_s, med_s, max_s = evaluate_trial_seed(params, episodes, eval_games, seed=s)
            print(f"    Seed {s} -> Mean: {mean_s:.2f} | Std: {std_s:.2f} | Median: {med_s:.1f} | Max: {max_s}")
            seed_means.append(mean_s)
            seed_stds.append(std_s)
            seed_medians.append(med_s)
            seed_maxs.append(max_s)

        avg_mean = float(np.mean(seed_means))
        avg_std = float(np.mean(seed_stds))
        avg_med = float(np.mean(seed_medians))
        overall_max = int(np.max(seed_maxs))

        # Risk-adjusted composite score: rewards high mean and median performance while penalizing instability
        risk_adjusted_score = float(avg_mean - 0.5 * avg_std + 0.1 * avg_med)

        print(f"  -> Aggregate Mean: {avg_mean:.2f} | Std: {avg_std:.2f} | Median: {avg_med:.1f} | Max: {overall_max} | Composite: {risk_adjusted_score:.2f}")

        trial_data = {
            "trial": trial_idx,
            "params": params,
            "eval_mean": avg_mean,
            "eval_std": avg_std,
            "eval_median": avg_med,
            "eval_max": overall_max,
            "composite_score": risk_adjusted_score,
            "seed_results": [
                {"seed": s, "mean": m, "std": sd, "median": md, "max": mx}
                for s, m, sd, md, mx in zip(seeds, seed_means, seed_stds, seed_medians, seed_maxs)
            ],
        }
        all_trial_results.append(trial_data)

    # Sort trials by risk-adjusted composite score
    all_trial_results.sort(key=lambda x: x["composite_score"], reverse=True)
    best_trial = all_trial_results[0]

    # Save complete JSON
    json_path = output_dir / "hyperparameter_tuning_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_trial_results, f, indent=4)

    # Save best hyperparameters JSON
    best_path = output_dir / "best_hyperparameters.json"
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_trial["params"], f, indent=4)

    # Save CSV Summary
    csv_path = output_dir / "hyperparameter_tuning_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Trial", "Composite_Score", "Eval_Mean", "Eval_Std", "Eval_Median", "Eval_Max", "LR", "Gamma", "Batch_Size", "Target_Sync", "PER_Alpha", "PER_Beta"])
        for rank, res in enumerate(all_trial_results, 1):
            p = res["params"]
            writer.writerow([
                rank,
                res["trial"],
                f"{res['composite_score']:.2f}",
                f"{res['eval_mean']:.2f}",
                f"{res['eval_std']:.2f}",
                f"{res['eval_median']:.1f}",
                res["eval_max"],
                p["lr"],
                p["gamma"],
                p["batch_size"],
                p["target_update_freq"],
                p["per_alpha"],
                p["per_beta"],
            ])

    print("\n" + "=" * 76)
    print("            HYPERPARAMETER OPTIMIZATION SUMMARY RANKING")
    print("=" * 76)
    print(f"{'Rank':<5} | {'Trial':<6} | {'Composite':<10} | {'Eval Mean':<10} | {'Std':<6} | {'Median':<6} | {'Max':<5} | {'LR':<8} | {'Batch':<6}")
    print("-" * 76)
    for rank, res in enumerate(all_trial_results, 1):
        p = res["params"]
        print(f"{rank:<5d} | {res['trial']:<6d} | {res['composite_score']:<10.2f} | {res['eval_mean']:<10.2f} | {res['eval_std']:<6.2f} | {res['eval_median']:<6.1f} | {res['eval_max']:<5d} | {p['lr']:<8.4f} | {p['batch_size']:<6d}")
    print("=" * 76)
    print(f"Best Configuration: Trial {best_trial['trial']} (Score: {best_trial['composite_score']:.2f})")
    print(f"Saved best parameters to: {best_path}")

    return best_trial


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization for DQN Snake AI")
    parser.add_argument("--trials", type=int, default=5, help="Number of random search trials (default: 5)")
    parser.add_argument("--episodes", type=int, default=80, help="Training episodes per trial seed (default: 80)")
    parser.add_argument("--eval-games", type=int, default=20, help="Evaluation games per trial seed (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Search master seed (default: 42)")
    args = parser.parse_args()

    run_hyperparameter_optimization(
        trials=args.trials,
        episodes=args.episodes,
        eval_games=args.eval_games,
        seeds=[42, 101, 777],
        search_seed=args.seed,
    )


if __name__ == "__main__":
    main()
