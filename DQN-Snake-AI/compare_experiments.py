"""
Phase 2 Experiment Orchestration and Comparative Benchmark.
Independently configures, trains, evaluates, and compares 4 model variants:
- Experiment A: Original DQN (Phase 1 Baseline)
- Experiment B: Double DQN (DDQN)
- Experiment C: Double Dueling DQN
- Experiment D: Double Dueling DQN + Prioritized Experience Replay (PER)

Generates comparison tables in CSV and TXT formats and multi-panel charts.
"""

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from agent import Agent, set_seed
from config import BASE_DIR, EXPERIMENTS_DIR, RESULTS_DIR
from model import load_model
from plot_utils import plot_experiment_comparisons
from snake_ai import SnakeGameAI


EXPERIMENT_CONFIGS = {
    "Experiment A (Original DQN)": {
        "double_dqn": False,
        "dueling": False,
        "use_per": False,
        "description": "Single Q-Network baseline with uniform replay",
    },
    "Experiment B (Double DQN)": {
        "double_dqn": True,
        "dueling": False,
        "use_per": False,
        "description": "Decoupled action selection and evaluation",
    },
    "Experiment C (Double Dueling DQN)": {
        "double_dqn": True,
        "dueling": True,
        "use_per": False,
        "description": "DDQN with decoupled Value V(s) and Advantage A(s,a) streams",
    },
    "Experiment D (Double Dueling + PER)": {
        "double_dqn": True,
        "dueling": True,
        "use_per": True,
        "description": "DDQN + Dueling + Prioritized Experience Replay via SumTree",
    },
}


def run_single_experiment_training(
    name: str,
    cfg: Dict[str, Any],
    episodes: int = 150,
    seed: int = 42,
    output_dir: Path = EXPERIMENTS_DIR,
) -> Tuple[Agent, List[int], List[float]]:
    """Train a specific variant for a given number of episodes."""
    set_seed(seed)
    exp_slug = name.split("(")[1].replace(")", "").strip().lower().replace(" ", "_").replace("+", "plus")
    exp_dir = output_dir / exp_slug
    exp_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = exp_dir / "model.pth"

    print("\n" + "=" * 65)
    print(f"  Training {name.upper()}")
    print(f"  Details: {cfg['description']}")
    print(f"  Episodes: {episodes} | Output: {exp_dir}")
    print("=" * 65)

    agent = Agent(
        use_double_dqn=cfg["double_dqn"],
        dueling=cfg["dueling"],
        use_per=cfg["use_per"],
        seed=seed,
    )
    game = SnakeGameAI(display=False)

    scores: List[int] = []
    losses: List[float] = []

    for ep in range(1, episodes + 1):
        game.reset()
        done = False
        step_losses = []

        while not done:
            s_old = game.get_state()
            action = agent.get_action(s_old)
            r, done, score = game.move(action.index(1))
            s_new = game.get_state()

            agent.remember(s_old, action, r, s_new, done)
            loss = agent.train_replay_batch()
            if loss is not None:
                step_losses.append(loss)

        agent.n_games = ep
        agent.decay_epsilon()
        scores.append(score)
        ep_loss = float(np.mean(step_losses)) if step_losses else (losses[-1] if losses else 0.0)
        losses.append(ep_loss)

        if ep % 25 == 0 or ep == 1 or ep == episodes:
            avg_25 = float(np.mean(scores[-min(len(scores), 25):]))
            print(f"  [{name}] Ep {ep:3d}/{episodes:3d} -> Score: {score:2d} | Avg(25): {avg_25:5.2f} | Loss: {ep_loss:6.4f} | Eps: {agent.epsilon:.3f}")

    # Save trained checkpoint
    agent.save_checkpoint(model_save_path, episode=episodes, scores=scores, losses=losses)
    print(f"  Model checkpoint saved to: {model_save_path}")

    return agent, scores, losses


def evaluate_agent_greedy(
    model_path: Path,
    num_games: int = 50,
    seed: int = 123,
) -> Dict[str, float]:
    """Perform greedy evaluation over independent games."""
    set_seed(seed)
    model, arch_type = load_model(model_path, map_location="cpu")
    agent = Agent()
    agent.model = model
    agent.dueling = (arch_type == "dueling")
    agent.model.eval()
    agent.training = False
    agent.epsilon = 0.0

    game = SnakeGameAI(display=False)
    scores = []
    steps = []

    for _ in range(num_games):
        game.reset()
        done = False
        while not done:
            s = game.get_state()
            act = agent.get_action(s, deterministic=True)
            _, done, _ = game.move(act.index(1))
        scores.append(game.score)
        steps.append(game.frame_iteration)

    scores_arr = np.array(scores)
    return {
        "eval_mean": float(np.mean(scores_arr)),
        "eval_max": int(np.max(scores_arr)),
        "eval_min": int(np.min(scores_arr)),
        "eval_median": float(np.median(scores_arr)),
        "eval_std": float(np.std(scores_arr)),
        "eval_steps_mean": float(np.mean(steps)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 Comparative Experiments Runner")
    parser.add_argument("--episodes", type=int, default=80, help="Training episodes per experiment (default: 80)")
    parser.add_argument("--eval-games", type=int, default=50, help="Greedy evaluation games per variant (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--skip-train", action="store_true", default=False, help="Skip training and evaluate existing checkpoints")
    args = parser.parse_args()

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_data: Dict[str, Dict[str, Any]] = {}

    print("=" * 65)
    print("      PHASE 2: DEEP REINFORCEMENT LEARNING ALGORITHM BENCHMARK")
    print("=" * 65)
    print(f"Experiments to test  : 4 variants (A, B, C, D)")
    print(f"Training episodes    : {args.episodes} episodes per variant")
    print(f"Greedy eval games    : {args.eval_games} games per variant")
    print("=" * 65)

    for name, cfg in EXPERIMENT_CONFIGS.items():
        exp_slug = name.split("(")[1].replace(")", "").strip().lower().replace(" ", "_").replace("+", "plus")
        exp_dir = EXPERIMENTS_DIR / exp_slug
        model_path = exp_dir / "model.pth"

        scores: List[int] = []
        losses: List[float] = []

        if not args.skip_train or not model_path.is_file():
            agent, scores, losses = run_single_experiment_training(
                name=name,
                cfg=cfg,
                episodes=args.episodes,
                seed=args.seed,
                output_dir=EXPERIMENTS_DIR,
            )
        else:
            print(f"\nSkipping training for {name}, found existing checkpoint at: {model_path}")
            ckpt = torch.load(model_path, map_location="cpu")
            if isinstance(ckpt, dict):
                scores = ckpt.get("scores", [])
                losses = ckpt.get("losses", [])

        # Evaluate model
        print(f"  Evaluating {name} over {args.eval_games} independent games...")
        eval_metrics = evaluate_agent_greedy(model_path, num_games=args.eval_games, seed=123)
        print(f"  [Result] Mean Score: {eval_metrics['eval_mean']:.2f} | Max: {eval_metrics['eval_max']} | Std: {eval_metrics['eval_std']:.2f}")

        summary_data[name] = {
            "description": cfg["description"],
            "model_path": str(model_path),
            "scores": scores,
            "losses": losses,
            **eval_metrics,
        }

    # Print summary comparative table
    sep_double = "=" * 80
    sep_single = "-" * 80

    table_lines = [
        "",
        sep_double,
        "                 PHASE 2 EXPERIMENT BENCHMARK RESULTS",
        sep_double,
        f"{'Experiment Variant':<35} | {'Mean Score':<10} | {'Max':<5} | {'Median':<6} | {'Std Dev':<8} | {'Avg Steps':<10}",
        sep_single,
    ]
    for name, data in summary_data.items():
        table_lines.append(
            f"{name:<35} | {data['eval_mean']:<10.2f} | {data['eval_max']:<5d} | {data['eval_median']:<6.1f} | {data['eval_std']:<8.2f} | {data['eval_steps_mean']:<10.1f}"
        )
    table_lines.append(sep_double)
    table_str = "\n".join(table_lines)
    print(table_str)

    # Save summary TXT
    txt_out = EXPERIMENTS_DIR / "experiments_comparison.txt"
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write(table_str)
    print(f"\nSummary report saved to: {txt_out}")

    # Save summary CSV
    csv_out = EXPERIMENTS_DIR / "experiments_comparison.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Experiment", "Description", "Mean_Score", "Max_Score", "Min_Score", "Median_Score", "Std_Dev", "Avg_Steps"])
        for name, data in summary_data.items():
            writer.writerow([
                name,
                data["description"],
                f"{data['eval_mean']:.2f}",
                data["eval_max"],
                data["eval_min"],
                f"{data['eval_median']:.1f}",
                f"{data['eval_std']:.2f}",
                f"{data['eval_steps_mean']:.1f}",
            ])
    print(f"Summary CSV saved to: {csv_out}")

    # Generate comparative plot
    plot_path = EXPERIMENTS_DIR / "experiments_learning_curves.png"
    plot_experiment_comparisons(summary_data, save_path=plot_path)
    print(f"Comparative charts saved to: {plot_path}")


if __name__ == "__main__":
    main()

