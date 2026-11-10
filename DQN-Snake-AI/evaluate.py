"""
Robust Evaluation System for Advanced DQN Snake AI (Phases 1, 2, & 3).
Supports:
- Standard DQN, Double DQN, Dueling DQN, PER
- Enhanced 24-D and Standard 11-D State Representations (auto-detected)
- Pure A* Agent evaluation
- DQN with A* Safety Shield Guidance evaluation
- Statistical metrics, CSV & TXT reporting, and distribution charts
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame
import torch

from agent import Agent, set_seed
from config import (
    BASE_DIR,
    BEST_MODEL_PATH,
    DEFAULT_EVAL_GAMES,
    EVAL_METRICS_CSV,
    EVAL_RESULTS_TXT,
    EVAL_SEED,
    LEGACY_EVAL_TXT,
    LEGACY_MODEL_PATH,
    RESULTS_DIR,
)
from model import load_model
from pathfinding import AStarAgent
from plot_utils import plot_evaluation_distribution
from snake_ai import SnakeGameAI


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate DQN / A* Snake AI Agent (Phase 3)")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model checkpoint (.pth). Defaults to best_model.pth or snake_dqn_improved.pth",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=DEFAULT_EVAL_GAMES,
        help=f"Number of independent evaluation games (default: {DEFAULT_EVAL_GAMES})",
    )
    parser.add_argument(
        "--pure-astar",
        action="store_true",
        default=False,
        help="Evaluate Pure A* Pathfinding Agent instead of DQN",
    )
    parser.add_argument(
        "--astar-shield",
        action="store_true",
        default=False,
        help="Enable A* Safety Shield Guidance to prevent suicidal DQN moves and traps",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Display pygame visual window during evaluation",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Rendering FPS when visual mode is enabled (default: 25)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=EVAL_SEED,
        help=f"Random seed for reproducible evaluation (default: {EVAL_SEED})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIR),
        help="Directory to save evaluation reports and plots",
    )
    return parser.parse_args()


def resolve_model_path(custom_path: Optional[str] = None) -> Path:
    """Resolve model checkpoint path with intelligent fallbacks."""
    if custom_path:
        p = Path(custom_path)
        if p.is_file():
            return p
        alt_p = BASE_DIR / custom_path
        if alt_p.is_file():
            return alt_p
        raise FileNotFoundError(f"Specified model checkpoint not found: {custom_path}")

    if BEST_MODEL_PATH.is_file():
        return BEST_MODEL_PATH
    if LEGACY_MODEL_PATH.is_file():
        return LEGACY_MODEL_PATH
    latest = BASE_DIR / "checkpoints" / "latest_checkpoint.pth"
    if latest.is_file():
        return latest

    raise FileNotFoundError("No model checkpoint found! Please run 'python train.py' first.")


def evaluate(args: Optional[argparse.Namespace] = None) -> Dict[str, float]:
    """
    Run greedy evaluation over N games for DQN, DQN + A* Shield, or Pure A*.
    
    Returns:
        Dictionary of computed statistical metrics.
    """
    if args is None:
        args = parse_arguments()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    astar_agent = None
    agent = None
    arch_type = "Pure A* (Algorithmic)"
    state_mode = "standard"
    model_name = "Pure A*"

    if args.pure_astar:
        astar_agent = AStarAgent()
        game = SnakeGameAI(display=args.render, fps=args.fps, state_mode="standard")
    else:
        model_path = resolve_model_path(args.model)
        model_name = model_path.name
        loaded_model, arch_type, detected_input_size = load_model(model_path, map_location="cpu")
        state_mode = "enhanced" if detected_input_size == 24 else "standard"

        agent = Agent(state_size=detected_input_size, seed=args.seed)
        agent.model = loaded_model
        agent.dueling = (arch_type == "dueling")
        agent.model.eval()
        agent.training = False
        agent.epsilon = 0.0

        game = SnakeGameAI(
            display=args.render,
            fps=args.fps,
            state_mode=state_mode,
        )

    mode_label = "Pure A* Pathfinding" if args.pure_astar else f"DQN ({arch_type.upper()})"
    if args.astar_shield and not args.pure_astar:
        mode_label += " + A* Safety Shield Guidance"

    print("=" * 65)
    print("      DQN SNAKE AI - MODEL EVALUATION (PHASE 3 INTELLIGENCE)")
    print("=" * 65)
    print(f"Agent Mode         : {mode_label}")
    print(f"Model Checkpoint   : {model_name}")
    print(f"State Mode         : {state_mode.upper()} ({game.state_size} features)")
    print(f"Evaluation Games   : {args.games}")
    print(f"Visual Rendering   : {'Enabled' if args.render else 'Headless'}")
    print(f"Random Seed        : {args.seed}")
    print(f"Output Directory   : {output_dir}")
    print("=" * 65)

    game_records: List[Dict[str, float]] = []
    scores: List[int] = []
    survival_steps: List[int] = []
    total_rewards: List[float] = []
    food_collected: List[int] = []

    try:
        for game_idx in range(1, args.games + 1):
            game.reset()
            done = False

            while not done:
                if args.render:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            print("\nEvaluation interrupted by user.")
                            return {}

                if args.pure_astar:
                    action_idx = astar_agent.get_action(game)
                else:
                    state = game.get_state()
                    action_idx = agent.get_action_index(
                        state,
                        deterministic=True,
                        astar_shield=args.astar_shield,
                        game=game if args.astar_shield else None,
                    )

                _, done, score = game.move(action_idx)

                if args.render:
                    path_overlay = astar_agent.last_path if args.pure_astar else None
                    game.draw(record=max(scores) if scores else 0, path=path_overlay)

            scores.append(game.score)
            survival_steps.append(game.frame_iteration)
            total_rewards.append(round(game.total_reward, 2))
            food_collected.append(game.food_eaten)

            game_records.append({
                "game": game_idx,
                "score": game.score,
                "food_collected": game.food_eaten,
                "survival_steps": game.frame_iteration,
                "total_reward": round(game.total_reward, 2),
            })

            if game_idx % 20 == 0 or game_idx == 1 or game_idx == args.games:
                print(
                    f"Game {game_idx:3d}/{args.games:3d} | "
                    f"Score: {game.score:2d} | "
                    f"Steps: {game.frame_iteration:3d} | "
                    f"Reward: {game.total_reward:6.2f}"
                )

    finally:
        if args.render:
            pygame.quit()

    if not scores:
        print("No evaluation games completed.")
        return {}

    scores_arr = np.array(scores)
    metrics = {
        "games_evaluated": len(scores),
        "average_score": float(np.mean(scores_arr)),
        "maximum_score": int(np.max(scores_arr)),
        "minimum_score": int(np.min(scores_arr)),
        "median_score": float(np.median(scores_arr)),
        "std_dev_score": float(np.std(scores_arr)),
        "average_food_collected": float(np.mean(food_collected)),
        "average_survival_steps": float(np.mean(survival_steps)),
        "average_reward": float(np.mean(total_rewards)),
    }

    print("\n" + "=" * 65)
    print("                 EVALUATION RESULTS SUMMARY")
    print("=" * 65)
    print(f"Agent Architecture     : {mode_label}")
    print(f"Total Games Evaluated  : {metrics['games_evaluated']}")
    print(f"Average Score          : {metrics['average_score']:.2f}")
    print(f"Maximum Score          : {metrics['maximum_score']}")
    print(f"Minimum Score          : {metrics['minimum_score']}")
    print(f"Median Score           : {metrics['median_score']:.1f}")
    print(f"Standard Deviation     : {metrics['std_dev_score']:.2f}")
    print(f"Average Food Collected : {metrics['average_food_collected']:.2f}")
    print(f"Average Survival Steps : {metrics['average_survival_steps']:.2f}")
    print(f"Average Total Reward   : {metrics['average_reward']:.2f}")
    print("=" * 65)

    csv_path = output_dir / "evaluation_metrics.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["# DQN Snake AI Evaluation Summary"])
        writer.writerow(["mode", mode_label])
        writer.writerow(["state_mode", state_mode])
        for k, v in metrics.items():
            writer.writerow([k, f"{v:.4f}" if isinstance(v, float) else v])
        writer.writerow([])
        writer.writerow(["Game", "Score", "Food_Collected", "Survival_Steps", "Total_Reward"])
        for rec in game_records:
            writer.writerow([
                rec["game"],
                rec["score"],
                rec["food_collected"],
                rec["survival_steps"],
                rec["total_reward"],
            ])
    print(f"CSV report saved to: {csv_path}")

    report_text = (
        "=========================================================\n"
        "             DQN SNAKE AI EVALUATION REPORT              \n"
        "=========================================================\n"
        f"Agent Mode             : {mode_label}\n"
        f"Model Evaluated        : {model_name}\n"
        f"State Mode             : {state_mode.upper()} ({game.state_size} features)\n"
        f"Total Games Played     : {metrics['games_evaluated']}\n"
        "---------------------------------------------------------\n"
        f"Average Score          : {metrics['average_score']:.2f}\n"
        f"Maximum Score          : {metrics['maximum_score']}\n"
        f"Minimum Score          : {metrics['minimum_score']}\n"
        f"Median Score           : {metrics['median_score']:.1f}\n"
        f"Standard Deviation     : {metrics['std_dev_score']:.2f}\n"
        f"Average Food Collected : {metrics['average_food_collected']:.2f}\n"
        f"Average Survival Steps : {metrics['average_survival_steps']:.2f}\n"
        f"Average Total Reward   : {metrics['average_reward']:.2f}\n"
        "=========================================================\n"
    )

    txt_path = output_dir / "evaluation_results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"TXT report saved to: {txt_path}")

    with open(LEGACY_EVAL_TXT, "w", encoding="utf-8") as f:
        f.write(report_text)

    dist_chart_path = output_dir / "evaluation_distribution.png"
    plot_evaluation_distribution(scores, dist_chart_path)
    print(f"Distribution plot saved to: {dist_chart_path}")

    return metrics


if __name__ == "__main__":
    evaluate()
