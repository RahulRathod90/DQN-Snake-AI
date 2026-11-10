"""
Benchmark comparison between Trained DQN Agent and Random-Action Agent.
Supports both Standard DQN and Dueling DQN models.
Evaluates both agents across independent games and outputs a comparison table.
"""

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame
import torch

from agent import Agent, set_seed
from config import BASE_DIR, BEST_MODEL_PATH, LEGACY_MODEL_PATH, RESULTS_DIR
from model import load_model
from snake_ai import SnakeGameAI


def resolve_model_path(custom_path: Optional[str] = None) -> Path:
    """Locate model checkpoint."""
    if custom_path:
        p = Path(custom_path)
        if p.is_file():
            return p
        alt_p = BASE_DIR / custom_path
        if alt_p.is_file():
            return alt_p
        raise FileNotFoundError(f"Model checkpoint not found: {custom_path}")

    if BEST_MODEL_PATH.is_file():
        return BEST_MODEL_PATH
    if LEGACY_MODEL_PATH.is_file():
        return LEGACY_MODEL_PATH
    raise FileNotFoundError("No trained model found. Please run train.py first.")


def evaluate_dqn(model_path: Path, num_games: int = 20) -> Tuple[List[int], List[int]]:
    """Evaluate trained DQN agent with greedy action selection."""
    agent = Agent()
    model, arch_type = load_model(model_path, map_location="cpu")
    agent.model = model
    agent.dueling = (arch_type == "dueling")
    agent.model.eval()
    agent.training = False
    agent.epsilon = 0.0

    scores: List[int] = []
    steps: List[int] = []

    for game_num in range(1, num_games + 1):
        game = SnakeGameAI(display=False)
        done = False

        while not done:
            state = game.get_state()
            action = agent.get_action(state, deterministic=True)
            action_idx = action.index(1)
            _, done, _ = game.move(action_idx)

        scores.append(game.score)
        steps.append(game.frame_iteration)
        if game_num % 5 == 0 or game_num == 1:
            print(f"  [DQN Agent] Game {game_num:2d}/{num_games:2d} -> Score: {game.score:2d}, Steps: {game.frame_iteration:3d}")

    return scores, steps


def evaluate_random(num_games: int = 20) -> Tuple[List[int], List[int]]:
    """Evaluate naive agent selecting uniform random actions."""
    scores: List[int] = []
    steps: List[int] = []

    for game_num in range(1, num_games + 1):
        game = SnakeGameAI(display=False)
        done = False

        while not done:
            action = random.randint(0, 2)
            _, done, _ = game.move(action)

        scores.append(game.score)
        steps.append(game.frame_iteration)
        if game_num % 5 == 0 or game_num == 1:
            print(f"  [Random Agent] Game {game_num:2d}/{num_games:2d} -> Score: {game.score:2d}, Steps: {game.frame_iteration:3d}")

    return scores, steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DQN Agent against Random Agent")
    parser.add_argument("--games", type=int, default=20, help="Number of benchmark games (default: 20)")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    set_seed(args.seed)
    model_path = resolve_model_path(args.model)

    print("=" * 65)
    print("        BENCHMARK: TRAINED DQN vs RANDOM BASELINE")
    print("=" * 65)
    print(f"Model Checkpoint : {model_path}")
    print(f"Benchmark Games  : {args.games} per agent")
    print("-" * 65)

    print("\n[1/2] Evaluating Trained DQN Agent...")
    dqn_scores, dqn_steps = evaluate_dqn(model_path, args.games)

    print("\n[2/2] Evaluating Random Agent...")
    rnd_scores, rnd_steps = evaluate_random(args.games)

    dqn_avg = float(np.mean(dqn_scores))
    rnd_avg = float(np.mean(rnd_scores))
    dqn_max = int(np.max(dqn_scores))
    rnd_max = int(np.max(rnd_scores))
    dqn_steps_avg = float(np.mean(dqn_steps))
    rnd_steps_avg = float(np.mean(rnd_steps))

    sep_double = "=" * 65
    sep_single = "-" * 65

    table = (
        f"\n{sep_double}\n"
        f"                  BENCHMARK COMPARISON RESULTS\n"
        f"{sep_double}\n"
        f"{'Metric':<25} | {'Random Agent':<16} | {'Trained DQN Agent':<16}\n"
        f"{sep_single}\n"
        f"{'Games Evaluated':<25} | {args.games:<16} | {args.games:<16}\n"
        f"{'Average Score':<25} | {rnd_avg:<16.2f} | {dqn_avg:<16.2f}\n"
        f"{'Maximum Score':<25} | {rnd_max:<16} | {dqn_max:<16}\n"
        f"{'Minimum Score':<25} | {min(rnd_scores):<16} | {min(dqn_scores):<16}\n"
        f"{'Avg Survival Steps':<25} | {rnd_steps_avg:<16.2f} | {dqn_steps_avg:<16.2f}\n"
        f"{sep_double}\n"
    )
    print(table)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comp_file = RESULTS_DIR / "comparison_results.txt"
    with open(comp_file, "w", encoding="utf-8") as f:
        f.write(table)
    print(f"Comparison report saved to: {comp_file}")


if __name__ == "__main__":
    main()
