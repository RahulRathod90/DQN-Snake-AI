"""
Phase 4 Model Training Pipeline for Experiment C (Enhanced State) and Experiment F (Optimized DQN).
Trains both models from scratch under verified, bug-free environments.
Saves:
- Best checkpoint based on periodic greedy validation
- Latest checkpoint
- Training histories (scores, losses, moving averages, epsilons)
- Training configuration JSON
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from agent import Agent, set_seed
from config import BASE_DIR, EXPERIMENTS_DIR, RESULTS_DIR
from snake_ai import SnakeGameAI

PHASE3_EXP_DIR = EXPERIMENTS_DIR / "phase3_comparison"
PHASE3_EXP_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_greedy(
    agent: Agent,
    state_mode: str = "standard",
    games: int = 5,
    seed: int = 999,
) -> float:
    """Evaluate agent performance using deterministic greedy policy."""
    was_training = agent.training
    old_eps = agent.epsilon
    agent.model.eval()
    agent.training = False
    agent.epsilon = 0.0

    eval_game = SnakeGameAI(display=False, state_mode=state_mode)
    scores = []

    for _ in range(games):
        eval_game.reset()
        done = False
        while not done:
            s = eval_game.get_state()
            act = agent.get_action(s, deterministic=True)
            _, done, _ = eval_game.move(act.index(1))
        scores.append(eval_game.score)

    agent.training = was_training
    agent.epsilon = old_eps
    if was_training:
        agent.model.train()

    return float(np.mean(scores))


def train_model(
    name: str,
    state_size: int,
    state_mode: str,
    reward_mode: str,
    lr: float,
    gamma: float,
    batch_size: int,
    epsilon_decay: float,
    target_update_freq: int,
    use_double_dqn: bool,
    dueling: bool,
    use_per: bool,
    per_alpha: float,
    per_beta: float,
    episodes: int,
    seed: int,
    best_output_path: Path,
    latest_output_path: Path,
    summary_output_path: Path,
) -> Dict[str, Any]:
    """Train a reinforcement learning agent with periodic greedy validation."""
    set_seed(seed)
    start_time = time.time()

    print("=" * 70)
    print(f"  Training {name}")
    print("=" * 70)
    print(f"State Size: {state_size} ({state_mode}) | Reward Mode: {reward_mode}")
    print(f"LR: {lr} | Gamma: {gamma} | Batch: {batch_size} | Eps Decay: {epsilon_decay}")
    print(f"Double DQN: {use_double_dqn} | Dueling: {dueling} | PER: {use_per} (a={per_alpha}, b={per_beta})")
    print(f"Episodes: {episodes} | Seed: {seed}")
    print("=" * 70)

    game = SnakeGameAI(display=False, state_mode=state_mode, reward_mode=reward_mode)
    agent = Agent(
        state_size=state_size,
        lr=lr,
        gamma=gamma,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        min_replay_size=max(batch_size, 64),
        target_update_freq=target_update_freq,
        use_double_dqn=use_double_dqn,
        dueling=dueling,
        use_per=use_per,
        per_alpha=per_alpha,
        per_beta=per_beta,
        seed=seed,
    )

    scores: List[int] = []
    losses: List[float] = []
    moving_avgs: List[float] = []
    epsilons: List[float] = []
    validation_history: List[Dict[str, Any]] = []

    best_val_score = -1.0
    best_episode = 0

    for ep in range(1, episodes + 1):
        game.reset()
        done = False
        step_losses = []

        while not done:
            s = game.get_state()
            act = agent.get_action(s)
            r, done, score = game.move(act.index(1))
            s_next = game.get_state()
            agent.remember(s, act, r, s_next, done)

            l = agent.train_replay_batch(batch_size)
            if l is not None:
                step_losses.append(l)

        agent.n_games = ep
        agent.decay_epsilon()

        avg_loss = float(np.mean(step_losses)) if step_losses else (losses[-1] if losses else 0.0)
        losses.append(avg_loss)
        scores.append(score)
        epsilons.append(agent.epsilon)

        window = scores[-min(len(scores), 50):]
        current_ma = float(np.mean(window))
        moving_avgs.append(current_ma)

        # Periodic Validation Check every 10 episodes
        if ep % 10 == 0 or ep == episodes:
            val_mean = evaluate_greedy(agent, state_mode=state_mode, games=5, seed=seed + 1000 + ep)
            validation_history.append({"episode": ep, "val_mean": val_mean, "moving_avg": current_ma})

            print(
                f"  Ep {ep:3d}/{episodes:3d} | Train Score: {score:2d} | "
                f"MA(50): {current_ma:5.2f} | Val Mean: {val_mean:5.2f} | "
                f"Loss: {avg_loss:6.4f} | Eps: {agent.epsilon:.3f}"
            )

            # Save best checkpoint if validation improves or moving average reaches record
            criterion = val_mean + 0.5 * current_ma
            if criterion > best_val_score:
                best_val_score = criterion
                best_episode = ep
                agent.save_checkpoint(
                    best_output_path,
                    episode=ep,
                    scores=scores,
                    losses=losses,
                    moving_avg_scores=moving_avgs,
                    best_score=int(np.max(scores)),
                    best_avg_score=current_ma,
                )
                print(f"    * New Best Checkpoint saved at Ep {ep} (Val Mean: {val_mean:.2f}, MA: {current_ma:.2f})")

    # Save latest checkpoint
    agent.save_checkpoint(
        latest_output_path,
        episode=episodes,
        scores=scores,
        losses=losses,
        moving_avg_scores=moving_avgs,
        best_score=int(np.max(scores)),
        best_avg_score=moving_avgs[-1],
    )

    duration = time.time() - start_time
    print(f"\nTraining completed in {duration:.1f}s. Best model at Ep {best_episode}.")

    summary = {
        "name": name,
        "state_size": state_size,
        "state_mode": state_mode,
        "reward_mode": reward_mode,
        "episodes": episodes,
        "training_duration_seconds": float(np.round(duration, 2)),
        "final_loss": float(np.round(losses[-1], 6)),
        "best_episode": best_episode,
        "best_criterion_score": float(np.round(best_val_score, 2)),
        "final_moving_avg": float(np.round(moving_avgs[-1], 2)),
        "max_train_score": int(np.max(scores)),
        "scores": scores,
        "losses": losses,
        "moving_avgs": moving_avgs,
        "epsilons": epsilons,
        "validation_history": validation_history,
        "hyperparameters": {
            "lr": lr,
            "gamma": gamma,
            "batch_size": batch_size,
            "epsilon_decay": epsilon_decay,
            "target_update_freq": target_update_freq,
            "use_double_dqn": use_double_dqn,
            "dueling": dueling,
            "use_per": use_per,
            "per_alpha": per_alpha,
            "per_beta": per_beta,
            "seed": seed,
        },
    }

    with open(summary_output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    return summary


def run_all_training(episodes: int = 150) -> None:
    """Train both Experiment C (Enhanced State) and Experiment F (Optimized DQN)."""
    # 1. Experiment C: Enhanced State (24 features)
    exp_c_best = PHASE3_EXP_DIR / "exp_c_enhanced_state.pth"
    exp_c_latest = PHASE3_EXP_DIR / "exp_c_enhanced_state_latest.pth"
    exp_c_summary = PHASE3_EXP_DIR / "exp_c_training_summary.json"

    train_model(
        name="Exp C: Enhanced State (24-D)",
        state_size=24,
        state_mode="enhanced",
        reward_mode="advanced",
        lr=0.0005,
        gamma=0.99,
        batch_size=64,
        epsilon_decay=0.995,
        target_update_freq=100,
        use_double_dqn=True,
        dueling=True,
        use_per=True,
        per_alpha=0.6,
        per_beta=0.4,
        episodes=episodes,
        seed=42,
        best_output_path=exp_c_best,
        latest_output_path=exp_c_latest,
        summary_output_path=exp_c_summary,
    )

    # 2. Experiment F: Best Hyperparameters (Validated from Sweep)
    hp_path = RESULTS_DIR / "best_hyperparameters.json"
    if hp_path.is_file():
        with open(hp_path, "r", encoding="utf-8") as f:
            hp = json.load(f)
    else:
        hp = {"lr": 0.001, "gamma": 0.95, "batch_size": 128, "epsilon_decay": 0.992, "target_update_freq": 100, "per_alpha": 0.4, "per_beta": 0.4}

    exp_f_best = PHASE3_EXP_DIR / "exp_f_best_hyperparams.pth"
    exp_f_latest = PHASE3_EXP_DIR / "exp_f_best_hyperparams_latest.pth"
    exp_f_summary = PHASE3_EXP_DIR / "exp_f_training_summary.json"

    train_model(
        name="Exp F: Optimized DQN (Tuned Hyperparams)",
        state_size=11,
        state_mode="standard",
        reward_mode="phase1",
        lr=hp.get("lr", 0.001),
        gamma=hp.get("gamma", 0.95),
        batch_size=hp.get("batch_size", 128),
        epsilon_decay=hp.get("epsilon_decay", 0.992),
        target_update_freq=hp.get("target_update_freq", 100),
        use_double_dqn=True,
        dueling=True,
        use_per=True,
        per_alpha=hp.get("per_alpha", 0.4),
        per_beta=hp.get("per_beta", 0.4),
        episodes=episodes,
        seed=42,
        best_output_path=exp_f_best,
        latest_output_path=exp_f_latest,
        summary_output_path=exp_f_summary,
    )


if __name__ == "__main__":
    run_all_training(episodes=150)

