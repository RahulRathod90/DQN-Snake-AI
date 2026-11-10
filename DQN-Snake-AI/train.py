"""
Training Pipeline for Advanced DQN Snake AI (Phases 1, 2, & 3).
Supports:
- State Representations: Standard 11-D or Enhanced 24-D
- Reward Modes: Phase 1 Baseline or Phase 3 Advanced (PBRS + Trap Avoidance)
- Algorithms: Standard DQN, Double DQN (DDQN), Dueling DQN, PER
- Headless execution, Checkpointing, and Training Resumption
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pygame
import torch

from agent import Agent, set_seed
from config import (
    BASE_DIR,
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CHECKPOINT_FREQ,
    DEFAULT_EPISODES,
    GAMMA,
    LATEST_CHECKPOINT_PATH,
    LEARNING_RATE,
    LEGACY_MODEL_PATH,
    MOVING_AVG_WINDOW,
    PER_ALPHA,
    PER_BETA_MAX,
    PER_BETA_START,
    PER_EPSILON,
    RANDOM_SEED,
    RENDER_FPS,
    RESULTS_DIR,
    REWARD_MODE,
    STATE_MODE,
    TARGET_UPDATE_FREQ,
    TRAIN_FREQ_STEPS,
    USE_DOUBLE_DQN,
    USE_DUELING,
    USE_PER,
)
from plot_utils import compute_moving_average, plot_training_results
from snake_ai import SnakeGameAI


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Train Advanced DQN Snake AI Agent (Phase 3)")
    
    # Schedule & Hyperparameters
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help=f"Number of training episodes (default: {DEFAULT_EPISODES})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Mini-batch size for training (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate for Adam optimizer (default: {LEARNING_RATE})",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=GAMMA,
        help=f"Discount factor gamma (default: {GAMMA})",
    )
    parser.add_argument(
        "--train-freq",
        type=int,
        default=TRAIN_FREQ_STEPS,
        help=f"Train mini-batch every N environment steps (default: {TRAIN_FREQ_STEPS})",
    )
    parser.add_argument(
        "--target-update-freq",
        type=int,
        default=TARGET_UPDATE_FREQ,
        help=f"Sync target network every N train steps (default: {TARGET_UPDATE_FREQ})",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=CHECKPOINT_FREQ,
        help=f"Save checkpoint every N episodes (default: {CHECKPOINT_FREQ})",
    )

    # Phase 2 & 3 Architecture and Representation
    parser.add_argument(
        "--state-mode",
        type=str,
        default=STATE_MODE,
        choices=["standard", "enhanced"],
        help="State representation: 'standard' (11 features) or 'enhanced' (24 features)",
    )
    parser.add_argument(
        "--reward-mode",
        type=str,
        default=REWARD_MODE,
        choices=["phase1", "advanced"],
        help="Reward function: 'phase1' (baseline) or 'advanced' (PBRS + trap avoidance)",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default=None,
        choices=["dqn", "ddqn", "dueling", "per"],
        help="Preset algorithm: 'dqn' (Standard), 'ddqn' (Double DQN), 'dueling' (Double Dueling), 'per' (Double Dueling + PER)",
    )
    parser.add_argument(
        "--double-dqn",
        dest="double_dqn",
        action="store_true",
        default=USE_DOUBLE_DQN,
        help="Enable Double DQN target calculation",
    )
    parser.add_argument(
        "--no-double-dqn",
        dest="double_dqn",
        action="store_false",
        help="Disable Double DQN (use standard single DQN target)",
    )
    parser.add_argument(
        "--dueling",
        action="store_true",
        default=USE_DUELING,
        help="Enable Dueling DQN architecture (Value V(s) + Advantage A(s, a))",
    )
    parser.add_argument(
        "--per",
        action="store_true",
        default=USE_PER,
        help="Enable Prioritized Experience Replay (PER)",
    )
    parser.add_argument(
        "--per-alpha",
        type=float,
        default=PER_ALPHA,
        help=f"PER prioritization exponent alpha (default: {PER_ALPHA})",
    )
    parser.add_argument(
        "--per-beta",
        type=float,
        default=PER_BETA_START,
        help=f"PER initial importance sampling beta (default: {PER_BETA_START})",
    )
    parser.add_argument(
        "--per-beta-max",
        type=float,
        default=PER_BETA_MAX,
        help=f"PER maximum importance sampling beta (default: {PER_BETA_MAX})",
    )

    # Runtime & Display Options
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run training in headless mode (no pygame GUI) for maximum speed",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Force enable visual window rendering during training",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=RENDER_FPS,
        help=f"Rendering FPS when visualization is enabled (default: {RENDER_FPS})",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {RANDOM_SEED})",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        default=False,
        help="Disable automatic generation of training graphs",
    )
    return parser.parse_args()


def train(args: Optional[argparse.Namespace] = None) -> Agent:
    """
    Main training loop for Advanced DQN Snake AI.
    
    Returns:
        Trained Agent instance.
    """
    if args is None:
        args = parse_arguments()

    if args.algo == "dqn":
        args.double_dqn = False
        args.dueling = False
        args.per = False
    elif args.algo == "ddqn":
        args.double_dqn = True
        args.dueling = False
        args.per = False
    elif args.algo == "dueling":
        args.double_dqn = True
        args.dueling = True
        args.per = False
    elif args.algo == "per":
        args.double_dqn = True
        args.dueling = True
        args.per = True

    set_seed(args.seed)
    display = False if args.headless else args.render

    algo_name = "Standard DQN"
    if args.double_dqn and not args.dueling and not args.per:
        algo_name = "Double DQN (DDQN)"
    elif args.double_dqn and args.dueling and not args.per:
        algo_name = "Double Dueling DQN"
    elif args.double_dqn and args.dueling and args.per:
        algo_name = "Double Dueling DQN + PER"
    elif args.dueling:
        algo_name = "Dueling DQN"

    # Initialize Environment
    game = SnakeGameAI(
        display=display,
        fps=args.fps,
        state_mode=args.state_mode,
        reward_mode=args.reward_mode,
    )

    print("=" * 65)
    print("      DQN SNAKE AI - ADVANCED REINFORCEMENT LEARNING (PHASE 3)")
    print("=" * 65)
    print(f"Algorithm Selected  : {algo_name}")
    print(f"State Representation: {args.state_mode.upper()} ({game.state_size} features)")
    print(f"Reward Function     : {args.reward_mode.upper()}")
    print(f"Double DQN Target   : {'Enabled' if args.double_dqn else 'Disabled'}")
    print(f"Dueling Architecture: {'Enabled' if args.dueling else 'Disabled'}")
    print(f"Experience Replay   : {'Prioritized (PER)' if args.per else 'Standard Uniform'}")
    if args.per:
        print(f"  PER Alpha / Beta  : alpha={args.per_alpha}, beta_start={args.per_beta}")
    print(f"Target Episodes     : {args.episodes}")
    print(f"Learning Rate       : {args.lr}")
    print(f"Discount Factor (gamma): {args.gamma}")
    print(f"Batch Size          : {args.batch_size}")
    print(f"Train Step Frequency: Every {args.train_freq} step(s)")
    print(f"Target Network Sync : Every {args.target_update_freq} train steps")
    print(f"Display Mode        : {'Visual Window' if display else 'Headless (Fast)'}")
    print(f"Random Seed         : {args.seed}")
    print("=" * 65)

    # Initialize Agent with dynamic state dimensionality
    agent = Agent(
        state_size=game.state_size,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        use_double_dqn=args.double_dqn,
        dueling=args.dueling,
        use_per=args.per,
        per_alpha=args.per_alpha,
        per_beta=args.per_beta,
        per_beta_max=args.per_beta_max,
        per_beta_anneal_episodes=args.episodes,
        seed=args.seed,
    )

    scores: List[int] = []
    losses: List[float] = []
    epsilons: List[float] = []
    moving_avg_scores: List[float] = []
    best_moving_avg: float = -float("inf")
    best_score: int = 0
    start_episode = 1
    total_env_steps = 0

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_file():
            print(f"Resuming training from checkpoint: {resume_path}")
            ckpt = agent.load_checkpoint(resume_path)
            start_episode = ckpt.get("episode", 0) + 1
            scores = ckpt.get("scores", [])
            losses = ckpt.get("losses", [])
            moving_avg_scores = ckpt.get("moving_avg_scores", [])
            best_score = ckpt.get("best_score", 0)
            best_moving_avg = ckpt.get("best_avg_score", -float("inf"))
            print(f"Resumed at Episode {start_episode} | Prior Best Avg: {best_moving_avg:.2f}")
        else:
            print(f"Warning: Checkpoint '{args.resume}' not found. Starting from scratch.")

    try:
        for episode in range(start_episode, args.episodes + 1):
            game.reset()
            done = False
            episode_losses: List[float] = []

            while not done:
                if display:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            print("\nTraining interrupted by window close.")
                            return agent

                state_old = game.get_state()
                action = agent.get_action(state_old)
                action_idx = action.index(1)

                reward, done, score = game.move(action_idx)
                state_new = game.get_state()

                agent.remember(state_old, action, reward, state_new, done)

                total_env_steps += 1
                if total_env_steps % args.train_freq == 0:
                    loss_val = agent.train_replay_batch(args.batch_size)
                    if loss_val is not None:
                        episode_losses.append(loss_val)

                if display:
                    game.draw(best_score)

            agent.n_games = episode
            agent.decay_epsilon()

            scores.append(score)
            epsilons.append(agent.epsilon)
            avg_loss = float(np.mean(episode_losses)) if episode_losses else (losses[-1] if losses else 0.0)
            losses.append(avg_loss)

            window_slice = scores[-min(len(scores), MOVING_AVG_WINDOW):]
            current_moving_avg = float(np.mean(window_slice))
            moving_avg_scores.append(current_moving_avg)

            if score > best_score:
                best_score = score

            if current_moving_avg > best_moving_avg and episode >= 10:
                best_moving_avg = current_moving_avg
                agent.save_checkpoint(
                    BEST_MODEL_PATH,
                    episode=episode,
                    scores=scores,
                    losses=losses,
                    moving_avg_scores=moving_avg_scores,
                    best_score=best_score,
                    best_avg_score=best_moving_avg,
                )

            if episode % args.checkpoint_freq == 0:
                agent.save_checkpoint(
                    LATEST_CHECKPOINT_PATH,
                    episode=episode,
                    scores=scores,
                    losses=losses,
                    moving_avg_scores=moving_avg_scores,
                    best_score=best_score,
                    best_avg_score=best_moving_avg,
                )

            if episode % 10 == 0 or episode == 1 or episode == args.episodes:
                log_extra = f" | Beta: {agent.memory.beta:.3f}" if args.per else ""
                print(
                    f"Episode {episode:4d}/{args.episodes:4d} | "
                    f"Score: {score:2d} | "
                    f"Avg(50): {current_moving_avg:5.2f} | "
                    f"Best Avg: {best_moving_avg:5.2f} | "
                    f"Best: {best_score:2d} | "
                    f"Loss: {avg_loss:7.4f} | "
                    f"Eps: {agent.epsilon:.3f}{log_extra}"
                )

        print(f"\nTraining for {algo_name} completed successfully!")

    except KeyboardInterrupt:
        print("\nTraining paused by user (KeyboardInterrupt).")

    finally:
        agent.save_checkpoint(
            LATEST_CHECKPOINT_PATH,
            episode=agent.n_games,
            scores=scores,
            losses=losses,
            moving_avg_scores=moving_avg_scores,
            best_score=best_score,
            best_avg_score=best_moving_avg,
        )
        print(f"Latest checkpoint saved to: {LATEST_CHECKPOINT_PATH}")

        if not args.dueling and game.state_size == 11:
            agent.model.save(LEGACY_MODEL_PATH)
            print(f"Legacy model weights saved to: {LEGACY_MODEL_PATH}")

        if not args.no_plots and scores:
            legacy_plot = BASE_DIR / "training_scores_improved.png"
            plot_training_results(
                scores=scores,
                moving_avg_scores=moving_avg_scores,
                losses=losses,
                epsilons=epsilons,
                save_dir=RESULTS_DIR,
                legacy_path=legacy_plot if (not args.dueling and game.state_size == 11) else None,
            )
            print(f"Training charts saved to: {RESULTS_DIR}")

        if display:
            pygame.quit()

    return agent


if __name__ == "__main__":
    train()
