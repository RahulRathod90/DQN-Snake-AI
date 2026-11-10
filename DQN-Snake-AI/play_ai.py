"""
Interactive Visual AI Demonstration & Phase 5 Professional Application Launcher.

Provides backward-compatible CLI visualizer alongside the modern Phase 5
Professional GUI Suite (Game Arena, Telemetry Dashboard, Algorithm Comparison,
Training Studio, and Analytics).
"""

import argparse
from pathlib import Path
import sys
from typing import List, Optional

import pygame
import torch

from agent import Agent, DQNAgent
from config import (
    BASE_DIR,
    BEST_MODEL_PATH,
    COLOR_BLACK,
    COLOR_BLUE,
    COLOR_CYAN,
    COLOR_DARK_GRAY,
    COLOR_DARK_GREEN,
    COLOR_GRAY,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_WHITE,
    COLOR_YELLOW,
    PLAY_FPS,
    RECORD_PATH,
    TOTAL_GRID_CELLS,
)
from gui.app import SnakeAIApp
from model import load_model
from pathfinding import AStarAgent, PathFinder, evaluate_action_safety
from snake_ai import SnakeGameAI


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Snake AI Decision Telemetry & Application Launcher (Phase 5)")
    parser.add_argument(
        "--gui",
        action="store_true",
        default=True,
        help="Launch the full Phase 5 Professional GUI Suite (default: True)",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Launch the legacy Phase 3 CLI/Pygame visualizer instead of the full GUI suite",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model checkpoint (.pth)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["dqn", "astar", "hybrid"],
        help="Agent mode: 'dqn' (pure DQN), 'astar' (pure A*), or 'hybrid' (DQN + A* Safety Shield)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=PLAY_FPS,
        help=f"Game framerate (default: {PLAY_FPS})",
    )
    parser.add_argument(
        "--show-path",
        action="store_true",
        default=True,
        help="Render A* planned path dots on the grid (default: True)",
    )
    return parser.parse_args()


def resolve_model_path(custom_path: Optional[str] = None) -> Path:
    """Find available model checkpoint."""
    if custom_path:
        p = Path(custom_path)
        if p.is_file():
            return p
        alt_p = BASE_DIR / custom_path
        if alt_p.is_file():
            return alt_p
        raise FileNotFoundError(f"Model not found: {custom_path}")

    if BEST_MODEL_PATH.is_file():
        return BEST_MODEL_PATH
    legacy = BASE_DIR / "snake_dqn_improved.pth"
    if legacy.is_file():
        return legacy
    latest = BASE_DIR / "checkpoints" / "latest_checkpoint.pth"
    if latest.is_file():
        return latest

    raise FileNotFoundError("No checkpoint found. Please train a model first using 'python train.py'.")


def play_legacy_cli(args: argparse.Namespace) -> None:
    """Legacy Phase 3 interactive visualizer execution loop for backwards compatibility."""
    astar_agent = AStarAgent()
    path_finder = PathFinder()
    agent = None
    state_mode = "standard"

    if args.mode in ["dqn", "hybrid"]:
        model_path = resolve_model_path(args.model)
        loaded_model, arch_type, detected_input_size = load_model(model_path, map_location="cpu")
        state_mode = "enhanced" if detected_input_size == 24 else "standard"
        print(f"Loaded {arch_type.upper()} model ({detected_input_size} features) from: {model_path}")

        agent = Agent(state_size=detected_input_size)
        agent.model = loaded_model
        agent.dueling = (arch_type == "dueling")
        agent.model.eval()
        agent.training = False
        agent.epsilon = 0.0

    # Initialize environment
    game = SnakeGameAI(display=True, fps=args.fps, state_mode=state_mode)

    # Load high score record
    record = 0
    if RECORD_PATH.is_file():
        try:
            with open(RECORD_PATH, "r", encoding="utf-8") as f:
                record = int(f.read().strip())
        except (ValueError, IOError):
            record = 0

    running = True
    paused = False
    show_hud = True
    show_path = args.show_path
    use_shield = (args.mode == "hybrid")
    current_fps = args.fps

    print("\n" + "=" * 65)
    print("      SNAKE AI INTERACTIVE DECISION VISUALIZER (LEGACY)")
    print("=" * 65)
    print("Controls:")
    print("  [Space] : Pause / Resume")
    print("  [H]     : Toggle Decision HUD Overlay")
    print("  [P]     : Toggle A* Planned Path Overlay")
    print("  [S]     : Toggle A* Safety Shield Guidance")
    print("  [+] / [-]: Increase / Decrease Game Speed")
    print("  [Esc]   : Exit")
    print("=" * 65 + "\n")

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        print("[PAUSED]" if paused else "[RESUMED]")
                    elif event.key == pygame.K_h:
                        show_hud = not show_hud
                    elif event.key == pygame.K_p:
                        show_path = not show_path
                    elif event.key == pygame.K_s:
                        use_shield = not use_shield
                        print(f"Safety Shield: {'ENABLED' if use_shield else 'DISABLED'}")
                    elif event.key in [pygame.K_PLUS, pygame.K_EQUALS]:
                        current_fps = min(60, current_fps + 3)
                        game.fps = current_fps
                    elif event.key == pygame.K_MINUS:
                        current_fps = max(3, current_fps - 3)
                        game.fps = current_fps

            if not running:
                break

            if paused:
                game.clock.tick(10)
                continue

            q_vals = [0.0, 0.0, 0.0]
            astar_act = astar_agent.get_action(game)
            astar_path = astar_agent.last_path if show_path else None

            if args.mode == "astar":
                chosen_move = astar_act
            else:
                state = game.get_state()
                _, q_vals = agent.get_action_and_q_values(state, deterministic=True)
                chosen_move = agent.get_action_index(
                    state,
                    deterministic=True,
                    astar_shield=use_shield,
                    game=game if use_shield else None,
                )

            _, done, score = game.move(chosen_move)

            if show_hud:
                game.draw(
                    record=record,
                    path=astar_path,
                    q_values=q_vals if args.mode != "astar" else None,
                    selected_action=chosen_move,
                    astar_action=astar_act,
                    epsilon=0.0,
                )
            else:
                game.draw(record=record)

            if done:
                if score > record:
                    record = score
                    try:
                        with open(RECORD_PATH, "w", encoding="utf-8") as f:
                            f.write(str(record))
                    except IOError:
                        pass
                print(f"Game Over! Score: {score} | Record: {record} | Steps: {game.frame_iteration}")
                game.reset()

    finally:
        pygame.quit()
        print("Visualizer terminated.")


def main() -> None:
    """Select between Phase 5 GUI Suite and CLI visualizer."""
    args = parse_arguments()
    if args.cli:
        play_legacy_cli(args)
    else:
        app = SnakeAIApp(start_mode=args.mode, fps=args.fps)
        app.run()


if __name__ == "__main__":
    main()
