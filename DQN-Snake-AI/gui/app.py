"""
Main Application Controller for the DQN Snake AI Professional Suite.
Integrates Game Arena, Real-Time AI Telemetry, Algorithm Comparison Matrix,
Interactive Training Studio, and Scientific Analytics into a unified GUI.
"""

from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pygame
import torch

from agent import Agent, DQNAgent
from config import (
    BASE_DIR,
    BEST_MODEL_PATH,
    BLOCK_SIZE,
    CHECKPOINTS_DIR,
    EXPERIMENTS_DIR,
    RECORD_PATH,
    RESULTS_DIR,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)
from gui.analytics_view import AnalyticsView
from gui.board_renderer import BoardRenderer
from gui.comparison_view import ComparisonView
from gui.telemetry_panel import TelemetryPanel
from gui.theme import (
    ACCENT_CYAN,
    ACCENT_GREEN,
    ACCENT_ORANGE,
    ACCENT_PURPLE,
    ACCENT_RED,
    ACCENT_YELLOW,
    BG_CARD,
    BG_DARK,
    BG_SURFACE,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    BOARD_X,
    BOARD_Y,
    BORDER_COLOR,
    BORDER_HIGHLIGHT,
    BTN_ACTIVE_BG,
    BTN_ACTIVE_TEXT,
    BTN_PRIMARY_BG,
    FOOTER_HEIGHT,
    HEADER_HEIGHT,
    SIDEBAR_HEIGHT,
    SIDEBAR_WIDTH,
    SIDEBAR_X,
    SIDEBAR_Y,
    TAB_ANALYTICS,
    TAB_ARENA,
    TAB_COMPARISON,
    TAB_TRAINING,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    get_font,
)
from gui.training_view import TrainingView
from gui.widgets import Button, DropdownSelector, TabSelector, draw_card
from model import load_model
from pathfinding import AStarAgent, AStarSafetyShield, PathFinder, to_grid
from snake_ai import SnakeGameAI

# ==============================================================================
# AVAILABLE VALIDATED AGENTS REGISTRY
# ==============================================================================
AGENT_REGISTRY: List[Dict[str, Any]] = [
    {
        "key": "hybrid",
        "name": "DQN + A* Safety Shield (Phase 4 Hybrid)",
        "short": "DQN + Shield",
        "model_file": "snake_dqn_improved.pth",
        "state_dim": 11,
        "state_mode": "standard",
        "arch": "dueling",
        "use_shield": True,
        "is_pure_astar": False,
        "is_human": False,
    },
    {
        "key": "pure_astar",
        "name": "Pure A* Agent (Deterministic Baseline)",
        "short": "Pure A*",
        "model_file": None,
        "state_dim": 11,
        "state_mode": "standard",
        "arch": "algorithmic",
        "use_shield": False,
        "is_pure_astar": True,
        "is_human": False,
    },
    {
        "key": "opt_dqn",
        "name": "Optimized DQN (Phase 4 Validated)",
        "short": "Optimized DQN",
        "model_file": "experiments/phase3_comparison/exp_f_best_hyperparams.pth",
        "state_dim": 11,
        "state_mode": "standard",
        "arch": "dueling",
        "use_shield": False,
        "is_pure_astar": False,
        "is_human": False,
    },
    {
        "key": "dueling_per",
        "name": "DDQN + Dueling + PER (Phase 2 Best)",
        "short": "DDQN + PER",
        "model_file": "snake_dqn_improved.pth",
        "state_dim": 11,
        "state_mode": "standard",
        "arch": "dueling",
        "use_shield": False,
        "is_pure_astar": False,
        "is_human": False,
    },
    {
        "key": "enhanced_24d",
        "name": "Enhanced 24-D DQN (Phase 4 Restored)",
        "short": "Enhanced 24-D",
        "model_file": "experiments/phase3_comparison/exp_c_enhanced_state.pth",
        "state_dim": 24,
        "state_mode": "enhanced",
        "arch": "dueling",
        "use_shield": False,
        "is_pure_astar": False,
        "is_human": False,
    },
    {
        "key": "orig_dqn",
        "name": "Original DQN (Phase 1 Baseline)",
        "short": "Original DQN",
        "model_file": "experiments/original_dqn/model.pth",
        "state_dim": 11,
        "state_mode": "standard",
        "arch": "standard",
        "use_shield": False,
        "is_pure_astar": False,
        "is_human": False,
    },
    {
        "key": "human",
        "name": "Human Player (Manual Keyboard)",
        "short": "Human",
        "model_file": None,
        "state_dim": 11,
        "state_mode": "standard",
        "arch": "manual",
        "use_shield": False,
        "is_pure_astar": False,
        "is_human": True,
    },
]


class SnakeAIApp:
    """Professional application coordinator managing rendering, game loop, and tabs."""

    def __init__(self, start_mode: str = "hybrid", fps: int = 15):
        pygame.init()
        pygame.display.set_caption("DQN Snake AI - Professional Reinforcement Learning Platform")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        # Framerate and speed control
        self.fps_options = [5, 15, 30, 60, 0]
        self.fps_labels = ["5 FPS", "15 FPS", "30 FPS", "60 FPS", "TURBO"]
        self.current_fps_idx = 1 if fps == 15 else 3
        self.target_fps = fps

        # Navigation Tabs
        self.tabs = [TAB_ARENA, TAB_COMPARISON, TAB_TRAINING, TAB_ANALYTICS]
        self.current_tab = TAB_ARENA
        self.tab_selector = TabSelector(
            (0, 0, WINDOW_WIDTH, HEADER_HEIGHT),
            tabs=self.tabs,
            active_index=0,
            on_change=self._on_tab_change,
        )

        # High-score record
        self.record = self._load_record()
        self.live_session_scores: Dict[str, List[int]] = {ag["key"]: [] for ag in AGENT_REGISTRY}

        # Views
        content_rect = (0, HEADER_HEIGHT, WINDOW_WIDTH, WINDOW_HEIGHT - HEADER_HEIGHT)
        self.board_renderer = BoardRenderer((BOARD_X, BOARD_Y, BOARD_WIDTH, BOARD_HEIGHT), block_size=BLOCK_SIZE)
        self.telemetry_panel = TelemetryPanel((SIDEBAR_X, SIDEBAR_Y, SIDEBAR_WIDTH, SIDEBAR_HEIGHT))
        self.comparison_view = ComparisonView(content_rect)
        self.training_view = TrainingView(content_rect)
        self.analytics_view = AnalyticsView(content_rect)

        # Agent & Environment
        self.selected_agent_key = start_mode
        self.current_agent_meta = self._get_agent_meta(start_mode)
        self.agent: Optional[DQNAgent] = None
        self.astar_agent = AStarAgent(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, block_size=BLOCK_SIZE)
        self.shield = AStarSafetyShield(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, block_size=BLOCK_SIZE)
        self.path_finder = PathFinder(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, block_size=BLOCK_SIZE)

        # Game State
        self.game = SnakeGameAI(display=False, state_mode=self.current_agent_meta["state_mode"])
        self.is_paused = False
        self.is_game_over = False
        self.game_over_reason = ""
        self.show_path = True
        self.human_buffered_direction: Optional[str] = None

        # Telemetry State
        self.cur_q_values: Optional[List[float]] = None
        self.cur_action = 0
        self.cur_risk = {"straight": False, "right": False, "left": False}
        self.cur_shield_info: Dict[str, Any] = {}
        self.cur_astar_info: Dict[str, Any] = {}

        # Footer Arena Buttons
        ft_y = BOARD_Y + BOARD_HEIGHT + 14
        self.btn_play_pause = Button(
            (BOARD_X, ft_y, 110, 36),
            "Pause",
            on_click=self._toggle_pause,
            active_color=ACCENT_YELLOW,
            font_size=12,
        )
        self.btn_restart = Button(
            (BOARD_X + 120, ft_y, 110, 36),
            "Restart",
            on_click=self.reset_game,
            active_color=ACCENT_CYAN,
            font_size=12,
        )
        self.btn_toggle_path = Button(
            (BOARD_X + 240, ft_y, 110, 36),
            "Path: ON",
            on_click=self._toggle_path,
            active_color=ACCENT_PURPLE,
            font_size=12,
        )

        # Speed buttons
        self.speed_buttons: List[Button] = []
        for i, (fval, flbl) in enumerate(zip(self.fps_options, self.fps_labels)):
            b = Button(
                (BOARD_X + 360 + i * 54, ft_y, 50, 36),
                flbl.replace(" FPS", "").replace("TURBO", "MAX"),
                is_active=(i == self.current_fps_idx),
                font_size=11,
            )
            self.speed_buttons.append(b)

        # Agent Dropdown
        dropdown_options = [{"key": a["key"], "name": a["name"]} for a in AGENT_REGISTRY]
        self.agent_dropdown = DropdownSelector(
            (SIDEBAR_X, ft_y, SIDEBAR_WIDTH, 36),
            options=dropdown_options,
            selected_key=self.selected_agent_key,
            on_select=self._on_agent_selected,
        )

        # Load initial agent
        self._load_current_agent()

    def _load_record(self) -> int:
        if RECORD_PATH.is_file():
            try:
                with open(RECORD_PATH, "r", encoding="utf-8") as f:
                    return int(f.read().strip())
            except Exception:
                return 0
        return 0

    def _save_record(self, new_record: int) -> None:
        try:
            with open(RECORD_PATH, "w", encoding="utf-8") as f:
                f.write(str(new_record))
        except Exception:
            pass

    def _get_agent_meta(self, key: str) -> Dict[str, Any]:
        for ag in AGENT_REGISTRY:
            if ag["key"] == key:
                return ag
        return AGENT_REGISTRY[0]

    def _on_tab_change(self, idx: int, tab_name: str) -> None:
        self.current_tab = tab_name

    def _toggle_pause(self) -> None:
        self.is_paused = not self.is_paused
        self.btn_play_pause.text = "Resume" if self.is_paused else "Pause"

    def _toggle_path(self) -> None:
        self.show_path = not self.show_path
        self.btn_toggle_path.text = "Path: ON" if self.show_path else "Path: OFF"

    def _on_agent_selected(self, option: Dict[str, str]) -> None:
        self.selected_agent_key = option["key"]
        self.current_agent_meta = self._get_agent_meta(self.selected_agent_key)
        self._load_current_agent()
        self.reset_game()

    def _load_current_agent(self) -> None:
        meta = self.current_agent_meta
        if meta["is_human"] or meta["is_pure_astar"]:
            self.agent = None
            return

        model_rel = meta.get("model_file")
        if not model_rel:
            return

        model_p = BASE_DIR / model_rel
        if not model_p.is_file():
            model_p = Path(model_rel)

        state_dim = meta["state_dim"]
        use_dueling = (meta["arch"] == "dueling")

        self.agent = DQNAgent(
            state_size=state_dim,
            action_size=3,
            dueling=use_dueling,
            use_double_dqn=True,
            use_per=True,
            device="cpu",
        )

        if model_p.is_file():
            try:
                self.agent.load_checkpoint(model_p)
                meta["checkpoint_status"] = f"Loaded ({model_p.name})"
            except Exception as e:
                meta["checkpoint_status"] = f"Init Default ({str(e)[:15]})"
        else:
            meta["checkpoint_status"] = "File Not Found (Untrained)"

        self.agent.model.eval()
        self.agent.training = False
        self.agent.epsilon = 0.0

    def reset_game(self) -> None:
        """Reset the game arena and reinitialize environment for chosen agent."""
        state_mode = self.current_agent_meta["state_mode"]
        if self.game.state_mode != state_mode:
            self.game = SnakeGameAI(display=False, state_mode=state_mode)
        else:
            self.game.reset()

        self.is_game_over = False
        self.game_over_reason = ""
        self.human_buffered_direction = None
        self.shield.reset_telemetry()

    def _step_human(self) -> Tuple[int, bool, int]:
        """Convert buffered keyboard direction into snake relative action (0=Straight, 1=Right, 2=Left)."""
        directions = ["UP", "RIGHT", "DOWN", "LEFT"]
        cur_idx = directions.index(self.game.direction)

        if self.human_buffered_direction:
            desired_dir = self.human_buffered_direction
            des_idx = directions.index(desired_dir)
            diff = (des_idx - cur_idx) % 4
            if diff == 0:
                action = 0
            elif diff == 1:
                action = 1
            elif diff == 3:
                action = 2
            else:
                action = 0  # Cannot turn directly backwards
        else:
            action = 0

        reward, done, score = self.game.move(action)
        return action, done, score

    def _step_ai(self) -> Tuple[int, bool, int]:
        """Execute one decision step for the selected AI agent."""
        meta = self.current_agent_meta
        s = self.game.get_state()

        # Compute collision risk
        hx, hy = self.game.snake[0]
        bx = self.game.block_size
        dirs = ["UP", "RIGHT", "DOWN", "LEFT"]
        c_idx = dirs.index(self.game.direction)
        pt_straight = self.game._get_candidate_point(dirs[c_idx], (hx, hy))
        pt_right = self.game._get_candidate_point(dirs[(c_idx + 1) % 4], (hx, hy))
        pt_left = self.game._get_candidate_point(dirs[(c_idx - 1) % 4], (hx, hy))
        self.cur_risk = {
            "straight": self.game.check_collision(pt_straight),
            "right": self.game.check_collision(pt_right),
            "left": self.game.check_collision(pt_left),
        }

        if meta["is_pure_astar"]:
            act_idx = self.astar_agent.get_action(self.game)
            self.cur_q_values = None
            self.cur_shield_info = {}
            path = self.astar_agent.last_path or []
            self.cur_astar_info = {
                "route_length": len(path),
                "target_type": "Food" if path else "Tail Chase",
                "tail_reachable": True,
            }
        elif meta["use_shield"]:
            act_idx, was_overridden, source = self.agent.get_action_with_telemetry(s, self.game, self.shield)
            _, self.cur_q_values = self.agent.get_action_and_q_values(s, deterministic=True)
            self.cur_shield_info = {
                "enabled": True,
                "was_overridden": was_overridden,
                "rejection_reason": self.shield.last_rejection_reason if was_overridden else "None",
                "fallback_source": source,
                "intervention_count": self.shield.actions_rejected,
            }
            self.cur_astar_info = {}
        else:
            # Standalone DQN
            act_idx = self.agent.get_action_index(s, deterministic=True)
            _, self.cur_q_values = self.agent.get_action_and_q_values(s, deterministic=True)
            self.cur_shield_info = {"enabled": False}
            self.cur_astar_info = {}

        self.cur_action = act_idx

        # Check termination reason right before fatal move
        directions = ["UP", "RIGHT", "DOWN", "LEFT"]
        cur_idx = directions.index(self.game.direction)
        if act_idx == 1:
            cur_idx = (cur_idx + 1) % 4
        elif act_idx == 2:
            cur_idx = (cur_idx - 1) % 4
        cand_dir = directions[cur_idx]
        cand_pt = self.game._get_candidate_point(cand_dir, (hx, hy))

        is_wall = (cand_pt[0] < 0 or cand_pt[0] >= self.game.width or cand_pt[1] < 0 or cand_pt[1] >= self.game.height)
        is_body = (cand_pt in self.game.snake[:-1])

        reward, done, score = self.game.move(act_idx)

        if done:
            if is_wall:
                self.game_over_reason = "Wall Collision"
            elif is_body:
                self.game_over_reason = "Self-Body Collision"
            else:
                self.game_over_reason = "Starvation Timeout"

        return act_idx, done, score

    def update(self) -> None:
        """Update game loop logic if running."""
        self.board_renderer.update_animation()

        if self.current_tab != TAB_ARENA or self.is_paused or self.is_game_over:
            return

        if self.current_agent_meta["is_human"]:
            act, done, score = self._step_human()
        else:
            act, done, score = self._step_ai()

        if done:
            self.is_game_over = True
            if score > self.record:
                self.record = score
                self._save_record(self.record)

            # Record in live session stats
            key = self.current_agent_meta["key"]
            if key in self.live_session_scores:
                self.live_session_scores[key].append(score)

    def handle_events(self) -> bool:
        """Process keyboard, mouse, and widget interactions. Returns False on QUIT."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            # Top tab switching
            if self.tab_selector.handle_event(event):
                continue

            # Tab-specific routing
            if self.current_tab == TAB_ARENA:
                # Agent dropdown
                if self.agent_dropdown.handle_event(event):
                    continue

                # Control buttons
                if self.btn_play_pause.handle_event(event):
                    continue
                if self.btn_restart.handle_event(event):
                    continue
                if self.btn_toggle_path.handle_event(event):
                    continue

                # Speed buttons
                for i, b in enumerate(self.speed_buttons):
                    if b.handle_event(event):
                        self.current_fps_idx = i
                        self.target_fps = self.fps_options[i]
                        for j, sb in enumerate(self.speed_buttons):
                            sb.is_active = (j == i)
                        break

                # Keyboard hotkeys
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self._toggle_pause()
                    elif event.key == pygame.K_r:
                        self.reset_game()
                    elif event.key == pygame.K_p:
                        self._toggle_path()
                    elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5]:
                        idx = event.key - pygame.K_1
                        self.current_fps_idx = idx
                        self.target_fps = self.fps_options[idx]
                        for j, sb in enumerate(self.speed_buttons):
                            sb.is_active = (j == idx)

                    # Human controls
                    if self.current_agent_meta["is_human"] and not self.is_paused and not self.is_game_over:
                        if event.key in [pygame.K_UP, pygame.K_w] and self.game.direction != "DOWN":
                            self.human_buffered_direction = "UP"
                        elif event.key in [pygame.K_DOWN, pygame.K_s] and self.game.direction != "UP":
                            self.human_buffered_direction = "DOWN"
                        elif event.key in [pygame.K_LEFT, pygame.K_a] and self.game.direction != "RIGHT":
                            self.human_buffered_direction = "LEFT"
                        elif event.key in [pygame.K_RIGHT, pygame.K_d] and self.game.direction != "LEFT":
                            self.human_buffered_direction = "RIGHT"

            elif self.current_tab == TAB_COMPARISON:
                self.comparison_view.handle_event(event)

            elif self.current_tab == TAB_TRAINING:
                self.training_view.handle_event(event)

            elif self.current_tab == TAB_ANALYTICS:
                self.analytics_view.handle_event(event)

        return True

    def render(self) -> None:
        """Render active tab and UI components onto the screen."""
        self.screen.fill(BG_DARK)

        # Draw Header
        self.tab_selector.draw(self.screen)

        if self.current_tab == TAB_ARENA:
            # 1. Render Game Board
            astar_path = None
            if self.show_path:
                if self.current_agent_meta["is_pure_astar"]:
                    astar_path = self.astar_agent.last_path
                else:
                    snake_grid = [to_grid(p, BLOCK_SIZE) for p in self.game.snake]
                    head_g = snake_grid[0]
                    food_g = to_grid(self.game.food, BLOCK_SIZE)
                    obstacles = set(snake_grid[:-1])
                    path_g = self.path_finder.a_star(head_g, food_g, obstacles)
                    if path_g:
                        astar_path = [(gx * BLOCK_SIZE, gy * BLOCK_SIZE) for gx, gy in path_g]

            board_surf = self.board_renderer.render(
                snake=self.game.snake,
                food=self.game.food,
                direction=self.game.direction,
                score=self.game.score,
                record=self.record,
                is_paused=self.is_paused,
                is_game_over=self.is_game_over,
                game_over_reason=self.game_over_reason,
                astar_path=astar_path,
                show_path=self.show_path,
                is_human=self.current_agent_meta["is_human"],
            )
            self.screen.blit(board_surf, (BOARD_X, BOARD_Y))

            # 2. Render Telemetry Sidebar
            hx, hy = self.game.snake[0]
            fx, fy = self.game.food
            bx = self.game.block_size
            food_dist = (abs(hx - fx) + abs(hy - fy)) // bx
            total_cells = getattr(self.game, "total_grid_cells", (self.game.width // bx) * (self.game.height // bx))
            free_space_ratio = self.game._flood_fill_space() / max(1, total_cells)

            self.telemetry_panel.draw(
                surface=self.screen,
                agent_info=self.current_agent_meta,
                score=self.game.score,
                record=self.record,
                food_eaten=self.game.food_eaten,
                steps=self.game.frame_iteration,
                direction=self.game.direction,
                collision_risk=self.cur_risk,
                food_dist=food_dist,
                free_space_ratio=free_space_ratio,
                selected_action=self.cur_action,
                q_values=self.cur_q_values,
                shield_telemetry=self.cur_shield_info,
                astar_info=self.cur_astar_info,
                is_human=self.current_agent_meta["is_human"],
            )

            # 3. Render Footer Buttons
            self.btn_play_pause.draw(self.screen)
            self.btn_restart.draw(self.screen)
            self.btn_toggle_path.draw(self.screen)
            for b in self.speed_buttons:
                b.draw(self.screen)

            # Agent Dropdown (drawn last so menu renders on top)
            self.agent_dropdown.draw(self.screen)
            self.agent_dropdown.draw_menu(self.screen)

        elif self.current_tab == TAB_COMPARISON:
            self.comparison_view.draw(self.screen, live_history=self.live_session_scores)

        elif self.current_tab == TAB_TRAINING:
            self.training_view.draw(self.screen)

        elif self.current_tab == TAB_ANALYTICS:
            self.analytics_view.draw(self.screen)

        pygame.display.flip()

    def run(self) -> None:
        """Main application execution loop."""
        running = True
        try:
            while running:
                running = self.handle_events()
                self.update()
                self.render()

                if self.target_fps > 0:
                    self.clock.tick(self.target_fps)
                else:
                    self.clock.tick(120)  # Turbo mode tick
        finally:
            pygame.quit()


def launch_app() -> None:
    """Entry point to launch the graphical platform."""
    app = SnakeAIApp()
    app.run()


if __name__ == "__main__":
    launch_app()

