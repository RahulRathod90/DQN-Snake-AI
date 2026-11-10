"""
Interactive Non-Blocking Training Studio for the Snake AI Application.
Enables real-time model training, checkpoint resumption, hyperparameter tuning,
and dynamic in-GUI loss and score plotting via an asynchronous background thread.
"""

from collections import deque
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pygame
import torch

from agent import Agent, DQNAgent
from config import BASE_DIR, CHECKPOINTS_DIR
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
    BORDER_COLOR,
    BORDER_HIGHLIGHT,
    BTN_ACTIVE_BG,
    BTN_ACTIVE_TEXT,
    BTN_PRIMARY_BG,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    get_font,
)
from gui.widgets import Button, LiveChart, ProgressBar, draw_card
from snake_ai import SnakeGameAI


class TrainingWorker(threading.Thread):
    """Background worker thread executing training episodes asynchronously without freezing Pygame."""

    def __init__(
        self,
        algorithm: str,
        target_episodes: int,
        learning_rate: float,
        batch_size: int,
        gamma: float,
        device: str,
        resume_checkpoint: Optional[str] = None,
    ):
        super().__init__(daemon=True)
        self.algorithm = algorithm
        self.target_episodes = target_episodes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device
        self.resume_checkpoint = resume_checkpoint

        self.is_running = True
        self.is_paused = False
        self.stop_requested = False

        # Thread-safe shared metrics
        self.current_episode = 0
        self.current_score = 0
        self.moving_avg_score = 0.0
        self.current_loss = 0.0
        self.current_epsilon = 1.0
        self.current_reward = 0.0
        self.buffer_size = 0
        self.status_message = "Initializing training environment..."

        self.scores_history: List[float] = []
        self.moving_avgs_history: List[float] = []
        self.loss_history: List[float] = []

    def request_stop(self) -> None:
        """Signal the worker to gracefully complete the current episode and save."""
        self.stop_requested = True
        self.status_message = "Stopping gracefully after current episode..."

    def run(self) -> None:
        try:
            state_mode = "enhanced" if "24" in self.algorithm else "standard"
            state_size = 24 if state_mode == "enhanced" else 11
            use_dueling = "dueling" in self.algorithm.lower() or "per" in self.algorithm.lower() or state_mode == "enhanced"
            use_double = "ddqn" in self.algorithm.lower() or use_dueling
            use_per = "per" in self.algorithm.lower() or use_dueling

            agent = DQNAgent(
                state_size=state_size,
                action_size=3,
                lr=self.learning_rate,
                gamma=self.gamma,
                batch_size=self.batch_size,
                use_double_dqn=use_double,
                dueling=use_dueling,
                use_per=use_per,
                device=self.device,
                min_replay_size=max(self.batch_size, 64),
            )

            start_ep = 1
            if self.resume_checkpoint:
                ckpt_p = Path(self.resume_checkpoint)
                if ckpt_p.is_file():
                    agent.load_checkpoint(ckpt_p)
                    start_ep = agent.n_games + 1
                    self.status_message = f"Resumed from {ckpt_p.name} at episode {start_ep}"

            game = SnakeGameAI(display=False, state_mode=state_mode)
            score_window = deque(maxlen=20)

            for ep in range(start_ep, self.target_episodes + 1):
                if self.stop_requested:
                    break

                self.current_episode = ep
                self.status_message = f"Training Episode {ep}/{self.target_episodes}..."
                game.reset()
                done = False
                ep_losses = []

                while not done:
                    if self.stop_requested:
                        break

                    s = game.get_state()
                    act_idx = agent.get_action_index(s, deterministic=False)
                    reward, done, score = game.move(act_idx)
                    next_s = game.get_state()

                    agent.remember(s, act_idx, reward, next_s, done)
                    loss = agent.train_replay_batch()
                    if loss is not None:
                        ep_losses.append(loss)

                agent.decay_epsilon()
                agent.n_games += 1

                score_window.append(game.score)
                m_avg = float(np.mean(score_window))
                mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0

                self.current_score = game.score
                self.moving_avg_score = m_avg
                self.current_loss = mean_loss
                self.current_epsilon = agent.epsilon
                self.current_reward = game.total_reward
                self.buffer_size = len(agent.memory)

                self.scores_history.append(float(game.score))
                self.moving_avgs_history.append(m_avg)
                if mean_loss > 0:
                    self.loss_history.append(mean_loss)

                # Save periodic checkpoint
                if ep % 25 == 0 or ep == self.target_episodes:
                    save_path = CHECKPOINTS_DIR / f"gui_training_{agent.state_size}d_ep{ep}.pth"
                    agent.save_checkpoint(
                        save_path,
                        episode=ep,
                        scores=[int(s) for s in self.scores_history],
                        losses=self.loss_history,
                        moving_avg_scores=self.moving_avgs_history,
                        best_score=int(np.max(self.scores_history)),
                        best_avg_score=float(np.max(self.moving_avgs_history)),
                    )

            self.status_message = "Training Completed Successfully!" if not self.stop_requested else "Training Stopped Safely."

        except Exception as e:
            self.status_message = f"Training Error: {str(e)}"
        finally:
            self.is_running = False


class TrainingView:
    """Interactive Training Studio Tab rendering configuration widgets and live curves."""

    def __init__(self, rect: Tuple[int, int, int, int]):
        self.rect = pygame.Rect(rect)
        self.worker: Optional[TrainingWorker] = None

        # Configuration options
        self.algorithms = ["Original DQN", "DDQN + Dueling + PER", "Enhanced 24-D DQN"]
        self.selected_algo_idx = 1
        self.episodes_options = [50, 100, 200, 500]
        self.selected_ep_idx = 1
        self.lr_options = [0.001, 0.0005, 0.0001]
        self.selected_lr_idx = 0
        self.batch_options = [32, 64, 128]
        self.selected_batch_idx = 2
        self.gamma_options = [0.90, 0.95, 0.99]
        self.selected_gamma_idx = 1
        self.devices = ["cpu"]
        if torch.cuda.is_available():
            self.devices.append("cuda")
        self.selected_device_idx = 0

        # Action Buttons
        ctrl_x = self.rect.x + 20
        self.btn_start = Button(
            (ctrl_x, self.rect.y + 440, 140, 38),
            "Start Training",
            on_click=self._start_training,
            active_color=ACCENT_GREEN,
            font_size=13,
        )
        self.btn_stop = Button(
            (ctrl_x + 150, self.rect.y + 440, 140, 38),
            "Stop Safely",
            on_click=self._stop_training,
            active_color=ACCENT_RED,
            font_size=13,
        )
        self.btn_stop.is_disabled = True

        # Live Curves
        chart_x = self.rect.x + 380
        chart_w = self.rect.width - 400
        self.score_chart = LiveChart(
            (chart_x, self.rect.y + 16, chart_w, 240),
            title="Live Episode Scores & 20-Ep Moving Average",
            x_label="Episode",
            y_label="Score",
        )
        self.score_chart.add_series("Score", ACCENT_CYAN, width=1)
        self.score_chart.add_series("Moving Avg", ACCENT_GREEN, width=2)

        self.loss_chart = LiveChart(
            (chart_x, self.rect.y + 266, chart_w, 230),
            title="Huber Training Loss Progression",
            x_label="Step",
            y_label="Loss",
        )
        self.loss_chart.add_series("Smooth L1 Loss", ACCENT_RED, width=2)

    def _start_training(self) -> None:
        if self.worker and self.worker.is_running:
            return

        algo = self.algorithms[self.selected_algo_idx]
        target_ep = self.episodes_options[self.selected_ep_idx]
        lr = self.lr_options[self.selected_lr_idx]
        batch_size = self.batch_options[self.selected_batch_idx]
        gamma = self.gamma_options[self.selected_gamma_idx]
        dev = self.devices[self.selected_device_idx]

        self.worker = TrainingWorker(
            algorithm=algo,
            target_episodes=target_ep,
            learning_rate=lr,
            batch_size=batch_size,
            gamma=gamma,
            device=dev,
        )
        self.worker.start()
        self.btn_start.is_disabled = True
        self.btn_stop.is_disabled = False

    def _stop_training(self) -> None:
        if self.worker and self.worker.is_running:
            self.worker.request_stop()
            self.btn_stop.is_disabled = True

    def handle_event(self, event: pygame.event.Event) -> bool:
        if self.btn_start.handle_event(event):
            return True
        if self.btn_stop.handle_event(event):
            return True

        # Click handling for option selectors
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            px, py = event.pos
            # Algorithm selector clicks
            for i in range(len(self.algorithms)):
                r = pygame.Rect(self.rect.x + 20, self.rect.y + 70 + i * 28, 320, 24)
                if r.collidepoint(px, py):
                    self.selected_algo_idx = i
                    return True
            # Episode clicks
            for i, _ in enumerate(self.episodes_options):
                r = pygame.Rect(self.rect.x + 20 + i * 78, self.rect.y + 190, 72, 26)
                if r.collidepoint(px, py):
                    self.selected_ep_idx = i
                    return True
            # Learning rate clicks
            for i, _ in enumerate(self.lr_options):
                r = pygame.Rect(self.rect.x + 20 + i * 105, self.rect.y + 246, 98, 26)
                if r.collidepoint(px, py):
                    self.selected_lr_idx = i
                    return True
            # Batch size clicks
            for i, _ in enumerate(self.batch_options):
                r = pygame.Rect(self.rect.x + 20 + i * 105, self.rect.y + 302, 98, 26)
                if r.collidepoint(px, py):
                    self.selected_batch_idx = i
                    return True
            # Gamma clicks
            for i, _ in enumerate(self.gamma_options):
                r = pygame.Rect(self.rect.x + 20 + i * 105, self.rect.y + 358, 98, 26)
                if r.collidepoint(px, py):
                    self.selected_gamma_idx = i
                    return True
        return False

    def draw(self, surface: pygame.Surface) -> None:
        """Render training studio controls and live charts."""
        pygame.draw.rect(surface, BG_DARK, self.rect)

        # 1. Left Config Container
        cfg_rect = pygame.Rect(self.rect.x + 12, self.rect.y + 16, 350, 480)
        draw_card(surface, cfg_rect, title="Hyperparameter & Model Settings", bg_color=BG_CARD)

        font_label = get_font(12, bold=True)
        font_opt = get_font(11)

        # Algorithm selection
        surface.blit(font_label.render("Select AI Architecture:", True, TEXT_SECONDARY), (self.rect.x + 20, self.rect.y + 48))
        for i, aname in enumerate(self.algorithms):
            r = pygame.Rect(self.rect.x + 20, self.rect.y + 70 + i * 28, 320, 24)
            is_sel = (i == self.selected_algo_idx)
            pygame.draw.rect(surface, (40, 60, 50) if is_sel else (24, 26, 38), r, border_radius=4)
            pygame.draw.rect(surface, ACCENT_GREEN if is_sel else BORDER_COLOR, r, width=1, border_radius=4)
            txt_col = ACCENT_GREEN if is_sel else TEXT_SECONDARY
            surface.blit(font_opt.render(aname, True, txt_col), (r.x + 10, r.centery - 6))

        # Target Episodes
        surface.blit(font_label.render("Target Training Episodes:", True, TEXT_SECONDARY), (self.rect.x + 20, self.rect.y + 168))
        for i, ep in enumerate(self.episodes_options):
            r = pygame.Rect(self.rect.x + 20 + i * 78, self.rect.y + 190, 72, 26)
            is_sel = (i == self.selected_ep_idx)
            pygame.draw.rect(surface, (40, 60, 50) if is_sel else (24, 26, 38), r, border_radius=4)
            pygame.draw.rect(surface, ACCENT_GREEN if is_sel else BORDER_COLOR, r, width=1, border_radius=4)
            txt = font_opt.render(f"{ep} Ep", True, ACCENT_GREEN if is_sel else TEXT_SECONDARY)
            surface.blit(txt, txt.get_rect(center=r.center))

        # Learning Rate
        surface.blit(font_label.render("Learning Rate (α):", True, TEXT_SECONDARY), (self.rect.x + 20, self.rect.y + 226))
        for i, lr in enumerate(self.lr_options):
            r = pygame.Rect(self.rect.x + 20 + i * 105, self.rect.y + 246, 98, 26)
            is_sel = (i == self.selected_lr_idx)
            pygame.draw.rect(surface, (40, 60, 50) if is_sel else (24, 26, 38), r, border_radius=4)
            pygame.draw.rect(surface, ACCENT_GREEN if is_sel else BORDER_COLOR, r, width=1, border_radius=4)
            txt = font_opt.render(f"{lr}", True, ACCENT_GREEN if is_sel else TEXT_SECONDARY)
            surface.blit(txt, txt.get_rect(center=r.center))

        # Batch Size
        surface.blit(font_label.render("Mini-Batch Size:", True, TEXT_SECONDARY), (self.rect.x + 20, self.rect.y + 282))
        for i, bs in enumerate(self.batch_options):
            r = pygame.Rect(self.rect.x + 20 + i * 105, self.rect.y + 302, 98, 26)
            is_sel = (i == self.selected_batch_idx)
            pygame.draw.rect(surface, (40, 60, 50) if is_sel else (24, 26, 38), r, border_radius=4)
            pygame.draw.rect(surface, ACCENT_GREEN if is_sel else BORDER_COLOR, r, width=1, border_radius=4)
            txt = font_opt.render(f"{bs}", True, ACCENT_GREEN if is_sel else TEXT_SECONDARY)
            surface.blit(txt, txt.get_rect(center=r.center))

        # Gamma Discount Factor
        surface.blit(font_label.render("Discount Factor (γ):", True, TEXT_SECONDARY), (self.rect.x + 20, self.rect.y + 338))
        for i, gm in enumerate(self.gamma_options):
            r = pygame.Rect(self.rect.x + 20 + i * 105, self.rect.y + 358, 98, 26)
            is_sel = (i == self.selected_gamma_idx)
            pygame.draw.rect(surface, (40, 60, 50) if is_sel else (24, 26, 38), r, border_radius=4)
            pygame.draw.rect(surface, ACCENT_GREEN if is_sel else BORDER_COLOR, r, width=1, border_radius=4)
            txt = font_opt.render(f"{gm}", True, ACCENT_GREEN if is_sel else TEXT_SECONDARY)
            surface.blit(txt, txt.get_rect(center=r.center))

        # Status text
        if self.worker:
            stat_str = self.worker.status_message
            if not self.worker.is_running:
                self.btn_start.is_disabled = False
                self.btn_stop.is_disabled = True
        else:
            stat_str = "Status: Ready to train"

        surface.blit(font_opt.render(stat_str, True, ACCENT_CYAN), (self.rect.x + 20, self.rect.y + 404))

        # Buttons
        self.btn_start.draw(surface)
        self.btn_stop.draw(surface)

        # 2. Right Charts Container
        if self.worker:
            self.score_chart.update_data("Score", self.worker.scores_history)
            self.score_chart.update_data("Moving Avg", self.worker.moving_avgs_history)
            self.loss_chart.update_data("Smooth L1 Loss", self.worker.loss_history)

        self.score_chart.draw(surface)
        self.loss_chart.draw(surface)

        # 3. Bottom Live Metrics Strip
        strip_rect = pygame.Rect(self.rect.x + 12, self.rect.y + 510, self.rect.width - 24, 80)
        draw_card(surface, strip_rect, bg_color=BG_CARD)

        if self.worker and self.worker.is_running:
            m_ep = f"{self.worker.current_episode}/{self.worker.target_episodes}"
            m_sc = f"{self.worker.current_score}"
            m_avg = f"{self.worker.moving_avg_score:.2f}"
            m_eps = f"{self.worker.current_epsilon:.3f}"
            m_loss = f"{self.worker.current_loss:.4f}"
            m_buf = f"{self.worker.buffer_size:,}"
        else:
            m_ep, m_sc, m_avg, m_eps, m_loss, m_buf = "-", "-", "-", "-", "-", "-"

        metrics = [
            ("EPISODE", m_ep, ACCENT_CYAN),
            ("LAST SCORE", m_sc, ACCENT_GREEN),
            ("MOVING AVG", m_avg, ACCENT_YELLOW),
            ("EPSILON", m_eps, TEXT_PRIMARY),
            ("HUBER LOSS", m_loss, ACCENT_RED),
            ("REPLAY MEMORY", m_buf, ACCENT_PURPLE),
        ]

        strip_col_w = strip_rect.width // 6
        font_sval = get_font(16, bold=True, mono=True)
        font_slbl = get_font(10, bold=True)

        for i, (lbl, val, col) in enumerate(metrics):
            cx = strip_rect.x + i * strip_col_w + strip_col_w // 2
            lsurf = font_slbl.render(lbl, True, TEXT_MUTED)
            surface.blit(lsurf, lsurf.get_rect(center=(cx, strip_rect.y + 24)))
            vsurf = font_sval.render(val, True, col)
            surface.blit(vsurf, vsurf.get_rect(center=(cx, strip_rect.y + 50)))

