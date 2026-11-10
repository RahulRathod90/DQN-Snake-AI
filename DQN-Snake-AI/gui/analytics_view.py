"""
Scientific Analytics & Publication Visualizations View.
Allows interactive inspection of all 7 Phase 4 benchmark plots,
in-depth statistical interpretations, and one-click CSV / JSON data exports.
"""

from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple
import pygame

from config import BASE_DIR, RESULTS_DIR
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
from gui.widgets import Button, draw_card

CHARTS_METADATA = [
    {
        "id": "avg_score",
        "title": "Mean Score Comparison",
        "filename": "phase4_average_score_comparison.png",
        "desc": "Compares 100-game mean food scores across all 6 variants with standard deviation error bars. Pure A* dominates with 222.75, followed by DQN + Shield (118.36), and Optimized DQN (31.52).",
    },
    {
        "id": "score_dist",
        "title": "Score Distribution Boxplot",
        "filename": "phase4_score_distribution.png",
        "desc": "Box-and-whisker plot illustrating median scores, interquartile ranges, and variance. Shows how the A* Safety Shield significantly reduces early mortality.",
    },
    {
        "id": "survival",
        "title": "Survival Steps per Game",
        "filename": "phase4_survival_comparison.png",
        "desc": "Logarithmic scale survival frame count. Pure A* survives 33,193 steps on average; Hybrid Shield enables over 4,367 steps, an 8x increase over baseline DQN.",
    },
    {
        "id": "training",
        "title": "Training & Loss Curves",
        "filename": "phase4_training_curves.png",
        "desc": "50-episode moving average reward and Smooth L1 Huber loss progression comparing the corrected 24-D Enhanced DQN (Exp C) and Tuned DQN (Exp F).",
    },
    {
        "id": "collision",
        "title": "Collision & Termination Modes",
        "filename": "phase4_collision_rate.png",
        "desc": "Detailed breakdown of wall collisions, self-body collisions, and starvation timeouts. Tuned DQN reduced wall collisions from 76% in baseline to only 1%.",
    },
    {
        "id": "inference",
        "title": "Inference Latency Profile",
        "filename": "phase4_inference_time.png",
        "desc": "Average decision time in milliseconds. Baseline DQN executes in 0.092ms; Tuned DQN in 0.269ms; Hybrid Safety Shield with multi-tier validation completes in 1.142ms.",
    },
    {
        "id": "hybrid",
        "title": "Hybrid Safety Shield Telemetry",
        "filename": "phase4_hybrid_shield_analysis.png",
        "desc": "Pie chart of approval vs rejection rates (10.7% override rate) alongside a histogram of fallback triggers (A* food routes, alternative DQN moves, tail chasing).",
    },
]


class AnalyticsView:
    """Analytics view displaying embedded matplotlib charts, interpretation notes, and export actions."""

    def __init__(self, rect: Tuple[int, int, int, int]):
        self.rect = pygame.Rect(rect)
        self.selected_chart_idx = 0
        self.loaded_surfaces: Dict[str, Optional[pygame.Surface]] = {}
        self.export_status = ""
        self.export_timer = 0.0

        # Chart selection buttons
        btn_y = self.rect.y + 16
        self.chart_buttons: List[Button] = []
        for i, meta in enumerate(CHARTS_METADATA):
            btn = Button(
                (self.rect.x + 20 + (i % 4) * 175, btn_y + (i // 4) * 36, 165, 30),
                meta["title"],
                font_size=11,
                is_active=(i == 0),
            )
            self.chart_buttons.append(btn)

        # Export Buttons
        exp_y = btn_y + 80
        self.btn_export_csv = Button(
            (self.rect.x + 740, exp_y, 190, 32),
            "Export Results (CSV)",
            on_click=self._export_csv,
            active_color=ACCENT_GREEN,
            font_size=12,
        )
        self.btn_export_json = Button(
            (self.rect.x + 940, exp_y, 190, 32),
            "Export Results (JSON)",
            on_click=self._export_json,
            active_color=ACCENT_CYAN,
            font_size=12,
        )

        self._load_chart_images()

    def _load_chart_images(self) -> None:
        """Load and cache scaled Pygame surfaces for the charts."""
        for meta in CHARTS_METADATA:
            fname = meta["filename"]
            p = RESULTS_DIR / fname
            if not p.is_file():
                p = BASE_DIR / "results" / fname

            if p.is_file():
                try:
                    img = pygame.image.load(str(p))
                    # Scale to fit image display viewport (700 x 420)
                    scaled = pygame.transform.smoothscale(img, (700, 420))
                    self.loaded_surfaces[fname] = scaled
                except Exception:
                    self.loaded_surfaces[fname] = None
            else:
                self.loaded_surfaces[fname] = None

    def _export_csv(self) -> None:
        src = RESULTS_DIR / "phase4_benchmark_results.csv"
        dst = BASE_DIR / "exported_phase4_results.csv"
        if src.is_file():
            shutil.copy(src, dst)
            self.export_status = f"Saved CSV to: {dst.name}"
            self.export_timer = 4.0

    def _export_json(self) -> None:
        src = RESULTS_DIR / "phase4_benchmark_results.json"
        dst = BASE_DIR / "exported_phase4_results.json"
        if src.is_file():
            shutil.copy(src, dst)
            self.export_status = f"Saved JSON to: {dst.name}"
            self.export_timer = 4.0

    def handle_event(self, event: pygame.event.Event) -> bool:
        if self.btn_export_csv.handle_event(event):
            return True
        if self.btn_export_json.handle_event(event):
            return True

        for i, btn in enumerate(self.chart_buttons):
            if btn.handle_event(event):
                self.selected_chart_idx = i
                for j, b in enumerate(self.chart_buttons):
                    b.is_active = (j == i)
                return True
        return False

    def draw(self, surface: pygame.Surface, dt: float = 0.05) -> None:
        """Render chart viewport, interpretation panel, and export status."""
        pygame.draw.rect(surface, BG_DARK, self.rect)

        # Draw chart selector buttons
        for btn in self.chart_buttons:
            btn.draw(surface)

        # Draw export buttons
        self.btn_export_csv.draw(surface)
        self.btn_export_json.draw(surface)

        # Export toast notification
        if self.export_timer > 0:
            self.export_timer -= dt
            font_toast = get_font(12, bold=True)
            toast = font_toast.render(self.export_status, True, ACCENT_GREEN)
            surface.blit(toast, (self.rect.x + 740, self.rect.y + 54))

        # 1. Main Chart Display Viewport (Left)
        view_rect = pygame.Rect(self.rect.x + 20, self.rect.y + 120, 720, 460)
        draw_card(surface, view_rect, bg_color=BG_CARD)

        cur_meta = CHARTS_METADATA[self.selected_chart_idx]
        fname = cur_meta["filename"]
        img_surf = self.loaded_surfaces.get(fname)

        if img_surf:
            surface.blit(img_surf, (view_rect.x + 10, view_rect.y + 20))
        else:
            font_err = get_font(14)
            err_txt = font_err.render(f"Chart file {fname} not found in results directory.", True, ACCENT_RED)
            surface.blit(err_txt, err_txt.get_rect(center=view_rect.center))

        # 2. Right Interpretation & Statistical Takeaways Card
        note_rect = pygame.Rect(self.rect.x + 760, self.rect.y + 120, self.rect.width - 780, 460)
        draw_card(surface, note_rect, title="Scientific Analysis & Insights", bg_color=BG_CARD)

        font_title = get_font(14, bold=True)
        surface.blit(font_title.render(cur_meta["title"], True, ACCENT_CYAN), (note_rect.x + 16, note_rect.y + 42))

        # Description text wrap
        font_body = get_font(12)
        words = cur_meta["desc"].split(" ")
        lines = []
        cur_line = []
        for w in words:
            cur_line.append(w)
            test_line = " ".join(cur_line)
            if font_body.size(test_line)[0] > (note_rect.width - 32):
                cur_line.pop()
                lines.append(" ".join(cur_line))
                cur_line = [w]
        if cur_line:
            lines.append(" ".join(cur_line))

        ty = note_rect.y + 72
        for l in lines:
            surface.blit(font_body.render(l, True, TEXT_SECONDARY), (note_rect.x + 16, ty))
            ty += 20

        # Verified Findings Summary Card inside right panel
        sum_rect = pygame.Rect(note_rect.x + 12, note_rect.y + 200, note_rect.width - 24, 240)
        draw_card(surface, sum_rect, title="Top Validated Models Summary", bg_color=(24, 26, 38))

        models_summary = [
            ("Pure A* Agent", "222.75 Mean", "Max Score 474", ACCENT_GREEN),
            ("DQN + A* Safety Shield", "118.36 Mean", "Max Score 182", ACCENT_GREEN),
            ("Optimized DQN (Var F)", "31.52 Mean", "Max Score 71", ACCENT_CYAN),
            ("Original DQN Baseline", "28.99 Mean", "Max Score 61", TEXT_SECONDARY),
        ]

        sy = sum_rect.y + 38
        font_sm = get_font(11, bold=True)
        font_det = get_font(10)
        for mname, mval, mmax, mcol in models_summary:
            surface.blit(font_sm.render(mname, True, TEXT_PRIMARY), (sum_rect.x + 14, sy))
            surface.blit(font_sm.render(mval, True, mcol), (sum_rect.right - 100, sy))
            surface.blit(font_det.render(mmax, True, TEXT_MUTED), (sum_rect.x + 14, sy + 18))
            pygame.draw.line(surface, BORDER_COLOR, (sum_rect.x + 10, sy + 38), (sum_rect.right - 10, sy + 38), width=1)
            sy += 48

