"""
Algorithm Comparison View for the Snake AI Application.
Provides interactive side-by-side analysis of all 6 validated AI variants,
rendering verified 100-game benchmark metrics alongside live session statistics.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pygame

from gui.theme import (
    ACCENT_CYAN,
    ACCENT_GREEN,
    ACCENT_ORANGE,
    ACCENT_PURPLE,
    ACCENT_RED,
    ACCENT_YELLOW,
    BG_CARD,
    BG_CARD_HOVER,
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

# Verified Phase 4 100-Game Benchmark Data
PHASE4_BENCHMARK_DATA: List[Dict[str, Any]] = [
    {
        "id": "Var A",
        "name": "Original DQN (Phase 1 Baseline)",
        "short_name": "Original DQN",
        "arch": "Standard FC (11-D)",
        "mean": 28.99,
        "max": 61,
        "median": 28.0,
        "std": 12.32,
        "steps": 550.1,
        "latency_ms": 0.092,
        "wall_pct": 76.0,
        "body_pct": 24.0,
        "badge": "Phase 1 Baseline",
        "badge_col": ACCENT_CYAN,
    },
    {
        "id": "Var B",
        "name": "DDQN + Dueling + PER (Phase 2 Best)",
        "short_name": "DDQN + PER",
        "arch": "Dueling FC (11-D)",
        "mean": 26.69,
        "max": 63,
        "median": 26.0,
        "std": 14.71,
        "steps": 683.8,
        "latency_ms": 0.350,
        "wall_pct": 68.0,
        "body_pct": 32.0,
        "badge": "Phase 2 Winner",
        "badge_col": ACCENT_PURPLE,
    },
    {
        "id": "Var C",
        "name": "Enhanced 24-feature DQN (Phase 4 Corrected)",
        "short_name": "Enhanced 24-D",
        "arch": "Dueling FC (24-D)",
        "mean": 21.68,
        "max": 52,
        "median": 22.0,
        "std": 11.83,
        "steps": 873.1,
        "latency_ms": 0.605,
        "wall_pct": 72.0,
        "body_pct": 28.0,
        "badge": "Phase 4 Restored",
        "badge_col": ACCENT_YELLOW,
    },
    {
        "id": "Var D",
        "name": "Pure A* Agent (Deterministic Algorithmic)",
        "short_name": "Pure A*",
        "arch": "Algorithmic Planner",
        "mean": 222.75,
        "max": 474,
        "median": 216.0,
        "std": 111.85,
        "steps": 33193.8,
        "latency_ms": 0.305,
        "wall_pct": 4.0,
        "body_pct": 91.0,
        "badge": "Global Ceiling",
        "badge_col": ACCENT_GREEN,
    },
    {
        "id": "Var E",
        "name": "DQN + A* Safety Shield (Phase 4 Hybrid)",
        "short_name": "DQN + Shield",
        "arch": "Dueling + A* Shield",
        "mean": 118.36,
        "max": 182,
        "median": 130.5,
        "std": 46.77,
        "steps": 4367.2,
        "latency_ms": 1.142,
        "wall_pct": 16.0,
        "body_pct": 74.0,
        "badge": "Best Hybrid Agent",
        "badge_col": ACCENT_GREEN,
    },
    {
        "id": "Var F",
        "name": "Optimized DQN (Validated Hyperparameters)",
        "short_name": "Optimized DQN",
        "arch": "Dueling FC (11-D)",
        "mean": 31.52,
        "max": 71,
        "median": 33.5,
        "std": 15.62,
        "steps": 882.6,
        "latency_ms": 0.269,
        "wall_pct": 1.0,
        "body_pct": 82.0,
        "badge": "Top Pure RL Agent",
        "badge_col": ACCENT_CYAN,
    },
]


class ComparisonView:
    """Renders the comprehensive multi-algorithm comparison matrix."""

    def __init__(self, rect: Tuple[int, int, int, int]):
        self.rect = pygame.Rect(rect)
        self.show_live = False
        self.btn_benchmark = Button(
            (self.rect.x + 20, self.rect.y + 16, 220, 32),
            "100-Game Benchmark (Verified)",
            on_click=self._select_benchmark,
            is_active=True,
            font_size=12,
        )
        self.btn_live = Button(
            (self.rect.x + 250, self.rect.y + 16, 220, 32),
            "Live Session Games",
            on_click=self._select_live,
            is_active=False,
            font_size=12,
        )

    def _select_benchmark(self) -> None:
        self.show_live = False
        self.btn_benchmark.is_active = True
        self.btn_live.is_active = False

    def _select_live(self) -> None:
        self.show_live = True
        self.btn_benchmark.is_active = False
        self.btn_live.is_active = True

    def handle_event(self, event: pygame.event.Event) -> bool:
        if self.btn_benchmark.handle_event(event):
            return True
        if self.btn_live.handle_event(event):
            return True
        return False

    def draw(self, surface: pygame.Surface, live_history: Optional[Dict[str, List[int]]] = None) -> None:
        """Render the complete comparison screen."""
        pygame.draw.rect(surface, BG_DARK, self.rect)

        # Header controls
        self.btn_benchmark.draw(surface)
        self.btn_live.draw(surface)

        font_desc = get_font(12)
        desc_txt = (
            "Source: Phase 4 100-game independent benchmark (Seed 123, Greedy Policy)"
            if not self.show_live
            else "Source: Live session gameplay recorded in current runtime"
        )
        surface.blit(font_desc.render(desc_txt, True, TEXT_MUTED), (self.rect.x + 500, self.rect.y + 24))

        # Main Table Container
        table_rect = pygame.Rect(self.rect.x + 20, self.rect.y + 60, self.rect.width - 40, 520)
        draw_card(surface, table_rect, bg_color=BG_CARD)

        # Table Header Row
        th_y = table_rect.y + 12
        cols = [
            ("Variant", 180),
            ("Architecture", 150),
            ("Mean Score", 110),
            ("Max Score", 90),
            ("Median", 80),
            ("Avg Steps", 110),
            ("Latency", 90),
            ("Wall Collision %", 130),
            ("Key Verdict", 150),
        ]

        th_x = table_rect.x + 16
        font_th = get_font(12, bold=True)
        for cname, cw in cols:
            surface.blit(font_th.render(cname, True, ACCENT_CYAN), (th_x, th_y))
            th_x += cw

        pygame.draw.line(surface, BORDER_COLOR, (table_rect.x + 12, th_y + 26), (table_rect.right - 12, th_y + 26), width=1)

        # Table Data Rows
        row_y = th_y + 36
        font_row = get_font(12)
        font_num = get_font(12, bold=True, mono=True)
        font_badge = get_font(10, bold=True)

        for i, var in enumerate(PHASE4_BENCHMARK_DATA):
            rx = table_rect.x + 16
            row_rect = pygame.Rect(table_rect.x + 8, row_y - 4, table_rect.width - 16, 42)
            if i % 2 == 1:
                pygame.draw.rect(surface, (28, 30, 44), row_rect, border_radius=4)

            # Compute live or benchmark stats
            if self.show_live and live_history and var["id"] in live_history and live_history[var["id"]]:
                lscores = live_history[var["id"]]
                mean_s = f"{np.mean(lscores):.2f}"
                max_s = f"{np.max(lscores)}"
                med_s = f"{np.median(lscores):.1f}"
                steps_s = "Session"
            else:
                mean_s = f"{var['mean']:.2f} ± {var['std']:.1f}"
                max_s = str(var["max"])
                med_s = str(var["median"])
                steps_s = f"{var['steps']:,.1f}"

            # Col 1: Name
            name_surf = font_row.render(var["short_name"], True, TEXT_PRIMARY)
            surface.blit(name_surf, (rx, row_y + 6))
            rx += cols[0][1]

            # Col 2: Arch
            arch_surf = font_row.render(var["arch"], True, TEXT_SECONDARY)
            surface.blit(arch_surf, (rx, row_y + 6))
            rx += cols[1][1]

            # Col 3: Mean
            mean_surf = font_num.render(mean_s, True, ACCENT_GREEN)
            surface.blit(mean_surf, (rx, row_y + 6))
            rx += cols[2][1]

            # Col 4: Max
            max_surf = font_num.render(max_s, True, ACCENT_YELLOW)
            surface.blit(max_surf, (rx, row_y + 6))
            rx += cols[3][1]

            # Col 5: Median
            med_surf = font_num.render(med_s, True, TEXT_PRIMARY)
            surface.blit(med_surf, (rx, row_y + 6))
            rx += cols[4][1]

            # Col 6: Steps
            step_surf = font_num.render(steps_s, True, ACCENT_CYAN)
            surface.blit(step_surf, (rx, row_y + 6))
            rx += cols[5][1]

            # Col 7: Latency
            lat_surf = font_num.render(f"{var['latency_ms']:.3f} ms", True, TEXT_SECONDARY)
            surface.blit(lat_surf, (rx, row_y + 6))
            rx += cols[6][1]

            # Col 8: Wall Collision %
            wall_surf = font_num.render(f"{var['wall_pct']:.1f}%", True, ACCENT_RED if var['wall_pct'] > 50 else ACCENT_GREEN)
            surface.blit(wall_surf, (rx, row_y + 6))
            rx += cols[7][1]

            # Col 9: Badge
            badge_r = pygame.Rect(rx, row_y + 4, 120, 22)
            pygame.draw.rect(surface, (36, 40, 58), badge_r, border_radius=4)
            pygame.draw.rect(surface, var["badge_col"], badge_r, width=1, border_radius=4)
            b_txt = font_badge.render(var["badge"], True, var["badge_col"])
            surface.blit(b_txt, b_txt.get_rect(center=badge_r.center))

            row_y += 46

        # Bottom Visual Score Bar Comparison
        chart_rect = pygame.Rect(self.rect.x + 20, table_rect.bottom + 12, self.rect.width - 40, 80)
        draw_card(surface, chart_rect, title="Relative Mean Score Comparison", bg_color=BG_CARD)

        bar_x = chart_rect.x + 16
        bar_y = chart_rect.y + 36
        max_possible = 250.0

        for var in PHASE4_BENCHMARK_DATA:
            b_w = int((var["mean"] / max_possible) * (chart_rect.width - 180))
            br = pygame.Rect(bar_x + 95, bar_y, b_w, 6)
            # Label
            font_bl = get_font(10)
            surface.blit(font_bl.render(var["short_name"], True, TEXT_SECONDARY), (bar_x, bar_y - 3))
            pygame.draw.rect(surface, var["badge_col"], br, border_radius=3)
            # Score
            surface.blit(font_bl.render(f"{var['mean']:.1f}", True, TEXT_PRIMARY), (br.right + 8, bar_y - 3))
            bar_y += 14

