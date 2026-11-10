"""
AI Intelligence Dashboard & Decision Telemetry Panel.
Provides live inspection of neural network Q-values, action distributions,
spatial risk matrix, flood-fill accessible space, and safety shield interventions.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pygame

from gui.theme import (
    ACCENT_CYAN,
    ACCENT_GREEN,
    ACCENT_ORANGE,
    ACCENT_PURPLE,
    ACCENT_RED,
    ACCENT_YELLOW,
    BG_CARD,
    BG_SURFACE,
    BORDER_COLOR,
    BORDER_HIGHLIGHT,
    TEXT_DARK,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    get_font,
)
from gui.widgets import ProgressBar, draw_card


class TelemetryPanel:
    """Renders the right-hand real-time AI telemetry and intelligence dashboard."""

    def __init__(self, rect: Tuple[int, int, int, int]):
        self.rect = pygame.Rect(rect)
        self.space_bar = ProgressBar(
            (self.rect.x + 16, self.rect.y + 242, self.rect.width - 32, 12),
            value=1.0,
            fill_color=ACCENT_GREEN,
        )

    def draw(
        self,
        surface: pygame.Surface,
        agent_info: Dict[str, Any],
        score: int,
        record: int,
        food_eaten: int,
        steps: int,
        direction: str,
        collision_risk: Dict[str, bool],
        food_dist: int,
        free_space_ratio: float,
        selected_action: int,
        q_values: Optional[List[float]] = None,
        shield_telemetry: Optional[Dict[str, Any]] = None,
        astar_info: Optional[Dict[str, Any]] = None,
        is_human: bool = False,
    ) -> None:
        """Render all telemetry components onto the surface."""
        # Main sidebar backdrop
        pygame.draw.rect(surface, BG_SURFACE, self.rect, border_radius=8)
        pygame.draw.rect(surface, BORDER_COLOR, self.rect, width=1, border_radius=8)

        # ----------------------------------------------------------------------
        # 1. Agent Architecture & Status Card (Top)
        # ----------------------------------------------------------------------
        card1_rect = pygame.Rect(self.rect.x + 12, self.rect.y + 12, self.rect.width - 24, 76)
        draw_card(surface, card1_rect, bg_color=BG_CARD)

        font_name = get_font(14, bold=True)
        agent_title = "Human Player" if is_human else agent_info.get("name", "Unknown Agent")
        title_surf = font_name.render(agent_title, True, ACCENT_GREEN if not is_human else ACCENT_CYAN)
        surface.blit(title_surf, (card1_rect.x + 12, card1_rect.y + 10))

        font_meta = get_font(11)
        if is_human:
            meta_str = "Control: Keyboard [Arrows / WASD]  |  Mode: Manual"
        else:
            arch = agent_info.get("architecture", "standard").upper()
            dim = agent_info.get("state_dim", 11)
            ckpt = agent_info.get("checkpoint_status", "Validated")
            meta_str = f"Arch: {arch}  |  State: {dim}-D  |  Weights: {ckpt}"
        surface.blit(font_meta.render(meta_str, True, TEXT_MUTED), (card1_rect.x + 12, card1_rect.y + 34))

        # Status badge
        badge_rect = pygame.Rect(card1_rect.right - 95, card1_rect.y + 10, 85, 20)
        badge_bg = (40, 60, 50) if not is_human else (40, 55, 70)
        pygame.draw.rect(surface, badge_bg, badge_rect, border_radius=4)
        badge_lbl = "ACTIVE AI" if not is_human else "HUMAN"
        font_badge = get_font(10, bold=True)
        b_surf = font_badge.render(badge_lbl, True, ACCENT_GREEN if not is_human else ACCENT_CYAN)
        surface.blit(b_surf, b_surf.get_rect(center=badge_rect.center))

        # ----------------------------------------------------------------------
        # 2. Live Game Metrics Grid
        # ----------------------------------------------------------------------
        card2_rect = pygame.Rect(self.rect.x + 12, card1_rect.bottom + 10, self.rect.width - 24, 78)
        draw_card(surface, card2_rect, bg_color=BG_CARD)

        col_w = card2_rect.width // 4
        metrics = [
            ("SCORE", str(score), ACCENT_GREEN),
            ("RECORD", str(record), ACCENT_YELLOW),
            ("FOOD", str(food_eaten), ACCENT_RED),
            ("STEPS", f"{steps:,}", ACCENT_CYAN),
        ]

        font_val = get_font(18, bold=True, mono=True)
        font_sub = get_font(10, bold=True)

        for i, (label, val, col) in enumerate(metrics):
            cx = card2_rect.x + i * col_w + col_w // 2
            # Label
            lbl_surf = font_sub.render(label, True, TEXT_MUTED)
            surface.blit(lbl_surf, lbl_surf.get_rect(center=(cx, card2_rect.y + 24)))
            # Value
            val_surf = font_val.render(val, True, col)
            surface.blit(val_surf, val_surf.get_rect(center=(cx, card2_rect.y + 50)))

            if i < 3:
                pygame.draw.line(
                    surface,
                    BORDER_COLOR,
                    (card2_rect.x + (i + 1) * col_w, card2_rect.y + 12),
                    (card2_rect.x + (i + 1) * col_w, card2_rect.bottom - 12),
                    width=1,
                )

        # ----------------------------------------------------------------------
        # 3. Spatial Perception & Collision Risk
        # ----------------------------------------------------------------------
        card3_rect = pygame.Rect(self.rect.x + 12, card2_rect.bottom + 10, self.rect.width - 24, 94)
        draw_card(surface, card3_rect, title="Spatial Perception & Safety Matrix", bg_color=BG_CARD)

        font_item = get_font(11, bold=True)
        # Collision chips: Front, Right, Left
        chip_y = card3_rect.y + 36
        chip_labels = [
            ("Front", collision_risk.get("straight", False)),
            ("Right", collision_risk.get("right", False)),
            ("Left", collision_risk.get("left", False)),
        ]

        chip_x = card3_rect.x + 12
        for name, is_danger in chip_labels:
            bg_col = (70, 25, 30) if is_danger else (30, 50, 40)
            txt_col = ACCENT_RED if is_danger else ACCENT_GREEN
            txt_stat = "DANGER" if is_danger else "SAFE"
            chip_r = pygame.Rect(chip_x, chip_y, 90, 22)
            pygame.draw.rect(surface, bg_col, chip_r, border_radius=4)
            pygame.draw.rect(surface, txt_col, chip_r, width=1, border_radius=4)

            chip_txt = font_item.render(f"{name}: {txt_stat}", True, txt_col)
            surface.blit(chip_txt, chip_txt.get_rect(center=chip_r.center))
            chip_x += 96

        # Food Distance & Heading
        font_info = get_font(11)
        dist_str = f"Food Distance: {food_dist} blocks  |  Heading: {direction}"
        surface.blit(font_info.render(dist_str, True, TEXT_SECONDARY), (card3_rect.x + 12, card3_rect.y + 68))

        # Free space ratio
        pct = max(0.0, min(1.0, free_space_ratio))
        pct_col = ACCENT_GREEN if pct > 0.40 else (ACCENT_YELLOW if pct > 0.20 else ACCENT_RED)
        space_txt = font_info.render(f"Reachable Open Space: {pct * 100:.1f}%", True, pct_col)
        surface.blit(space_txt, (card3_rect.right - space_txt.get_width() - 12, card3_rect.y + 68))

        # ----------------------------------------------------------------------
        # 4. Action Decision Engine / Q-Values
        # ----------------------------------------------------------------------
        card4_rect = pygame.Rect(self.rect.x + 12, card3_rect.bottom + 10, self.rect.width - 24, 116)

        if q_values and len(q_values) == 3 and not is_human:
            draw_card(surface, card4_rect, title="DQN Q-Values & Policy Distribution", bg_color=BG_CARD)
            # Softmax action probabilities
            q_arr = np.array(q_values, dtype=np.float32)
            exp_q = np.exp(q_arr - np.max(q_arr))
            probs = exp_q / np.sum(exp_q)

            action_names = ["Straight", "Turn Right", "Turn Left"]
            bar_y = card4_rect.y + 36
            font_bar = get_font(11, mono=True)

            for act_i, (aname, q_val, prob) in enumerate(zip(action_names, q_values, probs)):
                is_chosen = (act_i == selected_action)
                txt_col = ACCENT_GREEN if is_chosen else TEXT_SECONDARY

                # Action label
                lbl = font_bar.render(f"{aname:<10}: Q={q_val:+.2f} ({prob * 100:4.1f}%)", True, txt_col)
                surface.blit(lbl, (card4_rect.x + 14, bar_y + act_i * 24))

                # Bar graphic
                bar_r = pygame.Rect(card4_rect.x + 220, bar_y + act_i * 24 + 4, 180, 10)
                pygame.draw.rect(surface, (20, 22, 34), bar_r, border_radius=3)
                fill_w = int(180 * prob)
                fill_col = ACCENT_GREEN if is_chosen else (ACCENT_CYAN if prob > 0.3 else TEXT_MUTED)
                if fill_w > 0:
                    pygame.draw.rect(surface, fill_col, (bar_r.x, bar_r.y, fill_w, bar_r.height), border_radius=3)
                pygame.draw.rect(surface, BORDER_COLOR, bar_r, width=1, border_radius=3)

                if is_chosen:
                    # Arrow indicator
                    arr = font_bar.render("◄ CHOSEN", True, ACCENT_GREEN)
                    surface.blit(arr, (bar_r.right + 8, bar_y + act_i * 24))

        elif astar_info:
            draw_card(surface, card4_rect, title="A* Heuristic & Route Planning", bg_color=BG_CARD)
            font_as = get_font(12)
            route_len = astar_info.get("route_length", 0)
            target = astar_info.get("target_type", "Food")
            tail_safe = astar_info.get("tail_reachable", True)

            t_col = ACCENT_GREEN if tail_safe else ACCENT_RED
            tail_str = "YES (Route to tail is safe)" if tail_safe else "NO (Tail unreachable - risk!)"

            surface.blit(font_as.render(f"Planned Target : {target}", True, TEXT_PRIMARY), (card4_rect.x + 16, card4_rect.y + 38))
            surface.blit(font_as.render(f"Route Distance : {route_len} grid steps", True, ACCENT_CYAN), (card4_rect.x + 16, card4_rect.y + 60))
            surface.blit(font_as.render(f"Tail Accessible: {tail_str}", True, t_col), (card4_rect.x + 16, card4_rect.y + 82))

        else:
            draw_card(surface, card4_rect, title="User Input & Movement Control", bg_color=BG_CARD)
            font_hm = get_font(12)
            surface.blit(font_hm.render("Mode: Direct Human Keyboard Control", True, TEXT_PRIMARY), (card4_rect.x + 16, card4_rect.y + 42))
            surface.blit(font_hm.render("Controls: [Arrow Keys] or [W, A, S, D] to Steer", True, ACCENT_CYAN), (card4_rect.x + 16, card4_rect.y + 66))
            surface.blit(font_hm.render("Press [Space] to pause at any time.", True, TEXT_MUTED), (card4_rect.x + 16, card4_rect.y + 88))

        # ----------------------------------------------------------------------
        # 5. Safety Shield Telemetry Card (Bottom)
        # ----------------------------------------------------------------------
        card5_rect = pygame.Rect(self.rect.x + 12, card4_rect.bottom + 10, self.rect.width - 24, 78)
        draw_card(surface, card5_rect, title="A* Safety Shield Guardian", bg_color=BG_CARD)

        font_sh = get_font(11)
        if shield_telemetry and shield_telemetry.get("enabled", True):
            is_overridden = shield_telemetry.get("was_overridden", False)
            reason = shield_telemetry.get("rejection_reason", "None")
            fallback = shield_telemetry.get("fallback_source", "None")
            interventions = shield_telemetry.get("intervention_count", 0)

            if is_overridden:
                stat_col = ACCENT_RED
                stat_text = f"OVERRIDDEN ({reason})"
            else:
                stat_col = ACCENT_GREEN
                stat_text = "APPROVED (DQN action safe)"

            surface.blit(font_sh.render(f"Shield Status: {stat_text}", True, stat_col), (card5_rect.x + 14, card5_rect.y + 34))
            surface.blit(
                font_sh.render(f"Fallback Mode: {fallback}  |  Session Overrides: {interventions}", True, TEXT_SECONDARY),
                (card5_rect.x + 14, card5_rect.y + 54),
            )
        else:
            status_txt = "STANDBY (Shield only active for Hybrid Mode)"
            surface.blit(font_sh.render(status_txt, True, TEXT_MUTED), (card5_rect.x + 14, card5_rect.y + 42))

