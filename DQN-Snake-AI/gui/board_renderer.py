"""
High-Fidelity Game Board Renderer for the Snake AI Arena.
Renders smooth snake segments, direction-sensitive eyes, glowing food,
planned A* path overlays, danger indicators, and game-over/pause screens.
"""

import math
from typing import List, Optional, Set, Tuple
import pygame

from gui.theme import (
    ACCENT_CYAN,
    ACCENT_GREEN,
    ACCENT_PURPLE,
    ACCENT_RED,
    ACCENT_YELLOW,
    BG_CARD,
    BORDER_COLOR,
    FOOD_COLOR,
    FOOD_GLOW,
    GRID_BG,
    GRID_LINE,
    PATH_COLOR,
    SNAKE_BODY,
    SNAKE_BODY_TAIL,
    SNAKE_EYE,
    SNAKE_HEAD,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    get_font,
)


class BoardRenderer:
    """Renders the Snake game grid, entities, overlays, and game state banners."""

    def __init__(self, rect: Tuple[int, int, int, int], block_size: int = 20):
        self.rect = pygame.Rect(rect)
        self.block_size = block_size
        self.surface = pygame.Surface((self.rect.width, self.rect.height))
        self.pulse_phase = 0.0

    def update_animation(self, dt: float = 0.05) -> None:
        """Update continuous visual animations (food pulse, glows)."""
        self.pulse_phase = (self.pulse_phase + dt) % (2 * math.pi)

    def render(
        self,
        snake: List[Tuple[int, int]],
        food: Tuple[int, int],
        direction: str,
        score: int,
        record: int,
        is_paused: bool = False,
        is_game_over: bool = False,
        game_over_reason: str = "",
        astar_path: Optional[List[Tuple[int, int]]] = None,
        danger_cells: Optional[Set[Tuple[int, int]]] = None,
        show_path: bool = True,
        is_human: bool = False,
    ) -> pygame.Surface:
        """Draw the complete game board onto internal surface and return it."""
        # 1. Background and subtle grid
        self.surface.fill(GRID_BG)
        bx = self.block_size

        for x in range(0, self.rect.width, bx):
            pygame.draw.line(self.surface, GRID_LINE, (x, 0), (x, self.rect.height), width=1)
        for y in range(0, self.rect.height, bx):
            pygame.draw.line(self.surface, GRID_LINE, (0, y), (self.rect.width, y), width=1)

        # 2. Render danger cells if provided
        if danger_cells:
            for cell in danger_cells:
                r = pygame.Rect(cell[0], cell[1], bx, bx)
                pygame.draw.rect(self.surface, (70, 20, 30), r)
                pygame.draw.rect(self.surface, ACCENT_RED, r, width=1)

        # 3. Render A* planned path overlay
        if show_path and astar_path and len(astar_path) >= 2:
            pts = [(p[0] + bx // 2, p[1] + bx // 2) for p in astar_path]
            # Draw translucent connected line
            path_surf = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
            pygame.draw.lines(path_surf, (*PATH_COLOR, 130), False, pts, width=3)
            for pt in pts[1:-1]:
                pygame.draw.circle(path_surf, (*PATH_COLOR, 180), pt, 4)
            self.surface.blit(path_surf, (0, 0))

        # 4. Render glowing food item
        fx, fy = food
        pulse_scale = 1.0 + 0.12 * math.sin(self.pulse_phase * 3.0)
        radius = int((bx // 2 - 2) * pulse_scale)
        center = (fx + bx // 2, fy + bx // 2)

        # Outer soft glow
        glow_surf = pygame.Surface((bx * 2, bx * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*FOOD_GLOW, 60), (bx, bx), radius + 4)
        self.surface.blit(glow_surf, (center[0] - bx, center[1] - bx))

        # Core ruby fruit
        pygame.draw.circle(self.surface, FOOD_COLOR, center, radius)
        # Specular shine
        pygame.draw.circle(self.surface, (255, 220, 230), (center[0] - radius // 3, center[1] - radius // 3), max(2, radius // 4))

        # 5. Render Snake Body (gradient from head to tail)
        snake_len = len(snake)
        for i in range(snake_len - 1, 0, -1):
            seg = snake[i]
            t = i / max(1, snake_len - 1)
            # Interpolate between SNAKE_BODY and SNAKE_BODY_TAIL
            col = (
                int(SNAKE_BODY[0] * (1 - t) + SNAKE_BODY_TAIL[0] * t),
                int(SNAKE_BODY[1] * (1 - t) + SNAKE_BODY_TAIL[1] * t),
                int(SNAKE_BODY[2] * (1 - t) + SNAKE_BODY_TAIL[2] * t),
            )
            # Body segment with rounded corners
            seg_rect = pygame.Rect(seg[0] + 1, seg[1] + 1, bx - 2, bx - 2)
            pygame.draw.rect(self.surface, col, seg_rect, border_radius=5)

            # Connector to previous segment for seamless continuous body
            prev_seg = snake[i - 1]
            conn_x = min(seg[0], prev_seg[0]) + 1
            conn_y = min(seg[1], prev_seg[1]) + 1
            conn_w = abs(seg[0] - prev_seg[0]) + bx - 2
            conn_h = abs(seg[1] - prev_seg[1]) + bx - 2
            pygame.draw.rect(self.surface, col, (conn_x, conn_y, conn_w, conn_h), border_radius=4)

        # 6. Render Snake Head
        if snake:
            hx, hy = snake[0]
            head_rect = pygame.Rect(hx + 1, hy + 1, bx - 2, bx - 2)
            pygame.draw.rect(self.surface, SNAKE_HEAD, head_rect, border_radius=6)

            # Draw Direction-sensitive eyes
            eye_radius = max(2, bx // 7)
            pupil_radius = max(1, eye_radius // 2)

            if direction == "UP":
                eye_l = (hx + bx // 4, hy + bx // 4)
                eye_r = (hx + 3 * bx // 4, hy + bx // 4)
                pupil_off = (0, -1)
            elif direction == "DOWN":
                eye_l = (hx + bx // 4, hy + 3 * bx // 4)
                eye_r = (hx + 3 * bx // 4, hy + 3 * bx // 4)
                pupil_off = (0, 1)
            elif direction == "LEFT":
                eye_l = (hx + bx // 4, hy + bx // 4)
                eye_r = (hx + bx // 4, hy + 3 * bx // 4)
                pupil_off = (-1, 0)
            else:  # RIGHT
                eye_l = (hx + 3 * bx // 4, hy + bx // 4)
                eye_r = (hx + 3 * bx // 4, hy + 3 * bx // 4)
                pupil_off = (1, 0)

            # Eye whites
            pygame.draw.circle(self.surface, (255, 255, 255), eye_l, eye_radius)
            pygame.draw.circle(self.surface, (255, 255, 255), eye_r, eye_radius)
            # Pupils
            pygame.draw.circle(self.surface, SNAKE_EYE, (eye_l[0] + pupil_off[0], eye_l[1] + pupil_off[1]), pupil_radius)
            pygame.draw.circle(self.surface, SNAKE_EYE, (eye_r[0] + pupil_off[0], eye_r[1] + pupil_off[1]), pupil_radius)

        # 7. Render Board Border
        pygame.draw.rect(self.surface, BORDER_COLOR, (0, 0, self.rect.width, self.rect.height), width=2)

        # 8. Pause Overlay
        if is_paused and not is_game_over:
            overlay = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
            overlay.fill((14, 15, 23, 160))
            self.surface.blit(overlay, (0, 0))

            font_pause = get_font(28, bold=True)
            txt = font_pause.render("GAME PAUSED", True, ACCENT_YELLOW)
            self.surface.blit(txt, txt.get_rect(center=(self.rect.width // 2, self.rect.height // 2 - 15)))

            font_sub = get_font(14)
            sub = font_sub.render("Press [Space] or click Resume to continue", True, TEXT_SECONDARY)
            self.surface.blit(sub, sub.get_rect(center=(self.rect.width // 2, self.rect.height // 2 + 20)))

        # 9. Game Over Overlay
        if is_game_over:
            overlay = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
            overlay.fill((14, 15, 23, 190))
            self.surface.blit(overlay, (0, 0))

            card_w, card_h = 360, 220
            card_rect = pygame.Rect(
                (self.rect.width - card_w) // 2,
                (self.rect.height - card_h) // 2,
                card_w,
                card_h,
            )
            pygame.draw.rect(self.surface, BG_CARD, card_rect, border_radius=10)
            pygame.draw.rect(self.surface, ACCENT_RED, card_rect, width=2, border_radius=10)

            font_go = get_font(24, bold=True)
            t_go = font_go.render("GAME OVER", True, ACCENT_RED)
            self.surface.blit(t_go, t_go.get_rect(center=(self.rect.width // 2, card_rect.y + 35)))

            # Termination reason
            font_reason = get_font(13)
            reason_str = game_over_reason or "Collision with obstacle"
            t_reason = font_reason.render(f"Cause: {reason_str}", True, ACCENT_YELLOW)
            self.surface.blit(t_reason, t_reason.get_rect(center=(self.rect.width // 2, card_rect.y + 70)))

            # Score Summary
            font_stats = get_font(15, bold=True)
            t_score = font_stats.render(f"Final Score: {score}   |   Best: {record}", True, TEXT_PRIMARY)
            self.surface.blit(t_score, t_score.get_rect(center=(self.rect.width // 2, card_rect.y + 110)))

            # Restart instructions
            font_prompt = get_font(13)
            prompt = font_prompt.render("Press [R] or Click Restart to Play Again", True, ACCENT_GREEN)
            self.surface.blit(prompt, prompt.get_rect(center=(self.rect.width // 2, card_rect.y + 160)))

        return self.surface

