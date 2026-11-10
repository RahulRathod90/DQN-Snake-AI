"""
Reusable Pygame UI widgets for the DQN Snake AI Application:
- Button with hover and active states
- TabSelector for top navigation
- DropdownSelector / Modal for Agent selection
- SpeedControl button strip
- ProgressBar for free space and training progression
- LiveChart for dynamic in-GUI curve rendering
- Card containers for grouping telemetry sections
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pygame

from gui.theme import (
    ACCENT_CYAN,
    ACCENT_GREEN,
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
    BTN_PRIMARY_HOVER,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    get_font,
)


class Button:
    """Interactive clickable button with smooth hover and active state rendering."""

    def __init__(
        self,
        rect: Tuple[int, int, int, int],
        text: str,
        on_click: Optional[Callable[[], None]] = None,
        font_size: int = 14,
        is_active: bool = False,
        bg_color: Tuple[int, int, int] = BTN_PRIMARY_BG,
        hover_color: Tuple[int, int, int] = BTN_PRIMARY_HOVER,
        active_color: Tuple[int, int, int] = BTN_ACTIVE_BG,
        text_color: Tuple[int, int, int] = TEXT_PRIMARY,
        active_text_color: Tuple[int, int, int] = BTN_ACTIVE_TEXT,
        border_radius: int = 6,
    ):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.on_click = on_click
        self.font_size = font_size
        self.is_active = is_active
        self.is_hovered = False
        self.is_disabled = False
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.active_color = active_color
        self.text_color = text_color
        self.active_text_color = active_text_color
        self.border_radius = border_radius

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Process mouse movement and click events. Returns True if button was clicked."""
        if self.is_disabled:
            self.is_hovered = False
            return False

        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                if self.on_click:
                    self.on_click()
                return True
        return False

    def draw(self, surface: pygame.Surface) -> None:
        """Render the button onto the target surface."""
        if self.is_disabled:
            bg = (40, 42, 54)
            tx_col = TEXT_MUTED
            border_col = BORDER_COLOR
        elif self.is_active:
            bg = self.active_color
            tx_col = self.active_text_color
            border_col = BORDER_HIGHLIGHT
        elif self.is_hovered:
            bg = self.hover_color
            tx_col = TEXT_PRIMARY
            border_col = ACCENT_CYAN
        else:
            bg = self.bg_color
            tx_col = self.text_color
            border_col = BORDER_COLOR

        pygame.draw.rect(surface, bg, self.rect, border_radius=self.border_radius)
        pygame.draw.rect(surface, border_col, self.rect, width=1, border_radius=self.border_radius)

        font = get_font(self.font_size, bold=(self.is_active or self.is_hovered))
        label_surf = font.render(self.text, True, tx_col)
        label_rect = label_surf.get_rect(center=self.rect.center)
        surface.blit(label_surf, label_rect)


class TabSelector:
    """Header tab navigation strip."""

    def __init__(
        self,
        rect: Tuple[int, int, int, int],
        tabs: List[str],
        active_index: int = 0,
        on_change: Optional[Callable[[int, str], None]] = None,
    ):
        self.rect = pygame.Rect(rect)
        self.tabs = tabs
        self.active_index = active_index
        self.on_change = on_change
        self.hovered_index = -1
        self._calculate_tab_rects()

    def _calculate_tab_rects(self) -> None:
        count = len(self.tabs)
        tab_w = self.rect.width // max(1, count)
        self.tab_rects = [
            pygame.Rect(self.rect.x + i * tab_w, self.rect.y, tab_w, self.rect.height)
            for i in range(count)
        ]

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.hovered_index = -1
            for i, r in enumerate(self.tab_rects):
                if r.collidepoint(event.pos):
                    self.hovered_index = i
                    break
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for i, r in enumerate(self.tab_rects):
                if r.collidepoint(event.pos):
                    if self.active_index != i:
                        self.active_index = i
                        if self.on_change:
                            self.on_change(i, self.tabs[i])
                        return True
        return False

    def draw(self, surface: pygame.Surface) -> None:
        for i, (tab_name, r) in enumerate(zip(self.tabs, self.tab_rects)):
            is_active = (i == self.active_index)
            is_hover = (i == self.hovered_index and not is_active)

            bg = BG_SURFACE if not is_active else BG_CARD
            pygame.draw.rect(surface, bg, r)

            if is_active:
                # Active top border accent line
                pygame.draw.line(surface, ACCENT_GREEN, (r.left + 8, r.bottom - 2), (r.right - 8, r.bottom - 2), width=3)
                tx_col = ACCENT_GREEN
            elif is_hover:
                tx_col = TEXT_PRIMARY
            else:
                tx_col = TEXT_MUTED

            font = get_font(14, bold=is_active)
            surf = font.render(tab_name, True, tx_col)
            surface.blit(surf, surf.get_rect(center=r.center))


class DropdownSelector:
    """Clean popup dropdown for selecting agents or configurations."""

    def __init__(
        self,
        rect: Tuple[int, int, int, int],
        options: List[Dict[str, str]],
        selected_key: str,
        on_select: Optional[Callable[[Dict[str, str]], None]] = None,
        font_size: int = 13,
    ):
        self.rect = pygame.Rect(rect)
        self.options = options
        self.selected_key = selected_key
        self.on_select = on_select
        self.font_size = font_size
        self.is_open = False
        self.hovered_idx = -1

    @property
    def selected_option(self) -> Dict[str, str]:
        for opt in self.options:
            if opt["key"] == self.selected_key:
                return opt
        return self.options[0] if self.options else {"key": "", "name": "None"}

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.is_open = not self.is_open
                return True
            elif self.is_open:
                # Check option clicks
                opt_h = self.rect.height
                for i, opt in enumerate(self.options):
                    r = pygame.Rect(self.rect.x, self.rect.bottom + i * opt_h, self.rect.width, opt_h)
                    if r.collidepoint(event.pos):
                        self.selected_key = opt["key"]
                        self.is_open = False
                        if self.on_select:
                            self.on_select(opt)
                        return True
                self.is_open = False
        elif event.type == pygame.MOUSEMOTION and self.is_open:
            self.hovered_idx = -1
            opt_h = self.rect.height
            for i, _ in enumerate(self.options):
                r = pygame.Rect(self.rect.x, self.rect.bottom + i * opt_h, self.rect.width, opt_h)
                if r.collidepoint(event.pos):
                    self.hovered_idx = i
                    break
        return False

    def draw(self, surface: pygame.Surface) -> None:
        # Draw main box
        pygame.draw.rect(surface, BG_CARD, self.rect, border_radius=6)
        border_col = ACCENT_CYAN if self.is_open else BORDER_COLOR
        pygame.draw.rect(surface, border_col, self.rect, width=1, border_radius=6)

        font = get_font(self.font_size, bold=True)
        txt = font.render(self.selected_option.get("name", ""), True, TEXT_PRIMARY)
        surface.blit(txt, (self.rect.x + 10, self.rect.centery - txt.get_height() // 2))

        # Arrow indicator
        arrow = "▲" if self.is_open else "▼"
        arr_surf = font.render(arrow, True, ACCENT_CYAN)
        surface.blit(arr_surf, (self.rect.right - 20, self.rect.centery - arr_surf.get_height() // 2))

    def draw_menu(self, surface: pygame.Surface) -> None:
        """Draw overlay dropdown options if opened."""
        if not self.is_open:
            return

        opt_h = self.rect.height
        menu_h = opt_h * len(self.options)
        menu_rect = pygame.Rect(self.rect.x, self.rect.bottom, self.rect.width, menu_h)

        # Drop shadow / background
        pygame.draw.rect(surface, BG_DARK, menu_rect.inflate(2, 2), border_radius=6)
        pygame.draw.rect(surface, BG_SURFACE, menu_rect, border_radius=6)
        pygame.draw.rect(surface, BORDER_HIGHLIGHT, menu_rect, width=1, border_radius=6)

        font = get_font(self.font_size)
        for i, opt in enumerate(self.options):
            r = pygame.Rect(self.rect.x, self.rect.bottom + i * opt_h, self.rect.width, opt_h)
            is_sel = (opt["key"] == self.selected_key)
            is_hov = (i == self.hovered_idx)

            if is_sel:
                pygame.draw.rect(surface, (40, 60, 50), r)
                tx_col = ACCENT_GREEN
            elif is_hov:
                pygame.draw.rect(surface, BG_CARD_HOVER, r)
                tx_col = TEXT_PRIMARY
            else:
                tx_col = TEXT_SECONDARY

            txt = font.render(opt.get("name", ""), True, tx_col)
            surface.blit(txt, (r.x + 12, r.centery - txt.get_height() // 2))


class ProgressBar:
    """Smooth visual progress bar."""

    def __init__(
        self,
        rect: Tuple[int, int, int, int],
        value: float = 0.0,
        fill_color: Tuple[int, int, int] = ACCENT_GREEN,
        bg_color: Tuple[int, int, int] = (20, 22, 32),
        border_radius: int = 4,
    ):
        self.rect = pygame.Rect(rect)
        self.value = max(0.0, min(1.0, value))
        self.fill_color = fill_color
        self.bg_color = bg_color
        self.border_radius = border_radius

    def draw(self, surface: pygame.Surface, value: Optional[float] = None) -> None:
        if value is not None:
            self.value = max(0.0, min(1.0, value))

        pygame.draw.rect(surface, self.bg_color, self.rect, border_radius=self.border_radius)
        if self.value > 0.001:
            w = int(self.rect.width * self.value)
            fill_r = pygame.Rect(self.rect.x, self.rect.y, w, self.rect.height)
            pygame.draw.rect(surface, self.fill_color, fill_r, border_radius=self.border_radius)
        pygame.draw.rect(surface, BORDER_COLOR, self.rect, width=1, border_radius=self.border_radius)


class LiveChart:
    """In-GUI multi-series line plot for live score and training loss visualization."""

    def __init__(
        self,
        rect: Tuple[int, int, int, int],
        title: str = "Training Progress",
        x_label: str = "Episode",
        y_label: str = "Score",
    ):
        self.rect = pygame.Rect(rect)
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.series: Dict[str, Dict[str, Any]] = {}

    def add_series(self, name: str, color: Tuple[int, int, int], width: int = 2) -> None:
        self.series[name] = {"color": color, "width": width, "data": []}

    def update_data(self, name: str, data: List[float]) -> None:
        if name in self.series:
            self.series[name]["data"] = list(data)

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, BG_CARD, self.rect, border_radius=8)
        pygame.draw.rect(surface, BORDER_COLOR, self.rect, width=1, border_radius=8)

        # Title
        font_title = get_font(13, bold=True)
        t_surf = font_title.render(self.title, True, TEXT_PRIMARY)
        surface.blit(t_surf, (self.rect.x + 12, self.rect.y + 10))

        # Chart plotting area
        pad_l = 45
        pad_r = 15
        pad_t = 34
        pad_b = 30
        plot_rect = pygame.Rect(
            self.rect.x + pad_l,
            self.rect.y + pad_t,
            self.rect.width - pad_l - pad_r,
            self.rect.height - pad_t - pad_b,
        )

        # Plot background & grid
        pygame.draw.rect(surface, (20, 22, 34), plot_rect)
        pygame.draw.rect(surface, BORDER_COLOR, plot_rect, width=1)

        # Determine min and max
        all_vals = []
        max_len = 0
        for s in self.series.values():
            d = s["data"]
            if d:
                all_vals.extend(d)
                max_len = max(max_len, len(d))

        if not all_vals or max_len < 2:
            font_msg = get_font(12)
            msg = font_msg.render("Waiting for data points...", True, TEXT_MUTED)
            surface.blit(msg, msg.get_rect(center=plot_rect.center))
            return

        min_y = min(0.0, float(np.min(all_vals)))
        max_y = max(10.0, float(np.max(all_vals)) * 1.1)
        y_range = max(1.0, max_y - min_y)

        # Horizontal grid lines (4 lines)
        font_axis = get_font(10, mono=True)
        for i in range(5):
            y_val = min_y + (y_range / 4.0) * i
            py = int(plot_rect.bottom - (y_val - min_y) / y_range * plot_rect.height)
            pygame.draw.line(surface, (36, 38, 56), (plot_rect.left, py), (plot_rect.right, py), width=1)
            lbl = font_axis.render(f"{y_val:.0f}", True, TEXT_MUTED)
            surface.blit(lbl, (self.rect.x + 6, py - 6))

        # Draw series
        for name, s in self.series.items():
            data = s["data"]
            if len(data) < 2:
                continue

            pts = []
            for idx, val in enumerate(data):
                px = int(plot_rect.left + (idx / (max_len - 1)) * plot_rect.width)
                py = int(plot_rect.bottom - (val - min_y) / y_range * plot_rect.height)
                py = max(plot_rect.top, min(plot_rect.bottom, py))
                pts.append((px, py))

            if len(pts) >= 2:
                pygame.draw.lines(surface, s["color"], False, pts, width=s["width"])

        # Legend
        leg_x = self.rect.right - 15
        for name, s in reversed(list(self.series.items())):
            font_leg = get_font(11, bold=True)
            txt = font_leg.render(name, True, s["color"])
            leg_x -= (txt.get_width() + 14)
            surface.blit(txt, (leg_x, self.rect.y + 10))
            pygame.draw.rect(surface, s["color"], (leg_x - 8, self.rect.y + 15, 6, 6))


def draw_card(
    surface: pygame.Surface,
    rect: Tuple[int, int, int, int],
    title: Optional[str] = None,
    border_color: Tuple[int, int, int] = BORDER_COLOR,
    bg_color: Tuple[int, int, int] = BG_CARD,
    border_radius: int = 8,
) -> pygame.Rect:
    """Helper to draw styled card container with optional header."""
    r = pygame.Rect(rect)
    pygame.draw.rect(surface, bg_color, r, border_radius=border_radius)
    pygame.draw.rect(surface, border_color, r, width=1, border_radius=border_radius)

    if title:
        font = get_font(13, bold=True)
        lbl = font.render(title, True, ACCENT_CYAN)
        surface.blit(lbl, (r.x + 12, r.y + 10))
        pygame.draw.line(surface, BORDER_COLOR, (r.x + 8, r.y + 30), (r.right - 8, r.y + 30), width=1)
    return r

