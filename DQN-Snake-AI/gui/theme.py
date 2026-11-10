"""
Theme, Color Palettes, Fonts, and Layout Geometry for the Snake AI GUI Application.
Modern Dark Cyberpunk / Catppuccin-inspired professional theme.
"""

from typing import Dict, Tuple
import pygame

# ==============================================================================
# WINDOW & LAYOUT GEOMETRY
# ==============================================================================
WINDOW_WIDTH = 1180
WINDOW_HEIGHT = 760

HEADER_HEIGHT = 52
FOOTER_HEIGHT = 68

BOARD_X = 24
BOARD_Y = HEADER_HEIGHT + 16
BOARD_WIDTH = 640
BOARD_HEIGHT = 480

SIDEBAR_X = BOARD_X + BOARD_WIDTH + 20
SIDEBAR_Y = HEADER_HEIGHT + 16
SIDEBAR_WIDTH = WINDOW_WIDTH - SIDEBAR_X - 24
SIDEBAR_HEIGHT = BOARD_HEIGHT

TAB_ARENA = "Game Arena"
TAB_COMPARISON = "Algorithm Comparison"
TAB_TRAINING = "Training Studio"
TAB_ANALYTICS = "Results & Analytics"

# ==============================================================================
# COLOR PALETTE (RGB)
# ==============================================================================
# Base backgrounds
BG_DARK = (18, 18, 28)
BG_SURFACE = (26, 27, 40)
BG_CARD = (33, 35, 52)
BG_CARD_HOVER = (42, 45, 66)
BORDER_COLOR = (54, 58, 84)
BORDER_HIGHLIGHT = (80, 250, 123)

# Grid & Game Elements
GRID_BG = (14, 15, 23)
GRID_LINE = (28, 30, 44)
SNAKE_HEAD = (80, 250, 123)       # Vibrant Neon Green
SNAKE_BODY = (46, 204, 113)       # Emerald Green
SNAKE_BODY_TAIL = (30, 140, 75)   # Forest Green
SNAKE_EYE = (18, 18, 28)          # Pupil
FOOD_COLOR = (255, 85, 85)        # Ruby Red
FOOD_GLOW = (255, 121, 198)       # Magenta glow
PATH_COLOR = (139, 233, 253)      # Electric Cyan

# UI Accent & Status Colors
ACCENT_CYAN = (139, 233, 253)
ACCENT_GREEN = (80, 250, 123)
ACCENT_PURPLE = (189, 147, 249)
ACCENT_YELLOW = (241, 250, 140)
ACCENT_ORANGE = (255, 184, 108)
ACCENT_RED = (255, 85, 85)

# Text Colors
TEXT_PRIMARY = (248, 248, 242)
TEXT_SECONDARY = (180, 185, 210)
TEXT_MUTED = (120, 126, 155)
TEXT_DARK = (18, 18, 28)

# Buttons
BTN_PRIMARY_BG = (68, 71, 90)
BTN_PRIMARY_HOVER = (98, 114, 164)
BTN_ACTIVE_BG = (80, 250, 123)
BTN_ACTIVE_TEXT = (18, 18, 28)
BTN_DANGER_BG = (255, 85, 85)
BTN_DANGER_HOVER = (255, 110, 110)

# ==============================================================================
# FONT MANAGER
# ==============================================================================
_FONT_CACHE: Dict[Tuple[str, int, bool], pygame.font.Font] = {}


def get_font(size: int = 14, bold: bool = False, mono: bool = False) -> pygame.font.Font:
    """Return cached Pygame font for optimal rendering speed."""
    if not pygame.font.get_init():
        pygame.font.init()

    font_name = "consolas" if mono else "segoeui"
    cache_key = (font_name, size, bold)

    if cache_key not in _FONT_CACHE:
        try:
            # Attempt to load preferred system font
            font = pygame.font.SysFont(font_name, size, bold=bold)
        except Exception:
            font = pygame.font.SysFont(None, size, bold=bold)
        _FONT_CACHE[cache_key] = font

    return _FONT_CACHE[cache_key]

