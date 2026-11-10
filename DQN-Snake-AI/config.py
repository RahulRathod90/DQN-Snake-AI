"""
Centralized Configuration for DQN Snake AI (Phase 1, Phase 2, & Phase 3).
Contains all hyperparameters, game constants, reward definitions, algorithm flags,
state representation modes, and path management.
"""

import os
import tempfile
from pathlib import Path

# Suppress Matplotlib cache warnings on Windows restricted dirs
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

# ==========================================
# Paths & Directories (with PyInstaller / Standalone Support)
# ==========================================
import sys

if getattr(sys, "frozen", False):
    # Executable bundle environment
    BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    BASE_DIR = Path(sys.executable).resolve().parent
    # Fallback if checkpoints exist in bundle
    if not (BASE_DIR / "checkpoints").exists() and (BUNDLE_DIR / "checkpoints").exists():
        CHECKPOINTS_DIR = BUNDLE_DIR / "checkpoints"
    else:
        CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
else:
    BASE_DIR = Path(__file__).resolve().parent
    BUNDLE_DIR = BASE_DIR
    CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

RESULTS_DIR = BASE_DIR / "results"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINTS_DIR / "best_model.pth"
LATEST_CHECKPOINT_PATH = CHECKPOINTS_DIR / "latest_checkpoint.pth"
LEGACY_MODEL_PATH = BASE_DIR / "snake_dqn_improved.pth"

EVAL_METRICS_CSV = RESULTS_DIR / "evaluation_metrics.csv"
EVAL_RESULTS_TXT = RESULTS_DIR / "evaluation_results.txt"
LEGACY_EVAL_TXT = BASE_DIR / "evaluation_results.txt"
RECORD_PATH = BASE_DIR / "record.txt"

# ==========================================
# Game Environment Settings
# ==========================================
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400
BLOCK_SIZE = 20
GRID_WIDTH = SCREEN_WIDTH // BLOCK_SIZE   # 30 cells
GRID_HEIGHT = SCREEN_HEIGHT // BLOCK_SIZE # 20 cells
TOTAL_GRID_CELLS = GRID_WIDTH * GRID_HEIGHT # 600 cells

# Colors (RGB)
COLOR_BLACK = (0, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_DARK_GREEN = (0, 200, 0)
COLOR_RED = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_CYAN = (0, 255, 255)
COLOR_BLUE = (30, 144, 255)
COLOR_GRAY = (128, 128, 128)
COLOR_DARK_GRAY = (40, 40, 40)

RENDER_FPS = 30
PLAY_FPS = 15

# ==========================================
# State Representation (Phase 3)
# ==========================================
STATE_SIZE_STANDARD = 11  # Phase 1/2 Baseline
STATE_SIZE_ENHANCED = 24  # Phase 3 Enhanced State Representation
STATE_SIZE = STATE_SIZE_STANDARD
STATE_MODE = "standard"   # "standard" (11 features) or "enhanced" (24 features)

# ==========================================
# Reward Engineering (Phase 1 & Phase 3)
# ==========================================
REWARD_MODE = "phase1"       # "phase1" (baseline) or "advanced" (Phase 3 PBRS + trap penalty)
REWARD_FOOD = 10.0           # Eating food
REWARD_COLLISION = -10.0     # Collision with wall or body
REWARD_CLOSER = 0.1          # Moving closer to food (Phase 1)
REWARD_FARTHER = -0.1        # Moving farther from food (Phase 1)
REWARD_STEP = -0.01          # Base survival step penalty (Phase 1)
REWARD_TRAP_PENALTY = -2.0   # Entering enclosed trap smaller than snake length (Phase 3)
REWARD_SPACE_FACTOR = 0.05   # Bonus for maintaining accessible open space (Phase 3)

STARVATION_BASE_STEPS = 100
STARVATION_FACTOR = 100

# ==========================================
# DQN Neural Network Architecture
# ==========================================
HIDDEN_SIZE = 256            # 256 hidden neurons
ACTION_SIZE = 3              # 3 actions [Straight, Right, Left]

# ==========================================
# DQN Agent Hyperparameters
# ==========================================
LEARNING_RATE = 0.0005       # Adam learning rate
GAMMA = 0.99                 # Discount factor for Bellman target
BATCH_SIZE = 64              # Mini-batch size for experience replay
MEMORY_CAPACITY = 100_000    # Maximum transitions stored in replay buffer
MIN_REPLAY_SIZE = 1_000      # Transitions required before training begins
TARGET_UPDATE_FREQ = 100     # Training steps between target network updates
MAX_GRAD_NORM = 1.0          # Gradient clipping norm

# Epsilon-greedy exploration
EPSILON_START = 1.0          # Initial exploration rate
EPSILON_MIN = 0.01           # Minimum exploration rate
EPSILON_DECAY = 0.995        # Multiplicative decay per episode

# ==========================================
# Phase 2 Advanced RL Flags & Hyperparameters
# ==========================================
USE_DOUBLE_DQN = True        # Double Deep Q-Network (DDQN)
USE_DUELING = False          # Dueling Deep Q-Network Architecture
USE_PER = False              # Prioritized Experience Replay (PER)

PER_ALPHA = 0.6              # Prioritization exponent
PER_BETA_START = 0.4         # Initial importance sampling exponent
PER_BETA_MAX = 1.0           # Maximum importance sampling exponent
PER_EPSILON = 1e-5           # Small constant to prevent zero priorities

# ==========================================
# Training Pipeline Defaults
# ==========================================
DEFAULT_EPISODES = 1000      # Default training episodes
TRAIN_FREQ_STEPS = 1         # Train replay buffer every N environment steps
MOVING_AVG_WINDOW = 50       # Window size for moving average tracking
CHECKPOINT_FREQ = 100        # Save periodic checkpoint every N episodes
RANDOM_SEED = 42             # Default random seed for reproducibility

# ==========================================
# Evaluation Defaults
# ==========================================
DEFAULT_EVAL_GAMES = 100     # Independent evaluation games
EVAL_SEED = 123              # Evaluation random seed
