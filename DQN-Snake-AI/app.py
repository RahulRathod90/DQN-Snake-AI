"""
Professional Application Launcher for the DQN Snake AI Platform.
Usage:
    python app.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from gui.app import SnakeAIApp

if __name__ == "__main__":
    app = SnakeAIApp()
    app.run()

