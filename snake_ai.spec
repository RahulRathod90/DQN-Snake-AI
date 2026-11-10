# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller Specification File for DQN Snake AI Standalone Windows Executable.
Compiles the application, Pygame GUI, PyTorch inference engine, and verified model checkpoints.
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None
base_dir = Path.cwd()

# Data files to bundle with executable
datas = [
    ('gui', 'gui'),
    ('DQN-Snake-AI/gui', 'gui'),
    ('results/*.png', 'results'),
    ('DQN-Snake-AI/results/*.png', 'results'),
    ('results/*.json', 'results'),
    ('DQN-Snake-AI/results/*.json', 'results'),
    ('results/*.csv', 'results'),
    ('DQN-Snake-AI/results/*.csv', 'results'),
    ('experiments/phase3_comparison/*.pth', 'experiments/phase3_comparison'),
    ('DQN-Snake-AI/experiments/phase3_comparison/*.pth', 'experiments/phase3_comparison'),
    ('experiments/original_dqn/*.pth', 'experiments/original_dqn'),
    ('DQN-Snake-AI/experiments/original_dqn/*.pth', 'experiments/original_dqn'),
    ('snake_dqn_improved.pth', '.'),
    ('DQN-Snake-AI/snake_dqn_improved.pth', '.'),
    ('record.txt', '.'),
    ('DQN-Snake-AI/record.txt', '.'),
    ('README.md', '.'),
]

# Ensure files exist before bundling
datas = [(src, dst) for src, dst in datas if list(base_dir.glob(src)) or (base_dir / src).is_file()]

hiddenimports = [
    'pygame',
    'torch',
    'numpy',
    'matplotlib',
    'gui',
    'gui.theme',
    'gui.widgets',
    'gui.board_renderer',
    'gui.telemetry_panel',
    'gui.comparison_view',
    'gui.training_view',
    'gui.analytics_view',
    'gui.app',
    'agent',
    'config',
    'model',
    'pathfinding',
    'snake_ai',
    'prioritized_replay',
]

a = Analysis(
    ['app.py'],
    pathex=[str(base_dir), str(base_dir / "DQN-Snake-AI")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'IPython', 'notebook', 'scipy', 'pytest'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SnakeAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True, # Console enabled for logging & debug outputs
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SnakeAI',
)
