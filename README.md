# DQN Snake AI — Professional Reinforcement Learning Platform (v1.0.0)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Pygame](https://img.shields.io/badge/Pygame-2.6+-yellowgreen.svg)](https://www.pygame.org/)
[![Release](https://img.shields.io/badge/Release-v1.0.0%20Production-blueviolet.svg)](RELEASE_NOTES.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests Passing](https://img.shields.io/badge/Tests-24%2F24%20Passed-brightgreen.svg)]()
[![Memory Profile](https://img.shields.io/badge/Memory-Zero%20Leak%20Verified-success.svg)]()

A comprehensive, production-grade Deep Reinforcement Learning and Graph Search platform for the classic Snake game. This project spans the complete engineering lifecycle from baseline Deep Q-Networks to modern architectures (Double DQN, Dueling Networks, Prioritized Experience Replay), advanced state formulations (24-D spatial lookahead), deterministic algorithmic planning (A* with tail reachability), safe hybrid shielding, an interactive multi-tabbed desktop GUI suite, and standalone Windows deployment automation.

---

## 1. Project Architecture

```
                          ┌──────────────────────────────────────────────┐
                          │            SNAKE GAME ENVIRONMENT            │
                          │   (Standard 11-D State / Enhanced 24-D State) │
                          └──────────────────────┬───────────────────────┘
                                                 │
                                                 ▼
             ┌───────────────────────────────────┴───────────────────────────────────┐
             │                                                                       │
             ▼                                                                       ▼
┌─────────────────────────┐                                             ┌─────────────────────────┐
│     NEURAL Q-NETWORK    │                                             │   A* PATHFINDING BOT    │
│  - Linear_QNet (11-D)   │                                             │  - Manhattan Heuristic  │
│  - Dueling_QNet (11/24) │                                             │  - Tail Reachability    │
│  - Target Network Sync  │                                             │  - Tail-Chasing Fallback│
└────────────┬────────────┘                                             └────────────┬────────────┘
             │                                                                       │
             ▼                                                                       ▼
     Candidate Action                                                         Safe Alternative
             │                                                                       │
             └───────────────────────────────────┬───────────────────────────────────┘
                                                 │
                                                 ▼
                                 ┌───────────────────────────────┐
                                 │     A* SAFETY SHIELD FILTER   │
                                 │  - Immediate Collision Guard  │
                                 │  - BFS Trapped Pocket Guard   │
                                 │  - Conditional Tail Reachable │
                                 │  - 4-Tier Fallback Cascade    │
                                 └───────────────┬───────────────┘
                                                 │
                                                 ▼
                                          Executed Action
```

---

## 2. Project Evolution Across All 6 Phases

| Phase | Milestone Name | Core Contributions & Engineering Deliverables |
|:---|:---|:---|
| **Phase 1** | **Foundational DQN** | Stabilized single-network DQN, Experience Replay, Huber loss (Smooth L1), gradient clipping, reward shaping, target network synchronization, and model checkpointing. |
| **Phase 2** | **Advanced Deep RL** | Decoupled target selection with **Double DQN (DDQN)**; split value/advantage streams with **Dueling Networks**; error-weighted sampling with **Prioritized Experience Replay (PER)** via SumTree. |
| **Phase 3** | **Spatial & Heuristic Systems** | Formulated **Enhanced 24-D State Representation**, developed grid-based **A\* Pathfinding Engine**, implemented BFS flood-fill trap detection, and built real-time HUD telemetry. |
| **Phase 4** | **Scientific Validation & Retraining** | Isolated and resolved critical flood-fill self-collision bug (restored feature 23) and tuning minimum replay threshold. Retrained Exp C (up from 0.07 to **21.68**) and Exp F (up from 0.35 to **31.52**). Conducted **100-game fair benchmark** with 7 publication-quality graphs. |
| **Phase 5** | **Professional Application Suite** | Transformed project into a modern desktop platform featuring **Game Arena**, **AI Intelligence Dashboard**, **Multi-Algorithm Comparison Matrix**, non-blocking **Training Studio**, and **Analytics View**. |
| **Phase 6** | **QA, Optimization & Deployment** | Executed 24 automated unit, integration, and stress tests (**100% passed**); verified **zero memory leaks** over 5,000 steps; established one-click Windows automations (`install.bat`, `run.bat`, `build.bat`, `clean.bat`, `test.bat`); prepared PyInstaller standalone build configuration. |

---

## 3. Verified Empirical Benchmark Results

Evaluated across **100 independent evaluation games per variant** under identical environmental conditions (Seed: `123`, Greedy Policy $\epsilon=0.0$):

| Variant ID | Name | Architecture | Mean Score | Max Score | Min Score | Median Score | Std Dev | Avg Steps | Latency | Wall Collisions | Body Collisions | Starve Timeouts |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Var A** | Original DQN (Phase 1 Baseline) | Standard FC (11-D) | 28.99 | 61 | 6 | 28.00 | 12.32 | 550.1 | 0.092 ms | 76 | 24 | 0 |
| **Var B** | DDQN + Dueling + PER (Phase 2 Best) | Dueling FC (11-D) | 26.69 | 63 | 0 | 26.00 | 14.71 | 683.8 | 0.350 ms | 68 | 32 | 0 |
| **Var C** | Enhanced 24-D DQN (Phase 4 Corrected) | Dueling FC (24-D) | 21.68 | 52 | 0 | 22.00 | 11.83 | 873.1 | 0.605 ms | 72 | 28 | 0 |
| **Var D** | Pure A* Agent (Deterministic Baseline) | Algorithmic Search | **222.75** | **474** | 38 | **216.00** | 111.85 | **33,193.8** | 0.305 ms | 4 | 91 | 5 |
| **Var E** | DQN + A* Safety Shield (Phase 4 Hybrid) | Dueling + A* Shield | **118.36** | **182** | 0 | **130.50** | 46.77 | **4,367.2** | 1.142 ms | 16 | 74 | 10 |
| **Var F** | Optimized DQN (Phase 4 Validated) | Dueling FC (11-D) | **31.52** | **71** | 0 | **33.50** | 15.62 | **882.6** | 0.269 ms | **1** | 82 | 17 |

> [!NOTE]
> **Key Scientific Takeaways:**
> - **Pure A\*** achieves the absolute highest score ceiling (**222.75 mean**, **474 max**) through exhaustive graph exploration.
> - **DQN + A\* Safety Shield** provides the ideal balance: fast neural network movement backed by an intervention rate of only **10.74%**, yielding a **118.36 mean score** (a 4x improvement over pure DQN).
> - **Optimized DQN (Var F)** achieved the highest score among all standalone RL models (**31.52 mean**, **71 max**) while reducing suicidal wall collisions from 76% down to **1%**.

---

## 4. Installation and Execution Guide

### Option 1: One-Click Windows Scripts (Recommended)

All `.bat` launcher scripts are conveniently located in the project root directory:

| Script | Purpose | Command (PowerShell / Command Prompt) |
| :--- | :--- | :--- |
| **`install.bat`** | Creates `.venv` virtual environment & installs all dependencies | `.\install.bat` |
| **`run.bat`** | Launches the professional desktop GUI application suite | `.\run.bat` |
| **`test.bat`** | Runs all 24 automated unit, integration, and stress tests | `.\test.bat` |
| **`build.bat`** | Compiles standalone Windows `.exe` via PyInstaller (`dist/SnakeAI`) | `.\build.bat` |
| **`clean.bat`** | Removes temporary `__pycache__`, `.pyc`, and build directories | `.\clean.bat` |

### Option 2: Direct Python Execution (Cross-Platform)

If you prefer running commands directly via terminal:

1. **Set up Virtual Environment**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

2. **Install Required Packages**:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Launch Desktop Application**:
   ```powershell
   python app.py
   ```

4. **Execute Verification & Stress Tests**:
   ```powershell
   python -m unittest test_suite.py
   python stress_test.py
   ```

---

## 5. Professional Desktop GUI Suite (v1.0.0)

### Application Tabs:
1. **Game Arena**:
   - High-fidelity grid with animated snake (direction-sensitive eyes, gradient emerald body).
   - Glowing ruby food item with specular reflections and pulsing animations.
   - Connected translucent A* path overlay.
   - Interactive speed selector: `5 FPS`, `15 FPS`, `30 FPS`, `60 FPS`, and `TURBO (MAX)`.
   - Live dropdown selector with **7 Selectable Agent Modes**:
     1. Human Player (Keyboard `Arrows` / `WASD` with anti-reversal steering protection)
     2. Original DQN (Phase 1 Baseline)
     3. DDQN + Dueling + PER (Phase 2 Best)
     4. Enhanced 24-D DQN (Phase 4 Corrected)
     5. Pure A* Agent (Deterministic Baseline)
     6. DQN + A* Safety Shield (Phase 4 Hybrid)
     7. Optimized DQN (Phase 4 Validated)
2. **AI Intelligence Dashboard**:
   - Real-time Q-value bar charts ($Q_{Straight}, Q_{Right}, Q_{Left}$) and Softmax action probabilities.
   - Spatial perception matrix: Front, Right, and Left collision risk chips.
   - Manhattan food distance and accessible open space percentage bar.
   - Active Safety Shield telemetry displaying overrides, rejections reasons, and fallback modes.
3. **Algorithm Comparison Matrix**:
   - Live multi-agent comparison table displaying verified benchmark metrics.
   - Toggle between **100-Game Benchmark Data** and **Live Session Gameplay Statistics**.
   - Relative mean score visual comparison bars.
4. **Interactive Training Studio**:
   - Hyperparameter configuration sliders: Algorithm, Target Episodes, Learning Rate, Batch Size, Discount Factor ($\gamma$).
   - Non-blocking asynchronous worker thread (`TrainingWorker`) preventing UI lockups.
   - Real-time in-GUI score and loss curve plotting.
5. **Scientific Analytics & Export**:
   - Interactive viewer for all 7 Phase 4 benchmark plots.
   - One-click **Export Results as CSV** and **Export Results as JSON**.

### Keyboard Shortcuts:
| Key | Action |
|:---:|:---|
| `[Space]` | Pause / Resume Game |
| `[R]` | Restart Game Arena |
| `[P]` | Toggle A* Planned Path Overlay |
| `[1] .. [5]` | Adjust Framerate (5 FPS, 15 FPS, 30 FPS, 60 FPS, TURBO) |
| `[Up, Down, Left, Right]` or `[W, A, S, D]` | Steer Snake in Human Player mode |

---

## 6. Project Directory Layout

```
DQN-Snake-AI/
├── checkpoints/                        # Model checkpoints and training saves
├── experiments/                        # Experiment backups and Phase 2/3 outputs
│   ├── phase3_comparison/              # Phase 4 Retrained models & summaries
│   │   ├── exp_c_enhanced_state.pth    # Retrained 24-D DQN (Mean 21.68)
│   │   └── exp_f_best_hyperparams.pth  # Retrained Tuned DQN (Mean 31.52)
│   └── backups/                        # Pre-retraining backups
├── gui/                                # Phase 5 Professional Desktop Application
│   ├── __init__.py
│   ├── analytics_view.py               # Embedded matplotlib chart viewer & exports
│   ├── app.py                          # Unified Pygame GUI coordinator
│   ├── board_renderer.py               # High-fidelity board, snake & food rendering
│   ├── comparison_view.py              # Interactive multi-algorithm comparison matrix
│   ├── telemetry_panel.py              # Real-time Q-values & spatial dashboard
│   ├── theme.py                        # Cyberpunk/Catppuccin color palette & fonts
│   ├── training_view.py                # Asynchronous training studio with live plots
│   └── widgets.py                      # Reusable buttons, dropdowns, bars, and charts
├── results/                            # Benchmark reports, CSVs, and PNG charts
│   ├── phase4_average_score_comparison.png
│   ├── phase4_score_distribution.png
│   ├── phase4_survival_comparison.png
│   ├── phase4_training_curves.png
│   ├── phase4_collision_rate.png
│   ├── phase4_inference_time.png
│   ├── phase4_hybrid_shield_analysis.png
│   ├── phase4_benchmark_results.csv
│   ├── phase4_benchmark_results.json
│   ├── phase4_training_summary.csv
│   └── phase6_stress_test_results.json
├── agent.py                            # DQN / DDQN / Dueling / PER Agent & Safety Shield
├── app.py                              # Desktop Application Launcher
├── benchmark_phase4.py                 # Multi-variant 100-game benchmark runner
├── build.bat                           # Standalone Windows executable builder
├── clean.bat                           # Temporary artifact and cache cleaner
├── config.py                           # System configuration and hyperparameters
├── evaluate.py                         # Evaluation runner with telemetry
├── hyperparameter_tuning.py            # Multi-seed hyperparameter search
├── install.bat                         # Automated Windows setup script
├── model.py                            # PyTorch Linear_QNet & Dueling_QNet
├── pathfinding.py                      # A* search, tail reachability & AStarSafetyShield
├── play_ai.py                          # Dual GUI/CLI visualizer launcher
├── run.bat                             # One-click Windows application launcher
├── snake_ai.py                         # Gymnasium-style Snake environment
├── snake_ai.spec                       # PyInstaller packaging configuration
├── snake_game.py                       # Manual keyboard Snake game
├── stress_test.py                      # Phase 6 Stress & memory leak profiling suite
├── test.bat                            # Full automated test runner
├── test_suite.py                       # 17-test unit and integration test suite
├── .gitignore                          # Clean Git tracking configuration
├── PHASE_4_AUDIT_REPORT.md             # Formal engineering audit of Phase 3
├── PHASE_4_FINAL_REPORT.md             # Final scientific validation report
├── PHASE_6_QA_REPORT.md                # Phase 6 QA & reliability report
├── RELEASE_NOTES.md                    # v1.0.0 Production Release details
└── README.md                           # Documentation and user guide
```

---

## 7. Deliverables & College Presentation Suitability

This project has been structured for academic defense, technical conferences, and industry portfolio presentation:
- **Mathematical Formulations**: Bellman optimality equation, Double DQN loss decomposition, Huber loss gradient clipping, Dueling advantage centering, and A* admissible Manhattan heuristic.
- **Empirical Rigor**: No fabricated metrics; 100 independent evaluation games per variant under deterministic random seeding.
- **Stress-Tested Reliability**: 24 automated tests covering memory leak profiling, rapid resetting, agent hot-switching, and fault injection.
- **Zero Framework Bloat**: Clean vanilla PyTorch and Pygame implementation without unnecessary external wrappers.
