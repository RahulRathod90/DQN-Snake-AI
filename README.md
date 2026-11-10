# DQN Snake AI
### Deep Reinforcement Learning & Intelligent Pathfinding Platform

A Snake game AI platform that combines Deep Q-Learning with graph-search pathfinding. It progresses from a baseline DQN agent to advanced reinforcement-learning variants, A* planning, a safety-shielded hybrid agent, and an interactive desktop application for training, evaluation, and analysis.

---

## Project Links

- **GitHub Repository**: [DQN-Snake-AI](https://github.com/RahulRathod90/DQN-Snake-AI)
- **Live Demo**: This project is currently a desktop application and does not have a browser-hosted demo. Add a demo video or deployment URL here when one is available.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   SNAKE GAME ENVIRONMENT                    │
│             Game State • Food • Snake • Rewards             │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    STATE REPRESENTATION                     │
│         Standard 11-D State / Enhanced 24-D State           │
└──────────────────────────────┬──────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
┌──────────────────────────────┐ ┌────────────────────────────┐
│        DEEP RL AGENT         │ │       A* PATHFINDING       │
│                              │ │                            │
│ • DQN / Double DQN           │ │ • Manhattan Heuristic      │
│ • Dueling Network            │ │ • Tail Reachability        │
│ • Prioritized Replay (PER)   │ │ • BFS Trap Detection       │
│ • Q-value Action Selection   │ │ • Tail-chasing Fallback    │
└──────────────┬───────────────┘ └──────────────┬─────────────┘
               │ Candidate Action               │ Safe Alternative
               └───────────────┬────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      A* SAFETY SHIELD                       │
│        Collision Checks • Trap Checks • Fallback Logic      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    EXECUTED GAME ACTION                     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                REWARD & EXPERIENCE FEEDBACK                 │
│           Replay Memory • Learning • Evaluation             │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               └──────────────► Agent Update
```

---

## Application Architecture

```
┌───────────────────────────┐
│        Desktop GUI        │
│  Game Arena / Dashboard   │
│   Training / Comparison   │
│    Analytics & Export     │
└─────────────┬─────────────┘
              │
┌─────────────▼─────────────┐
│        Application        │
│          Launcher         │
└─────────────┬─────────────┘
              │
┌─────────────┼─────────────┐
▼             ▼             ▼
┌──────────────┐ ┌──────────────┐ ┌────────────────┐
│ RL Agent and │ │  Snake Game  │ │ A* Pathfinding │
│ Neural Models│ │ Environment  │ │    & Safety    │
└──────┬───────┘ └──────┬───────┘ └───────┬────────┘
       └────────────────┼─────────────────┘
                        ▼
┌───────────────────────────┐
│   Evaluation & Analytics  │
│    Benchmarks • Reports   │
│    CSV / JSON • Charts    │
└───────────────────────────┘
```

---

## Key Features

- **Deep Q-Learning**: Learns game actions from state, reward, and experience.
- **Advanced RL algorithms**: Double DQN, Dueling DQN, and Prioritized Experience Replay.
- **State representations**: Standard 11-dimensional and enhanced 24-dimensional state formulations.
- **A* pathfinding**: Manhattan-distance heuristic, tail-reachability checks, and fallback planning.
- **Hybrid safety shield**: Filters unsafe candidate actions and can select safer alternatives.
- **Desktop application**: Game Arena, AI Intelligence Dashboard, algorithm comparison, training studio, and analytics.
- **Evaluation tools**: Reproducible multi-agent benchmarks, plots, and CSV/JSON exports.
- **Windows automation**: Setup, run, test, build, and cleanup scripts.

---

## Algorithms and Development Phases

| Phase | Focus | Main Implementation |
|:---:|:---|:---|
| **1** | **Foundational DQN** | Experience replay, Huber loss, gradient clipping, reward shaping, target-network synchronization |
| **2** | **Advanced Deep RL** | Double DQN, Dueling Networks, Prioritized Experience Replay |
| **3** | **Spatial & Planning Systems** | 24-D state, A* pathfinding, BFS trap detection, telemetry |
| **4** | **Validation & Retraining** | Bug fixes, hyperparameter tuning, 100-game benchmark |
| **5** | **Desktop Application** | Game Arena, intelligence dashboard, comparison matrix, training studio, analytics |
| **6** | **QA & Deployment** | Automated tests, stress testing, Windows scripts, PyInstaller configuration |

---

## Benchmark Results

The following results are reported by the project's 100-game evaluation benchmark. Each variant was evaluated under the same stated conditions (seed `123`, greedy policy $\epsilon = 0.0$).

| Agent | Architecture | Mean Score | Maximum Score |
|:---|:---|:---:|:---:|
| **Original DQN** | Standard FC, 11-D | 28.99 | 61 |
| **DDQN + Dueling + PER** | Dueling FC, 11-D | 26.69 | 63 |
| **Enhanced 24-D DQN** | Dueling FC, 24-D | 21.68 | 52 |
| **Pure A\*** | Algorithmic search | **222.75** | **474** |
| **DQN + A\* Safety Shield** | Dueling DQN + A* | **118.36** | **182** |
| **Optimized DQN** | Dueling FC, 11-D | **31.52** | **71** |

> **Interpretation:** In the reported benchmark, Pure A* achieved the highest mean score. The DQN + A* Safety Shield hybrid also outperformed the standalone DQN variants by mean score. These are project benchmark results, not a guarantee of performance on other environments.

---

## Desktop Application

The application includes the following views:

1. **Game Arena** — play manually or select one of the available AI agents.
2. **AI Intelligence Dashboard** — inspect Q-values, action probabilities, spatial risk, and safety-shield activity.
3. **Algorithm Comparison Matrix** — compare benchmark results and live-session statistics.
4. **Training Studio** — configure training parameters and monitor training curves.
5. **Scientific Analytics** — view benchmark plots and export results as CSV or JSON.

### Keyboard Controls

| Key | Action |
|:---:|:---|
| `Space` | Pause / resume |
| `R` | Restart game |
| `P` | Toggle A* path overlay |
| `1`–`5` | Change game speed |
| `Arrow keys` / `W A S D` | Control the snake in Human Player mode |

---

## Tech Stack

| Area | Technologies |
|:---|:---|
| **Language** | Python 3.10+ |
| **Deep Learning** | PyTorch |
| **Game & Rendering** | Pygame |
| **Data Visualization** | Matplotlib |
| **Packaging** | PyInstaller |
| **Testing** | Python unittest and project stress tests |

---

## Installation

### Windows Quick Start
If the batch files are present in the project root:
```powershell
.\install.bat
.\run.bat
```
Run the automated test suite with:
```powershell
.\test.bat
```
Build the standalone Windows application with:
```powershell
.\build.bat
```

### Manual Setup

```bash
git clone https://github.com/RahulRathod90/DQN-Snake-AI.git
cd DQN-Snake-AI

python -m venv .venv
```

Activate the environment:
- **Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

Install dependencies and launch:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Run tests:
```bash
python -m unittest test_suite.py
python stress_test.py
```

*The Windows batch scripts are optional convenience launchers.*

---

## Repository Structure

```
DQN-Snake-AI/
├── checkpoints/          # Saved model checkpoints
├── experiments/          # Experiment outputs and backups
├── gui/                  # Desktop GUI modules
├── results/              # Benchmark reports, charts, and exports
├── agent.py              # DQN agents and safety-shield integration
├── app.py                # Desktop application launcher
├── config.py             # Configuration and hyperparameters
├── model.py              # Neural-network models
├── pathfinding.py        # A* search and safety logic
├── snake_ai.py           # Snake game environment
├── snake_game.py         # Manual Snake game
├── train.py              # Training entry point
├── evaluate.py           # Evaluation runner
├── requirements.txt      # Python dependencies
├── test_suite.py         # Automated tests
├── README.md             # Project documentation
└── LICENSE               # MIT License
```

---

## Future Improvements

- Publish a short gameplay and training demonstration.
- Add a browser-based interactive demo if web deployment is desired.
- Extend evaluation across more random seeds and game configurations.
- Document reproducible training commands and hardware requirements.

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE).

---
*Developed as a reinforcement learning project exploring Deep Q-Networks, intelligent path planning, and safe hybrid decision-making.*
