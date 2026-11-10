# 🐍 DQN Snake AI
### Deep Reinforcement Learning & Intelligent Pathfinding Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Pygame](https://img.shields.io/badge/Pygame-2.6%2B-yellowgreen.svg)](https://www.pygame.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Deep Reinforcement Learning project that trains an AI agent to play the classic Snake game using Deep Q-Networks (DQN). The platform combines advanced reinforcement learning algorithms, A* pathfinding, intelligent safety mechanisms, and an interactive desktop application.

---

## 🌐 Project Links

- **GitHub Repository:** https://github.com/RahulRathod90/DQN-Snake-AI

---

## 🏗️ System Architecture

```text
          ┌──────────────────────────────┐
          │      SNAKE ENVIRONMENT       │
          │   Snake • Food • Rewards    │
          └──────────────┬───────────────┘
                         │
                         ▼
          ┌──────────────────────────────┐
          │    STATE REPRESENTATION      │
          │    Standard 11-D / 24-D      │
          └──────────────┬───────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
   ┌────────────────────┐  ┌────────────────────┐
   │   DEEP RL AGENT    │  │  A* PATHFINDING    │
   │                    │  │                    │
   │ • DQN              │  │ • A* Search        │
   │ • Double DQN       │  │ • BFS Detection    │
   │ • Dueling Network  │  │ • Tail Reachability│
   │ • Prioritized Replay│ │ • Safe Fallback    │
   └──────────┬─────────┘  └──────────┬─────────┘
              │                       │
              └──────────┬────────────┘
                         ▼
          ┌──────────────────────────────┐
          │       A* SAFETY SHIELD       │
          │ Collision & Trap Prevention │
          └──────────────┬───────────────┘
                         │
                         ▼
          ┌──────────────────────────────┐
          │       EXECUTED ACTION        │
          └──────────────┬───────────────┘
                         │
                         ▼
          ┌──────────────────────────────┐
          │    REWARD & LEARNING        │
          │ Experience Replay & Updates │
          └──────────────┬───────────────┘
                         │
                         └──────► Agent
