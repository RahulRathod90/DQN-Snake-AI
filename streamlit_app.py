"""
DQN Snake AI — Live Interactive Web Application
Built with Streamlit & PyTorch.
Deploys directly on Streamlit Community Cloud (share.streamlit.io).
"""

import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch

# Ensure local modules can be loaded whether run from repo root or subfolder
ROOT_DIR = Path(__file__).resolve().parent
SUB_DIR = ROOT_DIR / "DQN-Snake-AI"
for p in [str(ROOT_DIR), str(SUB_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame

if not pygame.get_init():
    pygame.init()

from model import Dueling_QNet, Linear_QNet
from pathfinding import AStarAgent, AStarSafetyShield
from snake_ai import BLOCK_SIZE, SCREEN_HEIGHT, SCREEN_WIDTH, SnakeGameAI

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="DQN Snake AI — Autonomous Deep RL Platform",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply subtle styling enhancements
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .stButton > button {
        border-radius: 6px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# Model Loading & Caching
# ==============================================================================
@st.cache_resource
def load_all_models() -> Dict[str, dict]:
    """Cache and load all trained PyTorch neural network checkpoints."""
    models = {}

    def find_file(relative_paths: List[str]) -> Optional[Path]:
        for rp in relative_paths:
            for base in [SUB_DIR, ROOT_DIR]:
                p = base / rp
                if p.exists():
                    return p
        return None

    # 1. Optimized DQN (Phase 4 Validated, 11-D Dueling)
    p_opt = find_file([
        "experiments/phase3_comparison/exp_f_best_hyperparams.pth",
        "checkpoints/best_model.pth",
        "snake_dqn_improved.pth",
    ])
    if p_opt:
        try:
            ckpt = torch.load(p_opt, map_location="cpu", weights_only=False)
            sd = ckpt.get("model_state_dict", ckpt)
            net = Dueling_QNet(11, 256, 3)
            net.load_state_dict(sd)
            net.eval()
            models["opt_dqn"] = {"net": net, "dim": 11, "type": "dueling"}
        except Exception:
            pass

    # 2. DDQN + Dueling + PER (Phase 2 Best, 11-D Dueling)
    p_ddqn = find_file([
        "experiments/double_dueling_plus_per/model.pth",
        "checkpoints/best_model.pth",
    ])
    if p_ddqn:
        try:
            ckpt = torch.load(p_ddqn, map_location="cpu", weights_only=False)
            sd = ckpt.get("model_state_dict", ckpt)
            net = Dueling_QNet(11, 256, 3)
            net.load_state_dict(sd)
            net.eval()
            models["ddqn_per"] = {"net": net, "dim": 11, "type": "dueling"}
        except Exception:
            pass

    # 3. Enhanced 24-D DQN (Phase 4 Restored, 24-D Dueling)
    p_enh = find_file([
        "experiments/phase3_comparison/exp_c_enhanced_state.pth",
    ])
    if p_enh:
        try:
            ckpt = torch.load(p_enh, map_location="cpu", weights_only=False)
            sd = ckpt.get("model_state_dict", ckpt)
            net = Dueling_QNet(24, 256, 3)
            net.load_state_dict(sd)
            net.eval()
            models["enhanced_24d"] = {"net": net, "dim": 24, "type": "dueling"}
        except Exception:
            pass

    # 4. Original DQN (Phase 1 Baseline, 11-D Linear)
    p_orig = find_file([
        "experiments/original_dqn/model.pth",
        "snake_dqn_improved.pth",
    ])
    if p_orig:
        try:
            ckpt = torch.load(p_orig, map_location="cpu", weights_only=False)
            sd = ckpt.get("model_state_dict", ckpt)
            net = Linear_QNet(11, 256, 3)
            net.load_state_dict(sd)
            net.eval()
            models["orig_dqn"] = {"net": net, "dim": 11, "type": "linear"}
        except Exception:
            pass

    return models


# ==============================================================================
# Session State Initialization
# ==============================================================================
if "game" not in st.session_state:
    st.session_state.game = SnakeGameAI(display=False, width=SCREEN_WIDTH, height=SCREEN_HEIGHT)
    st.session_state.game.reset()

if "shield" not in st.session_state:
    st.session_state.shield = AStarSafetyShield(SCREEN_WIDTH, SCREEN_HEIGHT, BLOCK_SIZE)

if "astar_bot" not in st.session_state:
    st.session_state.astar_bot = AStarAgent(SCREEN_WIDTH, SCREEN_HEIGHT, BLOCK_SIZE)

if "is_playing" not in st.session_state:
    st.session_state.is_playing = False

if "step_count" not in st.session_state:
    st.session_state.step_count = 0

if "high_score" not in st.session_state:
    st.session_state.high_score = 0

if "last_action" not in st.session_state:
    st.session_state.last_action = 0

if "last_q_values" not in st.session_state:
    st.session_state.last_q_values = [0.0, 0.0, 0.0]

if "last_shield_override" not in st.session_state:
    st.session_state.last_shield_override = False

if "last_rejection_reason" not in st.session_state:
    st.session_state.last_rejection_reason = "None"

if "game_over_banner" not in st.session_state:
    st.session_state.game_over_banner = False


# ==============================================================================
# Helper: SVG Game Board Renderer
# ==============================================================================
def render_game_svg(game: SnakeGameAI, planned_path: Optional[List[Tuple[int, int]]] = None) -> str:
    """Generate high-contrast, scalable SVG graphic of current Snake board."""
    w, h = game.width, game.height
    bs = game.block_size
    svg_parts = [
        f'<svg viewBox="0 0 {w} {h}" width="100%" height="auto" '
        'style="border-radius:12px; border:2px solid #334155; background:#0F172A; '
        'box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5); display:block;">',
        '<defs>',
        '  <linearGradient id="snakeGrad" x1="0%" y1="0%" x2="100%" y2="100%">',
        '    <stop offset="0%" stop-color="#10B981" />',
        '    <stop offset="100%" stop-color="#059669" />',
        '  </linearGradient>',
        '  <linearGradient id="headGrad" x1="0%" y1="0%" x2="100%" y2="100%">',
        '    <stop offset="0%" stop-color="#34D399" />',
        '    <stop offset="100%" stop-color="#10B981" />',
        '  </linearGradient>',
        '  <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">',
        '    <feGaussianBlur stdDeviation="3" result="blur" />',
        '    <feComposite in="SourceGraphic" in2="blur" operator="over" />',
        '  </filter>',
        '</defs>',
    ]

    # 1. Subtle grid lines
    for x in range(0, w, bs * 2):
        svg_parts.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{h}" stroke="#1E293B" stroke-width="0.8" />')
    for y in range(0, h, bs * 2):
        svg_parts.append(f'<line x1="0" y1="{y}" x2="{w}" y2="{y}" stroke="#1E293B" stroke-width="0.8" />')

    # 2. Draw Planned A* Path Overlay
    if planned_path and len(planned_path) > 1:
        points_str = " ".join(f"{gx * bs + bs // 2},{gy * bs + bs // 2}" for gx, gy in planned_path)
        svg_parts.append(
            f'<polyline points="{points_str}" fill="none" stroke="#06B6D4" stroke-width="2.5" '
            'stroke-dasharray="4,4" opacity="0.65" />'
        )

    # 3. Draw Snake Body
    for i, seg in enumerate(game.snake[1:], start=1):
        sx, sy = seg
        svg_parts.append(
            f'<rect x="{sx + 1}" y="{sy + 1}" width="{bs - 2}" height="{bs - 2}" '
            f'rx="4" fill="url(#snakeGrad)" />'
        )

    # 4. Draw Snake Head with Eyes
    if game.snake:
        hx, hy = game.snake[0]
        svg_parts.append(
            f'<rect x="{hx + 0.5}" y="{hy + 0.5}" width="{bs - 1}" height="{bs - 1}" '
            f'rx="5" fill="url(#headGrad)" stroke="#6EE7B7" stroke-width="1.2" />'
        )
        # Direction-aware eyes
        eye_radius = 2.5
        pupil_radius = 1.2
        if game.direction == "UP":
            e1, e2 = (hx + 5, hy + 5), (hx + bs - 5, hy + 5)
            p1, p2 = (hx + 5, hy + 4), (hx + bs - 5, hy + 4)
        elif game.direction == "DOWN":
            e1, e2 = (hx + 5, hy + bs - 5), (hx + bs - 5, hy + bs - 5)
            p1, p2 = (hx + 5, hy + bs - 4), (hx + bs - 5, hy + bs - 4)
        elif game.direction == "LEFT":
            e1, e2 = (hx + 5, hy + 5), (hx + 5, hy + bs - 5)
            p1, p2 = (hx + 4, hy + 5), (hx + 4, hy + bs - 5)
        else:  # RIGHT
            e1, e2 = (hx + bs - 5, hy + 5), (hx + bs - 5, hy + bs - 5)
            p1, p2 = (hx + bs - 4, hy + 5), (hx + bs - 4, hy + bs - 5)

        svg_parts.append(f'<circle cx="{e1[0]}" cy="{e1[1]}" r="{eye_radius}" fill="#FFFFFF" />')
        svg_parts.append(f'<circle cx="{e2[0]}" cy="{e2[1]}" r="{eye_radius}" fill="#FFFFFF" />')
        svg_parts.append(f'<circle cx="{p1[0]}" cy="{p1[1]}" r="{pupil_radius}" fill="#0F172A" />')
        svg_parts.append(f'<circle cx="{p2[0]}" cy="{p2[1]}" r="{pupil_radius}" fill="#0F172A" />')

    # 5. Draw Glowing Ruby Food
    fx, fy = game.food
    cx, cy = fx + bs // 2, fy + bs // 2
    r = bs // 2 - 2
    svg_parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r + 3}" fill="#EF4444" opacity="0.3" filter="url(#glow)" />')
    svg_parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#EF4444" stroke="#FCA5A5" stroke-width="1.5" />')
    svg_parts.append(f'<circle cx="{cx - 2}" cy="{cy - 2}" r="2" fill="#FFFFFF" opacity="0.8" />')

    svg_parts.append("</svg>")
    return "".join(svg_parts)


# ==============================================================================
# Simulation Step Logic
# ==============================================================================
def execute_game_step(agent_label: str, models: Dict[str, dict]):
    """Advance simulation by 1 decision step according to selected agent."""
    game = st.session_state.game
    shield = st.session_state.shield
    astar_bot = st.session_state.astar_bot

    st.session_state.last_shield_override = False
    st.session_state.last_rejection_reason = "None"
    st.session_state.last_q_values = [0.0, 0.0, 0.0]

    if "Pure A*" in agent_label:
        action_idx = astar_bot.get_action(game)
    else:
        # Resolve target model
        model_info = None
        if "Safety Shield" in agent_label or "Optimized DQN" in agent_label:
            model_info = models.get("opt_dqn") or models.get("ddqn_per")
        elif "DDQN + Dueling + PER" in agent_label:
            model_info = models.get("ddqn_per") or models.get("opt_dqn")
        elif "Enhanced 24-D" in agent_label:
            model_info = models.get("enhanced_24d")
        else:
            model_info = models.get("orig_dqn") or models.get("opt_dqn")

        if model_info:
            net = model_info["net"]
            if model_info["dim"] == 24:
                state_vec = game._get_enhanced_state()
            else:
                state_vec = game.get_state()

            with torch.no_grad():
                s_t = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
                q_vals = net(s_t)[0].tolist()
                st.session_state.last_q_values = [round(float(v), 3) for v in q_vals]
                action_idx = int(np.argmax(q_vals))
        else:
            action_idx = 0

        # Apply Safety Shield if selected
        if "Safety Shield" in agent_label:
            filtered_act, overridden, reason = shield.filter_action(game, action_idx)
            st.session_state.last_shield_override = overridden
            st.session_state.last_rejection_reason = reason
            action_idx = filtered_act

    st.session_state.last_action = action_idx
    reward, done, score = game.move(action_idx)
    st.session_state.step_count += 1

    if score > st.session_state.high_score:
        st.session_state.high_score = score

    if done:
        st.session_state.game_over_banner = True
        st.session_state.is_playing = False
        game.reset()
        shield.reset_telemetry()


# ==============================================================================
# UI Layout
# ==============================================================================
st.title("🐍 DQN Snake AI — Deep Reinforcement Learning Platform")
st.caption(
    "Production-grade Reinforcement Learning, Autonomous Graph Search, "
    "and A* Safety Shielding for Snake. Deployed live on Streamlit Cloud."
)

tab_arena, tab_benchmark, tab_arch, tab_about = st.tabs([
    "🎮 Live AI Arena",
    "📊 Empirical Benchmarks",
    "🔬 System Architecture",
    "📦 Deployment & Source",
])

models = load_all_models()

# ------------------------------------------------------------------------------
# TAB 1: Live AI Arena
# ------------------------------------------------------------------------------
with tab_arena:
    col_ctrl, col_board, col_telemetry = st.columns([1.1, 2.2, 1.2], gap="medium")

    with col_ctrl:
        st.subheader("Agent Configuration")
        agent_choice = st.selectbox(
            "Select AI Agent",
            options=[
                "🏆 Pure A* Search (Deterministic Optimal)",
                "🛡️ DQN + A* Safety Shield (Phase 4 Hybrid)",
                "⚡ Optimized DQN (Phase 4 Validated)",
                "🧠 DDQN + Dueling + PER (Phase 2 Best)",
                "🌐 Enhanced 24-D DQN (Phase 4 Restored)",
                "🌱 Original DQN (Phase 1 Baseline)",
            ],
            index=0,
            key="agent_selector",
        )

        speed_choice = st.select_slider(
            "Simulation Speed",
            options=["Normal", "Fast", "Turbo"],
            value="Fast",
        )
        speed_intervals = {"Normal": 0.20, "Fast": 0.08, "Turbo": 0.02}
        interval = speed_intervals[speed_choice]

        show_astar_path = st.toggle("Show A* Path Overlay", value=True)

        st.divider()

        # Action Buttons
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            if st.session_state.is_playing:
                if st.button("⏸️ Pause", width="stretch"):
                    st.session_state.is_playing = False
                    st.rerun()
            else:
                if st.button("▶️ Play", type="primary", width="stretch"):
                    st.session_state.is_playing = True
                    st.session_state.game_over_banner = False
                    st.rerun()

        with btn_c2:
            if st.button("🔄 Reset Arena", width="stretch"):
                st.session_state.game.reset()
                st.session_state.shield.reset_telemetry()
                st.session_state.step_count = 0
                st.session_state.is_playing = False
                st.session_state.game_over_banner = False
                st.rerun()

        if not st.session_state.is_playing:
            if st.button("⏭️ Step Single Move", width="stretch"):
                execute_game_step(agent_choice, models)
                st.rerun()

        st.divider()

        # Score Metrics
        st.metric("Current Score", f"{st.session_state.game.score} 🍎")
        st.metric("High Score (Session)", f"{st.session_state.high_score} 🍎")
        st.metric("Step Counter", f"{st.session_state.step_count} moves")

    # Center: The Game Board
    with col_board:
        if st.session_state.game_over_banner:
            st.warning("Game Over! The snake collided. Click ▶️ Play or ⏭️ Step to restart.")

        # Fragment to execute live animation loops
        @st.fragment(run_every=interval if st.session_state.is_playing else None)
        def live_arena_fragment():
            if st.session_state.is_playing:
                execute_game_step(agent_choice, models)

            path_to_draw = None
            if show_astar_path:
                if "Pure A*" in agent_choice:
                    path_to_draw = st.session_state.astar_bot.last_path
                else:
                    path_to_draw = st.session_state.shield.last_path

            svg_html = render_game_svg(st.session_state.game, path_to_draw)
            st.html(svg_html)

        live_arena_fragment()

    # Right: Real-time Telemetry Dashboard
    with col_telemetry:
        st.subheader("Decision Telemetry")

        action_names = ["Straight (0)", "Turn Right (1)", "Turn Left (2)"]
        st.write(f"**Last Action:** `{action_names[st.session_state.last_action]}`")

        # Q-Values visualization
        if any(v != 0.0 for v in st.session_state.last_q_values):
            st.caption("Predicted Q-Values ($Q_{s, a}$)")
            q_df = pd.DataFrame({
                "Action": ["Straight", "Right", "Left"],
                "Q-Value": st.session_state.last_q_values,
            })
            st.bar_chart(q_df, x="Action", y="Q-Value", height=180)

        st.divider()

        # Spatial Risk Chips
        game = st.session_state.game
        dirs = ["UP", "RIGHT", "DOWN", "LEFT"]
        c_idx = dirs.index(game.direction)
        danger_s = game.check_collision(dirs[c_idx])
        danger_r = game.check_collision(dirs[(c_idx + 1) % 4])
        danger_l = game.check_collision(dirs[(c_idx - 1) % 4])

        st.caption("Immediate Collision Risk")
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.markdown(f"**Straight**\n{'🔴 Risk' if danger_s else '🟢 Clear'}")
        with rc2:
            st.markdown(f"**Right**\n{'🔴 Risk' if danger_r else '🟢 Clear'}")
        with rc3:
            st.markdown(f"**Left**\n{'🔴 Risk' if danger_l else '🟢 Clear'}")

        # Safety Shield Metrics
        if "Safety Shield" in agent_choice:
            st.divider()
            st.caption("A* Safety Shield Status")
            sh = st.session_state.shield
            override_status = "⚠️ Overridden" if st.session_state.last_shield_override else "✅ Approved"
            st.write(f"**Intervention:** {override_status}")
            st.write(f"**Rejection Reason:** `{st.session_state.last_rejection_reason}`")
            st.write(f"**Total Decisions:** {sh.total_decisions}")
            st.write(f"**Total Interventions:** {sh.actions_rejected}")


# ------------------------------------------------------------------------------
# TAB 2: Empirical Benchmarks
# ------------------------------------------------------------------------------
with tab_benchmark:
    st.subheader("100-Game Scientific Evaluation Benchmark")
    st.markdown(
        "Each agent was evaluated across **100 independent evaluation games** "
        "under identical environmental conditions (Seed: `123`, Greedy Policy $\\epsilon = 0.0$)."
    )

    benchmark_data = [
        {"Agent ID": "Var A", "Algorithm": "Original DQN (Baseline)", "Architecture": "Linear FC (11-D)", "Mean Score": 28.99, "Max Score": 61, "Wall Collisions": 76, "Avg Steps": 550.1, "Latency (ms)": 0.092},
        {"Agent ID": "Var B", "Algorithm": "DDQN + Dueling + PER", "Architecture": "Dueling FC (11-D)", "Mean Score": 26.69, "Max Score": 63, "Wall Collisions": 68, "Avg Steps": 683.8, "Latency (ms)": 0.350},
        {"Agent ID": "Var C", "Algorithm": "Enhanced 24-D DQN", "Architecture": "Dueling FC (24-D)", "Mean Score": 21.68, "Max Score": 52, "Wall Collisions": 72, "Avg Steps": 873.1, "Latency (ms)": 0.605},
        {"Agent ID": "Var D", "Algorithm": "Pure A* Search", "Architecture": "Algorithmic Search", "Mean Score": 222.75, "Max Score": 474, "Wall Collisions": 4, "Avg Steps": 33193.8, "Latency (ms)": 0.305},
        {"Agent ID": "Var E", "Algorithm": "DQN + A* Safety Shield", "Architecture": "Dueling + A* Shield", "Mean Score": 118.36, "Max Score": 182, "Wall Collisions": 16, "Avg Steps": 4367.2, "Latency (ms)": 1.142},
        {"Agent ID": "Var F", "Algorithm": "Optimized DQN (Phase 4)", "Architecture": "Dueling FC (11-D)", "Mean Score": 31.52, "Max Score": 71, "Wall Collisions": 1, "Avg Steps": 882.6, "Latency (ms)": 0.269},
    ]
    df_bm = pd.DataFrame(benchmark_data)
    st.dataframe(df_bm, width="stretch", hide_index=True)

    st.divider()

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        st.subheader("Mean Score Comparison")
        st.bar_chart(df_bm, x="Algorithm", y="Mean Score", height=320)

    with b_col2:
        st.subheader("Maximum Score Ceiling")
        st.bar_chart(df_bm, x="Algorithm", y="Max Score", height=320)

    st.info(
        "💡 **Key Scientific Finding:** While Pure A* achieves the absolute highest score ceiling (222.75), "
        "the **DQN + A* Safety Shield** provides the ideal balance: high-speed neural network inference with "
        "interventions only during fatal traps (10.7% override rate), yielding a 4x improvement over pure DQN.",
        icon="🔬",
    )


# ------------------------------------------------------------------------------
# TAB 3: System Architecture
# ------------------------------------------------------------------------------
with tab_arch:
    st.subheader("System Architecture & Algorithmic Design")

    st.markdown(
        """
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
        """
    )

    with st.expander("1. Double Deep Q-Networks (DDQN)", expanded=True):
        st.markdown(
            """
            In standard DQN, the max operator uses the same values to both select and evaluate an action,
            leading to severe maximization bias. **Double DQN** decouples action selection from action evaluation:

            $$Q_{target} = r + \\gamma Q(s', \\arg\\max_a Q(s', a; \\theta); \\theta^-)$$
            """
        )

    with st.expander("2. Dueling Architecture (Split Value & Advantage Streams)"):
        st.markdown(
            """
            Dueling networks explicitly separate state value $V(s)$ from the action advantage $A(s, a)$,
            enabling the network to evaluate state goodness without needing to learn the effect of each action:

            $$Q(s, a; \\theta, \\alpha, \\beta) = V(s; \\theta, \\beta) + \\left( A(s, a; \\theta, \\alpha) - \\frac{1}{|\\mathcal{A}|}\\sum_{a'} A(s, a'; \\theta, \\alpha) \\right)$$
            """
        )

    with st.expander("3. Prioritized Experience Replay (PER) via SumTree"):
        st.markdown(
            r"""
            Transitions are sampled proportionally to their TD-error magnitude $|\delta_i|$, corrected with
            importance sampling weights $w_i = (N \cdot P(i))^{-\beta}$ to preserve unbiased stochastic gradient descent.
            """
        )

    with st.expander("4. 4-Tier Hybrid Safety Shield"):
        st.markdown(
            """
            1. **Immediate Collision Guard**: Rejects actions driving the head directly into a wall or body segment.
            2. **BFS Trapped Pocket Guard**: Validates that accessible open space exceeds snake length.
            3. **Conditional Tail Reachability**: Ensures a valid exit path to the snake's tail always remains open.
            4. **Graceful Fallback Cascade**: If the proposed DQN move is dangerous, cascades to safe A* food route, tail-chasing loop, or maximal open space.
            """
        )


# ------------------------------------------------------------------------------
# TAB 4: Deployment & Source
# ------------------------------------------------------------------------------
with tab_about:
    st.subheader("Repository & Deployment Resources")

    st.markdown(
        """
        - **GitHub Repository:** [github.com/RahulRathod90/DQN-Snake-AI](https://github.com/RahulRathod90/DQN-Snake-AI)
        - **Author:** Rahul Rathod
        - **License:** MIT License
        """
    )

    st.divider()

    st.subheader("Deploy Live to Streamlit Community Cloud (Free)")
    st.markdown(
        """
        1. Visit **[share.streamlit.io](https://share.streamlit.io)** and log in with your GitHub account (`RahulRathod90`).
        2. Click **Create app** > **Yup, I have an app**.
        3. Fill in the fields:
           - **Repository:** `RahulRathod90/DQN-Snake-AI`
           - **Branch:** `main`
           - **Main file path:** `streamlit_app.py`
        4. Click **Deploy!** — Within 2 minutes, your project will be live at a public URL like `https://dqn-snake-ai.streamlit.app`!
        """
    )

    st.divider()

    st.subheader("Desktop GUI Execution (Pygame Native)")
    st.code(
        """# From PowerShell / Terminal in project directory:
.\\run.bat

# Or direct Python command:
python app.py""",
        language="powershell",
    )
