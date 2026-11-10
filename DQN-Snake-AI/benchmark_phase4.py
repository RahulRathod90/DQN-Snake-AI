"""
Phase 4 Final Benchmarking & Evaluation Suite (6 Variants).
Scientifically evaluates all 6 algorithms over 100 independent games under identical conditions:
- Variant A: Original DQN (Phase 1 Baseline)
- Variant B: DDQN + Dueling + PER (Phase 2 Best)
- Variant C: Enhanced 24-feature DQN (Phase 4 Corrected & Retrained)
- Variant D: Pure A* Agent (Deterministic Algorithmic Baseline)
- Variant E: DQN + A* Safety Shield (Phase 4 Multi-Tiered Hybrid)
- Variant F: Optimized DQN (Phase 4 Validated Hyperparameters)

Outputs:
- results/phase4_benchmark_results.csv
- results/phase4_benchmark_results.json
- results/phase4_training_summary.csv
- 7 publication-quality visualization charts in results/
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import Agent, set_seed
from config import BASE_DIR, EXPERIMENTS_DIR, RESULTS_DIR
from model import load_model
from pathfinding import AStarAgent, AStarSafetyShield
from snake_ai import SnakeGameAI

PHASE3_EXP_DIR = EXPERIMENTS_DIR / "phase3_comparison"
PHASE4_RESULTS_DIR = RESULTS_DIR
PHASE4_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_variant(
    variant_id: str,
    name: str,
    model_path: Optional[Path],
    is_pure_astar: bool = False,
    use_shield: bool = False,
    state_mode: str = "standard",
    games: int = 100,
    seed: int = 123,
) -> Dict[str, Any]:
    """Evaluate a single variant over N independent games with precise telemetry."""
    set_seed(seed)
    astar_bot = AStarAgent() if is_pure_astar else None
    shield = AStarSafetyShield() if use_shield else None
    agent = None
    arch_type = "Pure A*"

    if not is_pure_astar and model_path is not None:
        model, arch, detected_dim = load_model(model_path, map_location="cpu")
        arch_type = arch
        state_mode = "enhanced" if detected_dim == 24 else "standard"
        agent = Agent(state_size=detected_dim, seed=seed)
        agent.model = model
        agent.dueling = (arch == "dueling")
        agent.model.eval()
        agent.training = False
        agent.epsilon = 0.0

    game = SnakeGameAI(display=False, state_mode=state_mode)

    scores: List[int] = []
    steps: List[int] = []
    rewards: List[float] = []
    food_collected: List[int] = []
    termination_causes: Dict[str, int] = {"wall": 0, "body": 0, "starvation": 0}
    inference_times_ms: List[float] = []

    eval_start_time = time.time()

    for game_idx in range(1, games + 1):
        game.reset()
        done = False

        while not done:
            t0 = time.perf_counter()
            if is_pure_astar:
                act_idx = astar_bot.get_action(game)
            elif use_shield:
                s = game.get_state()
                act_idx, _, _ = agent.get_action_with_telemetry(s, game, shield)
            else:
                s = game.get_state()
                act_idx = agent.get_action_index(s, deterministic=True)

            inference_ms = (time.perf_counter() - t0) * 1000.0
            inference_times_ms.append(inference_ms)

            # Detect cause of collision before move if move is fatal
            directions = ["UP", "RIGHT", "DOWN", "LEFT"]
            cur_idx = directions.index(game.direction)
            if act_idx == 1:
                cur_idx = (cur_idx + 1) % 4
            elif act_idx == 2:
                cur_idx = (cur_idx - 1) % 4
            cand_dir = directions[cur_idx]

            hx, hy = game.snake[0]
            bx = game.block_size
            if cand_dir == "UP":
                cand_pt = (hx, hy - bx)
            elif cand_dir == "DOWN":
                cand_pt = (hx, hy + bx)
            elif cand_dir == "LEFT":
                cand_pt = (hx - bx, hy)
            else:
                cand_pt = (hx + bx, hy)

            is_wall_hit = (cand_pt[0] < 0 or cand_pt[0] >= game.width or cand_pt[1] < 0 or cand_pt[1] >= game.height)
            is_body_hit = (cand_pt in game.snake[:-1])

            _, done, _ = game.move(act_idx)

            if done:
                if is_wall_hit:
                    termination_causes["wall"] += 1
                elif is_body_hit:
                    termination_causes["body"] += 1
                else:
                    termination_causes["starvation"] += 1

        scores.append(game.score)
        food_collected.append(game.food_eaten)
        steps.append(game.frame_iteration)
        rewards.append(game.total_reward)

    total_eval_time = time.time() - eval_start_time
    scores_arr = np.array(scores)
    steps_arr = np.array(steps)
    rewards_arr = np.array(rewards)
    inf_arr = np.array(inference_times_ms)

    res = {
        "variant_id": variant_id,
        "name": name,
        "model_path": str(model_path) if model_path else "None (Algorithmic)",
        "architecture": arch_type,
        "state_mode": state_mode,
        "games_evaluated": games,
        "mean_score": float(np.round(np.mean(scores_arr), 2)),
        "max_score": int(np.max(scores_arr)),
        "min_score": int(np.min(scores_arr)),
        "median_score": float(np.round(np.median(scores_arr), 2)),
        "std_score": float(np.round(np.std(scores_arr), 2)),
        "avg_steps": float(np.round(np.mean(steps_arr), 2)),
        "avg_food": float(np.round(np.mean(food_collected), 2)),
        "avg_reward": float(np.round(np.mean(rewards_arr), 2)),
        "avg_inference_time_ms": float(np.round(np.mean(inf_arr), 3)),
        "total_eval_time_seconds": float(np.round(total_eval_time, 2)),
        "collision_stats": dict(termination_causes),
        "scores": scores,
        "steps": steps,
    }

    if use_shield and shield is not None:
        res["shield_telemetry"] = shield.get_telemetry()

    return res


def generate_all_visualizations(
    results: List[Dict[str, Any]],
    output_dir: Path = RESULTS_DIR,
) -> None:
    """Generate all 7 required publication-quality Phase 4 graphs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    names = [r["variant_id"] for r in results]
    full_names = [r["name"] for r in results]
    means = [r["mean_score"] for r in results]
    stds = [r["std_score"] for r in results]
    medians = [r["median_score"] for r in results]
    maxs = [r["max_score"] for r in results]
    steps = [r["avg_steps"] for r in results]
    inf_times = [r["avg_inference_time_ms"] for r in results]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#17becf"]

    # 1. phase4_average_score_comparison.png
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="black")
    for bar, m in zip(bars, means):
        ax.annotate(f"{m:.2f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Phase 4: Mean Evaluation Score (100 Independent Games)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Average Food Collected (Score)", fontsize=11)
    ax.set_xlabel("AI Variant", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()
    fig.savefig(output_dir / "phase4_average_score_comparison.png")
    plt.close(fig)

    # 2. phase4_score_distribution.png
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    scores_data = [r["scores"] for r in results]
    box = ax.boxplot(scores_data, tick_labels=names, patch_artist=True,
                     boxprops=dict(facecolor="#d0e1f9", color="#1f77b4"),
                     medianprops=dict(color="red", linewidth=2.0))
    for patch, col in zip(box["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    ax.set_title("Phase 4: Score Distribution Across 100 Games", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score (Food Collected)", fontsize=11)
    ax.set_xlabel("AI Variant", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()
    fig.savefig(output_dir / "phase4_score_distribution.png")
    plt.close(fig)

    # 3. phase4_survival_comparison.png
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    bars = ax.bar(names, steps, color=colors, alpha=0.85, edgecolor="black")
    for bar, st in zip(bars, steps):
        ax.annotate(f"{int(st):,}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Phase 4: Average Survival Steps per Game", fontsize=13, fontweight="bold")
    ax.set_ylabel("Survival Frames / Steps", fontsize=11)
    ax.set_xlabel("AI Variant", fontsize=11)
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()
    fig.savefig(output_dir / "phase4_survival_comparison.png")
    plt.close(fig)

    # 4. phase4_training_curves.png
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    # Check for available training summaries
    exp_c_sum_file = PHASE3_EXP_DIR / "exp_c_training_summary.json"
    exp_f_sum_file = PHASE3_EXP_DIR / "exp_f_training_summary.json"
    
    if exp_c_sum_file.is_file():
        with open(exp_c_sum_file, "r") as f:
            c_data = json.load(f)
            ax1.plot(c_data["moving_avgs"], label="Exp C: Enhanced 24-D", color="#2ca02c", linewidth=2.0)
            ax2.plot(c_data["losses"], label="Exp C: Enhanced 24-D", color="#2ca02c", alpha=0.6)

    if exp_f_sum_file.is_file():
        with open(exp_f_sum_file, "r") as f:
            f_data = json.load(f)
            ax1.plot(f_data["moving_avgs"], label="Exp F: Tuned DQN", color="#17becf", linewidth=2.0)
            ax2.plot(f_data["losses"], label="Exp F: Tuned DQN", color="#17becf", alpha=0.6)

    ax1.set_title("50-Episode Moving Average Training Scores", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Moving Average Score")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.7)

    ax2.set_title("Training Loss Progression (Smooth L1)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Huber Loss")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()
    fig.savefig(output_dir / "phase4_training_curves.png")
    plt.close(fig)

    # 5. phase4_collision_rate.png
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    wall_counts = [r["collision_stats"].get("wall", 0) for r in results]
    body_counts = [r["collision_stats"].get("body", 0) for r in results]
    starve_counts = [r["collision_stats"].get("starvation", 0) for r in results]

    x = np.arange(len(names))
    w = 0.25
    ax.bar(x - w, wall_counts, width=w, label="Wall Collision", color="#d62728", alpha=0.85)
    ax.bar(x, body_counts, width=w, label="Self-Body Collision", color="#ff7f0e", alpha=0.85)
    ax.bar(x + w, starve_counts, width=w, label="Starvation Timeout", color="#1f77b4", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title("Phase 4: Termination Cause Breakdown (out of 100 Games)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Game Count", fontsize=11)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()
    fig.savefig(output_dir / "phase4_collision_rate.png")
    plt.close(fig)

    # 6. phase4_inference_time.png
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    bars = ax.bar(names, inf_times, color=colors, alpha=0.85, edgecolor="black")
    for bar, it in zip(bars, inf_times):
        ax.annotate(f"{it:.3f} ms", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Phase 4: Average Decision Latency (Inference Time per Action)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Milliseconds (ms)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()
    fig.savefig(output_dir / "phase4_inference_time.png")
    plt.close(fig)

    # 7. phase4_hybrid_shield_analysis.png
    # Find variant E
    var_e = next((r for r in results if r["variant_id"] == "Var E"), None)
    if var_e and "shield_telemetry" in var_e:
        st = var_e["shield_telemetry"]
        fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

        # Decision breakdown
        approved = st["dqn_actions_approved"]
        rejected = st["actions_rejected"]
        ax_pie.pie([approved, rejected], labels=[f"DQN Approved ({approved:,})", f"Shield Overridden ({rejected:,})"],
                   autopct="%1.1f%%", colors=["#2ca02c", "#d62728"], explode=(0, 0.1), startangle=140)
        ax_pie.set_title(f"Hybrid Shield Intervention Rate ({st['intervention_rate_pct']}%)", fontsize=13, fontweight="bold")

        # Fallback distribution
        fallbacks = {
            "A* Food Route": st.get("fallback_astar_food", 0),
            "DQN Safe Alt": st.get("fallback_dqn_alt", 0),
            "Tail Chasing": st.get("fallback_tail_chase", 0),
            "Free Space BFS": st.get("fallback_free_space", 0),
        }
        fb_keys = list(fallbacks.keys())
        fb_vals = list(fallbacks.values())
        b = ax_bar.bar(fb_keys, fb_vals, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"], alpha=0.85)
        for bar, v in zip(b, fb_vals):
            ax_bar.annotate(f"{v:,}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
        ax_bar.set_title("Shield Fallback Mechanism Distribution", fontsize=13, fontweight="bold")
        ax_bar.set_ylabel("Intervention Count", fontsize=11)
        ax_bar.grid(True, linestyle="--", alpha=0.7)
        plt.setp(ax_bar.xaxis.get_majorticklabels(), rotation=15, ha="right")

        fig.suptitle("Phase 4: DQN + A* Hybrid Shield In-Depth Telemetry", fontsize=15, fontweight="bold")
        fig.tight_layout()
        fig.savefig(output_dir / "phase4_hybrid_shield_analysis.png")
        plt.close(fig)


def run_phase4_benchmark(games: int = 100, seed: int = 123) -> List[Dict[str, Any]]:
    """Execute complete 6-variant benchmark comparison."""
    # Resolve Model Paths
    model_a = EXPERIMENTS_DIR / "original_dqn" / "model.pth"
    if not model_a.is_file():
        model_a = BASE_DIR / "checkpoints" / "best_model.pth"

    model_b = EXPERIMENTS_DIR / "double_dueling_plus_per" / "model.pth"
    if not model_b.is_file():
        model_b = BASE_DIR / "checkpoints" / "best_model.pth"

    model_c = PHASE3_EXP_DIR / "exp_c_enhanced_state.pth"
    model_f = PHASE3_EXP_DIR / "exp_f_best_hyperparams.pth"

    variants = [
        {
            "variant_id": "Var A",
            "name": "Original DQN (Phase 1 Baseline)",
            "model_path": model_a,
            "is_pure_astar": False,
            "use_shield": False,
            "state_mode": "standard",
        },
        {
            "variant_id": "Var B",
            "name": "DDQN + Dueling + PER (Phase 2 Best)",
            "model_path": model_b,
            "is_pure_astar": False,
            "use_shield": False,
            "state_mode": "standard",
        },
        {
            "variant_id": "Var C",
            "name": "Enhanced 24-feature DQN (Phase 4 Corrected)",
            "model_path": model_c,
            "is_pure_astar": False,
            "use_shield": False,
            "state_mode": "enhanced",
        },
        {
            "variant_id": "Var D",
            "name": "Pure A* Agent (Deterministic Algorithmic)",
            "model_path": None,
            "is_pure_astar": True,
            "use_shield": False,
            "state_mode": "standard",
        },
        {
            "variant_id": "Var E",
            "name": "DQN + A* Safety Shield (Phase 4 Hybrid)",
            "model_path": model_b,
            "is_pure_astar": False,
            "use_shield": True,
            "state_mode": "standard",
        },
        {
            "variant_id": "Var F",
            "name": "Optimized DQN (Validated Hyperparameters)",
            "model_path": model_f,
            "is_pure_astar": False,
            "use_shield": False,
            "state_mode": "standard",
        },
    ]

    print("\n" + "=" * 80)
    print("      PHASE 4 SCIENTIFIC BENCHMARK & MULTI-ALGORITHM EVALUATION (6 VARIANTS)")
    print("=" * 80)
    print(f"Independent Games per Configuration : {games}")
    print(f"Evaluation Random Seed              : {seed}")
    print(f"Output Directory                    : {PHASE4_RESULTS_DIR}")
    print("=" * 80)

    cache_path = PHASE4_RESULTS_DIR / "phase4_benchmark_cache.json"
    cached_results = {}
    if cache_path.is_file():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                cached_results = {r["variant_id"]: r for r in cached_data}
            print(f"Loaded {len(cached_results)} cached variant benchmarks from: {cache_path}")
        except Exception:
            cached_results = {}

    results = []
    for var in variants:
        vid = var["variant_id"]
        if vid in cached_results and cached_results[vid].get("games_evaluated") == games:
            print(f"\n[RESUME / CACHE HIT] {vid}: {var['name']} (Mean: {cached_results[vid]['mean_score']:.2f})")
            r = cached_results[vid]
        else:
            print(f"\nEvaluating {var['variant_id']}: {var['name']} over {games} games...")
            r = evaluate_variant(
                variant_id=var["variant_id"],
                name=var["name"],
                model_path=var["model_path"],
                is_pure_astar=var["is_pure_astar"],
                use_shield=var["use_shield"],
                state_mode=var["state_mode"],
                games=games,
                seed=seed,
            )
            print(
                f"  -> Mean: {r['mean_score']:.2f} | Max: {r['max_score']} | Median: {r['median_score']:.1f} | "
                f"Std: {r['std_score']:.2f} | Avg Steps: {r['avg_steps']:.1f} | Latency: {r['avg_inference_time_ms']:.3f}ms"
            )
            cached_results[vid] = r
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(list(cached_results.values()), f, indent=4)
        results.append(r)

    # 1. Save results to JSON
    json_path = PHASE4_RESULTS_DIR / "phase4_benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved benchmark JSON to: {json_path}")

    # 2. Save results to CSV
    csv_path = PHASE4_RESULTS_DIR / "phase4_benchmark_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Variant_ID", "Name", "Architecture", "State_Mode", "Mean_Score",
            "Max_Score", "Min_Score", "Median_Score", "Std_Score", "Avg_Steps",
            "Avg_Food", "Avg_Reward", "Avg_Latency_ms", "Total_Eval_Time_s",
            "Wall_Collisions", "Body_Collisions", "Starvation_Timeouts"
        ])
        for r in results:
            writer.writerow([
                r["variant_id"],
                r["name"],
                r["architecture"],
                r["state_mode"],
                f"{r['mean_score']:.2f}",
                r["max_score"],
                r["min_score"],
                f"{r['median_score']:.2f}",
                f"{r['std_score']:.2f}",
                f"{r['avg_steps']:.1f}",
                f"{r['avg_food']:.2f}",
                f"{r['avg_reward']:.2f}",
                f"{r['avg_inference_time_ms']:.3f}",
                f"{r['total_eval_time_seconds']:.2f}",
                r["collision_stats"].get("wall", 0),
                r["collision_stats"].get("body", 0),
                r["collision_stats"].get("starvation", 0),
            ])
    print(f"Saved benchmark CSV to: {csv_path}")

    # 3. Save training summary CSV
    training_summary_csv = PHASE4_RESULTS_DIR / "phase4_training_summary.csv"
    with open(training_summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Variant_ID", "Name", "Training_Duration_s", "Final_Loss",
            "Best_Val_Score", "Best_Episode", "State_Dim", "Double_DQN", "Dueling", "PER"
        ])
        # Read from training summaries if available
        exp_c_sum = PHASE3_EXP_DIR / "exp_c_training_summary.json"
        exp_f_sum = PHASE3_EXP_DIR / "exp_f_training_summary.json"

        c_data = json.load(open(exp_c_sum)) if exp_c_sum.is_file() else {}
        f_data = json.load(open(exp_f_sum)) if exp_f_sum.is_file() else {}

        writer.writerow(["Var A", "Original DQN", "N/A (Pre-trained)", "N/A", "N/A", "60", "11", "False", "False", "False"])
        writer.writerow(["Var B", "DDQN + Dueling + PER", "N/A (Pre-trained)", "N/A", "N/A", "60", "11", "True", "True", "True"])
        writer.writerow([
            "Var C", "Enhanced 24-feature DQN",
            c_data.get("training_duration_seconds", "N/A"),
            c_data.get("final_loss", "N/A"),
            c_data.get("best_criterion_score", "N/A"),
            c_data.get("best_episode", "N/A"),
            "24", "True", "True", "True"
        ])
        writer.writerow(["Var D", "Pure A* Agent", "0 (Algorithmic)", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
        writer.writerow(["Var E", "DQN + A* Safety Shield", "0 (Hybrid Inference)", "N/A", "N/A", "60", "11", "True", "True", "True"])
        writer.writerow([
            "Var F", "Optimized DQN",
            f_data.get("training_duration_seconds", "N/A"),
            f_data.get("final_loss", "N/A"),
            f_data.get("best_criterion_score", "N/A"),
            f_data.get("best_episode", "N/A"),
            "11", "True", "True", "True"
        ])
    print(f"Saved training summary CSV to: {training_summary_csv}")

    # 4. Generate all 7 visualizations
    generate_all_visualizations(results, output_dir=PHASE4_RESULTS_DIR)
    print("Generated all 7 publication-quality visualization charts!")

    # Print summary table
    sep = "=" * 105
    print("\n" + sep)
    print("                     PHASE 4 FINAL SCIENTIFIC BENCHMARK RESULTS")
    print(sep)
    print(f"{'Variant':<8} | {'Name':<36} | {'Mean':<7} | {'Max':<5} | {'Median':<6} | {'Std':<6} | {'Avg Steps':<10} | {'Latency':<8}")
    print("-" * 105)
    for r in results:
        print(
            f"{r['variant_id']:<8} | {r['name']:<36} | {r['mean_score']:<7.2f} | {r['max_score']:<5d} | "
            f"{r['median_score']:<6.1f} | {r['std_score']:<6.2f} | {r['avg_steps']:<10.1f} | {r['avg_inference_time_ms']:<6.3f}ms"
        )
    print(sep)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 Benchmark Suite")
    parser.add_argument("--games", type=int, default=100, help="Evaluation games per variant (default: 100)")
    parser.add_argument("--seed", type=int, default=123, help="Evaluation random seed (default: 123)")
    args = parser.parse_args()

    run_phase4_benchmark(games=args.games, seed=args.seed)

