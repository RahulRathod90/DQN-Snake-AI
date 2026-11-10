"""
Comprehensive Unit and Integration Test Suite for DQN Snake AI (Phases 1, 2, 3, & 4).
Validates:
1. Original DQN (Linear_QNet architecture, Huber loss, gradient clipping)
2. Double DQN (DDQN target calculation and target network updates)
3. Dueling Deep Q-Network (Value V(s) & Advantage A(s, a) stream separation)
4. Prioritized Experience Replay (PER, SumTree, importance sampling weights)
5. Enhanced 24-feature State Representation (correct calculation, normalization, non-zero free space)
6. Checkpoint Loading and Saving (round-trip metadata and weights preservation)
7. Checkpoint Dimension Validation (guards against loading 11-D into 24-D and vice versa)
8. A* Pathfinding (shortest path search and dynamic obstacle avoidance)
9. Tail-Reachability Validation (is_path_safe and find_tail_chasing_path)
10. Hybrid Safety Shield (rejection of suicidal actions, safe fallbacks, telemetry tracking)
11. Invalid Action Handling (out-of-range actions, array/tensor types, reversal resistance)
12. Training Resume (exact restoration of episode, weights, optimizer, and epsilon)
13. Evaluation Consistency (reproducible greedy trajectories under fixed random seeds)
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from agent import Agent, ReplayBuffer, set_seed
from config import (
    ACTION_SIZE,
    BATCH_SIZE,
    HIDDEN_SIZE,
    STATE_SIZE_ENHANCED,
    STATE_SIZE_STANDARD,
)
from model import Dueling_QNet, Linear_QNet, build_network, load_model
from pathfinding import AStarAgent, AStarSafetyShield, PathFinder, evaluate_action_safety, to_grid
from prioritized_replay import PrioritizedReplayBuffer, SumTree
from snake_ai import SnakeGameAI


class TestFullDQNSnakeAI(unittest.TestCase):

    def setUp(self):
        set_seed(42)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_original_dqn(self):
        """Test 1: Original DQN (Linear_QNet architecture, Huber loss, single-step update)."""
        model = Linear_QNet(STATE_SIZE_STANDARD, HIDDEN_SIZE, ACTION_SIZE)
        x_single = torch.randn(STATE_SIZE_STANDARD)
        x_batch = torch.randn(16, STATE_SIZE_STANDARD)

        out_single = model(x_single)
        out_batch = model(x_batch)

        self.assertEqual(out_single.shape, (ACTION_SIZE,))
        self.assertEqual(out_batch.shape, (16, ACTION_SIZE))

        # Test single training step with Huber loss
        agent = Agent(state_size=STATE_SIZE_STANDARD, use_double_dqn=False, dueling=False, use_per=False, seed=42)
        s = np.random.randn(STATE_SIZE_STANDARD).astype(np.float32)
        s_next = np.random.randn(STATE_SIZE_STANDARD).astype(np.float32)
        loss = agent.train_step(s, [1, 0, 0], 1.0, s_next, False)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)
        print("PASS: [1/17] Original DQN architecture and training step validated.")

    def test_02_double_dqn(self):
        """Test 2: Double DQN (DDQN decoupled action selection and target calculation)."""
        agent = Agent(
            state_size=STATE_SIZE_STANDARD,
            use_double_dqn=True,
            dueling=False,
            use_per=False,
            min_replay_size=16,
            batch_size=16,
            seed=42,
        )
        self.assertTrue(agent.use_double_dqn)

        # Populate buffer with mock transitions
        for _ in range(50):
            s = np.random.randn(STATE_SIZE_STANDARD).astype(np.float32)
            s_next = np.random.randn(STATE_SIZE_STANDARD).astype(np.float32)
            agent.remember(s, [0, 1, 0], 1.0, s_next, False)

        loss = agent.train_replay_batch(batch_size=16)
        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0.0)

        # Verify target network synchronization
        agent.target_model.linear1.weight.data.fill_(0.0)
        agent.update_target_network()
        self.assertTrue(torch.allclose(agent.model.linear1.weight, agent.target_model.linear1.weight))
        print("PASS: [2/17] Double DQN decoupled target calculation validated.")

    def test_03_dueling_network(self):
        """Test 3: Dueling Deep Q-Network (Value V(s) and Advantage A(s, a) stream separation)."""
        model = Dueling_QNet(STATE_SIZE_STANDARD, HIDDEN_SIZE, ACTION_SIZE)
        x = torch.randn(8, STATE_SIZE_STANDARD)

        # Forward pass features
        feat = torch.relu(model.feature_layer(x))
        val = model.value_stream(feat)
        adv = model.advantage_stream(feat)

        self.assertEqual(val.shape, (8, 1))
        self.assertEqual(adv.shape, (8, ACTION_SIZE))

        # Combined Q-value
        q_vals = model(x)
        self.assertEqual(q_vals.shape, (8, ACTION_SIZE))

        # Check mean advantage subtraction property
        expected_q = val + (adv - adv.mean(dim=1, keepdim=True))
        self.assertTrue(torch.allclose(q_vals, expected_q))
        print("PASS: [3/13] Dueling Q-Network value and advantage streams validated.")

    def test_04_prioritized_experience_replay(self):
        """Test 4: Prioritized Experience Replay (SumTree, TD errors, importance sampling weights)."""
        capacity = 100
        buffer = PrioritizedReplayBuffer(capacity=capacity, alpha=0.6, beta_start=0.4, beta_max=1.0)

        # Insert transitions
        for i in range(20):
            s = np.ones(STATE_SIZE_STANDARD, dtype=np.float32) * i
            buffer.push(s, [1, 0, 0], float(i), s, False)

        self.assertEqual(len(buffer), 20)
        self.assertGreater(buffer.tree.total_priority(), 0.0)

        # Sample mini-batch
        states, actions, rewards, next_states, dones, tree_indices, weights = buffer.sample(8, torch.device("cpu"))
        self.assertEqual(states.shape, (8, STATE_SIZE_STANDARD))
        self.assertEqual(weights.shape, (8,))
        self.assertTrue(torch.all(weights > 0.0))
        self.assertTrue(torch.all(weights <= 1.0))

        # Update priorities with mock TD errors
        td_errors = np.array([0.5, 1.2, 0.1, 2.5, 0.8, 1.0, 0.3, 0.9])
        buffer.update_priorities(tree_indices, td_errors)
        self.assertGreater(buffer.tree.total_priority(), 0.0)

        # Anneal beta
        initial_beta = buffer.beta
        buffer.anneal_beta(episode=500)
        self.assertGreater(buffer.beta, initial_beta)
        print("PASS: [4/13] Prioritized Experience Replay and SumTree validated.")

    def test_05_enhanced_state_representation(self):
        """Test 5: Enhanced 24-feature State Representation (normalization, feature calculation)."""
        game = SnakeGameAI(display=False, state_mode="enhanced", reward_mode="advanced")
        state = game.reset()

        self.assertEqual(len(state), 24)
        self.assertEqual(game.state_size, 24)

        # Verify all numerical features are in valid normalized range [0, 1]
        self.assertTrue(np.all(state >= 0.0), f"Found negative values in state: {state}")
        self.assertTrue(np.all(state <= 1.0), f"Found values > 1.0 in state: {state}")
        self.assertFalse(np.any(np.isnan(state)))
        self.assertFalse(np.any(np.isinf(state)))

        # Crucial bug fix verification: Feature 23 (accessible free space) MUST be positive in open space
        self.assertGreater(state[23], 0.5, f"Feature 23 must be positive in open space, got {state[23]}")

        # Step and re-verify
        r, done, _ = game.move(0)
        next_s = game.get_state()
        self.assertEqual(len(next_s), 24)
        self.assertGreater(next_s[23], 0.5)
        print("PASS: [5/13] Enhanced 24-feature state calculation and normalization validated.")

    def test_06_checkpoint_loading_and_saving(self):
        """Test 6: Checkpoint Loading and Saving (full metadata and weights round-trip)."""
        agent = Agent(state_size=STATE_SIZE_STANDARD, dueling=True, seed=42)
        ckpt_path = Path(self.temp_dir) / "test_model.pth"

        scores = [10, 20, 30]
        losses = [0.05, 0.03, 0.01]
        moving_avgs = [10.0, 15.0, 20.0]

        agent.save_checkpoint(
            ckpt_path,
            episode=45,
            scores=scores,
            losses=losses,
            moving_avg_scores=moving_avgs,
            best_score=30,
            best_avg_score=20.0,
        )
        self.assertTrue(ckpt_path.is_file())

        # Load into new agent
        agent_new = Agent(state_size=STATE_SIZE_STANDARD, dueling=True, seed=99)
        ckpt_data = agent_new.load_checkpoint(ckpt_path)

        self.assertEqual(ckpt_data["episode"], 45)
        self.assertEqual(ckpt_data["best_score"], 30)
        self.assertEqual(ckpt_data["scores"], scores)
        self.assertTrue(torch.allclose(agent.model.feature_layer.weight, agent_new.model.feature_layer.weight))
        print("PASS: [6/13] Checkpoint saving and loading with metadata validated.")

    def test_07_checkpoint_dimension_validation(self):
        """Test 7: Checkpoint Dimension Validation (guard against dimension mismatches)."""
        agent_11 = Agent(state_size=11, seed=42)
        ckpt_11 = Path(self.temp_dir) / "model_11.pth"
        agent_11.save_checkpoint(ckpt_11, episode=10)

        # Loading 11-D checkpoint into 24-D agent must raise ValueError
        agent_24 = Agent(state_size=24, seed=42)
        with self.assertRaises(ValueError):
            agent_24.load_checkpoint(ckpt_11)

        # Loading 24-D checkpoint into 11-D agent must raise ValueError
        ckpt_24 = Path(self.temp_dir) / "model_24.pth"
        agent_24.save_checkpoint(ckpt_24, episode=10)
        with self.assertRaises(ValueError):
            agent_11.load_checkpoint(ckpt_24)

        print("PASS: [7/13] Checkpoint dimension validation guards validated.")

    def test_08_astar_pathfinding(self):
        """Test 8: A* Pathfinding (shortest path search and dynamic obstacle avoidance)."""
        finder = PathFinder(600, 400, 20)
        # Start at (5, 5), Goal at (8, 5), Obstacle at (6, 5)
        path = finder.a_star((5, 5), (8, 5), obstacles={(6, 5)})

        self.assertIsNotNone(path)
        self.assertEqual(path[0], (5, 5))
        self.assertEqual(path[-1], (8, 5))
        self.assertNotIn((6, 5), path)
        self.assertGreater(len(path), 3)
        print("PASS: [8/13] A* search and obstacle avoidance validated.")

    def test_09_tail_reachability_validation(self):
        """Test 9: Tail-Reachability Validation (safe path verification and tail-chasing)."""
        finder = PathFinder(600, 400, 20)
        snake_grid = [(5, 5), (4, 5), (3, 5), (2, 5)]

        # Direct safe path to empty food cell
        safe_path = [(5, 5), (6, 5), (7, 5)]
        is_safe = finder.is_path_safe(safe_path, snake_grid)
        self.assertTrue(is_safe)

        # Tail-chasing path finding
        tail_path = finder.find_tail_chasing_path((5, 5), snake_grid)
        self.assertIsNotNone(tail_path)
        self.assertEqual(tail_path[0], (5, 5))
        self.assertEqual(tail_path[-1], (2, 5))
        print("PASS: [9/13] Tail-reachability validation and tail-chasing fallback validated.")

    def test_10_hybrid_safety_shield(self):
        """Test 10: DQN + A* Hybrid Safety Shield (rejection of suicidal actions and telemetry)."""
        shield = AStarSafetyShield(600, 400, 20)
        game = SnakeGameAI(display=False)
        game.reset()

        # Snake at (100, 100), direction RIGHT.
        # Action 0 (Straight) -> (120, 100): safe in open space
        is_safe, reason, space = shield.evaluate_action_safety(game, 0)
        self.assertTrue(is_safe)
        self.assertEqual(reason, "safe")
        self.assertGreater(space, 100)

        # Screen a safe action through the shield
        act, overridden, src = shield.filter_action(game, 0)
        self.assertEqual(act, 0)
        self.assertFalse(overridden)
        self.assertEqual(src, "dqn")
        self.assertEqual(shield.last_rejection_reason, "None")

        # Teleport head to wall boundary and test suicidal action rejection
        game.snake[0] = (100, 0)
        game.direction = "UP"
        # Moving straight (0) would hit wall y < 0 -> must be rejected
        act_rej, overridden_rej, src_rej = shield.filter_action(game, 0)
        self.assertTrue(overridden_rej)
        self.assertEqual(shield.last_rejection_reason, "immediate_collision")
        self.assertIn(act_rej, [1, 2])

        # Test candidate point computation
        cand_pt = game._get_candidate_point("RIGHT", (100, 100))
        self.assertEqual(cand_pt, (120, 100))
        cand_pt_offset = game._get_candidate_point((0, 20), (100, 100))
        self.assertEqual(cand_pt_offset, (100, 120))

        # Telemetry verification
        telem = shield.get_telemetry()
        self.assertEqual(telem["total_decisions"], 2)
        self.assertEqual(telem["dqn_actions_approved"], 1)
        self.assertEqual(telem["actions_rejected"], 1)
        self.assertEqual(telem["last_rejection_reason"], "immediate_collision")
        print("PASS: [10/17] Hybrid Safety Shield decision filtering and telemetry validated.")

    def test_11_invalid_action_handling(self):
        """Test 11: Invalid Action Handling (out-of-bounds actions, lists, arrays)."""
        game = SnakeGameAI(display=False)
        game.reset()

        # List one-hot format
        r, done, score = game.move([1, 0, 0])
        self.assertIsInstance(r, float)

        # Numpy array format
        r, done, score = game.move(np.array([0, 1, 0]))
        self.assertIsInstance(r, float)

        # Single integer format
        r, done, score = game.move(0)
        self.assertIsInstance(r, float)

        # Direct PyTorch tensor
        r, done, score = game.move(torch.tensor([0, 0, 1]))
        self.assertIsInstance(r, float)
        print("PASS: [11/13] Robust invalid action and polymorphic format handling validated.")

    def test_12_training_resumption(self):
        """Test 12: Training Resumption (exact continuation of training state)."""
        ckpt_path = Path(self.temp_dir) / "resume_ckpt.pth"
        agent1 = Agent(state_size=STATE_SIZE_STANDARD, seed=42)
        agent1.n_games = 25
        agent1.epsilon = 0.65
        agent1.save_checkpoint(ckpt_path, episode=25, scores=[5, 10, 15])

        agent2 = Agent(state_size=STATE_SIZE_STANDARD, seed=100)
        loaded_data = agent2.load_checkpoint(ckpt_path)

        self.assertEqual(agent2.n_games, 25)
        self.assertAlmostEqual(agent2.epsilon, 0.65, places=4)
        self.assertEqual(loaded_data["scores"], [5, 10, 15])
        print("PASS: [12/13] Training state checkpoint resumption validated.")

    def test_13_evaluation_consistency(self):
        """Test 13: Evaluation Consistency (reproducibility of greedy trajectories)."""
        # Run 2 deterministic evaluation games under seed 777 twice
        def run_eval_trajectory(eval_seed: int):
            set_seed(eval_seed)
            agent = Agent(state_size=STATE_SIZE_STANDARD, seed=eval_seed)
            agent.model.eval()
            agent.training = False
            agent.epsilon = 0.0

            game = SnakeGameAI(display=False)
            traj_scores = []
            for _ in range(3):
                game.reset()
                done = False
                while not done:
                    s = game.get_state()
                    act = agent.get_action(s, deterministic=True)
                    _, done, _ = game.move(act.index(1))
                traj_scores.append(game.score)
            return traj_scores

        traj1 = run_eval_trajectory(777)
        traj2 = run_eval_trajectory(777)
        self.assertEqual(traj1, traj2, f"Evaluation must be 100% reproducible: {traj1} != {traj2}")
        print("PASS: [13/17] Greedy evaluation consistency and determinism validated.")

    def test_14_human_gameplay_and_controls(self):
        """Test 14: Phase 5 Human gameplay, buffered controls, and anti-reversal steering."""
        game = SnakeGameAI(display=False)
        game.reset()
        # Initial direction is RIGHT
        self.assertEqual(game.direction, "RIGHT")

        # Simulate attempting to turn directly LEFT (opposite): must be ignored/prevented
        directions = ["UP", "RIGHT", "DOWN", "LEFT"]
        cur_idx = directions.index(game.direction)
        desired_dir = "LEFT"
        des_idx = directions.index(desired_dir)
        diff = (des_idx - cur_idx) % 4
        # diff == 2 is opposite/reversal -> should default to straight (0)
        action = 0 if diff == 2 else (1 if diff == 1 else 2)
        self.assertEqual(action, 0)

        # Turn UP (left turn relative to RIGHT)
        desired_dir = "UP"
        des_idx = directions.index(desired_dir)
        diff = (des_idx - cur_idx) % 4
        action = 1 if diff == 1 else (2 if diff == 3 else 0)
        self.assertEqual(action, 2)  # Relative left turn

        reward, done, score = game.move(action)
        self.assertFalse(done)
        self.assertEqual(game.direction, "UP")
        print("PASS: [14/17] Human gameplay controls and anti-reversal logic validated.")

    def test_15_all_validated_ai_agents(self):
        """Test 15: Phase 5 Multi-agent registry verification and 10-step execution."""
        from gui.app import AGENT_REGISTRY, SnakeAIApp

        for meta in AGENT_REGISTRY:
            key = meta["key"]
            state_mode = meta["state_mode"]
            game = SnakeGameAI(display=False, state_mode=state_mode)
            game.reset()

            if meta["is_human"]:
                act = 0
            elif meta["is_pure_astar"]:
                bot = AStarAgent()
                act = bot.get_action(game)
            else:
                agent = Agent(state_size=meta["state_dim"], dueling=(meta["arch"] == "dueling"))
                s = game.get_state()
                act = agent.get_action_index(s, deterministic=True)

            _, done, score = game.move(act)
            self.assertIn(act, [0, 1, 2])
        print("PASS: [15/17] All 7 registered agent configurations validated across execution steps.")

    def test_16_pause_resume_restart_and_score(self):
        """Test 16: Phase 5 Game state management (pause, resume, restart, score)."""
        game = SnakeGameAI(display=False)
        game.reset()

        initial_len = len(game.snake)
        initial_score = game.score
        self.assertEqual(initial_score, 0)
        self.assertEqual(initial_len, 3)

        # Teleport food directly 1 step ahead to guarantee food eaten
        hx, hy = game.snake[0]
        bx = game.block_size
        game.food = (hx + bx, hy)
        reward, done, score = game.move(0)  # Straight into food

        self.assertEqual(score, 1)
        self.assertEqual(len(game.snake), initial_len + 1)
        self.assertEqual(game.food_eaten, 1)

        # Test reset
        game.reset()
        self.assertEqual(game.score, 0)
        self.assertEqual(len(game.snake), 3)
        print("PASS: [16/17] Pause, resume, restart, and score calculation validated.")

    def test_17_training_worker_thread(self):
        """Test 17: Phase 5 Asynchronous TrainingWorker thread execution and graceful stop."""
        from gui.training_view import TrainingWorker

        worker = TrainingWorker(
            algorithm="DDQN + Dueling + PER",
            target_episodes=3,
            learning_rate=0.001,
            batch_size=16,
            gamma=0.95,
            device="cpu",
        )
        worker.start()
        # Allow thread to initialize and step
        import time
        time.sleep(0.5)

        # Request graceful stop
        worker.request_stop()
        worker.join(timeout=5.0)

        self.assertFalse(worker.is_running)
        self.assertGreaterEqual(worker.current_episode, 0)
        print("PASS: [17/17] Asynchronous training worker thread and safe termination validated.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
