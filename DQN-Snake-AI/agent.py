"""
Advanced Deep Q-Network Agent Implementation for Snake AI (Phase 1, 2, & 3).
Supports:
1. Standard DQN (Phase 1 Baseline)
2. Double Deep Q-Network (DDQN)
3. Dueling Deep Q-Network
4. Prioritized Experience Replay (PER) with SumTree and Importance Sampling
5. Dynamic Input Dimensionality (11-D standard, 24-D enhanced)
6. Optional A* Safety Shield & Trap Avoidance Guidance
"""

import os
import random
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    ACTION_SIZE,
    BATCH_SIZE,
    EPSILON_DECAY,
    EPSILON_MIN,
    EPSILON_START,
    GAMMA,
    HIDDEN_SIZE,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    MEMORY_CAPACITY,
    MIN_REPLAY_SIZE,
    PER_ALPHA,
    PER_BETA_MAX,
    PER_BETA_START,
    PER_EPSILON,
    RANDOM_SEED,
    STATE_SIZE_STANDARD,
    TARGET_UPDATE_FREQ,
    USE_DOUBLE_DQN,
    USE_DUELING,
    USE_PER,
)
from model import Dueling_QNet, Linear_QNet, build_network
from prioritized_replay import PrioritizedReplayBuffer


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds across python, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ReplayBuffer:
    """
    Standard Uniform Experience Replay Buffer using deque for O(1) appends.
    Directly converts sampled batches into PyTorch tensors.
    """

    def __init__(self, capacity: int = MEMORY_CAPACITY):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: Union[np.ndarray, List[float]],
        action: Union[int, List[int], np.ndarray],
        reward: float,
        next_state: Union[np.ndarray, List[float]],
        done: bool,
    ) -> None:
        """Store a single transition in memory."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a uniform random mini-batch as PyTorch tensors."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_arr = np.array(states, dtype=np.float32)
        next_states_arr = np.array(next_states, dtype=np.float32)
        rewards_arr = np.array(rewards, dtype=np.float32)
        dones_arr = np.array(dones, dtype=np.float32)

        actions_arr = np.array(actions)
        if actions_arr.ndim > 1 and actions_arr.shape[1] > 1:
            action_indices = np.argmax(actions_arr, axis=1).astype(np.int64)
        else:
            action_indices = actions_arr.squeeze().astype(np.int64)

        states_t = torch.as_tensor(states_arr, dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(action_indices, dtype=torch.int64, device=device)
        rewards_t = torch.as_tensor(rewards_arr, dtype=torch.float32, device=device)
        next_states_t = torch.as_tensor(next_states_arr, dtype=torch.float32, device=device)
        dones_t = torch.as_tensor(dones_arr, dtype=torch.float32, device=device)

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self) -> int:
        return len(self.buffer)


class Agent:
    """
    Advanced Deep Q-Network Agent supporting:
        - Standard DQN & Double DQN (DDQN)
        - Standard Linear & Dueling architectures
        - Uniform & Prioritized (PER) Replay
        - Dynamic State Dimensions (11 standard or 24 enhanced)
        - Optional A* Safety Shield & Trap Avoidance Guidance
    """

    def __init__(
        self,
        state_size: int = STATE_SIZE_STANDARD,
        hidden_size: int = HIDDEN_SIZE,
        action_size: int = ACTION_SIZE,
        lr: float = LEARNING_RATE,
        gamma: float = GAMMA,
        epsilon_start: float = EPSILON_START,
        epsilon_min: float = EPSILON_MIN,
        epsilon_decay: float = EPSILON_DECAY,
        batch_size: int = BATCH_SIZE,
        memory_capacity: int = MEMORY_CAPACITY,
        min_replay_size: int = MIN_REPLAY_SIZE,
        target_update_freq: int = TARGET_UPDATE_FREQ,
        max_grad_norm: float = MAX_GRAD_NORM,
        use_double_dqn: bool = USE_DOUBLE_DQN,
        dueling: bool = USE_DUELING,
        use_per: bool = USE_PER,
        per_alpha: float = PER_ALPHA,
        per_beta: float = PER_BETA_START,
        per_beta_max: float = PER_BETA_MAX,
        per_beta_anneal_episodes: int = 1000,
        per_epsilon: float = PER_EPSILON,
        device: Optional[str] = None,
        seed: int = RANDOM_SEED,
    ):
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.min_replay_size = min_replay_size
        self.target_update_freq = target_update_freq
        self.max_grad_norm = max_grad_norm

        self.use_double_dqn = use_double_dqn
        self.dueling = dueling
        self.use_per = use_per

        self.n_games = 0
        self.training_steps = 0
        self.training = True

        set_seed(seed)

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        arch = "dueling" if self.dueling else "standard"
        self.model = build_network(arch, state_size, hidden_size, action_size).to(self.device)
        self.target_model = build_network(arch, state_size, hidden_size, action_size).to(self.device)
        self.update_target_network()
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()
        self.criterion_none = nn.SmoothL1Loss(reduction="none")

        if self.use_per:
            self.memory = PrioritizedReplayBuffer(
                capacity=memory_capacity,
                alpha=per_alpha,
                beta_start=per_beta,
                beta_max=per_beta_max,
                beta_anneal_episodes=per_beta_anneal_episodes,
                epsilon=per_epsilon,
            )
        else:
            self.memory = ReplayBuffer(capacity=memory_capacity)

    def update_target_network(self) -> None:
        """Synchronize target network weights with online policy network."""
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action_and_q_values(
        self,
        state: Union[np.ndarray, List[float]],
        deterministic: bool = False,
    ) -> Tuple[List[int], List[float]]:
        """
        Compute predicted Q-values and selected action.
        Useful for visualization and telemetry overlays.
        """
        final_move = [0, 0, 0]
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            q_values_tensor = self.model(state_tensor)
            q_values = q_values_tensor.squeeze(0).tolist()
            greedy_move = torch.argmax(q_values_tensor, dim=1).item()

        if self.training and not deterministic and random.random() < self.epsilon:
            move = random.randint(0, self.action_size - 1)
        else:
            move = greedy_move

        final_move[move] = 1
        return final_move, q_values

    def get_action(
        self,
        state: Union[np.ndarray, List[float]],
        deterministic: bool = False,
        astar_shield: bool = False,
        game: Optional[Any] = None,
        shield: Optional[Any] = None,
    ) -> List[int]:
        """
        Choose action using epsilon-greedy exploration or deterministic greedy selection.
        Optionally applies multi-tiered A* safety shielding to prevent collisions and traps.
        """
        final_move = [0, 0, 0]
        q_vals_list = None

        # 1. Base DQN decision
        if self.training and not deterministic and random.random() < self.epsilon:
            move = random.randint(0, self.action_size - 1)
        else:
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            if state_tensor.ndim == 1:
                state_tensor = state_tensor.unsqueeze(0)

            with torch.no_grad():
                q_values = self.model(state_tensor)
                move = torch.argmax(q_values, dim=1).item()
                q_vals_list = q_values.squeeze(0).tolist()

        # 2. Multi-tiered A* Safety Shield Guidance
        if astar_shield and game is not None:
            if shield is None:
                if not hasattr(self, "_default_shield"):
                    from pathfinding import AStarSafetyShield
                    self._default_shield = AStarSafetyShield(game.width, game.height, game.block_size)
                shield = self._default_shield

            move, _, _ = shield.filter_action(game, move, q_values=q_vals_list)

        final_move[move] = 1
        return final_move

    def get_action_index(
        self,
        state: Union[np.ndarray, List[float]],
        deterministic: bool = False,
        astar_shield: bool = False,
        game: Optional[Any] = None,
        shield: Optional[Any] = None,
    ) -> int:
        """Return chosen action integer index directly."""
        action_one_hot = self.get_action(
            state,
            deterministic=deterministic,
            astar_shield=astar_shield,
            game=game,
            shield=shield,
        )
        return action_one_hot.index(1)

    def get_action_with_telemetry(
        self,
        state: Union[np.ndarray, List[float]],
        game: Any,
        shield: Any,
        deterministic: bool = True,
    ) -> Tuple[int, bool, str]:
        """
        Choose action and return full shield decision telemetry.
        Returns:
            (action_index: int, was_overridden: bool, decision_source: str)
        """
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(state_tensor)
            dqn_move = torch.argmax(q_values, dim=1).item()
            q_vals_list = q_values.squeeze(0).tolist()

        return shield.filter_action(game, dqn_move, q_values=q_vals_list)

    def decay_epsilon(self) -> None:
        """Decay exploration rate per episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def remember(
        self,
        state: Union[np.ndarray, List[float]],
        action: Union[int, List[int], np.ndarray],
        reward: float,
        next_state: Union[np.ndarray, List[float]],
        done: bool,
    ) -> None:
        """Push transition into replay memory."""
        self.memory.push(state, action, reward, next_state, done)

    def train_replay_batch(self, batch_size: Optional[int] = None) -> Optional[float]:
        """
        Perform one mini-batch DQN update using:
            - Standard DQN or Double DQN (DDQN)
            - Standard Replay or Prioritized Experience Replay (PER)
            - Huber Loss with Gradient Clipping
        """
        batch_size = batch_size or self.batch_size
        if len(self.memory) < self.min_replay_size:
            return None

        tree_indices = None
        is_weights = None

        if self.use_per:
            states, actions, rewards, next_states, dones, tree_indices, is_weights = (
                self.memory.sample(batch_size, self.device)
            )
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(
                batch_size, self.device
            )

        current_q = self.model(states)
        state_action_values = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.use_double_dqn:
                best_next_actions = self.model(next_states).max(dim=1)[1]
                next_q = self.target_model(next_states).gather(
                    1, best_next_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                next_q = self.target_model(next_states).max(dim=1)[0]

            target_q_values = rewards + (1.0 - dones) * (self.gamma * next_q)

        if self.use_per and is_weights is not None:
            td_errors = torch.abs(state_action_values - target_q_values)
            loss_elements = self.criterion_none(state_action_values, target_q_values)
            loss = (is_weights * loss_elements).mean()

            self.memory.update_priorities(tree_indices, td_errors.detach().cpu().numpy())
            self.memory.anneal_beta()
        else:
            loss = self.criterion(state_action_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.update_target_network()

        return float(loss.item())

    def train_step(
        self,
        state: Any,
        action: Any,
        reward: Any,
        next_state: Any,
        done: Any,
    ) -> float:
        """Direct single-transition or batch training step for backward compatibility."""
        states_arr = np.array(state, dtype=np.float32)
        next_states_arr = np.array(next_state, dtype=np.float32)
        actions_arr = np.array(action)
        rewards_arr = np.array(reward, dtype=np.float32)
        dones_arr = np.array(done, dtype=np.float32)

        if states_arr.ndim == 1:
            states_arr = np.expand_dims(states_arr, 0)
            next_states_arr = np.expand_dims(next_states_arr, 0)
            actions_arr = np.expand_dims(actions_arr, 0)
            rewards_arr = np.expand_dims(rewards_arr, 0)
            dones_arr = np.expand_dims(dones_arr, 0)

        if actions_arr.ndim > 1 and actions_arr.shape[1] > 1:
            action_indices = np.argmax(actions_arr, axis=1).astype(np.int64)
        else:
            action_indices = actions_arr.squeeze().astype(np.int64)

        states_t = torch.as_tensor(states_arr, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states_arr, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(action_indices, dtype=torch.int64, device=self.device)
        rewards_t = torch.as_tensor(rewards_arr, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones_arr, dtype=torch.float32, device=self.device)

        if actions_t.ndim == 0:
            actions_t = actions_t.unsqueeze(0)
            rewards_t = rewards_t.unsqueeze(0)
            dones_t = dones_t.unsqueeze(0)

        current_q = self.model(states_t)
        state_action_values = current_q.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.use_double_dqn:
                best_next_actions = self.model(next_states_t).max(dim=1)[1]
                next_q = self.target_model(next_states_t).gather(
                    1, best_next_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                next_q = self.target_model(next_states_t).max(dim=1)[0]
            target_q_values = rewards_t + (1.0 - dones_t) * (self.gamma * next_q)

        loss = self.criterion(state_action_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.update_target_network()

        return float(loss.item())

    def train_long_memory(self) -> Optional[float]:
        """Legacy wrapper method."""
        return self.train_replay_batch()

    def save_checkpoint(
        self,
        file_path: Union[str, Path],
        episode: int,
        scores: Optional[List[int]] = None,
        losses: Optional[List[float]] = None,
        moving_avg_scores: Optional[List[float]] = None,
        best_score: int = 0,
        best_avg_score: float = 0.0,
    ) -> None:
        """Save full training checkpoint with architecture, dimension, and algorithm metadata."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "episode": episode,
            "training_steps": self.training_steps,
            "state_size": self.state_size,
            "state_mode": "enhanced" if self.state_size == 24 else "standard",
            "model_type": "dueling" if self.dueling else "standard",
            "dueling": self.dueling,
            "use_double_dqn": self.use_double_dqn,
            "use_per": self.use_per,
            "model_state_dict": self.model.state_dict(),
            "target_model_state_dict": self.target_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "scores": scores or [],
            "losses": losses or [],
            "moving_avg_scores": moving_avg_scores or [],
            "best_score": best_score,
            "best_avg_score": best_avg_score,
        }
        torch.save(checkpoint, str(path))

    def load_checkpoint(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load checkpoint and validate architecture dimensions against current agent."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        checkpoint = torch.load(str(path), map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # Check state_size compatibility
            ckpt_state_size = checkpoint.get("state_size")
            if ckpt_state_size is None:
                # Infer from state_dict
                if "feature_layer.weight" in checkpoint["model_state_dict"]:
                    ckpt_state_size = checkpoint["model_state_dict"]["feature_layer.weight"].shape[1]
                elif "linear1.weight" in checkpoint["model_state_dict"]:
                    ckpt_state_size = checkpoint["model_state_dict"]["linear1.weight"].shape[1]
                else:
                    ckpt_state_size = self.state_size

            if ckpt_state_size != self.state_size:
                raise ValueError(
                    f"Checkpoint architecture mismatch: Agent expects {self.state_size} features "
                    f"({'enhanced' if self.state_size == 24 else 'standard'}), "
                    f"but checkpoint was trained with {ckpt_state_size} features. "
                    f"Cannot load incompatible model weights."
                )

            is_dueling_ckpt = checkpoint.get("dueling", False) or checkpoint.get("model_type") == "dueling"
            if is_dueling_ckpt != self.dueling:
                print(f"Adapting agent architecture to: {'Dueling' if is_dueling_ckpt else 'Standard'}")
                self.dueling = is_dueling_ckpt
                arch = "dueling" if self.dueling else "standard"
                self.model = build_network(arch, self.state_size, self.hidden_size, self.action_size).to(self.device)
                self.target_model = build_network(arch, self.state_size, self.hidden_size, self.action_size).to(self.device)
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            if "target_model_state_dict" in checkpoint:
                self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
            else:
                self.update_target_network()

            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            self.epsilon = checkpoint.get("epsilon", self.epsilon)
            self.training_steps = checkpoint.get("training_steps", 0)
            self.n_games = checkpoint.get("episode", 0)
            self.use_double_dqn = checkpoint.get("use_double_dqn", self.use_double_dqn)
            self.use_per = checkpoint.get("use_per", self.use_per)
            return checkpoint
        elif isinstance(checkpoint, dict):
            # Check raw state dict input dimension and architecture
            is_raw_dueling = "feature_layer.weight" in checkpoint or "value_stream.0.weight" in checkpoint
            first_layer_key = "feature_layer.weight" if is_raw_dueling else "linear1.weight"
            if first_layer_key in checkpoint and checkpoint[first_layer_key].shape[1] != self.state_size:
                raise ValueError(
                    f"Raw weights input size ({checkpoint[first_layer_key].shape[1]}) does not match "
                    f"agent state_size ({self.state_size})."
                )

            # Auto-adapt model architecture if needed
            if is_raw_dueling != self.dueling:
                self.dueling = is_raw_dueling
                arch = "dueling" if self.dueling else "standard"
                self.model = build_network(arch, self.state_size, self.hidden_size, self.action_size).to(self.device)
                self.target_model = build_network(arch, self.state_size, self.hidden_size, self.action_size).to(self.device)
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

            self.model.load_state_dict(checkpoint)
            self.update_target_network()
            return {"episode": 0, "scores": [], "losses": []}
        else:
            raise ValueError(f"Unrecognized checkpoint format at: {path}")


# Semantic alias for RL Agent
DQNAgent = Agent
