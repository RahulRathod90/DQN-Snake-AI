"""
Prioritized Experience Replay (PER) for Deep Q-Networks.
Implements an efficient SumTree binary tree data structure for O(log N) updates
and importance sampling weight correction as described in Schaul et al. (2015).
"""

import math
import random
from typing import List, Optional, Tuple, Union

import numpy as np
import torch


class SumTree:
    """
    Binary SumTree data structure for Prioritized Experience Replay.
    
    In a SumTree:
        - Leaf nodes store transition priority values.
        - Internal nodes store the sum of their left and right children.
        - The root node (index 0) stores the total sum of all priorities.
    
    Array indexing:
        - Total capacity C (number of leaf nodes)
        - Tree array size: 2 * C - 1
        - Leaf indices: [C - 1, 2 * C - 2]
        - For node at index i:
            Parent: (i - 1) // 2
            Left child: 2 * i + 1
            Right child: 2 * i + 2
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.empty(capacity, dtype=object)
        self.write_ptr = 0
        self.size = 0
        self.max_priority = 1.0

    def _propagate(self, idx: int, change: float) -> None:
        """Propagate priority change up through parent nodes to the root."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, val: float) -> int:
        """Find the leaf index corresponding to a cumulative priority prefix value."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if val <= self.tree[left]:
            return self._retrieve(left, val)
        else:
            return self._retrieve(right, val - self.tree[left])

    def total_priority(self) -> float:
        """Return the sum of all priorities stored in the tree (root node value)."""
        return float(self.tree[0])

    def add(self, priority: float, data: object) -> None:
        """Insert transition with initial priority in O(log N)."""
        idx = self.write_ptr + self.capacity - 1

        self.data[self.write_ptr] = data
        self.update(idx, priority)

        self.write_ptr = (self.write_ptr + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def update(self, idx: int, priority: float) -> None:
        """Update leaf priority and propagate change in O(log N)."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

        if priority > self.max_priority:
            self.max_priority = priority

    def get(self, val: float) -> Tuple[int, float, object]:
        """
        Sample leaf transition given cumulative priority value val in [0, total_priority].
        Returns (tree_idx, priority, data).
        """
        idx = self._retrieve(0, val)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer.
    
    Transitions are sampled proportional to their temporal difference (TD) error:
        P(i) = (p_i^alpha) / sum_j (p_j^alpha)
    
    Importance-sampling weights are used to correct the sampling bias:
        w_i = (N * P(i))^(-beta) / max_j (w_j)
    """

    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_max: float = 1.0,
        beta_anneal_episodes: int = 1000,
        epsilon: float = 1e-5,
    ):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_max = beta_max
        self.beta_anneal_episodes = beta_anneal_episodes
        self.epsilon = epsilon
        self.current_episode = 0

    def push(
        self,
        state: Union[np.ndarray, List[float]],
        action: Union[int, List[int], np.ndarray],
        reward: float,
        next_state: Union[np.ndarray, List[float]],
        done: bool,
    ) -> None:
        """
        Insert new transition into the buffer.
        New experiences receive maximum priority to ensure they are visited at least once.
        """
        max_p = self.tree.max_priority
        if max_p <= 0.0:
            max_p = 1.0
        self.tree.add(max_p, (state, action, reward, next_state, done))

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor]:
        """
        Sample a mini-batch with priority proportional to TD errors.
        
        Returns:
            Tuple of:
                states: [B, STATE_SIZE] FloatTensor
                actions: [B] LongTensor
                rewards: [B] FloatTensor
                next_states: [B, STATE_SIZE] FloatTensor
                dones: [B] FloatTensor
                tree_indices: ndarray of tree indices for priority updates
                is_weights: [B] FloatTensor of normalized importance sampling weights
        """
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        tree_indices = np.empty(batch_size, dtype=np.int32)
        priorities = np.empty(batch_size, dtype=np.float64)

        total_p = self.tree.total_priority()
        if total_p <= 0.0 or math.isnan(total_p):
            total_p = 1.0

        segment = total_p / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            val = random.uniform(a, b)
            idx, priority, transition = self.tree.get(val)

            # Fallback if unpopulated leaf
            if transition is None:
                val = random.uniform(0, total_p)
                idx, priority, transition = self.tree.get(val)

            tree_indices[i] = idx
            priorities[i] = priority

            s, act, r, s_next, d = transition
            states.append(s)
            actions.append(act)
            rewards.append(r)
            next_states.append(s_next)
            dones.append(d)

        # Importance-sampling weight calculation:
        # P(i) = priority_i / total_priority
        # w_i = (N * P(i))^(-beta)
        probs = np.maximum(priorities, self.epsilon) / total_p
        n = len(self.tree)
        weights = (n * probs) ** (-self.beta)
        # Normalize weights by max weight to keep them in (0, 1]
        max_weight = np.max(weights)
        if max_weight > 0.0:
            weights = weights / max_weight
        else:
            weights = np.ones_like(weights)

        # Convert transitions to PyTorch tensors
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
        weights_t = torch.as_tensor(weights, dtype=torch.float32, device=device)

        return states_t, actions_t, rewards_t, next_states_t, dones_t, tree_indices, weights_t

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        Update transition priorities after a gradient descent step using TD errors:
            priority = (|TD_error| + epsilon)^alpha
        """
        td_errors = np.abs(np.squeeze(td_errors))
        clipped_errors = td_errors + self.epsilon
        priorities = clipped_errors ** self.alpha

        for idx, priority in zip(tree_indices, priorities):
            self.tree.update(int(idx), float(priority))

    def anneal_beta(self, episode: Optional[int] = None) -> float:
        """
        Linearly anneal beta from beta_start to beta_max as training progresses.
        """
        if episode is not None:
            self.current_episode = episode
        else:
            self.current_episode += 1

        fraction = min(1.0, self.current_episode / max(1, self.beta_anneal_episodes))
        self.beta = self.beta_start + fraction * (self.beta_max - self.beta_start)
        return self.beta

    def __len__(self) -> int:
        return len(self.tree)

