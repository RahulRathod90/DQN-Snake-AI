"""
DQN Neural Network Architectures for Snake AI (Phase 1, 2, & 3).
Implements:
1. Linear_QNet: Standard Fully Connected DQN (Phase 1 Baseline).
2. Dueling_QNet: Dueling Deep Q-Network with decoupled Value V(s) and Advantage A(s, a) streams.
3. Dynamic Input Dimensionality Support (11-D standard, 24-D enhanced).
4. Factory function `build_network` and automatic architecture/dimension loader `load_model`.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from config import ACTION_SIZE, HIDDEN_SIZE, STATE_SIZE_STANDARD


class Linear_QNet(nn.Module):
    """
    Standard Fully-Connected Q-Network (Baseline).
    
    Architecture:
        Input: input_size state features (11 standard or 24 enhanced)
        Hidden: 256 neurons with ReLU activation
        Output: 3 action Q-values (Straight, Right, Left)
    """

    def __init__(
        self,
        input_size: int = STATE_SIZE_STANDARD,
        hidden_size: int = HIDDEN_SIZE,
        output_size: int = ACTION_SIZE,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.architecture = "standard"

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

        self._init_weights()

    def _init_weights(self) -> None:
        """He normal initialization for ReLU layer, Xavier normal for linear output layer."""
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity="relu")
        nn.init.constant_(self.linear1.bias, 0.0)

        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for actions."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        elif x.dtype != torch.float32:
            x = x.to(torch.float32)

        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_path: Union[str, Path]) -> None:
        """Save model state dict to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), str(path))

    def load(self, file_path: Union[str, Path], map_location: str = "cpu") -> None:
        """Load weights from state dict or full checkpoint."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found at: {path}")

        checkpoint = torch.load(str(path), map_location=map_location)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise ValueError(f"Unrecognized checkpoint format at: {path}")

        # Check dimension compatibility
        ckpt_input_size = state_dict["linear1.weight"].shape[1]
        if ckpt_input_size != self.input_size:
            raise ValueError(
                f"Checkpoint input dimension mismatch: model expects {self.input_size} features, "
                f"but checkpoint contains {ckpt_input_size} features. Cannot load incompatible weights."
            )

        self.load_state_dict(state_dict)


class Dueling_QNet(nn.Module):
    """
    Dueling Deep Q-Network (Wang et al., 2016).
    
    Decouples state value V(s) and action advantage A(s, a):
        - Shared Feature Extractor: Input [input_size] -> Hidden [256] ReLU
        - State-Value Stream V(s): Hidden [256] -> Hidden [128] ReLU -> Value [1]
        - Advantage Stream A(s, a): Hidden [256] -> Hidden [128] ReLU -> Advantages [3]
    
    Q-value Aggregation:
        Q(s, a) = V(s) + (A(s, a) - (1 / |A|) * sum_a' A(s, a'))
    """

    def __init__(
        self,
        input_size: int = STATE_SIZE_STANDARD,
        hidden_size: int = HIDDEN_SIZE,
        output_size: int = ACTION_SIZE,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.architecture = "dueling"

        self.feature_layer = nn.Linear(input_size, hidden_size)

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming He normal initialization for ReLU layers and Xavier for stream heads."""
        nn.init.kaiming_normal_(self.feature_layer.weight, nonlinearity="relu")
        nn.init.constant_(self.feature_layer.bias, 0.0)

        nn.init.kaiming_normal_(self.value_stream[0].weight, nonlinearity="relu")
        nn.init.constant_(self.value_stream[0].bias, 0.0)
        nn.init.xavier_normal_(self.value_stream[2].weight)
        nn.init.constant_(self.value_stream[2].bias, 0.0)

        nn.init.kaiming_normal_(self.advantage_stream[0].weight, nonlinearity="relu")
        nn.init.constant_(self.advantage_stream[0].bias, 0.0)
        nn.init.xavier_normal_(self.advantage_stream[2].weight)
        nn.init.constant_(self.advantage_stream[2].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute combined Q-values with mean advantage subtraction."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        elif x.dtype != torch.float32:
            x = x.to(torch.float32)

        squeeze_output = False
        if x.ndim == 1:
            x = x.unsqueeze(0)
            squeeze_output = True

        features = torch.relu(self.feature_layer(x))
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))

        if squeeze_output:
            q_values = q_values.squeeze(0)

        return q_values

    def save(self, file_path: Union[str, Path]) -> None:
        """Save model state dict to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), str(path))

    def load(self, file_path: Union[str, Path], map_location: str = "cpu") -> None:
        """Load weights from state dict or full checkpoint."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found at: {path}")

        checkpoint = torch.load(str(path), map_location=map_location)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise ValueError(f"Unrecognized checkpoint format at: {path}")

        ckpt_input_size = state_dict["feature_layer.weight"].shape[1]
        if ckpt_input_size != self.input_size:
            raise ValueError(
                f"Checkpoint input dimension mismatch: model expects {self.input_size} features, "
                f"but checkpoint contains {ckpt_input_size} features. Cannot load incompatible weights."
            )

        self.load_state_dict(state_dict)


def build_network(
    architecture: str = "standard",
    input_size: int = STATE_SIZE_STANDARD,
    hidden_size: int = HIDDEN_SIZE,
    output_size: int = ACTION_SIZE,
) -> nn.Module:
    """Factory function to instantiate either Linear_QNet or Dueling_QNet."""
    arch = architecture.lower()
    if arch in ["dueling", "duel", "dueling_dqn"]:
        return Dueling_QNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    elif arch in ["standard", "dqn", "linear"]:
        return Linear_QNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    else:
        raise ValueError(f"Unknown architecture '{architecture}'. Choose 'standard' or 'dueling'.")


def load_model(
    file_path: Union[str, Path],
    map_location: str = "cpu",
    expected_input_size: Optional[int] = None,
) -> Tuple[nn.Module, str, int]:
    """
    Inspect checkpoint contents, infer architecture and input dimensions dynamically,
    and instantiate the corresponding network.
    
    Returns:
        Tuple of (model: nn.Module, architecture_name: str, input_size: int)
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(str(path), map_location=map_location)

    is_dueling = False
    state_dict = checkpoint

    if isinstance(checkpoint, dict):
        if checkpoint.get("model_type") == "dueling" or checkpoint.get("dueling") is True:
            is_dueling = True

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        if any(k.startswith("value_stream") or k.startswith("advantage_stream") for k in state_dict.keys()):
            is_dueling = True

    # Infer input size from tensor shapes
    if is_dueling:
        detected_input_size = state_dict["feature_layer.weight"].shape[1]
    else:
        detected_input_size = state_dict["linear1.weight"].shape[1]

    if expected_input_size is not None and detected_input_size != expected_input_size:
        raise ValueError(
            f"Checkpoint input size mismatch for '{path.name}': "
            f"expected {expected_input_size} features, but checkpoint has {detected_input_size} features."
        )

    model = build_network(
        architecture="dueling" if is_dueling else "standard",
        input_size=detected_input_size,
    )
    model.load_state_dict(state_dict)
    model.to(map_location)
    model.eval()

    arch_name = "dueling" if is_dueling else "standard"
    return model, arch_name, detected_input_size
