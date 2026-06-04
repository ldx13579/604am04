import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Q-network: maps state -> Q-values for all actions."""

    def __init__(self, state_dim: int = 10, action_dim: int = 100, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.LayerNorm(h)])
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class PolicyNetwork(nn.Module):
    """Policy network for Behavior Cloning: maps state -> action logits."""

    def __init__(self, state_dim: int = 10, action_dim: int = 100, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.LayerNorm(h)])
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

    def get_action(self, state: torch.Tensor) -> int:
        with torch.no_grad():
            logits = self.forward(state.unsqueeze(0))
            return logits.argmax(dim=1).item()
