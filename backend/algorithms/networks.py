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


class UserStateEncoder(nn.Module):
    """LSTM-based encoder for user temporal behavior sequences.

    Input features per timestep:
      - action (one-hot, dim=action_dim)
      - clicked (1)
      - dwell_time (1)
      - purchased (1)
      - state (state_dim)
    """

    def __init__(self, state_dim: int = 10, action_dim: int = 100,
                 hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_dim = action_dim + 3 + state_dim

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """Encode behavior sequences.

        Args:
            sequences: (batch, seq_len, input_dim)
            lengths: (batch,) actual lengths for packed sequences

        Returns:
            encoded: (batch, hidden_size) - final hidden state
        """
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                sequences, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.lstm(packed)
        else:
            _, (hidden, _) = self.lstm(sequences)

        final_hidden = hidden[-1]
        return self.layer_norm(final_hidden)

    def encode_features(self, actions: torch.Tensor, clicked: torch.Tensor,
                        dwell_times: torch.Tensor, purchased: torch.Tensor,
                        states: torch.Tensor) -> torch.Tensor:
        """Build input features from raw behavior data.

        Args:
            actions: (batch, seq_len) int action indices
            clicked: (batch, seq_len) binary
            dwell_times: (batch, seq_len) float
            purchased: (batch, seq_len) binary
            states: (batch, seq_len, state_dim) float

        Returns:
            features: (batch, seq_len, input_dim)
        """
        batch_size, seq_len = actions.shape
        action_onehot = torch.zeros(batch_size, seq_len, self.action_dim, device=actions.device)
        action_onehot.scatter_(2, actions.unsqueeze(-1).long(), 1.0)

        features = torch.cat([
            action_onehot,
            clicked.unsqueeze(-1),
            dwell_times.unsqueeze(-1),
            purchased.unsqueeze(-1),
            states,
        ], dim=-1)
        return features


class RNNQNetwork(nn.Module):
    """Q-network that takes LSTM-encoded user state combined with current state."""

    def __init__(self, state_dim: int = 10, action_dim: int = 100,
                 lstm_hidden_size: int = 128, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        combined_dim = state_dim + lstm_hidden_size

        layers = []
        in_dim = combined_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.LayerNorm(h)])
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, user_encoding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, state_dim) current observation
            user_encoding: (batch, lstm_hidden_size) from UserStateEncoder
        """
        combined = torch.cat([state, user_encoding], dim=-1)
        return self.net(combined)
