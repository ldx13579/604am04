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

    Features:
      - Adaptive input gate that scales feature projections based on input statistics
      - Learned default embedding used when no history data is available
    """

    def __init__(self, state_dim: int = 10, action_dim: int = 100,
                 hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_dim = action_dim + 3 + state_dim

        self.input_projection = nn.Linear(self.input_dim, self.input_dim)
        self.adaptive_gate = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.Sigmoid(),
        )

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.default_embedding = nn.Parameter(torch.zeros(hidden_size))
        self._embedding_accum = None
        self._embedding_count = 0

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """Encode behavior sequences.

        Args:
            sequences: (batch, seq_len, input_dim)
            lengths: (batch,) actual lengths for packed sequences

        Returns:
            encoded: (batch, hidden_size) - final hidden state
        """
        gate = self.adaptive_gate(sequences)
        projected = self.input_projection(sequences) * gate

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                projected, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.lstm(packed)
        else:
            _, (hidden, _) = self.lstm(projected)

        final_hidden = hidden[-1]
        output = self.layer_norm(final_hidden)

        if self.training:
            self._update_default_embedding(output)

        return output

    def get_default_embedding(self, batch_size: int = 1) -> torch.Tensor:
        """Return the learned default user representation for missing history."""
        return self.default_embedding.unsqueeze(0).expand(batch_size, -1)

    def _update_default_embedding(self, encodings: torch.Tensor):
        """Update running mean for default embedding during training."""
        with torch.no_grad():
            batch_mean = encodings.mean(dim=0)
            if self._embedding_accum is None:
                self._embedding_accum = batch_mean.clone()
            else:
                self._embedding_accum = self._embedding_accum.to(batch_mean.device)
                self._embedding_accum = 0.99 * self._embedding_accum + 0.01 * batch_mean
            self._embedding_count += 1

            if self._embedding_count % 100 == 0:
                self.default_embedding.data.copy_(self._embedding_accum)

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
