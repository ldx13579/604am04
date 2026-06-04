import torch
import torch.nn.functional as F
import numpy as np
from backend.algorithms.networks import UserStateEncoder, RNNQNetwork
from backend.config import N_CATEGORIES, N_ITEMS


class CQL_RNN:
    """CQL combined with LSTM-encoded user state sequences.

    Uses UserStateEncoder to process temporal behavior (clicks, dwell time, purchases)
    and RNNQNetwork to produce Q-values conditioned on both current state and history.
    """

    def __init__(self, state_dim=N_CATEGORIES, action_dim=N_ITEMS,
                 alpha=1.0, gamma=0.99, lr=3e-4, hidden_dims=None,
                 lstm_hidden_size=128, lstm_num_layers=2,
                 target_update_tau=0.005, seq_len=10):
        self.gamma = gamma
        self.alpha = alpha
        self.tau = target_update_tau
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.seq_len = seq_len

        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.user_encoder = UserStateEncoder(
            state_dim=state_dim, action_dim=action_dim,
            hidden_size=lstm_hidden_size, num_layers=lstm_num_layers,
        ).to(self.device)

        self.q_network = RNNQNetwork(
            state_dim=state_dim, action_dim=action_dim,
            lstm_hidden_size=lstm_hidden_size, hidden_dims=hidden_dims,
        ).to(self.device)

        self.target_q_network = RNNQNetwork(
            state_dim=state_dim, action_dim=action_dim,
            lstm_hidden_size=lstm_hidden_size, hidden_dims=hidden_dims,
        ).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.target_encoder = UserStateEncoder(
            state_dim=state_dim, action_dim=action_dim,
            hidden_size=lstm_hidden_size, num_layers=lstm_num_layers,
        ).to(self.device)
        self.target_encoder.load_state_dict(self.user_encoder.state_dict())

        all_params = (
            list(self.q_network.parameters()) +
            list(self.user_encoder.parameters())
        )
        self.optimizer = torch.optim.Adam(all_params, lr=lr)

    def compute_loss(self, batch):
        """Compute CQL loss with LSTM-encoded user sequences.

        batch: (states, actions, rewards, next_states, dones,
                seq_actions, seq_clicked, seq_dwell, seq_purchased, seq_states, seq_lengths)
        """
        (states, actions, rewards, next_states, dones,
         seq_actions, seq_clicked, seq_dwell, seq_purchased, seq_states, seq_lengths) = [
            b.to(self.device) for b in batch
        ]

        # Encode user history with LSTM
        features = self.user_encoder.encode_features(
            seq_actions, seq_clicked, seq_dwell, seq_purchased, seq_states
        )
        user_encoding = self.user_encoder(features, seq_lengths)

        # Current Q-values
        current_q = self.q_network(states, user_encoding)
        q_values = current_q.gather(1, actions.unsqueeze(1).long()).squeeze(1)

        # Target Q-values (use target networks)
        with torch.no_grad():
            target_features = self.target_encoder.encode_features(
                seq_actions, seq_clicked, seq_dwell, seq_purchased, seq_states
            )
            target_user_encoding = self.target_encoder(target_features, seq_lengths)
            next_q = self.target_q_network(next_states, target_user_encoding)
            max_next_q = next_q.max(dim=1)[0]
            targets = rewards + self.gamma * (1.0 - dones) * max_next_q

        bellman_loss = F.mse_loss(q_values, targets)

        # CQL conservative penalty
        logsumexp_q = torch.logsumexp(current_q, dim=1).mean()
        dataset_q = q_values.mean()
        cql_penalty = logsumexp_q - dataset_q

        total_loss = bellman_loss + self.alpha * cql_penalty

        metrics = {
            "loss": total_loss.item(),
            "q_value_mean": q_values.mean().item(),
            "q_value_max": q_values.max().item(),
            "q_value_min": q_values.min().item(),
            "cql_penalty": cql_penalty.item(),
        }
        return total_loss, metrics

    def update(self, batch):
        loss, metrics = self.compute_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q_network.parameters()) + list(self.user_encoder.parameters()), 1.0
        )
        self.optimizer.step()
        self._soft_update_targets()
        return metrics

    def _soft_update_targets(self):
        for param, target_param in zip(self.q_network.parameters(), self.target_q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for param, target_param in zip(self.user_encoder.parameters(), self.target_encoder.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def get_action(self, state: np.ndarray, sequence=None) -> int:
        """Get action given current state and optional history sequence."""
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            if sequence is not None:
                user_enc = self._encode_sequence(sequence)
            else:
                user_enc = self.user_encoder.get_default_embedding(1).to(self.device)

            q_values = self.q_network(state_t, user_enc)
            return q_values.argmax(dim=1).item()

    def _encode_sequence(self, sequence: dict) -> torch.Tensor:
        """Encode a single sequence dict into user embedding."""
        seq_len = len(sequence["actions"])
        actions = torch.tensor([sequence["actions"]], dtype=torch.float32, device=self.device)
        clicked = torch.tensor([sequence["clicked"]], dtype=torch.float32, device=self.device)
        dwell = torch.tensor([sequence["dwell_times"]], dtype=torch.float32, device=self.device)
        purchased = torch.tensor([sequence["purchased"]], dtype=torch.float32, device=self.device)
        states = torch.tensor([sequence["states"]], dtype=torch.float32, device=self.device)
        lengths = torch.tensor([seq_len], dtype=torch.long, device=self.device)

        features = self.user_encoder.encode_features(actions, clicked, dwell, purchased, states)
        return self.user_encoder(features, lengths)

    def get_q_distribution(self, state: np.ndarray, sequence=None) -> np.ndarray:
        """Return Q-values for all actions (used for visualization)."""
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            if sequence is not None:
                user_enc = self._encode_sequence(sequence)
            else:
                user_enc = self.user_encoder.get_default_embedding(1).to(self.device)
            q_values = self.q_network(state_t, user_enc)
            return q_values.squeeze(0).cpu().numpy()

    def get_state_dict(self) -> dict:
        return {
            "q_network": self.q_network.state_dict(),
            "target_q_network": self.target_q_network.state_dict(),
            "user_encoder": self.user_encoder.state_dict(),
            "target_encoder": self.target_encoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict):
        self.q_network.load_state_dict(state_dict["q_network"])
        self.target_q_network.load_state_dict(state_dict["target_q_network"])
        self.user_encoder.load_state_dict(state_dict["user_encoder"])
        self.target_encoder.load_state_dict(state_dict["target_encoder"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
