import torch
import torch.nn.functional as F
import numpy as np
from backend.algorithms.networks import QNetwork
from backend.config import N_CATEGORIES, N_ITEMS


class DQN:
    """Standard DQN for offline RL (no conservative penalty)."""

    def __init__(self, state_dim=N_CATEGORIES, action_dim=N_ITEMS,
                 gamma=0.99, lr=3e-4, hidden_dims=None, target_update_tau=0.005):
        self.gamma = gamma
        self.tau = target_update_tau
        self.action_dim = action_dim

        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = [b.to(self.device) for b in batch]

        with torch.no_grad():
            next_q = self.target_network(next_states)
            max_next_q = next_q.max(dim=1)[0]
            targets = rewards + self.gamma * (1.0 - dones) * max_next_q

        current_q = self.q_network(states)
        q_values = current_q.gather(1, actions.unsqueeze(1).long()).squeeze(1)
        bellman_loss = F.mse_loss(q_values, targets)

        metrics = {
            "loss": bellman_loss.item(),
            "q_value_mean": q_values.mean().item(),
            "q_value_max": q_values.max().item(),
            "q_value_min": q_values.min().item(),
            "cql_penalty": 0.0,
        }
        return bellman_loss, metrics

    def update(self, batch):
        loss, metrics = self.compute_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._soft_update_target()
        return metrics

    def _soft_update_target(self):
        for param, target_param in zip(self.q_network.parameters(), self.target_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def get_action(self, state: np.ndarray) -> int:
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_network(state_t)
            return q_values.argmax(dim=1).item()
