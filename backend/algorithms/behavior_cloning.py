import torch
import torch.nn.functional as F
import numpy as np
from backend.algorithms.networks import PolicyNetwork
from backend.config import N_CATEGORIES, N_ITEMS


class BehaviorCloning:
    """Behavior Cloning: supervised imitation of the behavior policy (ignores rewards)."""

    def __init__(self, state_dim=N_CATEGORIES, action_dim=N_ITEMS,
                 lr=3e-4, hidden_dims=None):
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.action_dim = action_dim

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = [b.to(self.device) for b in batch]
        logits = self.policy(states)
        loss = F.cross_entropy(logits, actions.long())

        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

        metrics = {
            "loss": loss.item(),
            "q_value_mean": entropy.item(),
            "q_value_max": probs.max(dim=1)[0].mean().item(),
            "q_value_min": probs.min(dim=1)[0].mean().item(),
            "cql_penalty": None,
        }
        return loss, metrics

    def update(self, batch):
        loss, metrics = self.compute_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return metrics

    def get_action(self, state: np.ndarray) -> int:
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.policy(state_t)
            return logits.argmax(dim=1).item()
