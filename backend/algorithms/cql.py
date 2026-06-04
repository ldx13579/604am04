import torch
import torch.nn.functional as F
from backend.algorithms.dqn import DQN
from backend.config import N_CATEGORIES, N_ITEMS


class CQL(DQN):
    """Conservative Q-Learning: DQN + conservative penalty to avoid OOD overestimation.

    The key addition is the CQL penalty:
        L_cql = E_s[logsumexp(Q(s, .))] - E_{(s,a)~D}[Q(s, a)]

    This pushes down Q-values for all actions uniformly, then pulls up Q-values
    for actions observed in the dataset. Net effect: OOD actions get lower Q-values.
    """

    def __init__(self, state_dim=N_CATEGORIES, action_dim=N_ITEMS,
                 alpha=1.0, gamma=0.99, lr=3e-4, hidden_dims=None,
                 target_update_tau=0.005):
        super().__init__(state_dim, action_dim, gamma, lr, hidden_dims, target_update_tau)
        self.alpha = alpha

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = [b.to(self.device) for b in batch]

        # Standard Bellman backup
        with torch.no_grad():
            next_q = self.target_network(next_states)
            max_next_q = next_q.max(dim=1)[0]
            targets = rewards + self.gamma * (1.0 - dones) * max_next_q

        current_q = self.q_network(states)
        q_values = current_q.gather(1, actions.unsqueeze(1).long()).squeeze(1)
        bellman_loss = F.mse_loss(q_values, targets)

        # CQL conservative penalty
        # logsumexp over all actions pushes all Q-values down
        logsumexp_q = torch.logsumexp(current_q, dim=1).mean()
        # Mean Q for dataset actions pulls those Q-values back up
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
