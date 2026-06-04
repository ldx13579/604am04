import torch
import torch.nn.functional as F
import numpy as np
from typing import Callable, List
from backend.algorithms.networks import QNetwork
from backend.config import N_CATEGORIES, N_ITEMS


class PolicyValidationResult:
    """Results of pre-FQE policy quality validation."""

    def __init__(self):
        self.is_valid = True
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.metrics = {}

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.is_valid = False

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "warnings": self.warnings,
            "errors": self.errors,
            "metrics": self.metrics,
        }


class PolicyValidator:
    """Validates a fixed policy before running FQE to ensure reliable evaluation.

    Checks:
    1. Action determinism: policy produces consistent outputs for same inputs.
    2. Action coverage: policy doesn't collapse to a single action.
    3. Action validity: all actions are within the valid range.
    4. Policy responsiveness: different states produce different actions.
    5. Reward alignment: policy actions align with positive-reward transitions in data.
    """

    def __init__(self, state_dim=N_CATEGORIES, action_dim=N_ITEMS,
                 min_action_coverage=0.05,
                 min_entropy_ratio=0.1,
                 max_collapse_ratio=0.95,
                 min_responsiveness=0.1,
                 min_reward_alignment=0.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.min_action_coverage = min_action_coverage
        self.min_entropy_ratio = min_entropy_ratio
        self.max_collapse_ratio = max_collapse_ratio
        self.min_responsiveness = min_responsiveness
        self.min_reward_alignment = min_reward_alignment

    def validate(self, policy_fn: Callable, replay_buffer,
                 n_samples=1000) -> PolicyValidationResult:
        result = PolicyValidationResult()

        batch = replay_buffer.sample(min(n_samples, replay_buffer.size))
        states = batch[0].cpu().numpy()
        actions_in_data = batch[1].cpu().numpy().astype(int)
        rewards = batch[2].cpu().numpy()

        self._check_determinism(policy_fn, states, result)
        policy_actions = self._get_policy_actions(policy_fn, states)
        self._check_action_validity(policy_actions, result)
        self._check_action_coverage(policy_actions, result)
        self._check_responsiveness(policy_actions, states, result)
        self._check_reward_alignment(policy_actions, actions_in_data, rewards, result)

        return result

    def _get_policy_actions(self, policy_fn: Callable,
                            states: np.ndarray) -> np.ndarray:
        actions = []
        for state in states:
            try:
                action = policy_fn(state)
                actions.append(int(action))
            except Exception:
                actions.append(-1)
        return np.array(actions)

    def _check_determinism(self, policy_fn: Callable, states: np.ndarray,
                           result: PolicyValidationResult):
        """Verify policy produces consistent outputs for the same inputs."""
        n_check = min(50, len(states))
        inconsistent_count = 0

        for i in range(n_check):
            state = states[i]
            try:
                a1 = policy_fn(state)
                a2 = policy_fn(state)
                if a1 != a2:
                    inconsistent_count += 1
            except Exception as e:
                result.add_error(f"Policy raised exception on state {i}: {e}")
                return

        ratio = inconsistent_count / n_check
        result.metrics["determinism_ratio"] = 1.0 - ratio

        if ratio > 0.5:
            result.add_warning(
                f"Policy is highly stochastic ({ratio:.1%} inconsistent). "
                "FQE assumes deterministic policy; results may have high variance."
            )

    def _check_action_validity(self, actions: np.ndarray,
                               result: PolicyValidationResult):
        """Ensure all actions are within valid range."""
        invalid_mask = (actions < 0) | (actions >= self.action_dim)
        invalid_ratio = invalid_mask.sum() / len(actions)
        result.metrics["invalid_action_ratio"] = float(invalid_ratio)

        if invalid_ratio > 0.0:
            result.add_error(
                f"Policy produces {invalid_ratio:.1%} invalid actions "
                f"(outside [0, {self.action_dim})). Cannot evaluate."
            )

    def _check_action_coverage(self, actions: np.ndarray,
                               result: PolicyValidationResult):
        """Check that the policy doesn't collapse to too few actions."""
        valid_actions = actions[actions >= 0]
        if len(valid_actions) == 0:
            result.add_error("No valid actions produced by policy.")
            return

        unique_actions = np.unique(valid_actions)
        coverage = len(unique_actions) / self.action_dim
        result.metrics["action_coverage"] = float(coverage)

        action_counts = np.bincount(valid_actions, minlength=self.action_dim)
        action_probs = action_counts / action_counts.sum()
        nonzero_probs = action_probs[action_probs > 0]
        entropy = -np.sum(nonzero_probs * np.log(nonzero_probs + 1e-10))
        max_entropy = np.log(self.action_dim)
        entropy_ratio = entropy / max_entropy
        result.metrics["action_entropy_ratio"] = float(entropy_ratio)

        most_common_ratio = action_counts.max() / len(valid_actions)
        result.metrics["most_common_action_ratio"] = float(most_common_ratio)

        if most_common_ratio > self.max_collapse_ratio:
            result.add_warning(
                f"Policy has collapsed: {most_common_ratio:.1%} of actions "
                f"are action {action_counts.argmax()}. FQE value estimate "
                "may reflect only this single action's value."
            )

        if coverage < self.min_action_coverage:
            result.add_warning(
                f"Policy covers only {coverage:.1%} of action space "
                f"({len(unique_actions)}/{self.action_dim} actions)."
            )

        if entropy_ratio < self.min_entropy_ratio:
            result.add_warning(
                f"Action entropy is very low ({entropy_ratio:.3f}). "
                "Policy may be near-deterministic on a small subset."
            )

    def _check_responsiveness(self, actions: np.ndarray, states: np.ndarray,
                              result: PolicyValidationResult):
        """Check that different states lead to different actions."""
        n_states = len(states)
        if n_states < 10:
            return

        n_pairs = min(200, n_states * (n_states - 1) // 2)
        idx_pairs = np.random.choice(n_states, size=(n_pairs, 2), replace=True)
        different_state_pairs = np.linalg.norm(
            states[idx_pairs[:, 0]] - states[idx_pairs[:, 1]], axis=1
        ) > 0.01

        valid_pairs = different_state_pairs.sum()
        if valid_pairs == 0:
            return

        different_actions = actions[idx_pairs[:, 0]] != actions[idx_pairs[:, 1]]
        responsive_pairs = (different_state_pairs & different_actions).sum()
        responsiveness = responsive_pairs / max(valid_pairs, 1)
        result.metrics["responsiveness"] = float(responsiveness)

        if responsiveness < self.min_responsiveness:
            result.add_warning(
                f"Policy responsiveness is low ({responsiveness:.1%}). "
                "Policy may not be meaningfully differentiating between states."
            )

    def _check_reward_alignment(self, policy_actions: np.ndarray,
                                data_actions: np.ndarray, rewards: np.ndarray,
                                result: PolicyValidationResult):
        """Check if policy actions align with rewarded actions in the dataset."""
        positive_reward_mask = rewards > 0
        if positive_reward_mask.sum() == 0:
            result.metrics["reward_alignment"] = 0.0
            return

        matching_positive = (
            (policy_actions == data_actions) & positive_reward_mask
        ).sum()
        alignment = matching_positive / positive_reward_mask.sum()
        result.metrics["reward_alignment"] = float(alignment)

        random_baseline = 1.0 / self.action_dim
        if alignment < random_baseline:
            result.add_warning(
                f"Policy-reward alignment ({alignment:.3f}) is below random "
                f"baseline ({random_baseline:.3f}). Policy may be poorly trained."
            )

        if alignment < self.min_reward_alignment:
            result.add_warning(
                f"Reward alignment {alignment:.3f} below threshold "
                f"{self.min_reward_alignment:.3f}. FQE results may be unreliable."
            )


class FittedQEvaluation:
    """Fitted Q Evaluation: estimate expected return of a fixed policy offline.

    Unlike DQN which uses max_a Q(s', a), FQE uses Q(s', pi(s')) where pi is
    the fixed policy being evaluated. This gives an unbiased estimate of the
    policy's value without online deployment.

    Includes pre-evaluation policy validation to ensure result reliability.
    """

    def __init__(self, policy_fn: Callable, state_dim=N_CATEGORIES,
                 action_dim=N_ITEMS, gamma=0.99, lr=1e-3,
                 hidden_dims=None, target_update_tau=0.005,
                 validate_policy=True):
        self.policy_fn = policy_fn
        self.gamma = gamma
        self.tau = target_update_tau
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.validate_policy = validate_policy
        self.validation_result: PolicyValidationResult = None

        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self._policy_validator = PolicyValidator(state_dim, action_dim)

    def validate_before_evaluation(self, replay_buffer,
                                   n_samples=1000) -> PolicyValidationResult:
        """Run policy quality checks before starting FQE."""
        self.validation_result = self._policy_validator.validate(
            self.policy_fn, replay_buffer, n_samples
        )
        return self.validation_result

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = [b.to(self.device) for b in batch]

        with torch.no_grad():
            next_actions = self._get_policy_actions(next_states)
            next_q = self.target_network(next_states)
            next_q_pi = next_q.gather(1, next_actions.unsqueeze(1).long()).squeeze(1)
            targets = rewards + self.gamma * (1.0 - dones) * next_q_pi

        current_q = self.q_network(states)
        q_values = current_q.gather(1, actions.unsqueeze(1).long()).squeeze(1)
        loss = F.mse_loss(q_values, targets)

        metrics = {
            "fqe_loss": loss.item(),
            "estimated_value": q_values.mean().item(),
        }
        return loss, metrics

    def update(self, batch) -> dict:
        loss, metrics = self.compute_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._soft_update_target()
        return metrics

    def _soft_update_target(self):
        for param, target_param in zip(self.q_network.parameters(), self.target_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def _get_policy_actions(self, states: torch.Tensor) -> torch.Tensor:
        states_np = states.cpu().numpy()
        actions = []
        for state in states_np:
            action = self.policy_fn(state)
            actions.append(action)
        return torch.tensor(actions, dtype=torch.long, device=self.device)

    def evaluate(self, replay_buffer, epochs=50, steps_per_epoch=500,
                 batch_size=256, skip_validation=False) -> List[dict]:
        if self.validate_policy and not skip_validation:
            validation = self.validate_before_evaluation(replay_buffer)
            if not validation.is_valid:
                return [{
                    "epoch": 0,
                    "fqe_loss": 0.0,
                    "estimated_value": 0.0,
                    "validation_failed": True,
                    "validation_errors": validation.errors,
                }]

        results = []

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            epoch_value = 0.0
            n_steps = 0

            for _ in range(steps_per_epoch):
                batch = replay_buffer.sample(batch_size)
                metrics = self.update(batch)
                epoch_loss += metrics["fqe_loss"]
                epoch_value += metrics["estimated_value"]
                n_steps += 1

            avg_loss = epoch_loss / max(n_steps, 1)
            eval_value = self.get_estimated_value(replay_buffer, batch_size)

            results.append({
                "epoch": epoch,
                "fqe_loss": avg_loss,
                "estimated_value": eval_value,
            })

        return results

    def get_estimated_value(self, replay_buffer, batch_size=256) -> float:
        batch = replay_buffer.sample(batch_size)
        states = batch[0].to(self.device)

        with torch.no_grad():
            actions = self._get_policy_actions(states)
            q_values = self.q_network(states)
            q_pi = q_values.gather(1, actions.unsqueeze(1).long()).squeeze(1)
            return q_pi.mean().item()
