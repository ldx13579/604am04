import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from backend.algorithms.cql import CQL
from backend.config import N_CATEGORIES, N_ITEMS


class EnsembleCQL:
    """Ensemble of CQL models with correlation-aware uncertainty, adaptive exploration,
    dynamic model count, and on-demand loading.

    Key improvements over naive ensemble:
    - Correlation-adjusted uncertainty: accounts for inter-model covariance to avoid
      overestimating uncertainty when models agree for correlated reasons.
    - UCB-inspired exploration: replaces pure random with uncertainty-weighted softmax,
      capped by an exploration budget to prevent performance degradation.
    - Dynamic ensemble sizing: prunes redundant (highly correlated) models and can
      activate dormant models when uncertainty is persistently high.
    - On-demand loading: models are lazily initialized and can be evicted from GPU
      when not actively needed.
    """

    def __init__(self, state_dim=N_CATEGORIES, action_dim=N_ITEMS,
                 alpha=1.0, gamma=0.99, lr=3e-4, hidden_dims=None,
                 target_update_tau=0.005, n_models=5,
                 uncertainty_threshold=1.0,
                 exploration_budget=0.3,
                 correlation_threshold=0.95,
                 min_active_models=3,
                 max_models=7,
                 ucb_coefficient=1.0,
                 lazy_load=True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.gamma = gamma
        self.lr = lr
        self.hidden_dims = hidden_dims
        self.target_update_tau = target_update_tau
        self.n_models = n_models
        self.uncertainty_threshold = uncertainty_threshold
        self.exploration_budget = exploration_budget
        self.correlation_threshold = correlation_threshold
        self.min_active_models = min_active_models
        self.max_models = max_models
        self.ucb_coefficient = ucb_coefficient
        self.lazy_load = lazy_load

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model_configs = []
        self.models: List[Optional[CQL]] = []
        self._model_loaded: List[bool] = []
        self._active_mask: List[bool] = []

        for i in range(n_models):
            self._model_configs.append({
                "state_dim": state_dim, "action_dim": action_dim,
                "alpha": alpha, "gamma": gamma, "lr": lr,
                "hidden_dims": hidden_dims, "target_update_tau": target_update_tau,
            })
            if lazy_load:
                self.models.append(None)
                self._model_loaded.append(False)
            else:
                self.models.append(self._create_model(i))
                self._model_loaded.append(True)
            self._active_mask.append(True)

        if lazy_load:
            for i in range(min(min_active_models, n_models)):
                self._ensure_loaded(i)

        self._exploration_count = 0
        self._total_action_count = 0
        self._correlation_matrix: Optional[np.ndarray] = None
        self._correlation_update_interval = 50
        self._steps_since_correlation_update = 0
        self._q_history: List[List[np.ndarray]] = [[] for _ in range(n_models)]
        self._q_history_maxlen = 100
        self._exploration_decay = 1.0
        self._exploration_decay_rate = 0.995

    def _create_model(self, idx: int) -> CQL:
        cfg = self._model_configs[idx]
        return CQL(
            state_dim=cfg["state_dim"], action_dim=cfg["action_dim"],
            alpha=cfg["alpha"], gamma=cfg["gamma"], lr=cfg["lr"],
            hidden_dims=cfg["hidden_dims"], target_update_tau=cfg["target_update_tau"],
        )

    def _ensure_loaded(self, idx: int):
        if not self._model_loaded[idx]:
            self.models[idx] = self._create_model(idx)
            self._model_loaded[idx] = True

    def _evict_model(self, idx: int):
        if self._model_loaded[idx] and not self._active_mask[idx]:
            self.models[idx] = None
            self._model_loaded[idx] = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _get_active_indices(self) -> List[int]:
        return [i for i in range(self.n_models) if self._active_mask[i]]

    def _get_active_models(self) -> List[CQL]:
        indices = self._get_active_indices()
        for idx in indices:
            self._ensure_loaded(idx)
        return [self.models[i] for i in indices]

    # ===== Correlation-Aware Uncertainty =====

    def _update_correlation_matrix(self, q_values_stacked: torch.Tensor):
        """Compute pairwise correlation between models based on Q-value predictions."""
        n_active = q_values_stacked.shape[0]
        if n_active < 2:
            self._correlation_matrix = np.eye(n_active)
            return

        q_flat = q_values_stacked.reshape(n_active, -1).cpu().numpy()
        q_centered = q_flat - q_flat.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(q_centered, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        q_normalized = q_centered / norms
        self._correlation_matrix = q_normalized @ q_normalized.T

    def _compute_correlation_adjusted_uncertainty(
        self, q_values_stacked: torch.Tensor
    ) -> Tuple[float, float]:
        """Compute uncertainty adjusted for inter-model correlation.

        Standard ensemble uncertainty (std) overestimates true epistemic uncertainty
        when models are correlated. We adjust by computing the effective number of
        independent models: n_eff = n^2 / sum(|corr_ij|), then scale std by
        sqrt(n/n_eff) to account for reduced effective diversity.
        """
        n_models = q_values_stacked.shape[0]
        raw_std = q_values_stacked.std(dim=0)

        if self._correlation_matrix is None or n_models < 2:
            return raw_std.mean().item(), raw_std.max().item()

        corr = self._correlation_matrix[:n_models, :n_models]
        abs_corr_sum = np.abs(corr).sum()
        n_eff = max((n_models ** 2) / max(abs_corr_sum, 1e-8), 1.0)
        n_eff = min(n_eff, float(n_models))

        correction_factor = np.sqrt(n_models / n_eff)
        adjusted_std = raw_std * correction_factor

        return adjusted_std.mean().item(), adjusted_std.max().item()

    # ===== Optimized Exploration Strategy =====

    def _ucb_exploration_action(self, state_t: torch.Tensor,
                                q_values_stacked: torch.Tensor) -> int:
        """UCB-inspired exploration: select action balancing mean Q and uncertainty.

        Instead of pure random exploration, uses softmax over:
            score(a) = mean_Q(a) + ucb_coefficient * std_Q(a)
        This explores uncertain actions while still preferring high-value ones.
        """
        mean_q = q_values_stacked.mean(dim=0).squeeze(0)
        std_q = q_values_stacked.std(dim=0).squeeze(0)

        if self._correlation_matrix is not None and q_values_stacked.shape[0] >= 2:
            n_models = q_values_stacked.shape[0]
            abs_corr_sum = np.abs(self._correlation_matrix[:n_models, :n_models]).sum()
            n_eff = max((n_models ** 2) / max(abs_corr_sum, 1e-8), 1.0)
            correction = np.sqrt(n_models / n_eff)
            std_q = std_q * correction

        scores = mean_q + self.ucb_coefficient * std_q

        temperature = max(0.1, std_q.mean().item())
        probs = F.softmax(scores / temperature, dim=-1)
        action = torch.multinomial(probs, 1).item()
        return action

    def _should_explore(self, uncertainty: float) -> bool:
        """Determine whether to explore, respecting exploration budget."""
        if uncertainty <= self.uncertainty_threshold:
            return False

        current_ratio = self._get_exploration_ratio()
        if current_ratio >= self.exploration_budget:
            return False

        explore_prob = min(
            (uncertainty / self.uncertainty_threshold - 1.0) * self._exploration_decay,
            1.0
        )
        self._exploration_decay *= self._exploration_decay_rate
        self._exploration_decay = max(self._exploration_decay, 0.1)

        return np.random.random() < explore_prob

    # ===== Dynamic Ensemble Sizing =====

    def _evaluate_model_redundancy(self):
        """Check if any active models are too correlated and can be deactivated."""
        if self._correlation_matrix is None:
            return

        active_indices = self._get_active_indices()
        n_active = len(active_indices)
        if n_active <= self.min_active_models:
            return

        redundant = set()
        for i in range(n_active):
            if active_indices[i] in redundant:
                continue
            for j in range(i + 1, n_active):
                if active_indices[j] in redundant:
                    continue
                corr = abs(self._correlation_matrix[i, j])
                if corr > self.correlation_threshold:
                    redundant.add(active_indices[j])
                    if n_active - len(redundant) <= self.min_active_models:
                        break
            if n_active - len(redundant) <= self.min_active_models:
                break

        for idx in redundant:
            self._active_mask[idx] = False
            if self.lazy_load:
                self._evict_model(idx)

    def _consider_expanding_ensemble(self, uncertainty_mean: float):
        """If uncertainty remains high, activate dormant models or create new ones."""
        if uncertainty_mean <= self.uncertainty_threshold * 1.5:
            return

        inactive_indices = [i for i in range(self.n_models) if not self._active_mask[i]]
        if inactive_indices:
            idx = inactive_indices[0]
            self._active_mask[idx] = True
            self._ensure_loaded(idx)
            return

        if self.n_models < self.max_models:
            new_idx = self.n_models
            self._model_configs.append({
                "state_dim": self.state_dim, "action_dim": self.action_dim,
                "alpha": self.alpha, "gamma": self.gamma, "lr": self.lr,
                "hidden_dims": self.hidden_dims, "target_update_tau": self.target_update_tau,
            })
            new_model = self._create_model(new_idx)
            self.models.append(new_model)
            self._model_loaded.append(True)
            self._active_mask.append(True)
            self._q_history.append([])
            self.n_models += 1

    # ===== Core Methods =====

    def compute_loss(self, batch) -> Tuple[torch.Tensor, dict]:
        active_models = self._get_active_models()
        all_losses = []
        all_metrics = []

        for model in active_models:
            loss, metrics = model.compute_loss(batch)
            all_losses.append(loss)
            all_metrics.append(metrics)

        mean_loss = torch.stack(all_losses).mean()

        avg_metrics = {}
        for key in all_metrics[0]:
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        states = batch[0].to(self.device)
        uncertainty_mean, uncertainty_max = self._compute_batch_uncertainty(states)
        avg_metrics["uncertainty_mean"] = uncertainty_mean
        avg_metrics["uncertainty_max"] = uncertainty_max
        avg_metrics["exploration_ratio"] = self._get_exploration_ratio()
        avg_metrics["active_models"] = float(len(active_models))

        return mean_loss, avg_metrics

    def update(self, batch) -> dict:
        active_models = self._get_active_models()
        all_metrics = []

        for model in active_models:
            metrics = model.update(batch)
            all_metrics.append(metrics)

        avg_metrics = {}
        for key in all_metrics[0]:
            if all_metrics[0][key] is not None:
                avg_metrics[key] = np.mean([m[key] for m in all_metrics])
            else:
                avg_metrics[key] = None

        states = batch[0].to(self.device)
        uncertainty_mean, uncertainty_max = self._compute_batch_uncertainty(states)
        avg_metrics["uncertainty_mean"] = uncertainty_mean
        avg_metrics["uncertainty_max"] = uncertainty_max
        avg_metrics["exploration_ratio"] = self._get_exploration_ratio()
        avg_metrics["active_models"] = float(len(active_models))

        self._steps_since_correlation_update += 1
        if self._steps_since_correlation_update >= self._correlation_update_interval:
            self._steps_since_correlation_update = 0
            self._evaluate_model_redundancy()
            self._consider_expanding_ensemble(uncertainty_mean)

        return avg_metrics

    def get_action(self, state: np.ndarray) -> int:
        self._total_action_count += 1

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            active_models = self._get_active_models()
            q_values_list = [model.q_network(state_t) for model in active_models]
            q_stacked = torch.stack(q_values_list)

            if self._steps_since_correlation_update == 0 or self._correlation_matrix is None:
                self._update_correlation_matrix(q_stacked)

            uncertainty_mean, _ = self._compute_correlation_adjusted_uncertainty(q_stacked)

            if self._should_explore(uncertainty_mean):
                self._exploration_count += 1
                return self._ucb_exploration_action(state_t, q_stacked)

            mean_q = q_stacked.mean(dim=0)
            return mean_q.argmax(dim=1).item()

    def get_uncertainty(self, state: np.ndarray) -> float:
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            active_models = self._get_active_models()
            q_values_list = [model.q_network(state_t) for model in active_models]
            q_stacked = torch.stack(q_values_list)
            self._update_correlation_matrix(q_stacked)
            adj_mean, _ = self._compute_correlation_adjusted_uncertainty(q_stacked)
            return adj_mean

    def _compute_batch_uncertainty(self, states: torch.Tensor) -> Tuple[float, float]:
        with torch.no_grad():
            sample_size = min(64, states.shape[0])
            sample_states = states[:sample_size]
            active_models = self._get_active_models()
            q_values_list = [model.q_network(sample_states) for model in active_models]
            q_stacked = torch.stack(q_values_list)

            self._update_correlation_matrix(q_stacked)
            return self._compute_correlation_adjusted_uncertainty(q_stacked)

    def _get_exploration_ratio(self) -> float:
        if self._total_action_count == 0:
            return 0.0
        return self._exploration_count / self._total_action_count

    def get_state_dict(self) -> dict:
        state = {"n_models": self.n_models, "active_mask": self._active_mask}
        for i in range(self.n_models):
            if self._model_loaded[i] and self.models[i] is not None:
                state[f"model_{i}"] = {
                    "q_network": self.models[i].q_network.state_dict(),
                    "target_network": self.models[i].target_network.state_dict(),
                }
        return state

    def load_state_dict(self, state_dict: dict):
        if "n_models" in state_dict:
            saved_n = state_dict["n_models"]
            while self.n_models < saved_n:
                self._model_configs.append({
                    "state_dim": self.state_dim, "action_dim": self.action_dim,
                    "alpha": self.alpha, "gamma": self.gamma, "lr": self.lr,
                    "hidden_dims": self.hidden_dims, "target_update_tau": self.target_update_tau,
                })
                self.models.append(None)
                self._model_loaded.append(False)
                self._active_mask.append(True)
                self._q_history.append([])
                self.n_models += 1

        if "active_mask" in state_dict:
            for i, active in enumerate(state_dict["active_mask"]):
                if i < self.n_models:
                    self._active_mask[i] = active

        for i in range(self.n_models):
            key = f"model_{i}"
            if key in state_dict:
                self._ensure_loaded(i)
                self.models[i].q_network.load_state_dict(state_dict[key]["q_network"])
                self.models[i].target_network.load_state_dict(state_dict[key]["target_network"])

    def get_per_model_losses(self, batch) -> List[float]:
        active_models = self._get_active_models()
        losses = []
        for model in active_models:
            loss, _ = model.compute_loss(batch)
            losses.append(loss.item())
        return losses

    def get_per_model_q_means(self, batch) -> List[float]:
        states = batch[0].to(self.device)
        active_models = self._get_active_models()
        means = []
        with torch.no_grad():
            for model in active_models:
                q = model.q_network(states)
                means.append(q.mean().item())
        return means

    def get_diagnostics(self) -> dict:
        """Return ensemble health diagnostics for monitoring."""
        active_count = sum(self._active_mask)
        return {
            "total_models": self.n_models,
            "active_models": active_count,
            "loaded_models": sum(self._model_loaded),
            "exploration_decay": self._exploration_decay,
            "exploration_ratio": self._get_exploration_ratio(),
            "mean_correlation": (
                float(np.abs(self._correlation_matrix).mean())
                if self._correlation_matrix is not None else 0.0
            ),
        }
