import io
import time
import hashlib
import threading
import numpy as np
import torch
from backend.database import SessionLocal
from backend.models import PolicyVersion, ModelSnapshot, TrainingRun
from backend.algorithms.dqn import DQN
from backend.algorithms.cql import CQL
from backend.algorithms.cql_rnn import CQL_RNN
from backend.algorithms.ensemble_cql import EnsembleCQL
from backend.config import N_CATEGORIES, N_ITEMS

CACHE_TTL_SECONDS = 30


class TrafficAllocator:
    """Consistent traffic allocation with balance correction.

    Uses deterministic hashing for session stickiness while tracking
    cumulative allocation to correct drift from the target split.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._counts = {}

    def assign_group(self, session_id: str, experiment_id: int, traffic_split: float) -> str:
        h = int(hashlib.sha256(
            f"{experiment_id}:{session_id}".encode()
        ).hexdigest(), 16)
        bucket = (h % 10000) / 10000.0

        with self._lock:
            key = experiment_id
            if key not in self._counts:
                self._counts[key] = {"A": 0, "B": 0}

            counts = self._counts[key]
            total = counts["A"] + counts["B"]

            if total > 0 and total >= 20:
                actual_ratio_a = counts["A"] / total
                drift = actual_ratio_a - traffic_split
                correction = drift * 0.3
                effective_split = traffic_split - correction
                effective_split = max(0.05, min(0.95, effective_split))
            else:
                effective_split = traffic_split

            group = "A" if bucket < effective_split else "B"
            counts[group] += 1

        return group

    def get_balance(self, experiment_id: int) -> dict:
        with self._lock:
            counts = self._counts.get(experiment_id, {"A": 0, "B": 0})
            total = counts["A"] + counts["B"]
            return {
                "group_a_count": counts["A"],
                "group_b_count": counts["B"],
                "actual_split_a": counts["A"] / total if total > 0 else 0.0,
                "total_allocations": total,
            }

    def reset(self, experiment_id: int):
        with self._lock:
            self._counts.pop(experiment_id, None)


class PolicyLoader:
    """Singleton that caches the current production policy with TTL-based refresh."""

    def __init__(self, cache_ttl: float = CACHE_TTL_SECONDS):
        self._lock = threading.Lock()
        self._agent = None
        self._policy_version_id = None
        self._algorithm = None
        self._hyperparams = None
        self._cache_ttl = cache_ttl
        self._last_check_time = 0.0

    def _load_agent_from_snapshot(self, snapshot, algorithm, hyperparams):
        buffer = io.BytesIO(snapshot.parameters_blob)
        state_dict = torch.load(buffer, map_location="cpu", weights_only=False)

        hidden_dims = hyperparams.get("hidden_dims", [256, 256])
        lr = hyperparams.get("lr", 3e-4)
        gamma = hyperparams.get("gamma", 0.99)
        tau = hyperparams.get("target_update_tau", 0.005)

        if algorithm == "ensemble_cql":
            agent = EnsembleCQL(
                state_dim=N_CATEGORIES, action_dim=N_ITEMS,
                alpha=hyperparams.get("alpha", 1.0),
                gamma=gamma, lr=lr, hidden_dims=hidden_dims,
                target_update_tau=tau,
                n_models=hyperparams.get("n_models", 5),
                uncertainty_threshold=hyperparams.get("uncertainty_threshold", 1.0),
            )
            agent.load_state_dict(state_dict)
        elif algorithm == "cql_rnn":
            agent = CQL_RNN(
                state_dim=N_CATEGORIES, action_dim=N_ITEMS,
                alpha=hyperparams.get("alpha", 1.0),
                gamma=gamma, lr=lr, hidden_dims=hidden_dims,
                target_update_tau=tau,
                lstm_hidden_size=hyperparams.get("lstm_hidden_size", 128),
                lstm_num_layers=hyperparams.get("lstm_num_layers", 2),
            )
            agent.q_network.load_state_dict(state_dict["q_network"])
            agent.target_network.load_state_dict(state_dict["target_network"])
            if "user_encoder" in state_dict:
                agent.user_encoder.load_state_dict(state_dict["user_encoder"])
        elif algorithm in ("cql", "dqn"):
            AgentClass = CQL if algorithm == "cql" else DQN
            kwargs = dict(state_dim=N_CATEGORIES, action_dim=N_ITEMS,
                          gamma=gamma, lr=lr, hidden_dims=hidden_dims,
                          target_update_tau=tau)
            if algorithm == "cql":
                kwargs["alpha"] = hyperparams.get("alpha", 1.0)
            agent = AgentClass(**kwargs)
            agent.q_network.load_state_dict(state_dict["q_network"])
            agent.target_network.load_state_dict(state_dict["target_network"])
        else:
            agent = DQN(
                state_dim=N_CATEGORIES, action_dim=N_ITEMS,
                gamma=gamma, lr=lr, hidden_dims=hidden_dims,
                target_update_tau=tau,
            )
            if "q_network" in state_dict:
                agent.q_network.load_state_dict(state_dict["q_network"])

        return agent

    def _ensure_loaded(self):
        """Load or reload production policy, respecting cache TTL."""
        now = time.time()
        if now - self._last_check_time < self._cache_ttl:
            return
        self._last_check_time = now

        db = SessionLocal()
        try:
            pv = db.query(PolicyVersion).filter(
                PolicyVersion.stage == "production"
            ).order_by(PolicyVersion.created_at.desc()).first()

            if pv is None:
                if self._agent is not None:
                    return
                run = db.query(TrainingRun).filter(
                    TrainingRun.status == "completed",
                    TrainingRun.algorithm.in_(["cql", "ensemble_cql", "cql_rnn", "dqn"])
                ).order_by(TrainingRun.best_reward.desc().nullslast()).first()
                if run is None:
                    return
                snapshot = db.query(ModelSnapshot).filter(
                    ModelSnapshot.run_id == run.id
                ).order_by(ModelSnapshot.performance_reward.desc().nullslast()).first()
                if snapshot is None:
                    return
                with self._lock:
                    self._agent = self._load_agent_from_snapshot(
                        snapshot, run.algorithm, run.hyperparameters)
                    self._policy_version_id = None
                    self._algorithm = run.algorithm
                    self._hyperparams = run.hyperparameters
                return

            if pv.id == self._policy_version_id:
                return

            snapshot = db.query(ModelSnapshot).get(pv.snapshot_id)
            if snapshot is None:
                return
            run = db.query(TrainingRun).get(pv.run_id)
            with self._lock:
                self._agent = self._load_agent_from_snapshot(
                    snapshot, run.algorithm, run.hyperparameters)
                self._policy_version_id = pv.id
                self._algorithm = run.algorithm
                self._hyperparams = run.hyperparameters
        finally:
            db.close()

    def invalidate_cache(self):
        """Force next call to re-check DB."""
        self._last_check_time = 0.0

    def get_action(self, state: np.ndarray) -> int:
        self._ensure_loaded()
        with self._lock:
            if self._agent is None:
                return self.random_action(state)
            return self._agent.get_action(state)

    def get_top_k(self, state: np.ndarray, k: int = 5) -> list:
        self._ensure_loaded()
        with self._lock:
            if self._agent is None:
                return self._random_top_k(k)
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self._agent.q_network(state_t.to(self._agent.device))
            scores, indices = torch.topk(q_values[0], k)
            return list(zip(indices.cpu().tolist(), scores.cpu().tolist()))

    def random_action(self, state: np.ndarray) -> int:
        return int(np.random.randint(0, N_ITEMS))

    def _random_top_k(self, k: int) -> list:
        items = np.random.choice(N_ITEMS, size=k, replace=False)
        return [(int(i), 0.0) for i in items]

    @property
    def is_loaded(self) -> bool:
        return self._agent is not None

    @property
    def current_version_id(self):
        return self._policy_version_id

    @property
    def current_algorithm(self):
        return self._algorithm

    @property
    def current_hyperparams(self):
        return self._hyperparams


traffic_allocator = TrafficAllocator()
policy_loader = PolicyLoader()
