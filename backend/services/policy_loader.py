import io
import time
import hashlib
import logging
import threading
import numpy as np
import torch
from backend.database import SessionLocal
from backend.models import PolicyVersion, ModelSnapshot, TrainingRun
from backend.algorithms.dqn import DQN
from backend.algorithms.cql import CQL
from backend.algorithms.cql_rnn import CQL_RNN
from backend.algorithms.ensemble_cql import EnsembleCQL
from backend.config import N_CATEGORIES, N_ITEMS, POLICY_CACHE_CONFIG

logger = logging.getLogger(__name__)


COMPATIBLE_TRANSFERS = {
    ("dqn", "cql"): ["q_network", "target_network"],
    ("cql", "dqn"): ["q_network", "target_network"],
    ("cql", "cql_rnn"): ["q_network", "target_network"],
    ("dqn", "cql_rnn"): ["q_network", "target_network"],
    ("cql", "ensemble_cql"): [],
    ("dqn", "ensemble_cql"): [],
    ("behavior_cloning", "cql"): [],
    ("behavior_cloning", "dqn"): [],
}


def check_state_dict_compatibility(source_sd: dict, target_sd: dict) -> dict:
    """Check layer-by-layer compatibility between two state dicts.

    Returns a report with transferable keys, shape mismatches, and missing keys.
    """
    transferable = []
    shape_mismatch = []
    missing_in_source = []

    for key in target_sd:
        if key not in source_sd:
            missing_in_source.append(key)
        elif source_sd[key].shape != target_sd[key].shape:
            shape_mismatch.append({
                "key": key,
                "source_shape": list(source_sd[key].shape),
                "target_shape": list(target_sd[key].shape),
            })
        else:
            transferable.append(key)

    return {
        "transferable": transferable,
        "shape_mismatch": shape_mismatch,
        "missing_in_source": missing_in_source,
        "transfer_ratio": len(transferable) / max(len(target_sd), 1),
    }


def safe_partial_load(target_module, source_state_dict: dict, strict: bool = False) -> dict:
    """Load compatible weights into a module, skipping incompatible layers.

    Returns a migration report.
    """
    target_sd = target_module.state_dict()
    report = check_state_dict_compatibility(source_state_dict, target_sd)

    if report["transfer_ratio"] == 0:
        return {**report, "status": "no_compatible_weights", "loaded": 0}

    filtered_sd = {k: source_state_dict[k] for k in report["transferable"]}
    target_module.load_state_dict(filtered_sd, strict=False)

    return {
        **report,
        "status": "partial" if report["shape_mismatch"] or report["missing_in_source"] else "full",
        "loaded": len(filtered_sd),
    }


def migrate_weights_cross_algorithm(
    source_state_dict: dict,
    source_algorithm: str,
    target_agent,
    target_algorithm: str,
) -> dict:
    """Attempt cross-algorithm weight transfer with safety checks.

    Handles architecture differences gracefully:
    - Same algorithm: direct load with shape validation
    - Compatible algorithms (e.g., DQN<->CQL): transfer shared components
    - Incompatible algorithms: skip transfer, return report
    """
    pair = (source_algorithm, target_algorithm)
    reports = {}

    if source_algorithm == target_algorithm:
        if target_algorithm == "ensemble_cql":
            try:
                target_agent.load_state_dict(source_state_dict)
                reports["ensemble"] = {"status": "full", "loaded": len(source_state_dict)}
            except (RuntimeError, KeyError) as e:
                reports["ensemble"] = {"status": "failed", "error": str(e)[:200]}
        elif target_algorithm == "behavior_cloning":
            if "policy_network" in source_state_dict:
                r = safe_partial_load(target_agent.policy, source_state_dict["policy_network"])
                reports["policy_network"] = r
            elif "policy" in source_state_dict:
                r = safe_partial_load(target_agent.policy, source_state_dict["policy"])
                reports["policy_network"] = r
        else:
            for component in ("q_network", "target_network", "user_encoder"):
                if component in source_state_dict and hasattr(target_agent, component):
                    module = getattr(target_agent, component)
                    r = safe_partial_load(module, source_state_dict[component])
                    reports[component] = r
        return {"migration_type": "same_algorithm", "reports": reports}

    if pair in COMPATIBLE_TRANSFERS:
        shared_components = COMPATIBLE_TRANSFERS[pair]
        for component in shared_components:
            if component in source_state_dict and hasattr(target_agent, component):
                module = getattr(target_agent, component)
                r = safe_partial_load(module, source_state_dict[component])
                reports[component] = r

        if not shared_components and "q_network" in source_state_dict:
            if hasattr(target_agent, "q_network"):
                r = safe_partial_load(target_agent.q_network, source_state_dict["q_network"])
                reports["q_network_fallback"] = r

        return {"migration_type": "cross_algorithm", "pair": pair, "reports": reports}

    if "q_network" in source_state_dict and hasattr(target_agent, "q_network"):
        r = safe_partial_load(target_agent.q_network, source_state_dict["q_network"])
        if r["transfer_ratio"] > 0.5:
            reports["q_network_best_effort"] = r
            if "target_network" in source_state_dict and hasattr(target_agent, "target_network"):
                r2 = safe_partial_load(target_agent.target_network, source_state_dict["target_network"])
                reports["target_network_best_effort"] = r2
            return {"migration_type": "best_effort", "reports": reports}

    return {"migration_type": "incompatible", "reports": {}, "reason": f"No safe transfer path from {source_algorithm} to {target_algorithm}"}


class TrafficAllocator:
    """Consistent traffic allocation with balance correction."""

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


class PolicyCache:
    """Adaptive caching strategy with configurable TTL modes.

    Modes:
    - aggressive: short TTL (5s), for high-frequency policy iteration
    - balanced: moderate TTL (30s), default for most scenarios
    - lazy: long TTL (120s), for stable production with infrequent updates
    - manual: only refreshes on explicit invalidation
    """

    PRESETS = {
        "aggressive": 5,
        "balanced": 30,
        "lazy": 120,
        "manual": 86400,
    }

    def __init__(self, mode: str = None, ttl_seconds: float = None):
        config = POLICY_CACHE_CONFIG
        self._mode = mode or config.get("mode", "balanced")
        self._ttl = ttl_seconds or config.get("ttl_seconds") or self.PRESETS.get(self._mode, 30)
        self._last_check = 0.0
        self._hit_count = 0
        self._miss_count = 0

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def ttl(self) -> float:
        return self._ttl

    @property
    def stats(self) -> dict:
        total = self._hit_count + self._miss_count
        return {
            "mode": self._mode,
            "ttl_seconds": self._ttl,
            "hits": self._hit_count,
            "misses": self._miss_count,
            "hit_rate": self._hit_count / total if total > 0 else 0.0,
        }

    def should_refresh(self) -> bool:
        now = time.time()
        if now - self._last_check >= self._ttl:
            self._last_check = now
            self._miss_count += 1
            return True
        self._hit_count += 1
        return False

    def invalidate(self):
        self._last_check = 0.0

    def update_config(self, mode: str = None, ttl_seconds: float = None):
        if mode and mode in self.PRESETS:
            self._mode = mode
            if ttl_seconds is None:
                self._ttl = self.PRESETS[mode]
        if ttl_seconds is not None:
            self._ttl = max(1.0, ttl_seconds)
            if mode is None:
                self._mode = "custom"


class PolicyLoader:
    """Singleton that caches the current production policy with adaptive refresh."""

    def __init__(self):
        self._lock = threading.Lock()
        self._agent = None
        self._policy_version_id = None
        self._algorithm = None
        self._hyperparams = None
        self._last_migration_report = None
        self.cache = PolicyCache()

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
            try:
                agent.load_state_dict(state_dict)
            except (RuntimeError, KeyError) as e:
                logger.warning(f"Ensemble load failed, attempting partial: {e}")
        elif algorithm == "cql_rnn":
            agent = CQL_RNN(
                state_dim=N_CATEGORIES, action_dim=N_ITEMS,
                alpha=hyperparams.get("alpha", 1.0),
                gamma=gamma, lr=lr, hidden_dims=hidden_dims,
                target_update_tau=tau,
                lstm_hidden_size=hyperparams.get("lstm_hidden_size", 128),
                lstm_num_layers=hyperparams.get("lstm_num_layers", 2),
            )
            if "q_network" in state_dict:
                safe_partial_load(agent.q_network, state_dict["q_network"])
            if "target_network" in state_dict:
                safe_partial_load(agent.target_network, state_dict["target_network"])
            if "user_encoder" in state_dict:
                safe_partial_load(agent.user_encoder, state_dict["user_encoder"])
        elif algorithm in ("cql", "dqn"):
            AgentClass = CQL if algorithm == "cql" else DQN
            kwargs = dict(state_dim=N_CATEGORIES, action_dim=N_ITEMS,
                          gamma=gamma, lr=lr, hidden_dims=hidden_dims,
                          target_update_tau=tau)
            if algorithm == "cql":
                kwargs["alpha"] = hyperparams.get("alpha", 1.0)
            agent = AgentClass(**kwargs)
            if "q_network" in state_dict:
                safe_partial_load(agent.q_network, state_dict["q_network"])
            if "target_network" in state_dict:
                safe_partial_load(agent.target_network, state_dict["target_network"])
        elif algorithm == "behavior_cloning":
            from backend.algorithms.behavior_cloning import BehaviorCloning
            agent = BehaviorCloning(
                state_dim=N_CATEGORIES, action_dim=N_ITEMS,
                lr=lr, hidden_dims=hidden_dims,
            )
            if "policy_network" in state_dict:
                safe_partial_load(agent.policy, state_dict["policy_network"])
            elif "policy" in state_dict:
                safe_partial_load(agent.policy, state_dict["policy"])
        else:
            agent = DQN(
                state_dim=N_CATEGORIES, action_dim=N_ITEMS,
                gamma=gamma, lr=lr, hidden_dims=hidden_dims,
                target_update_tau=tau,
            )
            if "q_network" in state_dict:
                safe_partial_load(agent.q_network, state_dict["q_network"])

        return agent

    def _ensure_loaded(self):
        """Load or reload production policy, respecting cache strategy."""
        if not self.cache.should_refresh():
            return

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
        self.cache.invalidate()

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

    @property
    def last_migration_report(self):
        return self._last_migration_report


traffic_allocator = TrafficAllocator()
policy_loader = PolicyLoader()
