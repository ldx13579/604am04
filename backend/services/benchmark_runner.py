import time
import threading
import numpy as np
from datetime import datetime
from backend.database import SessionLocal
from backend.models import PerfReportRun
from backend.data.dataset import ReplayBuffer
from backend.algorithms.cql import CQL
from backend.algorithms.dqn import DQN
from backend.algorithms.behavior_cloning import BehaviorCloning
from backend.algorithms.ensemble_cql import EnsembleCQL
from backend.environment.simulator import RecommendationEnv
from backend.config import N_CATEGORIES, N_ITEMS, DEFAULT_HYPERPARAMS


class BenchmarkRunner:
    """Runs training benchmarks with different dataset sizes and algorithms.

    Supports multi-algorithm comparison and computes efficiency metrics:
    time/1K transitions, reward per compute second, sample efficiency ratio.
    """

    def __init__(self):
        self.is_running = False
        self.current_size = None
        self.current_algorithm = None
        self.progress = {}
        self.results_cache = []
        self._lock = threading.Lock()

    def get_status(self) -> dict:
        return {
            "is_running": self.is_running,
            "current_size": self.current_size,
            "current_algorithm": self.current_algorithm,
            "progress": dict(self.progress),
        }

    def run_benchmark(self, dataset_sizes: list, algorithm: str = "cql",
                      epochs: int = 50, algorithms: list = None):
        """Run benchmark. If algorithms list provided, compare all of them."""
        if self.is_running:
            return

        algo_list = algorithms or [algorithm]

        with self._lock:
            self.is_running = True
            self.progress = {}
            self.results_cache = []
            for alg in algo_list:
                for size in dataset_sizes:
                    self.progress[f"{alg}:{size}"] = "pending"

        try:
            for alg in algo_list:
                for size in dataset_sizes:
                    key = f"{alg}:{size}"
                    self.current_size = size
                    self.current_algorithm = alg
                    self.progress[key] = "running"
                    self._run_single(size, alg, epochs)
                    self.progress[key] = "completed"
        finally:
            with self._lock:
                self.is_running = False
                self.current_size = None
                self.current_algorithm = None

    def _create_agent(self, algorithm: str, hyperparams: dict):
        lr = hyperparams.get("lr", 3e-4)
        hidden_dims = hyperparams.get("hidden_dims", [256, 256])
        gamma = hyperparams.get("gamma", 0.99)
        tau = hyperparams.get("target_update_tau", 0.005)

        if algorithm == "cql":
            return CQL(
                state_dim=N_CATEGORIES, action_dim=N_ITEMS,
                alpha=hyperparams.get("alpha", 1.0),
                gamma=gamma, lr=lr, hidden_dims=hidden_dims,
                target_update_tau=tau,
            )
        elif algorithm == "dqn":
            return DQN(
                state_dim=N_CATEGORIES, action_dim=N_ITEMS,
                gamma=gamma, lr=lr, hidden_dims=hidden_dims,
                target_update_tau=tau,
            )
        elif algorithm == "ensemble_cql":
            return EnsembleCQL(
                state_dim=N_CATEGORIES, action_dim=N_ITEMS,
                alpha=hyperparams.get("alpha", 1.0),
                gamma=gamma, lr=lr, hidden_dims=hidden_dims,
                target_update_tau=tau,
                n_models=hyperparams.get("n_models", 3),
                uncertainty_threshold=hyperparams.get("uncertainty_threshold", 1.0),
            )
        elif algorithm == "behavior_cloning":
            return BehaviorCloning(
                state_dim=N_CATEGORIES, action_dim=N_ITEMS,
                lr=lr, hidden_dims=hidden_dims,
            )
        else:
            return CQL(
                state_dim=N_CATEGORIES, action_dim=N_ITEMS,
                alpha=hyperparams.get("alpha", 1.0),
                gamma=gamma, lr=lr, hidden_dims=hidden_dims,
                target_update_tau=tau,
            )

    def _run_single(self, dataset_size: int, algorithm: str, epochs: int):
        replay_buffer = ReplayBuffer(
            capacity=dataset_size,
            db_limit=dataset_size,
        )

        if replay_buffer.size == 0:
            return

        hyperparams = DEFAULT_HYPERPARAMS.get(algorithm, DEFAULT_HYPERPARAMS["cql"])
        steps_per_epoch = min(hyperparams.get("steps_per_epoch", 1000), replay_buffer.size // 256)
        steps_per_epoch = max(steps_per_epoch, 1)
        batch_size = min(256, replay_buffer.size)

        agent = self._create_agent(algorithm, hyperparams)
        env = RecommendationEnv(seed=42)

        best_reward = -float("inf")
        convergence_epoch = None
        reward_history = []
        first_positive_epoch = None

        start_time = time.time()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for _ in range(steps_per_epoch):
                batch = replay_buffer.sample(batch_size)
                metrics = agent.update(batch)
                if metrics and "loss" in metrics:
                    epoch_loss += metrics["loss"]

            reward = env.evaluate_policy(agent.get_action, n_episodes=10)
            reward_history.append(reward)

            if reward > best_reward:
                best_reward = reward

            if convergence_epoch is None and epoch > 5:
                recent = reward_history[-3:]
                if len(recent) == 3 and all(r >= 0.9 * best_reward for r in recent):
                    convergence_epoch = epoch

            if first_positive_epoch is None and reward > 0:
                first_positive_epoch = epoch

        training_time = time.time() - start_time

        reward_stability = float(np.std(reward_history[-10:])) if len(reward_history) >= 10 else None
        sample_efficiency = best_reward / (dataset_size / 1000.0) if dataset_size > 0 else 0.0
        compute_efficiency = best_reward / training_time if training_time > 0 else 0.0

        db = SessionLocal()
        try:
            report = PerfReportRun(
                dataset_size=dataset_size,
                algorithm=algorithm,
                training_time_seconds=training_time,
                convergence_epoch=convergence_epoch,
                final_reward=best_reward,
                total_epochs_run=epochs,
            )
            db.add(report)
            db.commit()
        finally:
            db.close()

        self.results_cache.append({
            "dataset_size": dataset_size,
            "algorithm": algorithm,
            "training_time_seconds": training_time,
            "convergence_epoch": convergence_epoch,
            "final_reward": best_reward,
            "reward_stability": reward_stability,
            "sample_efficiency": sample_efficiency,
            "compute_efficiency": compute_efficiency,
            "first_positive_epoch": first_positive_epoch,
            "time_per_1k": training_time / (dataset_size / 1000.0) if dataset_size > 0 else 0,
        })

    def get_comparison(self) -> dict:
        """Generate cross-algorithm comparison from cached results."""
        if not self.results_cache:
            return {"algorithms": [], "comparison": []}

        algorithms = list(set(r["algorithm"] for r in self.results_cache))
        comparison = []

        for alg in algorithms:
            alg_results = [r for r in self.results_cache if r["algorithm"] == alg]
            if not alg_results:
                continue
            comparison.append({
                "algorithm": alg,
                "avg_reward": float(np.mean([r["final_reward"] for r in alg_results])),
                "avg_time": float(np.mean([r["training_time_seconds"] for r in alg_results])),
                "avg_sample_efficiency": float(np.mean([r["sample_efficiency"] for r in alg_results])),
                "avg_compute_efficiency": float(np.mean([r["compute_efficiency"] for r in alg_results])),
                "best_reward": float(max(r["final_reward"] for r in alg_results)),
                "fastest_convergence": min(
                    (r["convergence_epoch"] for r in alg_results if r["convergence_epoch"] is not None),
                    default=None
                ),
                "runs": alg_results,
            })

        comparison.sort(key=lambda x: x["avg_reward"], reverse=True)

        return {
            "algorithms": algorithms,
            "comparison": comparison,
            "winner": comparison[0]["algorithm"] if comparison else None,
        }


benchmark_runner = BenchmarkRunner()
