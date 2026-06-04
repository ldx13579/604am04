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
from backend.environment.simulator import RecommendationEnv
from backend.config import N_CATEGORIES, N_ITEMS, DEFAULT_HYPERPARAMS


class BenchmarkRunner:
    """Runs training benchmarks with different dataset sizes."""

    def __init__(self):
        self.is_running = False
        self.current_size = None
        self.progress = {}
        self._lock = threading.Lock()

    def get_status(self) -> dict:
        return {
            "is_running": self.is_running,
            "current_size": self.current_size,
            "progress": dict(self.progress),
        }

    def run_benchmark(self, dataset_sizes: list, algorithm: str = "cql", epochs: int = 50):
        if self.is_running:
            return

        with self._lock:
            self.is_running = True
            self.progress = {size: "pending" for size in dataset_sizes}

        try:
            for size in dataset_sizes:
                self.current_size = size
                self.progress[size] = "running"
                self._run_single(size, algorithm, epochs)
                self.progress[size] = "completed"
        finally:
            with self._lock:
                self.is_running = False
                self.current_size = None

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

        if algorithm == "cql":
            agent = CQL(
                state_dim=N_CATEGORIES, action_dim=N_ITEMS,
                alpha=hyperparams.get("alpha", 1.0),
                lr=hyperparams.get("lr", 3e-4),
            )
        elif algorithm == "dqn":
            agent = DQN(
                state_dim=N_CATEGORIES, action_dim=N_ITEMS,
                lr=hyperparams.get("lr", 3e-4),
            )
        else:
            agent = CQL(
                state_dim=N_CATEGORIES, action_dim=N_ITEMS,
                alpha=hyperparams.get("alpha", 1.0),
                lr=hyperparams.get("lr", 3e-4),
            )

        env = RecommendationEnv(seed=42)
        best_reward = -float("inf")
        convergence_epoch = None
        threshold = 0.95

        start_time = time.time()

        for epoch in range(epochs):
            for _ in range(steps_per_epoch):
                batch = replay_buffer.sample(256)
                agent.update(batch)

            reward = env.evaluate_policy(agent.get_action, n_episodes=10)
            if reward > best_reward:
                best_reward = reward
            if convergence_epoch is None and reward >= threshold * best_reward and epoch > 5:
                convergence_epoch = epoch

        training_time = time.time() - start_time

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


benchmark_runner = BenchmarkRunner()
