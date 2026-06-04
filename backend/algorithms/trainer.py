import time
from datetime import datetime
from sqlalchemy.orm import Session
from backend.database import SessionLocal
from backend.models import TrainingRun, TrainingMetric
from backend.data.dataset import ReplayBuffer
from backend.environment.simulator import RecommendationEnv
from backend.algorithms.dqn import DQN
from backend.algorithms.cql import CQL
from backend.algorithms.behavior_cloning import BehaviorCloning
from backend.config import DEFAULT_HYPERPARAMS, N_CATEGORIES, N_ITEMS


class Trainer:
    """Unified training loop for all offline RL algorithms."""

    def __init__(self):
        self.active_runs = {}

    def start_training(self, algorithm: str, hyperparams: dict = None, run_id: int = None):
        if hyperparams is None:
            hyperparams = DEFAULT_HYPERPARAMS.get(algorithm, {})

        merged = {**DEFAULT_HYPERPARAMS.get(algorithm, {}), **hyperparams}

        db = SessionLocal()
        try:
            if run_id is None:
                run = TrainingRun(
                    algorithm=algorithm,
                    hyperparameters=merged,
                    status="running",
                    started_at=datetime.utcnow(),
                    total_epochs=merged.get("epochs", 200),
                )
                db.add(run)
                db.commit()
                db.refresh(run)
                run_id = run.id
            else:
                run = db.query(TrainingRun).get(run_id)
                run.status = "running"
                run.started_at = datetime.utcnow()
                db.commit()
        finally:
            db.close()

        self._train_loop(run_id, algorithm, merged)
        return run_id

    def _train_loop(self, run_id: int, algorithm: str, params: dict):
        replay_buffer = ReplayBuffer(limit=500000)
        env = RecommendationEnv(seed=42)

        agent = self._create_agent(algorithm, params)
        epochs = params.get("epochs", 200)
        steps_per_epoch = params.get("steps_per_epoch", 1000)
        batch_size = params.get("batch_size", 256)
        best_reward = -float("inf")

        db = SessionLocal()
        try:
            for epoch in range(1, epochs + 1):
                epoch_metrics = {"loss": 0, "q_value_mean": 0, "q_value_max": 0,
                                 "q_value_min": 0, "cql_penalty": 0}
                n_steps = 0

                for _ in range(steps_per_epoch):
                    batch = replay_buffer.sample(batch_size)
                    metrics = agent.update(batch)
                    for k, v in metrics.items():
                        if v is not None:
                            epoch_metrics[k] += v
                    n_steps += 1

                for k in epoch_metrics:
                    epoch_metrics[k] /= max(n_steps, 1)

                cumulative_reward = env.evaluate_policy(agent.get_action, n_episodes=10)

                if cumulative_reward > best_reward:
                    best_reward = cumulative_reward

                metric = TrainingMetric(
                    run_id=run_id,
                    epoch=epoch,
                    loss=epoch_metrics["loss"],
                    q_value_mean=epoch_metrics["q_value_mean"],
                    q_value_max=epoch_metrics["q_value_max"],
                    q_value_min=epoch_metrics["q_value_min"],
                    cumulative_reward=cumulative_reward,
                    cql_penalty=epoch_metrics["cql_penalty"] if algorithm == "cql" else None,
                )
                db.add(metric)

                run = db.query(TrainingRun).get(run_id)
                run.total_epochs = epoch
                run.best_reward = best_reward
                db.commit()

                self.active_runs[run_id] = {
                    "epoch": epoch,
                    "total_epochs": epochs,
                    "metrics": epoch_metrics,
                    "cumulative_reward": cumulative_reward,
                }

            run = db.query(TrainingRun).get(run_id)
            run.status = "completed"
            run.completed_at = datetime.utcnow()
            db.commit()
        except Exception as e:
            run = db.query(TrainingRun).get(run_id)
            run.status = "failed"
            db.commit()
            raise e
        finally:
            if run_id in self.active_runs:
                del self.active_runs[run_id]
            db.close()

    def _create_agent(self, algorithm: str, params: dict):
        hidden_dims = params.get("hidden_dims", [256, 256])
        lr = params.get("lr", 3e-4)
        gamma = params.get("gamma", 0.99)
        tau = params.get("target_update_tau", 0.005)

        if algorithm == "cql":
            return CQL(
                state_dim=N_CATEGORIES, action_dim=N_ITEMS,
                alpha=params.get("alpha", 1.0),
                gamma=gamma, lr=lr, hidden_dims=hidden_dims,
                target_update_tau=tau,
            )
        elif algorithm == "dqn":
            return DQN(
                state_dim=N_CATEGORIES, action_dim=N_ITEMS,
                gamma=gamma, lr=lr, hidden_dims=hidden_dims,
                target_update_tau=tau,
            )
        elif algorithm == "behavior_cloning":
            return BehaviorCloning(
                state_dim=N_CATEGORIES, action_dim=N_ITEMS,
                lr=lr, hidden_dims=hidden_dims,
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")


trainer = Trainer()
