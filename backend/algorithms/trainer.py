import io
import traceback
from datetime import datetime
import torch
from backend.database import SessionLocal
from backend.models import TrainingRun, TrainingMetric, ModelSnapshot
from backend.data.dataset import ReplayBuffer, SequenceReplayBuffer
from backend.environment.simulator import RecommendationEnv
from backend.algorithms.dqn import DQN
from backend.algorithms.cql import CQL
from backend.algorithms.cql_rnn import CQL_RNN
from backend.algorithms.behavior_cloning import BehaviorCloning
from backend.config import DEFAULT_HYPERPARAMS, SHIFT_DETECTION_CONFIG, N_CATEGORIES, N_ITEMS

DEFAULT_CHUNK_REFRESH_INTERVAL = 20
ERROR_DETAIL_MAX_LENGTH = 5000
ERROR_DETAIL_TAIL_RESERVED = 2000


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
                    current_epoch=0,
                )
                db.add(run)
                db.commit()
                db.refresh(run)
                run_id = run.id
            else:
                run = db.query(TrainingRun).get(run_id)
                run.status = "running"
                run.started_at = datetime.utcnow()
                run.error_detail = None
                db.commit()
        finally:
            db.close()

        self._train_loop(run_id, algorithm, merged)
        return run_id

    def _train_loop(self, run_id: int, algorithm: str, params: dict):
        if algorithm == "cql_rnn":
            seq_len = params.get("seq_len", 10)
            replay_buffer = SequenceReplayBuffer(
                capacity=200_000, chunk_size=20_000, seq_len=seq_len
            )
        else:
            replay_buffer = ReplayBuffer(capacity=500_000, chunk_size=50_000)

        if replay_buffer.size == 0:
            self._fail_run(run_id, "ReplayBuffer is empty - no offline data in database. Generate data first.")
            return

        env = RecommendationEnv(seed=42)
        agent = self._create_agent(algorithm, params)
        epochs = params.get("epochs", 200)
        steps_per_epoch = params.get("steps_per_epoch", 1000)
        batch_size = params.get("batch_size", 256)
        chunk_refresh_interval = params.get("chunk_refresh_interval", DEFAULT_CHUNK_REFRESH_INTERVAL)
        best_reward = -float("inf")
        snapshot_interval = params.get("snapshot_interval", 50)

        db = SessionLocal()
        try:
            run = db.query(TrainingRun).get(run_id)

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

                if algorithm == "cql_rnn":
                    cumulative_reward = env.evaluate_policy(
                        lambda s: agent.get_action(s), n_episodes=10
                    )
                else:
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
                    cql_penalty=epoch_metrics["cql_penalty"] if algorithm in ("cql", "cql_rnn") else None,
                )
                db.add(metric)

                run.current_epoch = epoch
                run.best_reward = best_reward
                db.commit()

                self.active_runs[run_id] = {
                    "epoch": epoch,
                    "total_epochs": epochs,
                    "metrics": epoch_metrics,
                    "cumulative_reward": cumulative_reward,
                }

                if chunk_refresh_interval > 0 and epoch % chunk_refresh_interval == 0:
                    replay_buffer.refresh_chunk()

                if epoch % snapshot_interval == 0 or epoch == epochs:
                    self._save_snapshot(db, run_id, epoch, algorithm, agent, params, cumulative_reward)

            run.status = "completed"
            run.completed_at = datetime.utcnow()
            run.current_epoch = epochs
            db.commit()
        except Exception as e:
            error_msg = self._format_error(e)
            self._fail_run(run_id, error_msg, db=db)
        finally:
            if run_id in self.active_runs:
                del self.active_runs[run_id]
            db.close()

    def _format_error(self, exc: Exception) -> str:
        """Format exception preserving short messages fully, truncating long ones smartly."""
        summary = f"{type(exc).__name__}: {str(exc)}"
        tb = traceback.format_exc()
        full_msg = f"{summary}\n{tb}"

        if len(full_msg) <= ERROR_DETAIL_MAX_LENGTH:
            return full_msg

        # For long errors: keep the summary + beginning of traceback, and the tail
        # (tail often has the most relevant frames closest to the raise site)
        head_budget = ERROR_DETAIL_MAX_LENGTH - ERROR_DETAIL_TAIL_RESERVED - len("\n...[truncated]...\n")
        head_part = full_msg[:head_budget]
        tail_part = full_msg[-ERROR_DETAIL_TAIL_RESERVED:]
        return f"{head_part}\n...[truncated]...\n{tail_part}"

    def _fail_run(self, run_id: int, error_detail: str, db=None):
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True
        try:
            run = db.query(TrainingRun).get(run_id)
            if run:
                run.status = "failed"
                run.error_detail = error_detail
                run.completed_at = datetime.utcnow()
                db.commit()
        finally:
            if close_db:
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
        elif algorithm == "cql_rnn":
            return CQL_RNN(
                state_dim=N_CATEGORIES, action_dim=N_ITEMS,
                alpha=params.get("alpha", 1.0),
                gamma=gamma, lr=lr, hidden_dims=hidden_dims,
                lstm_hidden_size=params.get("lstm_hidden_size", 128),
                lstm_num_layers=params.get("lstm_num_layers", 2),
                target_update_tau=tau,
                seq_len=params.get("seq_len", 10),
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

    def _save_snapshot(self, db, run_id: int, epoch: int, algorithm: str,
                       agent, params: dict, reward: float):
        """Serialize model parameters and save to database.

        Enforces retention policy: only keeps the most recent N snapshots per run,
        deleting older ones to prevent unbounded storage growth.
        """
        if hasattr(agent, "get_state_dict"):
            state_dict = agent.get_state_dict()
        elif hasattr(agent, "q_network"):
            state_dict = {
                "q_network": agent.q_network.state_dict(),
                "target_network": agent.target_network.state_dict(),
            }
        else:
            state_dict = {}

        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        blob = buffer.getvalue()

        snapshot = ModelSnapshot(
            run_id=run_id,
            epoch=epoch,
            algorithm=algorithm,
            parameters_blob=blob,
            hyperparameters=params,
            state_dim=N_CATEGORIES,
            action_dim=N_ITEMS,
            performance_reward=reward,
        )
        db.add(snapshot)
        db.commit()

        self._enforce_snapshot_retention(db, run_id)

    def _enforce_snapshot_retention(self, db, run_id: int):
        """Delete oldest snapshots exceeding the max retention limit per run."""
        max_snapshots = SHIFT_DETECTION_CONFIG.get("max_snapshots_per_run", 5)

        all_snapshots = db.query(ModelSnapshot).filter(
            ModelSnapshot.run_id == run_id
        ).order_by(ModelSnapshot.epoch.desc()).all()

        if len(all_snapshots) > max_snapshots:
            to_delete = all_snapshots[max_snapshots:]
            for old_snapshot in to_delete:
                db.delete(old_snapshot)
            db.commit()

    @staticmethod
    def cleanup_all_snapshots():
        """Run retention cleanup across all training runs. Call periodically."""
        max_snapshots = SHIFT_DETECTION_CONFIG.get("max_snapshots_per_run", 5)
        db = SessionLocal()
        try:
            run_ids = [r[0] for r in db.query(TrainingRun.id).all()]
            for run_id in run_ids:
                snapshots = db.query(ModelSnapshot).filter(
                    ModelSnapshot.run_id == run_id
                ).order_by(ModelSnapshot.epoch.desc()).all()

                if len(snapshots) > max_snapshots:
                    for old in snapshots[max_snapshots:]:
                        db.delete(old)
            db.commit()
        finally:
            db.close()


trainer = Trainer()
