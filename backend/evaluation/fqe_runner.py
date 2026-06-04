import io
from datetime import datetime
import torch
from backend.database import SessionLocal
from backend.models import TrainingRun, ModelSnapshot, FQEEvaluation, FQEMetric
from backend.data.dataset import ReplayBuffer
from backend.algorithms.fqe import FittedQEvaluation, PolicyValidator
from backend.algorithms.dqn import DQN
from backend.algorithms.cql import CQL
from backend.algorithms.cql_rnn import CQL_RNN
from backend.algorithms.ensemble_cql import EnsembleCQL
from backend.config import N_CATEGORIES, N_ITEMS, FQE_DEFAULTS


class FQERunner:
    """Manages FQE evaluation lifecycle: load policy, validate, run FQE, save results.

    Includes pre-evaluation policy validation to catch poorly-trained or
    degenerate policies before spending compute on FQE.
    """

    def __init__(self):
        self.active_evaluations = {}

    def start_evaluation(self, source_run_id: int, hyperparams: dict = None,
                         evaluation_id: int = None):
        merged = {**FQE_DEFAULTS, **(hyperparams or {})}

        db = SessionLocal()
        try:
            run = db.query(TrainingRun).get(source_run_id)
            if run is None:
                raise ValueError(f"Source run {source_run_id} not found")
            if run.status != "completed":
                raise ValueError(f"Source run {source_run_id} is not completed (status: {run.status})")

            if evaluation_id is None:
                evaluation = FQEEvaluation(
                    source_run_id=source_run_id,
                    algorithm=run.algorithm,
                    hyperparameters=merged,
                    status="running",
                    total_epochs=merged.get("epochs", 50),
                    current_epoch=0,
                    started_at=datetime.utcnow(),
                )
                db.add(evaluation)
                db.commit()
                db.refresh(evaluation)
                evaluation_id = evaluation.id
            else:
                evaluation = db.query(FQEEvaluation).get(evaluation_id)
                evaluation.status = "running"
                evaluation.started_at = datetime.utcnow()
                db.commit()

            policy_fn = self._load_policy(db, source_run_id, run.algorithm, run.hyperparameters)
        finally:
            db.close()

        self._run_fqe(evaluation_id, policy_fn, merged)
        return evaluation_id

    def _load_policy(self, db, run_id: int, algorithm: str, hyperparams: dict):
        snapshot = db.query(ModelSnapshot).filter(
            ModelSnapshot.run_id == run_id
        ).order_by(ModelSnapshot.performance_reward.desc().nullslast()).first()

        if snapshot is None:
            raise ValueError(f"No model snapshot found for run {run_id}")

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
        elif algorithm == "cql":
            agent = CQL(
                state_dim=N_CATEGORIES, action_dim=N_ITEMS,
                alpha=hyperparams.get("alpha", 1.0),
                gamma=gamma, lr=lr, hidden_dims=hidden_dims,
                target_update_tau=tau,
            )
            agent.q_network.load_state_dict(state_dict["q_network"])
            agent.target_network.load_state_dict(state_dict["target_network"])
        elif algorithm == "dqn":
            agent = DQN(
                state_dim=N_CATEGORIES, action_dim=N_ITEMS,
                gamma=gamma, lr=lr, hidden_dims=hidden_dims,
                target_update_tau=tau,
            )
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

        return agent.get_action

    def _validate_policy(self, policy_fn, replay_buffer) -> dict:
        """Run policy quality validation before FQE.

        Returns validation results dict. If validation fails critically,
        the FQE run is aborted early.
        """
        validator = PolicyValidator(
            state_dim=N_CATEGORIES, action_dim=N_ITEMS
        )
        result = validator.validate(policy_fn, replay_buffer, n_samples=1000)
        return result.to_dict()

    def _run_fqe(self, evaluation_id: int, policy_fn, params: dict):
        replay_buffer = ReplayBuffer(capacity=500_000, chunk_size=50_000)

        if replay_buffer.size == 0:
            self._fail_evaluation(evaluation_id, "ReplayBuffer empty")
            return

        validation_result = self._validate_policy(policy_fn, replay_buffer)

        db = SessionLocal()
        try:
            evaluation = db.query(FQEEvaluation).get(evaluation_id)

            if not validation_result["is_valid"]:
                evaluation.status = "failed"
                evaluation.completed_at = datetime.utcnow()
                evaluation.hyperparameters = {
                    **evaluation.hyperparameters,
                    "validation_result": validation_result,
                }
                db.commit()
                return

            if validation_result["warnings"]:
                evaluation.hyperparameters = {
                    **evaluation.hyperparameters,
                    "validation_warnings": validation_result["warnings"],
                    "validation_metrics": validation_result["metrics"],
                }
                db.commit()
        finally:
            db.close()

        fqe = FittedQEvaluation(
            policy_fn=policy_fn,
            state_dim=N_CATEGORIES,
            action_dim=N_ITEMS,
            gamma=params.get("gamma", 0.99),
            lr=params.get("lr", 1e-3),
            hidden_dims=params.get("hidden_dims", [256, 256]),
            target_update_tau=params.get("target_update_tau", 0.005),
            validate_policy=False,
        )

        epochs = params.get("epochs", 50)
        steps_per_epoch = params.get("steps_per_epoch", 500)
        batch_size = params.get("batch_size", 256)

        db = SessionLocal()
        try:
            evaluation = db.query(FQEEvaluation).get(evaluation_id)

            for epoch in range(1, epochs + 1):
                epoch_loss = 0.0
                epoch_value = 0.0
                n_steps = 0

                for _ in range(steps_per_epoch):
                    batch = replay_buffer.sample(batch_size)
                    metrics = fqe.update(batch)
                    epoch_loss += metrics["fqe_loss"]
                    epoch_value += metrics["estimated_value"]
                    n_steps += 1

                avg_loss = epoch_loss / max(n_steps, 1)
                eval_value = fqe.get_estimated_value(replay_buffer, batch_size)

                fqe_metric = FQEMetric(
                    evaluation_id=evaluation_id,
                    epoch=epoch,
                    estimated_value=eval_value,
                    fqe_loss=avg_loss,
                )
                db.add(fqe_metric)

                evaluation.current_epoch = epoch
                evaluation.estimated_value = eval_value
                db.commit()

                self.active_evaluations[evaluation_id] = {
                    "epoch": epoch,
                    "total_epochs": epochs,
                    "estimated_value": eval_value,
                    "fqe_loss": avg_loss,
                    "status": "running",
                }

            evaluation.status = "completed"
            evaluation.completed_at = datetime.utcnow()
            db.commit()
        except Exception as e:
            self._fail_evaluation(evaluation_id, str(e), db=db)
        finally:
            if evaluation_id in self.active_evaluations:
                del self.active_evaluations[evaluation_id]
            db.close()

    def _fail_evaluation(self, evaluation_id: int, reason: str, db=None):
        close_db = db is None
        if db is None:
            db = SessionLocal()
        try:
            evaluation = db.query(FQEEvaluation).get(evaluation_id)
            if evaluation:
                evaluation.status = "failed"
                evaluation.completed_at = datetime.utcnow()
                db.commit()
        finally:
            if close_db:
                db.close()


fqe_runner = FQERunner()
