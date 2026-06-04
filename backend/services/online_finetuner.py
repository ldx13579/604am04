import io
import time
import threading
import numpy as np
import torch
from datetime import datetime
from backend.database import SessionLocal
from backend.models import (
    OnlineInteraction, FinetuneRun, PolicyVersion, ModelSnapshot, TrainingRun
)
from backend.services.policy_loader import policy_loader
from backend.algorithms.cql import CQL
from backend.algorithms.dqn import DQN
from backend.algorithms.cql_rnn import CQL_RNN
from backend.algorithms.ensemble_cql import EnsembleCQL
from backend.algorithms.behavior_cloning import BehaviorCloning
from backend.environment.simulator import RecommendationEnv
from backend.config import N_CATEGORIES, N_ITEMS, FINETUNE_CONFIG


FINETUNE_STRATEGY = {
    "cql": {
        "class": CQL,
        "extra_kwargs": lambda hp: {"alpha": hp.get("alpha", 0.5)},
        "state_dict_keys": ("q_network", "target_network"),
        "save_fn": lambda agent: {
            "q_network": agent.q_network.state_dict(),
            "target_network": agent.target_network.state_dict(),
        },
    },
    "dqn": {
        "class": DQN,
        "extra_kwargs": lambda hp: {},
        "state_dict_keys": ("q_network", "target_network"),
        "save_fn": lambda agent: {
            "q_network": agent.q_network.state_dict(),
            "target_network": agent.target_network.state_dict(),
        },
    },
    "cql_rnn": {
        "class": CQL_RNN,
        "extra_kwargs": lambda hp: {
            "alpha": hp.get("alpha", 0.5),
            "lstm_hidden_size": hp.get("lstm_hidden_size", 128),
            "lstm_num_layers": hp.get("lstm_num_layers", 2),
        },
        "state_dict_keys": ("q_network", "target_network", "user_encoder"),
        "save_fn": lambda agent: {
            "q_network": agent.q_network.state_dict(),
            "target_network": agent.target_network.state_dict(),
            "user_encoder": agent.user_encoder.state_dict(),
        },
    },
    "ensemble_cql": {
        "class": EnsembleCQL,
        "extra_kwargs": lambda hp: {
            "alpha": hp.get("alpha", 0.5),
            "n_models": hp.get("n_models", 5),
            "uncertainty_threshold": hp.get("uncertainty_threshold", 1.0),
        },
        "state_dict_keys": None,
        "save_fn": lambda agent: agent.state_dict(),
    },
    "behavior_cloning": {
        "class": BehaviorCloning,
        "extra_kwargs": lambda hp: {},
        "state_dict_keys": ("policy_network",),
        "save_fn": lambda agent: {
            "policy_network": agent.policy_network.state_dict(),
        },
    },
}


class OnlineFinetuner:
    """Periodically fine-tunes the production policy using collected interaction data.

    Adaptively selects the fine-tuning algorithm based on the current production
    policy's algorithm type.
    """

    def __init__(self):
        self.is_running = False
        self.last_run_at = None
        self.last_reward_improvement = None
        self.last_algorithm_used = None
        self._lock = threading.Lock()

    def get_status(self) -> dict:
        db = SessionLocal()
        try:
            buffer_size = db.query(OnlineInteraction).filter(
                OnlineInteraction.consumed == False
            ).count()
        finally:
            db.close()

        return {
            "is_running": self.is_running,
            "buffer_size": buffer_size,
            "last_run_at": self.last_run_at,
            "last_reward_improvement": self.last_reward_improvement,
            "last_algorithm_used": self.last_algorithm_used,
        }

    def run_finetune(self):
        if self.is_running:
            return None

        with self._lock:
            self.is_running = True

        try:
            return self._do_finetune()
        finally:
            with self._lock:
                self.is_running = False

    def _resolve_algorithm(self, prod_version, db):
        """Determine which algorithm and hyperparams to use for fine-tuning."""
        if prod_version:
            run = db.query(TrainingRun).get(prod_version.run_id)
            if run:
                return run.algorithm, run.hyperparameters
        algorithm = policy_loader.current_algorithm or "cql"
        hyperparams = policy_loader.current_hyperparams or {}
        return algorithm, hyperparams

    def _create_agent(self, algorithm: str, hyperparams: dict):
        """Instantiate the correct agent class for fine-tuning."""
        config = FINETUNE_CONFIG
        strategy = FINETUNE_STRATEGY.get(algorithm)
        if strategy is None:
            strategy = FINETUNE_STRATEGY["cql"]
            algorithm = "cql"

        base_kwargs = dict(
            state_dim=N_CATEGORIES,
            action_dim=N_ITEMS,
            lr=config["lr"],
            gamma=hyperparams.get("gamma", 0.99),
            hidden_dims=hyperparams.get("hidden_dims", [256, 256]),
            target_update_tau=hyperparams.get("target_update_tau", 0.005),
        )

        finetune_hp = {**hyperparams, "alpha": config["alpha"], "lr": config["lr"]}
        extra = strategy["extra_kwargs"](finetune_hp)
        base_kwargs.update(extra)

        if algorithm == "behavior_cloning":
            base_kwargs = {
                "state_dim": N_CATEGORIES,
                "action_dim": N_ITEMS,
                "lr": config["lr"],
                "hidden_dims": hyperparams.get("hidden_dims", [256, 256]),
            }

        agent = strategy["class"](**base_kwargs)
        return agent, strategy

    def _load_weights(self, agent, strategy, snapshot, algorithm):
        """Load pre-trained weights into the agent."""
        buffer = io.BytesIO(snapshot.parameters_blob)
        state_dict = torch.load(buffer, map_location="cpu", weights_only=False)

        if algorithm == "ensemble_cql":
            agent.load_state_dict(state_dict)
        elif strategy["state_dict_keys"]:
            for key in strategy["state_dict_keys"]:
                if key in state_dict and hasattr(agent, key):
                    getattr(agent, key).load_state_dict(state_dict[key])

    def _do_finetune(self):
        config = FINETUNE_CONFIG
        db = SessionLocal()
        try:
            interactions = db.query(OnlineInteraction).filter(
                OnlineInteraction.consumed == False
            ).order_by(OnlineInteraction.timestamp).limit(10000).all()

            if len(interactions) < config["min_buffer_size"]:
                return None

            prod_version = db.query(PolicyVersion).filter(
                PolicyVersion.stage == "production"
            ).order_by(PolicyVersion.created_at.desc()).first()

            algorithm, hyperparams = self._resolve_algorithm(prod_version, db)

            finetune_run = FinetuneRun(
                status="running",
                n_interactions_used=len(interactions),
                started_at=datetime.utcnow(),
            )
            if prod_version:
                finetune_run.source_policy_version_id = prod_version.id
            db.add(finetune_run)
            db.commit()
            db.refresh(finetune_run)
            run_id = finetune_run.id

            states = np.array([i.state for i in interactions], dtype=np.float32)
            actions = np.array([i.action for i in interactions], dtype=np.float32)
            rewards = np.array([i.reward for i in interactions], dtype=np.float32)
            next_states = np.array([i.next_state for i in interactions], dtype=np.float32)
            dones = np.array([1.0 if i.done else 0.0 for i in interactions], dtype=np.float32)

            snapshot = None
            if prod_version:
                snapshot = db.query(ModelSnapshot).get(prod_version.snapshot_id)
        finally:
            db.close()

        agent, strategy = self._create_agent(algorithm, hyperparams)

        if snapshot:
            self._load_weights(agent, strategy, snapshot, algorithm)

        env = RecommendationEnv(seed=42)
        reward_before = env.evaluate_policy(agent.get_action, n_episodes=10)

        n = len(states)
        batch_size = config["batch_size"]
        total_loss = 0.0
        loss_count = 0

        for epoch in range(config["epochs"]):
            for step in range(config["steps_per_epoch"]):
                idx = np.random.randint(0, n, size=batch_size)
                batch = (
                    torch.tensor(states[idx]),
                    torch.tensor(actions[idx]),
                    torch.tensor(rewards[idx]),
                    torch.tensor(next_states[idx]),
                    torch.tensor(dones[idx]),
                )
                metrics = agent.update(batch)
                if metrics and "loss" in metrics:
                    total_loss += metrics["loss"]
                    loss_count += 1

        reward_after = env.evaluate_policy(agent.get_action, n_episodes=10)
        avg_loss = total_loss / loss_count if loss_count > 0 else None

        db = SessionLocal()
        try:
            finetune_run = db.query(FinetuneRun).get(run_id)
            finetune_run.reward_before = reward_before
            finetune_run.reward_after = reward_after
            finetune_run.loss_before = avg_loss
            finetune_run.status = "completed"
            finetune_run.completed_at = datetime.utcnow()

            if reward_after > reward_before:
                buf = io.BytesIO()
                save_data = strategy["save_fn"](agent)
                torch.save(save_data, buf)

                new_snapshot = ModelSnapshot(
                    run_id=prod_version.run_id if prod_version else 0,
                    epoch=0,
                    algorithm=algorithm,
                    parameters_blob=buf.getvalue(),
                    hyperparameters={
                        **hyperparams,
                        "finetune_alpha": config["alpha"],
                        "finetune_lr": config["lr"],
                    },
                    state_dim=N_CATEGORIES,
                    action_dim=N_ITEMS,
                    performance_reward=reward_after,
                )
                db.add(new_snapshot)
                db.commit()
                db.refresh(new_snapshot)

                new_version = PolicyVersion(
                    run_id=prod_version.run_id if prod_version else 0,
                    snapshot_id=new_snapshot.id,
                    version_tag=f"finetune-{algorithm}-{run_id}",
                    stage="candidate",
                    notes=f"Online finetune ({algorithm}): reward {reward_before:.3f} -> {reward_after:.3f}",
                )
                db.add(new_version)

            interaction_ids = [i.id for i in db.query(OnlineInteraction).filter(
                OnlineInteraction.consumed == False
            ).order_by(OnlineInteraction.timestamp).limit(n).all()]

            if interaction_ids:
                db.query(OnlineInteraction).filter(
                    OnlineInteraction.id.in_(interaction_ids)
                ).update({"consumed": True}, synchronize_session=False)

            db.commit()

            self.last_run_at = datetime.utcnow()
            self.last_reward_improvement = reward_after - reward_before
            self.last_algorithm_used = algorithm
            return run_id
        finally:
            db.close()


online_finetuner = OnlineFinetuner()
