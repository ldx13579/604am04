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
from backend.environment.simulator import RecommendationEnv
from backend.config import N_CATEGORIES, N_ITEMS, FINETUNE_CONFIG


class OnlineFinetuner:
    """Periodically fine-tunes the production policy using collected interaction data.

    Uses CQL loss with lower alpha to maintain conservatism while adapting.
    """

    def __init__(self):
        self.is_running = False
        self.last_run_at = None
        self.last_reward_improvement = None
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

    def _do_finetune(self):
        config = FINETUNE_CONFIG
        db = SessionLocal()
        try:
            interactions = db.query(OnlineInteraction).filter(
                OnlineInteraction.consumed == False
            ).order_by(OnlineInteraction.timestamp).limit(10000).all()

            if len(interactions) < config["min_buffer_size"]:
                return None

            finetune_run = FinetuneRun(
                status="running",
                n_interactions_used=len(interactions),
                started_at=datetime.utcnow(),
            )

            prod_version = db.query(PolicyVersion).filter(
                PolicyVersion.stage == "production"
            ).order_by(PolicyVersion.created_at.desc()).first()

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

        finally:
            db.close()

        agent = CQL(
            state_dim=N_CATEGORIES,
            action_dim=N_ITEMS,
            alpha=config["alpha"],
            lr=config["lr"],
            gamma=0.99,
        )

        if prod_version:
            db = SessionLocal()
            try:
                snapshot = db.query(ModelSnapshot).get(prod_version.snapshot_id)
                if snapshot:
                    buffer = io.BytesIO(snapshot.parameters_blob)
                    state_dict = torch.load(buffer, map_location="cpu", weights_only=False)
                    if "q_network" in state_dict:
                        agent.q_network.load_state_dict(state_dict["q_network"])
                        agent.target_network.load_state_dict(state_dict["target_network"])
            finally:
                db.close()

        env = RecommendationEnv(seed=42)
        reward_before = env.evaluate_policy(agent.get_action, n_episodes=10)

        n = len(states)
        batch_size = config["batch_size"]
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
                agent.update(batch)

        reward_after = env.evaluate_policy(agent.get_action, n_episodes=10)

        db = SessionLocal()
        try:
            finetune_run = db.query(FinetuneRun).get(run_id)
            finetune_run.reward_before = reward_before
            finetune_run.reward_after = reward_after
            finetune_run.status = "completed"
            finetune_run.completed_at = datetime.utcnow()

            if reward_after > reward_before:
                buf = io.BytesIO()
                torch.save({
                    "q_network": agent.q_network.state_dict(),
                    "target_network": agent.target_network.state_dict(),
                }, buf)

                source_run = None
                if prod_version:
                    source_run = db.query(TrainingRun).get(prod_version.run_id)

                snapshot = ModelSnapshot(
                    run_id=prod_version.run_id if prod_version else 0,
                    epoch=0,
                    algorithm="cql",
                    parameters_blob=buf.getvalue(),
                    hyperparameters={"alpha": config["alpha"], "lr": config["lr"]},
                    state_dim=N_CATEGORIES,
                    action_dim=N_ITEMS,
                    performance_reward=reward_after,
                )
                db.add(snapshot)
                db.commit()
                db.refresh(snapshot)

                new_version = PolicyVersion(
                    run_id=prod_version.run_id if prod_version else 0,
                    snapshot_id=snapshot.id,
                    version_tag=f"finetune-{run_id}",
                    stage="candidate",
                    notes=f"Online finetune: reward {reward_before:.3f} -> {reward_after:.3f}",
                )
                db.add(new_version)

            for interaction_id in [i.id for i in db.query(OnlineInteraction).filter(
                OnlineInteraction.consumed == False
            ).limit(len(states)).all()]:
                db.query(OnlineInteraction).filter(
                    OnlineInteraction.id == interaction_id
                ).update({"consumed": True})

            db.commit()

            self.last_run_at = datetime.utcnow()
            self.last_reward_improvement = reward_after - reward_before
            return run_id
        finally:
            db.close()


online_finetuner = OnlineFinetuner()
