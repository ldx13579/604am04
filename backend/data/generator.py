import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func as sa_func
from backend.config import N_ITEMS, N_TRANSITIONS, BATCH_INSERT_SIZE, N_CATEGORIES
from backend.environment.simulator import RecommendationEnv
from backend.models import OfflineTransition, Item, GenerationStatus
from backend.database import SessionLocal


class DataGenerator:
    """Generates offline transitions using random policy and inserts into PostgreSQL.

    State is persisted to the generation_status table so progress survives restarts.
    """

    def __init__(self):
        self.env = RecommendationEnv()
        self.progress = 0.0
        self.is_running = False
        self.total_generated = 0

    def restore_state(self):
        """Restore generator state from database after service restart."""
        db = SessionLocal()
        try:
            status = db.query(GenerationStatus).get(1)
            if status:
                self.progress = status.progress
                self.total_generated = status.total_generated
                if status.is_running:
                    status.is_running = False
                    db.commit()
            else:
                count = db.query(sa_func.count(OfflineTransition.id)).scalar() or 0
                self.total_generated = count
                self.progress = min(count / N_TRANSITIONS, 1.0)
        finally:
            db.close()

    def _get_or_create_status(self, db: Session) -> GenerationStatus:
        status = db.query(GenerationStatus).get(1)
        if not status:
            status = GenerationStatus(
                id=1,
                is_running=False,
                progress=0.0,
                total_generated=0,
                target_count=N_TRANSITIONS,
                last_episode_id=0,
            )
            db.add(status)
            db.commit()
            db.refresh(status)
        return status

    def _update_status(self, db: Session, **kwargs):
        status = self._get_or_create_status(db)
        for k, v in kwargs.items():
            setattr(status, k, v)
        db.commit()

    def generate(self, n_transitions: int = N_TRANSITIONS, callback=None):
        self.is_running = True

        db = SessionLocal()
        try:
            self._seed_items(db)

            status = self._get_or_create_status(db)
            already_generated = db.query(sa_func.count(OfflineTransition.id)).scalar() or 0
            remaining = n_transitions - already_generated

            if remaining <= 0:
                self.total_generated = already_generated
                self.progress = 1.0
                self._update_status(db, is_running=False, progress=1.0,
                                    total_generated=already_generated)
                return

            episode_id = status.last_episode_id
            self.total_generated = already_generated

            self._update_status(db, is_running=True, progress=already_generated / n_transitions,
                                total_generated=already_generated)

            buffer = []
            state = self.env.reset()
            step_in_episode = 0

            for i in range(remaining):
                action = np.random.randint(0, N_ITEMS)
                next_state, reward, done = self.env.step(action)

                buffer.append(OfflineTransition(
                    state=state.tolist(),
                    action=int(action),
                    reward=float(reward),
                    next_state=next_state.tolist(),
                    done=done,
                    episode_id=episode_id,
                    timestamp_step=step_in_episode,
                ))

                state = next_state
                step_in_episode += 1

                if done:
                    state = self.env.reset()
                    episode_id += 1
                    step_in_episode = 0

                if len(buffer) >= BATCH_INSERT_SIZE:
                    db.bulk_save_objects(buffer)
                    db.commit()
                    buffer = []
                    self.total_generated = already_generated + i + 1
                    self.progress = self.total_generated / n_transitions
                    self._update_status(
                        db,
                        progress=self.progress,
                        total_generated=self.total_generated,
                        last_episode_id=episode_id,
                    )
                    if callback:
                        callback(self.progress)

            if buffer:
                db.bulk_save_objects(buffer)
                db.commit()
                self.total_generated = n_transitions

            self.progress = 1.0
            self._update_status(db, is_running=False, progress=1.0,
                                total_generated=n_transitions, last_episode_id=episode_id)
        except Exception:
            self._update_status(db, is_running=False)
            raise
        finally:
            db.close()
            self.is_running = False

    def _seed_items(self, db: Session):
        if db.query(Item).count() > 0:
            return
        items = []
        for i in range(N_ITEMS):
            category_id = i // (N_ITEMS // N_CATEGORIES)
            embedding = [0.0] * N_CATEGORIES
            embedding[category_id] = 1.0
            items.append(Item(
                id=i,
                category_id=category_id,
                popularity=float(np.random.uniform(0.5, 1.5)),
                embedding=embedding,
            ))
        db.bulk_save_objects(items)
        db.commit()


data_generator = DataGenerator()
