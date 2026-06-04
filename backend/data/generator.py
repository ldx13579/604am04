import numpy as np
from sqlalchemy.orm import Session
from backend.config import N_ITEMS, N_TRANSITIONS, BATCH_INSERT_SIZE, N_CATEGORIES
from backend.environment.simulator import RecommendationEnv
from backend.models import OfflineTransition, Item
from backend.database import SessionLocal


class DataGenerator:
    """Generates offline transitions using random policy and inserts into PostgreSQL."""

    def __init__(self):
        self.env = RecommendationEnv()
        self.progress = 0.0
        self.is_running = False
        self.total_generated = 0

    def generate(self, n_transitions: int = N_TRANSITIONS, callback=None):
        self.is_running = True
        self.total_generated = 0

        db = SessionLocal()
        try:
            self._seed_items(db)
            buffer = []
            episode_id = 0
            state = self.env.reset()
            step_in_episode = 0

            for i in range(n_transitions):
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
                    self.total_generated = i + 1
                    self.progress = (i + 1) / n_transitions
                    if callback:
                        callback(self.progress)

            if buffer:
                db.bulk_save_objects(buffer)
                db.commit()
                self.total_generated = n_transitions

            self.progress = 1.0
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
