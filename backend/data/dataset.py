import numpy as np
import torch
from torch.utils.data import IterableDataset
from sqlalchemy import text
from backend.database import SessionLocal
from backend.config import N_CATEGORIES


CHUNK_SIZE = 50_000


class OfflineRLDataset(IterableDataset):
    """PyTorch IterableDataset that loads offline transitions in chunks from PostgreSQL.

    Only loads CHUNK_SIZE rows at a time to limit memory usage.
    """

    def __init__(self, total_limit: int = None, chunk_size: int = CHUNK_SIZE):
        self.total_limit = total_limit
        self.chunk_size = chunk_size
        db = SessionLocal()
        try:
            row = db.execute(text("SELECT COUNT(*) FROM offline_transitions")).fetchone()
            count = row[0]
            self.size = min(count, total_limit) if total_limit else count
        finally:
            db.close()

    def __len__(self):
        return self.size

    def __iter__(self):
        offset = 0
        remaining = self.size
        while remaining > 0:
            fetch_count = min(self.chunk_size, remaining)
            db = SessionLocal()
            try:
                result = db.execute(text(
                    f"SELECT state, action, reward, next_state, done "
                    f"FROM offline_transitions ORDER BY id "
                    f"LIMIT {fetch_count} OFFSET {offset}"
                )).fetchall()
            finally:
                db.close()

            if not result:
                break

            for row in result:
                yield (
                    torch.tensor(row[0], dtype=torch.float32),
                    torch.tensor(row[1], dtype=torch.int64),
                    torch.tensor(row[2], dtype=torch.float32),
                    torch.tensor(row[3], dtype=torch.float32),
                    torch.tensor(row[4], dtype=torch.float32),
                )

            offset += len(result)
            remaining -= len(result)


class ReplayBuffer:
    """Replay buffer that loads data in chunks from PostgreSQL.

    Keeps a fixed-size window in memory. Samples uniformly from the loaded chunk,
    and periodically rotates to a new random chunk for diversity.
    """

    def __init__(self, capacity: int = 500_000, chunk_size: int = CHUNK_SIZE):
        self.capacity = capacity
        self.chunk_size = chunk_size
        self.states = None
        self.actions = None
        self.rewards = None
        self.next_states = None
        self.dones = None
        self.size = 0
        self.db_total = 0
        self._loaded_chunks = 0
        self._initialize()

    def _initialize(self):
        """Load initial chunks from database up to capacity."""
        db = SessionLocal()
        try:
            row = db.execute(text("SELECT COUNT(*) FROM offline_transitions")).fetchone()
            self.db_total = row[0]
        finally:
            db.close()

        if self.db_total == 0:
            return

        load_total = min(self.capacity, self.db_total)
        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_dones = []

        offset = 0
        remaining = load_total
        while remaining > 0:
            fetch_count = min(self.chunk_size, remaining)
            db = SessionLocal()
            try:
                result = db.execute(text(
                    f"SELECT state, action, reward, next_state, done "
                    f"FROM offline_transitions ORDER BY RANDOM() "
                    f"LIMIT {fetch_count}"
                )).fetchall()
            finally:
                db.close()

            if not result:
                break

            for row in result:
                all_states.append(row[0])
                all_actions.append(row[1])
                all_rewards.append(row[2])
                all_next_states.append(row[3])
                all_dones.append(float(row[4]))

            remaining -= len(result)
            offset += len(result)
            self._loaded_chunks += 1

        if all_states:
            self.states = np.array(all_states, dtype=np.float32)
            self.actions = np.array(all_actions, dtype=np.int64)
            self.rewards = np.array(all_rewards, dtype=np.float32)
            self.next_states = np.array(all_next_states, dtype=np.float32)
            self.dones = np.array(all_dones, dtype=np.float32)
            self.size = len(self.states)

    def refresh_chunk(self):
        """Replace a portion of the buffer with new random samples from DB."""
        if self.db_total == 0:
            return

        replace_count = min(self.chunk_size, self.size)
        db = SessionLocal()
        try:
            result = db.execute(text(
                f"SELECT state, action, reward, next_state, done "
                f"FROM offline_transitions ORDER BY RANDOM() "
                f"LIMIT {replace_count}"
            )).fetchall()
        finally:
            db.close()

        if not result:
            return

        indices = np.random.choice(self.size, size=len(result), replace=False)
        for i, row in enumerate(result):
            idx = indices[i]
            self.states[idx] = row[0]
            self.actions[idx] = row[1]
            self.rewards[idx] = row[2]
            self.next_states[idx] = row[3]
            self.dones[idx] = float(row[4])

        self._loaded_chunks += 1

    def sample(self, batch_size: int):
        if self.size == 0:
            raise RuntimeError("ReplayBuffer is empty. Generate data first.")
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.states[indices]),
            torch.tensor(self.actions[indices]),
            torch.tensor(self.rewards[indices]),
            torch.tensor(self.next_states[indices]),
            torch.tensor(self.dones[indices]),
        )
