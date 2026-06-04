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

    def __init__(self, capacity: int = 500_000, chunk_size: int = CHUNK_SIZE, db_limit: int = None):
        self.capacity = capacity
        self.chunk_size = chunk_size
        self.db_limit = db_limit
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

        effective_total = self.db_total
        if self.db_limit:
            effective_total = min(effective_total, self.db_limit)
        load_total = min(self.capacity, effective_total)
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


class SequenceReplayBuffer:
    """Replay buffer that provides sequences of transitions for RNN-based training.

    Each sample returns a (state, action, reward, next_state, done) tuple plus
    the preceding sequence of (action, clicked, dwell_time, purchased, state).
    """

    def __init__(self, capacity: int = 200_000, chunk_size: int = 20_000, seq_len: int = 10):
        self.capacity = capacity
        self.chunk_size = chunk_size
        self.seq_len = seq_len
        self.size = 0
        self.db_total = 0

        self.states = None
        self.actions = None
        self.rewards = None
        self.next_states = None
        self.dones = None
        self.episode_ids = None
        self.steps = None

        self._initialize()

    def _initialize(self):
        db = SessionLocal()
        try:
            row = db.execute(text("SELECT COUNT(*) FROM offline_transitions")).fetchone()
            self.db_total = row[0]
        finally:
            db.close()

        if self.db_total == 0:
            return

        load_total = min(self.capacity, self.db_total)
        all_states, all_actions, all_rewards, all_next_states, all_dones = [], [], [], [], []
        all_episode_ids, all_steps = [], []

        remaining = load_total
        while remaining > 0:
            fetch_count = min(self.chunk_size, remaining)
            db = SessionLocal()
            try:
                result = db.execute(text(
                    f"SELECT state, action, reward, next_state, done, episode_id, timestamp_step "
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
                all_episode_ids.append(row[5])
                all_steps.append(row[6])

            remaining -= len(result)

        if all_states:
            self.states = np.array(all_states, dtype=np.float32)
            self.actions = np.array(all_actions, dtype=np.int64)
            self.rewards = np.array(all_rewards, dtype=np.float32)
            self.next_states = np.array(all_next_states, dtype=np.float32)
            self.dones = np.array(all_dones, dtype=np.float32)
            self.episode_ids = np.array(all_episode_ids, dtype=np.int64)
            self.steps = np.array(all_steps, dtype=np.int64)
            self.size = len(self.states)

            sort_idx = np.lexsort((self.steps, self.episode_ids))
            self.states = self.states[sort_idx]
            self.actions = self.actions[sort_idx]
            self.rewards = self.rewards[sort_idx]
            self.next_states = self.next_states[sort_idx]
            self.dones = self.dones[sort_idx]
            self.episode_ids = self.episode_ids[sort_idx]
            self.steps = self.steps[sort_idx]

    def refresh_chunk(self):
        if self.db_total == 0 or self.size == 0:
            return
        replace_count = min(self.chunk_size, self.size)
        db = SessionLocal()
        try:
            result = db.execute(text(
                f"SELECT state, action, reward, next_state, done, episode_id, timestamp_step "
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
            self.episode_ids[idx] = row[5]
            self.steps[idx] = row[6]

    def sample(self, batch_size: int):
        """Sample transitions with their preceding sequences.

        Returns tuple of 11 tensors for CQL_RNN.
        """
        if self.size == 0:
            raise RuntimeError("SequenceReplayBuffer is empty. Generate data first.")

        indices = np.random.randint(0, self.size, size=batch_size)
        state_dim = self.states.shape[1]

        batch_states = self.states[indices]
        batch_actions = self.actions[indices]
        batch_rewards = self.rewards[indices]
        batch_next_states = self.next_states[indices]
        batch_dones = self.dones[indices]

        seq_actions = np.zeros((batch_size, self.seq_len), dtype=np.float32)
        seq_clicked = np.zeros((batch_size, self.seq_len), dtype=np.float32)
        seq_dwell = np.zeros((batch_size, self.seq_len), dtype=np.float32)
        seq_purchased = np.zeros((batch_size, self.seq_len), dtype=np.float32)
        seq_states = np.zeros((batch_size, self.seq_len, state_dim), dtype=np.float32)
        seq_lengths = np.zeros(batch_size, dtype=np.int64)

        for i, idx in enumerate(indices):
            ep_id = self.episode_ids[idx]
            step = self.steps[idx]

            ep_mask = self.episode_ids == ep_id
            ep_indices = np.where(ep_mask)[0]
            ep_steps = self.steps[ep_indices]

            preceding_mask = ep_steps < step
            preceding_idx = ep_indices[preceding_mask]

            if len(preceding_idx) > self.seq_len:
                preceding_idx = preceding_idx[-self.seq_len:]

            actual_len = len(preceding_idx)
            seq_lengths[i] = max(actual_len, 1)

            if actual_len > 0:
                offset = self.seq_len - actual_len
                seq_actions[i, offset:] = self.actions[preceding_idx].astype(np.float32)
                seq_clicked[i, offset:] = (self.rewards[preceding_idx] > 0).astype(np.float32)
                seq_dwell[i, offset:] = np.random.exponential(2.0, size=actual_len).astype(np.float32)
                seq_purchased[i, offset:] = (
                    (self.rewards[preceding_idx] > 0) &
                    (np.random.random(actual_len) < 0.1)
                ).astype(np.float32)
                seq_states[i, offset:] = self.states[preceding_idx]

        return (
            torch.tensor(batch_states),
            torch.tensor(batch_actions),
            torch.tensor(batch_rewards),
            torch.tensor(batch_next_states),
            torch.tensor(batch_dones),
            torch.tensor(seq_actions),
            torch.tensor(seq_clicked),
            torch.tensor(seq_dwell),
            torch.tensor(seq_purchased),
            torch.tensor(seq_states),
            torch.tensor(seq_lengths),
        )
