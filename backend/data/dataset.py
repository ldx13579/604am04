import numpy as np
import torch
from torch.utils.data import Dataset
from sqlalchemy import text
from backend.database import SessionLocal
from backend.config import N_CATEGORIES


class OfflineRLDataset(Dataset):
    """PyTorch Dataset that loads offline transitions from PostgreSQL."""

    def __init__(self, limit: int = None):
        db = SessionLocal()
        try:
            query = "SELECT state, action, reward, next_state, done FROM offline_transitions"
            if limit:
                query += f" LIMIT {limit}"
            result = db.execute(text(query)).fetchall()

            self.states = np.array([row[0] for row in result], dtype=np.float32)
            self.actions = np.array([row[1] for row in result], dtype=np.int64)
            self.rewards = np.array([row[2] for row in result], dtype=np.float32)
            self.next_states = np.array([row[3] for row in result], dtype=np.float32)
            self.dones = np.array([row[4] for row in result], dtype=np.float32)
        finally:
            db.close()

        self.size = len(self.states)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return (
            torch.tensor(self.states[idx]),
            torch.tensor(self.actions[idx]),
            torch.tensor(self.rewards[idx]),
            torch.tensor(self.next_states[idx]),
            torch.tensor(self.dones[idx]),
        )


class ReplayBuffer:
    """In-memory replay buffer loaded from PostgreSQL for fast batch sampling."""

    def __init__(self, limit: int = None):
        db = SessionLocal()
        try:
            query = "SELECT state, action, reward, next_state, done FROM offline_transitions"
            if limit:
                query += f" ORDER BY RANDOM() LIMIT {limit}"
            result = db.execute(text(query)).fetchall()

            self.states = np.array([row[0] for row in result], dtype=np.float32)
            self.actions = np.array([row[1] for row in result], dtype=np.int64)
            self.rewards = np.array([row[2] for row in result], dtype=np.float32)
            self.next_states = np.array([row[3] for row in result], dtype=np.float32)
            self.dones = np.array([row[4] for row in result], dtype=np.float32)
        finally:
            db.close()

        self.size = len(self.states)

    def sample(self, batch_size: int):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.states[indices]),
            torch.tensor(self.actions[indices]),
            torch.tensor(self.rewards[indices]),
            torch.tensor(self.next_states[indices]),
            torch.tensor(self.dones[indices]),
        )
