import numpy as np
from backend.config import N_CATEGORIES, N_ITEMS, ITEMS_PER_CATEGORY, EPISODE_LENGTH


class UserModel:
    """Models user interest state transitions across 10 categories."""

    ALPHA = 0.1      # interest reinforcement on click
    BETA = 0.05      # interest decay on no-click
    DECAY = 0.95     # passive decay for non-target categories

    @staticmethod
    def initial_state() -> np.ndarray:
        state = np.random.dirichlet(np.ones(N_CATEGORIES))
        return state.astype(np.float32)

    @staticmethod
    def transition(state: np.ndarray, category: int, clicked: bool) -> np.ndarray:
        next_state = state.copy()
        if clicked:
            next_state[category] = min(1.0, state[category] + UserModel.ALPHA)
            for j in range(N_CATEGORIES):
                if j != category:
                    next_state[j] = state[j] * UserModel.DECAY
        else:
            next_state[category] = max(0.0, state[category] - UserModel.BETA)
            for j in range(N_CATEGORIES):
                if j != category:
                    next_state[j] = state[j] * UserModel.DECAY
        total = next_state.sum()
        if total > 0:
            next_state = next_state / total
        return next_state.astype(np.float32)
