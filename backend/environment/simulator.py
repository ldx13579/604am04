import numpy as np
from backend.config import N_CATEGORIES, N_ITEMS, ITEMS_PER_CATEGORY, EPISODE_LENGTH
from backend.environment.user_model import UserModel


class ItemCatalog:
    """Static catalog of items with categories and popularity scores."""

    def __init__(self):
        self.n_items = N_ITEMS
        self.categories = np.zeros(N_ITEMS, dtype=np.int32)
        self.popularities = np.zeros(N_ITEMS, dtype=np.float32)

        for i in range(N_ITEMS):
            self.categories[i] = i // ITEMS_PER_CATEGORY
            self.popularities[i] = np.random.uniform(0.5, 1.5)

    def get_embedding(self, item_id: int) -> np.ndarray:
        emb = np.zeros(N_CATEGORIES, dtype=np.float32)
        emb[self.categories[item_id]] = 1.0
        return emb


class RecommendationEnv:
    """E-commerce recommendation environment simulator.

    State: 10-dim user interest vector (normalized).
    Action: item index in [0, N_ITEMS).
    Reward: 1.0 if click, 0.0 otherwise.
    """

    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.catalog = ItemCatalog()
        self.state = None
        self.step_count = 0

    def reset(self) -> np.ndarray:
        self.state = UserModel.initial_state()
        self.step_count = 0
        return self.state.copy()

    def step(self, action: int):
        category = self.catalog.categories[action]
        click_prob = self._compute_click_prob(self.state, action)
        clicked = np.random.random() < click_prob

        reward = 1.0 if clicked else 0.0
        self.state = UserModel.transition(self.state, category, clicked)
        self.step_count += 1
        done = self.step_count >= EPISODE_LENGTH

        return self.state.copy(), reward, done

    def _compute_click_prob(self, state: np.ndarray, action: int) -> float:
        category = self.catalog.categories[action]
        popularity = self.catalog.popularities[action]
        base_prob = state[category] * popularity
        noise = np.random.normal(0, 0.02)
        return float(np.clip(base_prob + noise, 0.01, 0.95))

    def evaluate_policy(self, policy_fn, n_episodes=20) -> float:
        total_reward = 0.0
        for _ in range(n_episodes):
            state = self.reset()
            ep_reward = 0.0
            while True:
                action = policy_fn(state)
                state, reward, done = self.step(action)
                ep_reward += reward
                if done:
                    break
            total_reward += ep_reward
        return total_reward / n_episodes
