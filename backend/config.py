import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rl_user:rl_password@localhost:5432/offline_rl")

N_CATEGORIES = 10
N_ITEMS = 100
ITEMS_PER_CATEGORY = 10
EPISODE_LENGTH = 50
N_TRANSITIONS = 1_000_000
BATCH_INSERT_SIZE = 10_000

DEFAULT_HYPERPARAMS = {
    "cql": {
        "alpha": 1.0,
        "gamma": 0.99,
        "lr": 3e-4,
        "batch_size": 256,
        "epochs": 200,
        "steps_per_epoch": 1000,
        "hidden_dims": [256, 256],
        "target_update_tau": 0.005,
    },
    "dqn": {
        "gamma": 0.99,
        "lr": 3e-4,
        "batch_size": 256,
        "epochs": 200,
        "steps_per_epoch": 1000,
        "hidden_dims": [256, 256],
        "target_update_tau": 0.005,
    },
    "behavior_cloning": {
        "lr": 3e-4,
        "batch_size": 256,
        "epochs": 200,
        "steps_per_epoch": 1000,
        "hidden_dims": [256, 256],
    },
}
