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
        "chunk_refresh_interval": 20,
    },
    "cql_rnn": {
        "alpha": 1.0,
        "gamma": 0.99,
        "lr": 3e-4,
        "batch_size": 128,
        "epochs": 200,
        "steps_per_epoch": 1000,
        "hidden_dims": [256, 256],
        "lstm_hidden_size": 128,
        "lstm_num_layers": 2,
        "seq_len": 10,
        "target_update_tau": 0.005,
        "chunk_refresh_interval": 20,
    },
    "dqn": {
        "gamma": 0.99,
        "lr": 3e-4,
        "batch_size": 256,
        "epochs": 200,
        "steps_per_epoch": 1000,
        "hidden_dims": [256, 256],
        "target_update_tau": 0.005,
        "chunk_refresh_interval": 20,
    },
    "behavior_cloning": {
        "lr": 3e-4,
        "batch_size": 256,
        "epochs": 200,
        "steps_per_epoch": 1000,
        "hidden_dims": [256, 256],
        "chunk_refresh_interval": 20,
    },
    "ensemble_cql": {
        "alpha": 1.0,
        "gamma": 0.99,
        "lr": 3e-4,
        "batch_size": 256,
        "epochs": 200,
        "steps_per_epoch": 1000,
        "hidden_dims": [256, 256],
        "target_update_tau": 0.005,
        "n_models": 5,
        "uncertainty_threshold": 1.0,
        "exploration_budget": 0.3,
        "correlation_threshold": 0.95,
        "min_active_models": 3,
        "max_models": 7,
        "ucb_coefficient": 1.0,
        "chunk_refresh_interval": 20,
    },
}

FQE_DEFAULTS = {
    "gamma": 0.99,
    "lr": 1e-3,
    "epochs": 50,
    "steps_per_epoch": 500,
    "batch_size": 256,
    "hidden_dims": [256, 256],
    "target_update_tau": 0.005,
}

SHIFT_DETECTION_CONFIG = {
    "kl_threshold": 0.5,
    "action_dist_threshold": 0.3,
    "new_item_ratio_threshold": 0.1,
    "check_interval_seconds": 300,
    "auto_retrain": True,
    "min_baseline_samples": 10000,
    "min_action_entropy_ratio": 0.3,
    "min_action_coverage_ratio": 0.1,
    "min_reward_std": 0.01,
    "max_collapsed_state_dims_ratio": 0.5,
    "retrain_min_severe_alerts": 2,
    "severity_multiplier": 2.0,
    "composite_retrain_threshold": 4.0,
    "max_snapshots_per_run": 5,
}

FINETUNE_CONFIG = {
    "interval_seconds": 3600,
    "min_buffer_size": 500,
    "epochs": 5,
    "steps_per_epoch": 100,
    "batch_size": 128,
    "lr": 1e-4,
    "alpha": 0.5,
    "auto_promote": False,
}

AB_TEST_CONFIG = {
    "aggregation_interval_seconds": 60,
    "window_minutes": 5,
}

BENCHMARK_CONFIG = {
    "default_sizes": [10000, 50000, 100000, 500000, 1000000],
    "default_epochs": 50,
}
