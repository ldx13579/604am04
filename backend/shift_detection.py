import numpy as np
from datetime import datetime
from sqlalchemy import text
from backend.database import SessionLocal
from backend.models import ShiftDetectionRecord, TrainingRun
from backend.config import SHIFT_DETECTION_CONFIG, N_ITEMS, N_CATEGORIES


class DistributionShiftDetector:
    """Detects data distribution shifts between offline data and online simulation.

    Monitors:
    1. Action distribution divergence (KL divergence)
    2. New item ratio (items in online env not seen in offline data)
    3. Reward distribution shift
    4. State space coverage changes
    """

    def __init__(self, config: dict = None):
        self.config = config or SHIFT_DETECTION_CONFIG
        self.offline_action_dist = None
        self.offline_reward_stats = None
        self.offline_state_mean = None
        self.offline_state_std = None
        self.known_items = set()
        self._initialized = False
        self._baseline_sample_count = 0

    def initialize_baseline(self):
        """Compute baseline statistics from offline data.

        Only proceeds if sample count meets the configured minimum threshold,
        ensuring statistical features are computed on sufficient data.
        """
        min_samples = self.config.get("min_baseline_samples", 10000)

        db = SessionLocal()
        try:
            count_row = db.execute(text(
                "SELECT COUNT(*) FROM offline_transitions"
            )).fetchone()
            total_available = count_row[0] if count_row else 0

            if total_available < min_samples:
                self._initialized = False
                self._baseline_sample_count = total_available
                return

            sample_size = min(total_available, 100000)
            result = db.execute(text(
                f"SELECT action, reward, state FROM offline_transitions "
                f"ORDER BY RANDOM() LIMIT {sample_size}"
            )).fetchall()
        finally:
            db.close()

        if not result or len(result) < min_samples:
            self._initialized = False
            self._baseline_sample_count = len(result) if result else 0
            return

        self._baseline_sample_count = len(result)

        actions = np.array([r[0] for r in result], dtype=np.int64)
        rewards = np.array([r[1] for r in result], dtype=np.float32)
        states = np.array([r[2] for r in result], dtype=np.float32)

        action_counts = np.bincount(actions, minlength=N_ITEMS).astype(np.float64)
        self.offline_action_dist = action_counts / action_counts.sum()

        self.offline_reward_stats = {
            "mean": float(rewards.mean()),
            "std": float(rewards.std()),
        }

        self.offline_state_mean = states.mean(axis=0)
        self.offline_state_std = states.std(axis=0)

        self.known_items = set(actions.tolist())
        self._initialized = True

    def detect_shift(self, online_actions: np.ndarray, online_rewards: np.ndarray,
                     online_states: np.ndarray, current_items: set = None) -> list:
        """Run all shift detection checks.

        Returns list of detection results (dicts with metric info and alert status).
        """
        if not self._initialized:
            self.initialize_baseline()

        if not self._initialized:
            return []

        results = []

        kl_result = self._check_action_distribution(online_actions)
        results.append(kl_result)

        reward_result = self._check_reward_shift(online_rewards)
        results.append(reward_result)

        state_result = self._check_state_shift(online_states)
        results.append(state_result)

        if current_items is not None:
            item_result = self._check_new_items(current_items)
            results.append(item_result)

        return results

    def _check_action_distribution(self, online_actions: np.ndarray) -> dict:
        """Compute KL divergence between offline and online action distributions."""
        online_counts = np.bincount(online_actions, minlength=N_ITEMS).astype(np.float64)
        online_dist = online_counts / max(online_counts.sum(), 1)

        epsilon = 1e-8
        p = self.offline_action_dist + epsilon
        q = online_dist + epsilon
        p = p / p.sum()
        q = q / q.sum()

        kl_div = float(np.sum(p * np.log(p / q)))

        threshold = self.config["kl_threshold"]
        return {
            "shift_type": "action_distribution",
            "metric_name": "kl_divergence",
            "metric_value": kl_div,
            "threshold": threshold,
            "is_alert": kl_div > threshold,
            "details": {
                "top_offline_actions": np.argsort(self.offline_action_dist)[-5:].tolist(),
                "top_online_actions": np.argsort(online_dist)[-5:].tolist(),
            },
        }

    def _check_reward_shift(self, online_rewards: np.ndarray) -> dict:
        """Check if reward distribution has shifted significantly."""
        online_mean = float(online_rewards.mean())
        offline_mean = self.offline_reward_stats["mean"]
        offline_std = max(self.offline_reward_stats["std"], 1e-6)

        z_score = abs(online_mean - offline_mean) / offline_std
        threshold = 2.0

        return {
            "shift_type": "reward_distribution",
            "metric_name": "reward_z_score",
            "metric_value": z_score,
            "threshold": threshold,
            "is_alert": z_score > threshold,
            "details": {
                "online_mean": online_mean,
                "offline_mean": offline_mean,
                "offline_std": float(offline_std),
            },
        }

    def _check_state_shift(self, online_states: np.ndarray) -> dict:
        """Check state space distribution shift using Mahalanobis-like distance."""
        online_mean = online_states.mean(axis=0)
        diff = online_mean - self.offline_state_mean
        safe_std = np.maximum(self.offline_state_std, 1e-6)
        normalized_dist = float(np.sqrt(np.sum((diff / safe_std) ** 2)))

        threshold = float(np.sqrt(N_CATEGORIES)) * 2.0

        return {
            "shift_type": "state_distribution",
            "metric_name": "state_mahalanobis",
            "metric_value": normalized_dist,
            "threshold": threshold,
            "is_alert": normalized_dist > threshold,
            "details": {
                "online_state_mean": online_mean.tolist(),
                "offline_state_mean": self.offline_state_mean.tolist(),
            },
        }

    def _check_new_items(self, current_items: set) -> dict:
        """Check ratio of items not present in offline data."""
        new_items = current_items - self.known_items
        ratio = len(new_items) / max(len(current_items), 1)
        threshold = self.config["new_item_ratio_threshold"]

        return {
            "shift_type": "new_items",
            "metric_name": "new_item_ratio",
            "metric_value": ratio,
            "threshold": threshold,
            "is_alert": ratio > threshold,
            "details": {
                "new_item_count": len(new_items),
                "total_items": len(current_items),
                "new_item_ids": list(new_items)[:20],
            },
        }

    def run_detection_and_record(self, online_actions: np.ndarray,
                                 online_rewards: np.ndarray,
                                 online_states: np.ndarray,
                                 current_items: set = None) -> list:
        """Run detection, save results to DB, and optionally trigger retraining.

        Retraining is only triggered when the number of severe alerts
        (metric_value >= threshold * severity_multiplier) meets the configured minimum.
        """
        results = self.detect_shift(online_actions, online_rewards, online_states, current_items)

        db = SessionLocal()
        alerts = []
        severe_alerts = []
        severity_multiplier = self.config.get("severity_multiplier", 2.0)
        min_severe_for_retrain = self.config.get("retrain_min_severe_alerts", 2)

        try:
            for result in results:
                is_severe = result["is_alert"] and (
                    result["metric_value"] >= result["threshold"] * severity_multiplier
                )
                result["is_severe"] = is_severe

                record = ShiftDetectionRecord(
                    detection_time=datetime.utcnow(),
                    shift_type=result["shift_type"],
                    metric_name=result["metric_name"],
                    metric_value=result["metric_value"],
                    threshold=result["threshold"],
                    is_alert=result["is_alert"],
                    details=result.get("details"),
                    triggered_retrain=False,
                )
                db.add(record)
                if result["is_alert"]:
                    alerts.append(result)
                if is_severe:
                    severe_alerts.append(result)

            db.commit()

            should_retrain = (
                self.config.get("auto_retrain", False) and
                len(severe_alerts) >= min_severe_for_retrain
            )

            if should_retrain:
                retrain_run_id = self._trigger_retrain(db)
                for record in db.query(ShiftDetectionRecord).filter(
                    ShiftDetectionRecord.is_alert == True,
                    ShiftDetectionRecord.triggered_retrain == False,
                ).order_by(ShiftDetectionRecord.id.desc()).limit(len(alerts)).all():
                    record.triggered_retrain = True
                    record.retrain_run_id = retrain_run_id
                db.commit()

        finally:
            db.close()

        return results

    def _trigger_retrain(self, db) -> int:
        """Create a new training run to retrain the policy."""
        from backend.config import DEFAULT_HYPERPARAMS
        params = DEFAULT_HYPERPARAMS["cql"].copy()

        run = TrainingRun(
            algorithm="cql",
            hyperparameters=params,
            status="pending",
            total_epochs=params.get("epochs", 200),
            current_epoch=0,
        )
        db.add(run)
        db.commit()
        db.refresh(run)

        from backend.algorithms.trainer import trainer
        import threading
        t = threading.Thread(
            target=trainer.start_training,
            args=("cql", params, run.id),
            daemon=True,
        )
        t.start()

        return run.id

    def simulate_online_data(self, n_episodes: int = 50) -> tuple:
        """Generate online simulation data for shift comparison."""
        from backend.environment.simulator import RecommendationEnv
        env = RecommendationEnv()

        all_actions = []
        all_rewards = []
        all_states = []

        for _ in range(n_episodes):
            state = env.reset()
            done = False
            while not done:
                action = np.random.randint(0, N_ITEMS)
                all_states.append(state.copy())
                all_actions.append(action)
                state, reward, done = env.step(action)
                all_rewards.append(reward)

        return (
            np.array(all_actions),
            np.array(all_rewards),
            np.array(all_states),
        )


shift_detector = DistributionShiftDetector()
