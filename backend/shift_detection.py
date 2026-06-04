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

        Two-stage validation:
        1. Sample count must meet the configured minimum threshold.
        2. Data quality checks ensure the sample is representative:
           - Action distribution entropy not too low (not degenerate)
           - Reward variance is non-trivial (signal exists)
           - State dimensions have meaningful variation (not collapsed)
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

        quality_issues = self._validate_data_quality(actions, rewards, states)
        if quality_issues:
            self._initialized = False
            self._quality_issues = quality_issues
            return

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
        self._quality_issues = []

    def _validate_data_quality(self, actions: np.ndarray, rewards: np.ndarray,
                               states: np.ndarray) -> list:
        """Check data distribution quality before using it as a baseline.

        Returns a list of issue descriptions. Empty list means data is acceptable.
        """
        issues = []

        action_counts = np.bincount(actions, minlength=N_ITEMS).astype(np.float64)
        action_dist = action_counts / action_counts.sum()
        nonzero_dist = action_dist[action_dist > 0]
        entropy = -np.sum(nonzero_dist * np.log(nonzero_dist))
        max_entropy = np.log(N_ITEMS)
        entropy_ratio = entropy / max_entropy
        min_entropy_ratio = self.config.get("min_action_entropy_ratio", 0.3)
        if entropy_ratio < min_entropy_ratio:
            issues.append(
                f"Action distribution entropy too low: ratio={entropy_ratio:.3f} "
                f"(min={min_entropy_ratio}). Data may be concentrated on few actions."
            )

        unique_actions = np.unique(actions)
        min_coverage = self.config.get("min_action_coverage_ratio", 0.1)
        coverage = len(unique_actions) / N_ITEMS
        if coverage < min_coverage:
            issues.append(
                f"Action coverage too sparse: {len(unique_actions)}/{N_ITEMS} "
                f"({coverage:.1%}, min={min_coverage:.0%}). "
                f"Baseline cannot represent unseen actions."
            )

        reward_std = float(rewards.std())
        min_reward_std = self.config.get("min_reward_std", 0.01)
        if reward_std < min_reward_std:
            issues.append(
                f"Reward variance too small: std={reward_std:.6f} "
                f"(min={min_reward_std}). No meaningful signal in rewards."
            )

        state_stds = states.std(axis=0)
        collapsed_dims = np.sum(state_stds < 1e-6)
        max_collapsed_ratio = self.config.get("max_collapsed_state_dims_ratio", 0.5)
        collapsed_ratio = collapsed_dims / states.shape[1]
        if collapsed_ratio > max_collapsed_ratio:
            issues.append(
                f"State space partially collapsed: {collapsed_dims}/{states.shape[1]} "
                f"dimensions have near-zero variance ({collapsed_ratio:.0%} > "
                f"{max_collapsed_ratio:.0%})."
            )

        return issues

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

        Retraining decision uses a composite score that accounts for:
        - Individual severity (how far each metric exceeds its threshold)
        - Cross-metric correlation (co-occurring shifts reinforce each other)
        - Minimum severe alert count as a baseline gate
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
                self._evaluate_retrain_decision(results, severe_alerts, min_severe_for_retrain)
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

    def _evaluate_retrain_decision(self, results: list, severe_alerts: list,
                                   min_severe: int) -> bool:
        """Decide whether to retrain by evaluating cross-metric relationships.

        The decision considers:
        1. Baseline gate: at least min_severe individual severe alerts must exist.
        2. Composite severity score: each alert contributes its normalized overshoot,
           with a correlation bonus when multiple related metrics shift together.
        3. The composite score must exceed a threshold to trigger retraining.

        Correlation logic:
        - action_distribution + reward_distribution shifting together indicates the
          underlying user behavior has changed (strong signal).
        - state_distribution + action_distribution co-shift indicates environment
          dynamics changed (strong signal).
        - new_items alone is weaker unless accompanied by action/state shifts.
        """
        if len(severe_alerts) < min_severe:
            return False

        normalized_scores = {}
        for r in results:
            if r["is_alert"]:
                overshoot = r["metric_value"] / max(r["threshold"], 1e-8)
                normalized_scores[r["shift_type"]] = overshoot

        correlation_pairs = [
            ("action_distribution", "reward_distribution", 1.5),
            ("state_distribution", "action_distribution", 1.4),
            ("new_items", "action_distribution", 1.3),
            ("new_items", "state_distribution", 1.2),
        ]

        composite_score = sum(normalized_scores.values())

        for type_a, type_b, boost in correlation_pairs:
            if type_a in normalized_scores and type_b in normalized_scores:
                pair_contribution = (
                    normalized_scores[type_a] + normalized_scores[type_b]
                ) * (boost - 1.0)
                composite_score += pair_contribution

        composite_threshold = self.config.get("composite_retrain_threshold", 4.0)
        return composite_score >= composite_threshold

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
