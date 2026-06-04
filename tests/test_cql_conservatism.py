"""Unit tests verifying CQL's conservatism property.

Core property: Q-values for out-of-distribution (OOD) actions should be
significantly lower than Q-values for in-distribution actions after training.
"""
import pytest
import numpy as np
import torch
from backend.algorithms.cql import CQL
from backend.algorithms.dqn import DQN
from backend.algorithms.cql_rnn import CQL_RNN
from backend.algorithms.networks import UserStateEncoder, RNNQNetwork


def generate_synthetic_batch(batch_size=256, state_dim=10, action_dim=100,
                             in_dist_actions=None):
    """Generate synthetic offline data concentrated on a subset of actions."""
    if in_dist_actions is None:
        in_dist_actions = list(range(20))

    states = torch.randn(batch_size, state_dim)
    actions = torch.tensor(
        np.random.choice(in_dist_actions, size=batch_size), dtype=torch.float32
    )
    rewards = torch.rand(batch_size)
    next_states = torch.randn(batch_size, state_dim)
    dones = torch.zeros(batch_size)

    return states, actions, rewards, next_states, dones


def generate_synthetic_sequence_batch(batch_size=64, state_dim=10, action_dim=100,
                                      seq_len=10, in_dist_actions=None):
    """Generate synthetic sequence batch for CQL_RNN."""
    if in_dist_actions is None:
        in_dist_actions = list(range(20))

    states = torch.randn(batch_size, state_dim)
    actions = torch.tensor(
        np.random.choice(in_dist_actions, size=batch_size), dtype=torch.float32
    )
    rewards = torch.rand(batch_size)
    next_states = torch.randn(batch_size, state_dim)
    dones = torch.zeros(batch_size)

    seq_actions = torch.tensor(
        np.random.choice(in_dist_actions, size=(batch_size, seq_len)), dtype=torch.float32
    )
    seq_clicked = (torch.rand(batch_size, seq_len) > 0.5).float()
    seq_dwell = torch.rand(batch_size, seq_len) * 5.0
    seq_purchased = (torch.rand(batch_size, seq_len) > 0.9).float()
    seq_states = torch.randn(batch_size, seq_len, state_dim)
    seq_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)

    return (states, actions, rewards, next_states, dones,
            seq_actions, seq_clicked, seq_dwell, seq_purchased, seq_states, seq_lengths)


class TestCQLConservatism:
    """Tests verifying CQL's conservative Q-value estimates."""

    def setup_method(self):
        self.state_dim = 10
        self.action_dim = 100
        self.in_dist_actions = list(range(20))
        self.ood_actions = list(range(20, 100))

    def _train_agent(self, agent, n_steps=200, in_dist_actions=None):
        """Train agent on synthetic data for n_steps."""
        if in_dist_actions is None:
            in_dist_actions = self.in_dist_actions
        for _ in range(n_steps):
            batch = generate_synthetic_batch(
                batch_size=128,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                in_dist_actions=in_dist_actions,
            )
            agent.update(batch)

    def test_cql_ood_q_values_lower_than_in_dist(self):
        """OOD actions should have significantly lower Q-values than in-distribution actions."""
        cql = CQL(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            alpha=5.0,
            lr=1e-3,
        )

        self._train_agent(cql, n_steps=300)

        test_states = torch.randn(50, self.state_dim)
        with torch.no_grad():
            q_values = cql.q_network(test_states.to(cql.device))

        in_dist_q = q_values[:, self.in_dist_actions].mean().item()
        ood_q = q_values[:, self.ood_actions].mean().item()

        assert ood_q < in_dist_q, (
            f"CQL conservatism violated: OOD Q-values ({ood_q:.4f}) should be "
            f"lower than in-distribution Q-values ({in_dist_q:.4f})"
        )

    def test_cql_penalty_positive(self):
        """CQL penalty should be positive, indicating conservative regularization."""
        cql = CQL(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            alpha=2.0,
            lr=1e-3,
        )

        batch = generate_synthetic_batch(
            batch_size=256,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            in_dist_actions=self.in_dist_actions,
        )
        _, metrics = cql.compute_loss(batch)

        assert metrics["cql_penalty"] > 0, (
            f"CQL penalty should be positive, got {metrics['cql_penalty']:.4f}"
        )

    def test_higher_alpha_more_conservative(self):
        """Higher alpha should suppress overall Q-values more aggressively."""
        torch.manual_seed(42)
        np.random.seed(42)

        cql_low = CQL(state_dim=self.state_dim, action_dim=self.action_dim, alpha=0.1, lr=1e-3)

        torch.manual_seed(42)
        cql_high = CQL(state_dim=self.state_dim, action_dim=self.action_dim, alpha=10.0, lr=1e-3)

        torch.manual_seed(0)
        for _ in range(300):
            batch = generate_synthetic_batch(
                batch_size=128,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                in_dist_actions=self.in_dist_actions,
            )
            cql_low.update(batch)
            cql_high.update(batch)

        test_states = torch.randn(100, self.state_dim)
        with torch.no_grad():
            q_low = cql_low.q_network(test_states.to(cql_low.device))
            q_high = cql_high.q_network(test_states.to(cql_high.device))

        all_q_low = q_low.mean().item()
        all_q_high = q_high.mean().item()

        assert all_q_high < all_q_low, (
            f"Higher alpha should produce lower overall Q-values: "
            f"alpha=10.0 mean Q={all_q_high:.4f}, alpha=0.1 mean Q={all_q_low:.4f}"
        )

    def test_cql_vs_dqn_conservatism(self):
        """CQL should have lower OOD Q-values compared to standard DQN."""
        cql = CQL(state_dim=self.state_dim, action_dim=self.action_dim, alpha=3.0, lr=1e-3)
        dqn = DQN(state_dim=self.state_dim, action_dim=self.action_dim, lr=1e-3)

        self._train_agent(cql, n_steps=200)
        self._train_agent(dqn, n_steps=200)

        test_states = torch.randn(50, self.state_dim)
        with torch.no_grad():
            cql_q = cql.q_network(test_states.to(cql.device))
            dqn_q = dqn.q_network(test_states.to(dqn.device))

        cql_ood = cql_q[:, self.ood_actions].mean().item()
        dqn_ood = dqn_q[:, self.ood_actions].mean().item()

        cql_gap = cql_q[:, self.in_dist_actions].mean().item() - cql_ood
        dqn_gap = dqn_q[:, self.in_dist_actions].mean().item() - dqn_ood

        assert cql_gap > dqn_gap, (
            f"CQL should have larger gap between in-dist and OOD Q-values: "
            f"CQL gap={cql_gap:.4f}, DQN gap={dqn_gap:.4f}"
        )

    def test_cql_q_value_distribution_separation(self):
        """Q-value distributions for in-dist vs OOD should be clearly separated."""
        cql = CQL(state_dim=self.state_dim, action_dim=self.action_dim, alpha=5.0, lr=1e-3)
        self._train_agent(cql, n_steps=300)

        test_states = torch.randn(100, self.state_dim)
        with torch.no_grad():
            q_values = cql.q_network(test_states.to(cql.device))

        in_dist_q = q_values[:, self.in_dist_actions].flatten().cpu().numpy()
        ood_q = q_values[:, self.ood_actions].flatten().cpu().numpy()

        in_dist_mean = in_dist_q.mean()
        ood_mean = ood_q.mean()
        pooled_std = np.sqrt((in_dist_q.std() ** 2 + ood_q.std() ** 2) / 2)

        effect_size = (in_dist_mean - ood_mean) / max(pooled_std, 1e-8)

        assert effect_size > 0.3, (
            f"Effect size between in-dist and OOD Q-values should be meaningful: "
            f"got {effect_size:.4f} (in_dist_mean={in_dist_mean:.4f}, "
            f"ood_mean={ood_mean:.4f})"
        )


class TestCQLRNNConservatism:
    """Tests for CQL+RNN conservatism with sequence-based state encoding."""

    def setup_method(self):
        self.state_dim = 10
        self.action_dim = 100
        self.seq_len = 10
        self.in_dist_actions = list(range(20))
        self.ood_actions = list(range(20, 100))

    def test_cql_rnn_ood_lower(self):
        """CQL_RNN should also produce lower Q-values for OOD actions."""
        agent = CQL_RNN(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            alpha=5.0,
            lr=1e-3,
            lstm_hidden_size=64,
            lstm_num_layers=1,
            seq_len=self.seq_len,
        )

        for _ in range(200):
            batch = generate_synthetic_sequence_batch(
                batch_size=64,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                seq_len=self.seq_len,
                in_dist_actions=self.in_dist_actions,
            )
            agent.update(batch)

        test_state = np.random.randn(self.state_dim).astype(np.float32)
        q_values = agent.get_q_distribution(test_state)

        in_dist_q = q_values[self.in_dist_actions].mean()
        ood_q = q_values[self.ood_actions].mean()

        assert ood_q < in_dist_q, (
            f"CQL_RNN conservatism violated: OOD Q ({ood_q:.4f}) >= in-dist Q ({in_dist_q:.4f})"
        )

    def test_cql_rnn_penalty_positive(self):
        """CQL_RNN penalty should be positive."""
        agent = CQL_RNN(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            alpha=2.0,
            lr=1e-3,
            lstm_hidden_size=64,
            lstm_num_layers=1,
            seq_len=self.seq_len,
        )

        batch = generate_synthetic_sequence_batch(
            batch_size=64,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            seq_len=self.seq_len,
            in_dist_actions=self.in_dist_actions,
        )
        _, metrics = agent.compute_loss(batch)

        assert metrics["cql_penalty"] > 0, (
            f"CQL_RNN penalty should be positive, got {metrics['cql_penalty']:.4f}"
        )

    def test_cql_rnn_with_realistic_sequences(self):
        """CQL_RNN with realistic user sequences (varying clicks, dwell, purchases)
        should produce significantly lower Q-values for OOD actions."""
        agent = CQL_RNN(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            alpha=5.0,
            lr=1e-3,
            lstm_hidden_size=64,
            lstm_num_layers=1,
            seq_len=self.seq_len,
        )

        for _ in range(300):
            batch = generate_synthetic_sequence_batch(
                batch_size=64,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                seq_len=self.seq_len,
                in_dist_actions=self.in_dist_actions,
            )
            agent.update(batch)

        n_test_states = 30
        all_in_dist_q = []
        all_ood_q = []

        for _ in range(n_test_states):
            test_state = np.random.randn(self.state_dim).astype(np.float32)
            sequence = {
                "actions": np.random.choice(self.in_dist_actions, size=self.seq_len).tolist(),
                "clicked": (np.random.random(self.seq_len) > 0.4).astype(float).tolist(),
                "dwell_times": np.random.exponential(3.0, size=self.seq_len).tolist(),
                "purchased": (np.random.random(self.seq_len) > 0.85).astype(float).tolist(),
                "states": np.random.randn(self.seq_len, self.state_dim).tolist(),
            }
            q_values = agent.get_q_distribution(test_state, sequence)
            all_in_dist_q.extend(q_values[self.in_dist_actions].tolist())
            all_ood_q.extend(q_values[self.ood_actions].tolist())

        in_dist_arr = np.array(all_in_dist_q)
        ood_arr = np.array(all_ood_q)

        in_dist_mean = in_dist_arr.mean()
        ood_mean = ood_arr.mean()
        pooled_std = np.sqrt((in_dist_arr.std() ** 2 + ood_arr.std() ** 2) / 2)
        effect_size = (in_dist_mean - ood_mean) / max(pooled_std, 1e-8)

        assert ood_mean < in_dist_mean, (
            f"CQL_RNN with sequences: OOD Q ({ood_mean:.4f}) should be < in-dist Q ({in_dist_mean:.4f})"
        )
        assert effect_size > 0.2, (
            f"Effect size between in-dist and OOD should be meaningful: "
            f"got {effect_size:.4f} (in_dist_mean={in_dist_mean:.4f}, ood_mean={ood_mean:.4f})"
        )

    def test_cql_rnn_default_vs_sequence_encoding(self):
        """Q-values should differ based on whether user history is provided.
        An agent with history context should produce different action rankings
        than one using the default embedding."""
        agent = CQL_RNN(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            alpha=3.0,
            lr=1e-3,
            lstm_hidden_size=64,
            lstm_num_layers=1,
            seq_len=self.seq_len,
        )

        for _ in range(200):
            batch = generate_synthetic_sequence_batch(
                batch_size=64,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                seq_len=self.seq_len,
                in_dist_actions=self.in_dist_actions,
            )
            agent.update(batch)

        test_state = np.random.randn(self.state_dim).astype(np.float32)

        q_no_seq = agent.get_q_distribution(test_state)

        sequence = {
            "actions": np.random.choice(self.in_dist_actions, size=self.seq_len).tolist(),
            "clicked": [1.0] * self.seq_len,
            "dwell_times": [5.0] * self.seq_len,
            "purchased": [0.0] * (self.seq_len - 1) + [1.0],
            "states": np.random.randn(self.seq_len, self.state_dim).tolist(),
        }
        q_with_seq = agent.get_q_distribution(test_state, sequence)

        diff = np.abs(q_with_seq - q_no_seq).mean()
        assert diff > 1e-4, (
            f"Q-values should change with user history encoding, "
            f"but mean absolute difference is only {diff:.6f}"
        )


class TestUserStateEncoder:
    """Tests for the LSTM-based user state encoder."""

    def test_encoder_output_shape(self):
        encoder = UserStateEncoder(state_dim=10, action_dim=100, hidden_size=64)
        batch_size, seq_len = 16, 8

        features = torch.randn(batch_size, seq_len, encoder.input_dim)
        lengths = torch.full((batch_size,), seq_len, dtype=torch.long)

        output = encoder(features, lengths)
        assert output.shape == (batch_size, 64)

    def test_encoder_handles_variable_lengths(self):
        encoder = UserStateEncoder(state_dim=10, action_dim=100, hidden_size=64)
        batch_size, max_seq_len = 8, 10

        features = torch.randn(batch_size, max_seq_len, encoder.input_dim)
        lengths = torch.tensor([3, 5, 10, 7, 2, 8, 6, 4], dtype=torch.long)

        output = encoder(features, lengths)
        assert output.shape == (batch_size, 64)
        assert not torch.isnan(output).any()

    def test_encode_features_builds_correct_dims(self):
        encoder = UserStateEncoder(state_dim=10, action_dim=100, hidden_size=64)
        batch_size, seq_len = 4, 5

        actions = torch.randint(0, 100, (batch_size, seq_len)).float()
        clicked = torch.randint(0, 2, (batch_size, seq_len)).float()
        dwell = torch.rand(batch_size, seq_len)
        purchased = torch.randint(0, 2, (batch_size, seq_len)).float()
        states = torch.randn(batch_size, seq_len, 10)

        features = encoder.encode_features(actions, clicked, dwell, purchased, states)
        assert features.shape == (batch_size, seq_len, 100 + 3 + 10)


class TestShiftDetection:
    """Tests for distribution shift detection logic (pure math, no DB)."""

    def test_kl_divergence_identical_distributions(self):
        offline_dist = np.ones(100) / 100
        online_actions = np.random.randint(0, 100, size=10000)

        online_counts = np.bincount(online_actions, minlength=100).astype(np.float64)
        online_dist = online_counts / online_counts.sum()

        epsilon = 1e-8
        p = offline_dist + epsilon
        q = online_dist + epsilon
        p = p / p.sum()
        q = q / q.sum()
        kl_div = float(np.sum(p * np.log(p / q)))

        assert kl_div < 0.1, (
            f"KL divergence for similar distributions should be small, got {kl_div:.4f}"
        )

    def test_kl_divergence_shifted_distribution(self):
        offline_dist = np.ones(100) / 100
        online_actions = np.random.choice([0, 1, 2], size=10000)

        online_counts = np.bincount(online_actions, minlength=100).astype(np.float64)
        online_dist = online_counts / online_counts.sum()

        epsilon = 1e-8
        p = offline_dist + epsilon
        q = online_dist + epsilon
        p = p / p.sum()
        q = q / q.sum()
        kl_div = float(np.sum(p * np.log(p / q)))

        assert kl_div > 0.5, (
            f"KL divergence for shifted distributions should be large, got {kl_div:.4f}"
        )

    def test_new_items_detection(self):
        known_items = set(range(100))
        threshold = 0.1

        current_no_new = set(range(100))
        new_items = current_no_new - known_items
        ratio = len(new_items) / max(len(current_no_new), 1)
        assert ratio == 0.0
        assert ratio <= threshold

        current_with_new = set(range(120))
        new_items = current_with_new - known_items
        ratio = len(new_items) / max(len(current_with_new), 1)
        assert ratio > threshold


class TestDataQualityValidation:
    """Tests for data distribution quality validation."""

    def _validate_data_quality(self, actions, rewards, states, config):
        """Inline version of DistributionShiftDetector._validate_data_quality."""
        issues = []
        N_ITEMS = 100

        action_counts = np.bincount(actions, minlength=N_ITEMS).astype(np.float64)
        action_dist = action_counts / action_counts.sum()
        nonzero_dist = action_dist[action_dist > 0]
        entropy = -np.sum(nonzero_dist * np.log(nonzero_dist))
        max_entropy = np.log(N_ITEMS)
        entropy_ratio = entropy / max_entropy
        min_entropy_ratio = config.get("min_action_entropy_ratio", 0.3)
        if entropy_ratio < min_entropy_ratio:
            issues.append(f"Action distribution entropy too low: ratio={entropy_ratio:.3f}")

        unique_actions = np.unique(actions)
        min_coverage = config.get("min_action_coverage_ratio", 0.1)
        coverage = len(unique_actions) / N_ITEMS
        if coverage < min_coverage:
            issues.append(f"Action coverage too sparse: {len(unique_actions)}/{N_ITEMS}")

        reward_std = float(rewards.std())
        min_reward_std = config.get("min_reward_std", 0.01)
        if reward_std < min_reward_std:
            issues.append(f"Reward variance too small: std={reward_std:.6f}")

        state_stds = states.std(axis=0)
        collapsed_dims = np.sum(state_stds < 1e-6)
        max_collapsed_ratio = config.get("max_collapsed_state_dims_ratio", 0.5)
        collapsed_ratio = collapsed_dims / states.shape[1]
        if collapsed_ratio > max_collapsed_ratio:
            issues.append(f"State space partially collapsed: {collapsed_dims}/{states.shape[1]}")

        return issues

    def test_good_data_passes_validation(self):
        """Well-distributed data should pass quality checks."""
        config = {
            "min_action_entropy_ratio": 0.3,
            "min_action_coverage_ratio": 0.1,
            "min_reward_std": 0.01,
            "max_collapsed_state_dims_ratio": 0.5,
        }
        actions = np.random.randint(0, 100, size=50000)
        rewards = np.random.choice([0.0, 1.0], size=50000, p=[0.7, 0.3])
        states = np.random.randn(50000, 10).astype(np.float32)

        issues = self._validate_data_quality(actions, rewards, states, config)
        assert len(issues) == 0, f"Good data should have no issues, got: {issues}"

    def test_degenerate_actions_detected(self):
        """Data concentrated on very few actions should fail entropy check."""
        config = {
            "min_action_entropy_ratio": 0.3,
            "min_action_coverage_ratio": 0.1,
            "min_reward_std": 0.01,
            "max_collapsed_state_dims_ratio": 0.5,
        }
        actions = np.random.choice([0, 1], size=50000)
        rewards = np.random.choice([0.0, 1.0], size=50000)
        states = np.random.randn(50000, 10).astype(np.float32)

        issues = self._validate_data_quality(actions, rewards, states, config)
        assert any("entropy" in i.lower() for i in issues)

    def test_zero_variance_rewards_detected(self):
        """Constant rewards should fail variance check."""
        config = {
            "min_action_entropy_ratio": 0.3,
            "min_action_coverage_ratio": 0.1,
            "min_reward_std": 0.01,
            "max_collapsed_state_dims_ratio": 0.5,
        }
        actions = np.random.randint(0, 100, size=50000)
        rewards = np.ones(50000, dtype=np.float32)
        states = np.random.randn(50000, 10).astype(np.float32)

        issues = self._validate_data_quality(actions, rewards, states, config)
        assert any("reward" in i.lower() for i in issues)

    def test_collapsed_state_dims_detected(self):
        """States with many zero-variance dimensions should fail."""
        config = {
            "min_action_entropy_ratio": 0.3,
            "min_action_coverage_ratio": 0.1,
            "min_reward_std": 0.01,
            "max_collapsed_state_dims_ratio": 0.5,
        }
        actions = np.random.randint(0, 100, size=50000)
        rewards = np.random.choice([0.0, 1.0], size=50000)
        states = np.zeros((50000, 10), dtype=np.float32)
        states[:, :3] = np.random.randn(50000, 3)

        issues = self._validate_data_quality(actions, rewards, states, config)
        assert any("collapsed" in i.lower() for i in issues)


class TestCompositeRetrainDecision:
    """Tests for cross-metric correlation retrain logic."""

    def _evaluate_retrain_decision(self, results, severe_alerts, min_severe, config):
        """Inline version of _evaluate_retrain_decision logic."""
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

        composite_threshold = config.get("composite_retrain_threshold", 4.0)
        return composite_score >= composite_threshold

    def test_single_alert_not_enough(self):
        """One severe alert alone should not trigger retraining."""
        config = {"composite_retrain_threshold": 4.0}
        results = [
            {"shift_type": "action_distribution", "metric_value": 2.0,
             "threshold": 0.5, "is_alert": True, "is_severe": True},
            {"shift_type": "reward_distribution", "metric_value": 0.3,
             "threshold": 2.0, "is_alert": False, "is_severe": False},
        ]
        severe = [r for r in results if r["is_severe"]]
        assert not self._evaluate_retrain_decision(results, severe, 2, config)

    def test_correlated_alerts_trigger_retrain(self):
        """Two correlated severe alerts should trigger retraining."""
        config = {"composite_retrain_threshold": 4.0}
        results = [
            {"shift_type": "action_distribution", "metric_value": 3.0,
             "threshold": 0.5, "is_alert": True, "is_severe": True},
            {"shift_type": "reward_distribution", "metric_value": 6.0,
             "threshold": 2.0, "is_alert": True, "is_severe": True},
            {"shift_type": "state_distribution", "metric_value": 1.0,
             "threshold": 6.0, "is_alert": False, "is_severe": False},
        ]
        severe = [r for r in results if r["is_severe"]]
        assert self._evaluate_retrain_decision(results, severe, 2, config)

    def test_weak_alerts_below_composite_threshold(self):
        """Multiple alerts that are barely over threshold should not retrain."""
        config = {"composite_retrain_threshold": 6.0}
        results = [
            {"shift_type": "action_distribution", "metric_value": 1.1,
             "threshold": 0.5, "is_alert": True, "is_severe": True},
            {"shift_type": "new_items", "metric_value": 0.22,
             "threshold": 0.1, "is_alert": True, "is_severe": True},
        ]
        severe = [r for r in results if r["is_severe"]]
        decision = self._evaluate_retrain_decision(results, severe, 2, config)
        assert not decision


class TestSnapshotRetention:
    """Tests for performance-aware snapshot retention logic."""

    def test_best_performer_protected(self):
        """The best-performing snapshot should never be deleted."""
        class FakeSnapshot:
            def __init__(self, id, epoch, performance_reward):
                self.id = id
                self.epoch = epoch
                self.performance_reward = performance_reward

        snapshots = [
            FakeSnapshot(6, 60, 10.0),
            FakeSnapshot(5, 50, 25.0),   # best performer
            FakeSnapshot(4, 40, 15.0),
            FakeSnapshot(3, 30, 20.0),
            FakeSnapshot(2, 20, 12.0),
            FakeSnapshot(1, 10, 8.0),
        ]

        max_snapshots = 3
        best = max(snapshots, key=lambda s: s.performance_reward if s.performance_reward is not None else -float("inf"))
        assert best.id == 5

        recent = snapshots[:max_snapshots]
        if best not in recent:
            retained = set(s.id for s in recent[:max_snapshots - 1])
            retained.add(best.id)
        else:
            retained = set(s.id for s in recent)

        assert 5 in retained, "Best performer (id=5) must be retained"
        assert 6 in retained, "Most recent (id=6) must be retained"
        assert len(retained) <= max_snapshots


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
