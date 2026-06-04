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
        """Higher alpha should result in lower OOD Q-values (more conservative)."""
        cql_low = CQL(state_dim=self.state_dim, action_dim=self.action_dim, alpha=0.5, lr=1e-3)
        cql_high = CQL(state_dim=self.state_dim, action_dim=self.action_dim, alpha=5.0, lr=1e-3)

        self._train_agent(cql_low, n_steps=200)
        self._train_agent(cql_high, n_steps=200)

        test_states = torch.randn(50, self.state_dim)
        with torch.no_grad():
            q_low = cql_low.q_network(test_states.to(cql_low.device))
            q_high = cql_high.q_network(test_states.to(cql_high.device))

        ood_q_low_alpha = q_low[:, self.ood_actions].mean().item()
        ood_q_high_alpha = q_high[:, self.ood_actions].mean().item()

        assert ood_q_high_alpha < ood_q_low_alpha, (
            f"Higher alpha should produce lower OOD Q-values: "
            f"alpha=5.0 got {ood_q_high_alpha:.4f}, alpha=0.5 got {ood_q_low_alpha:.4f}"
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
