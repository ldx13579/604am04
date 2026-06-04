"""A/B testing, recommendation service, online fine-tuning, and performance benchmarks

Revision ID: 006
Revises: 005
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

revision = '006'
down_revision = '005'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'ab_experiments',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('name', sa.String(100), nullable=False, unique=True),
        sa.Column('strategy_a', sa.String(50), nullable=False, server_default='cql'),
        sa.Column('strategy_b', sa.String(50), nullable=False, server_default='random'),
        sa.Column('traffic_split', sa.Float(), nullable=False, server_default='0.5'),
        sa.Column('status', sa.String(20), nullable=False, server_default='running'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('ended_at', sa.DateTime(), nullable=True),
    )

    op.create_table(
        'ab_impressions',
        sa.Column('id', sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column('experiment_id', sa.Integer(), sa.ForeignKey('ab_experiments.id'), nullable=False),
        sa.Column('group_name', sa.String(10), nullable=False),
        sa.Column('user_state', ARRAY(sa.Float()), nullable=False),
        sa.Column('recommended_items', ARRAY(sa.Integer()), nullable=False),
        sa.Column('session_id', sa.String(64), nullable=True),
        sa.Column('timestamp', sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index('ix_ab_impressions_experiment_id', 'ab_impressions', ['experiment_id'])
    op.create_index('ix_ab_impressions_timestamp', 'ab_impressions', ['timestamp'])

    op.create_table(
        'ab_clicks',
        sa.Column('id', sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column('impression_id', sa.BigInteger(), sa.ForeignKey('ab_impressions.id'), nullable=False),
        sa.Column('item_id', sa.Integer(), nullable=False),
        sa.Column('clicked', sa.Boolean(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index('ix_ab_clicks_impression_id', 'ab_clicks', ['impression_id'])
    op.create_index('ix_ab_clicks_timestamp', 'ab_clicks', ['timestamp'])

    op.create_table(
        'ab_ctr_snapshots',
        sa.Column('id', sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column('experiment_id', sa.Integer(), sa.ForeignKey('ab_experiments.id'), nullable=False),
        sa.Column('group_name', sa.String(10), nullable=False),
        sa.Column('window_start', sa.DateTime(), nullable=False),
        sa.Column('window_end', sa.DateTime(), nullable=False),
        sa.Column('impressions_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('clicks_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('ctr', sa.Float(), nullable=False, server_default='0.0'),
    )
    op.create_index('ix_ab_ctr_snapshots_experiment_id', 'ab_ctr_snapshots', ['experiment_id'])
    op.create_index('ix_ab_ctr_snapshots_window_start', 'ab_ctr_snapshots', ['window_start'])

    op.create_table(
        'online_interaction_buffer',
        sa.Column('id', sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column('state', ARRAY(sa.Float()), nullable=False),
        sa.Column('action', sa.Integer(), nullable=False),
        sa.Column('reward', sa.Float(), nullable=False),
        sa.Column('next_state', ARRAY(sa.Float()), nullable=False),
        sa.Column('done', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('consumed', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('timestamp', sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index('ix_online_interaction_buffer_consumed', 'online_interaction_buffer', ['consumed'])
    op.create_index('ix_online_interaction_buffer_timestamp', 'online_interaction_buffer', ['timestamp'])

    op.create_table(
        'finetune_runs',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('source_policy_version_id', sa.Integer(), sa.ForeignKey('policy_versions.id'), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='pending'),
        sa.Column('n_interactions_used', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('loss_before', sa.Float(), nullable=True),
        sa.Column('loss_after', sa.Float(), nullable=True),
        sa.Column('reward_before', sa.Float(), nullable=True),
        sa.Column('reward_after', sa.Float(), nullable=True),
        sa.Column('started_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
    )

    op.create_table(
        'perf_report_runs',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('dataset_size', sa.Integer(), nullable=False),
        sa.Column('algorithm', sa.String(50), nullable=False),
        sa.Column('training_time_seconds', sa.Float(), nullable=False),
        sa.Column('convergence_epoch', sa.Integer(), nullable=True),
        sa.Column('final_reward', sa.Float(), nullable=False),
        sa.Column('total_epochs_run', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )

    # Convert time-series tables to TimescaleDB hypertables
    op.execute("SELECT create_hypertable('ab_impressions', 'timestamp', if_not_exists => TRUE, migrate_data => TRUE)")
    op.execute("SELECT create_hypertable('ab_clicks', 'timestamp', if_not_exists => TRUE, migrate_data => TRUE)")
    op.execute("SELECT create_hypertable('ab_ctr_snapshots', 'window_start', if_not_exists => TRUE, migrate_data => TRUE)")
    op.execute("SELECT create_hypertable('online_interaction_buffer', 'timestamp', if_not_exists => TRUE, migrate_data => TRUE)")


def downgrade():
    op.drop_table('perf_report_runs')
    op.drop_table('finetune_runs')
    op.drop_table('online_interaction_buffer')
    op.drop_table('ab_ctr_snapshots')
    op.drop_table('ab_clicks')
    op.drop_table('ab_impressions')
    op.drop_table('ab_experiments')
