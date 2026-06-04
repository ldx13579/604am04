"""initial schema

Revision ID: 001
Revises:
Create Date: 2026-06-04
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "items",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("category_id", sa.Integer(), nullable=False),
        sa.Column("popularity", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("embedding", ARRAY(sa.Float()), nullable=False),
    )
    op.create_index("ix_items_category_id", "items", ["category_id"])

    op.create_table(
        "offline_transitions",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("state", ARRAY(sa.Float()), nullable=False),
        sa.Column("action", sa.Integer(), nullable=False),
        sa.Column("reward", sa.Float(), nullable=False),
        sa.Column("next_state", ARRAY(sa.Float()), nullable=False),
        sa.Column("done", sa.Boolean(), nullable=False),
        sa.Column("episode_id", sa.Integer(), nullable=False),
        sa.Column("timestamp_step", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_offline_transitions_episode_id", "offline_transitions", ["episode_id"])

    op.create_table(
        "training_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("algorithm", sa.String(50), nullable=False),
        sa.Column("hyperparameters", JSONB(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("started_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("total_epochs", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("best_reward", sa.Float(), nullable=True),
    )
    op.create_index("ix_training_runs_algorithm", "training_runs", ["algorithm"])

    op.create_table(
        "training_metrics",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("training_runs.id"), nullable=False),
        sa.Column("epoch", sa.Integer(), nullable=False),
        sa.Column("loss", sa.Float(), nullable=False),
        sa.Column("q_value_mean", sa.Float(), nullable=True),
        sa.Column("q_value_max", sa.Float(), nullable=True),
        sa.Column("q_value_min", sa.Float(), nullable=True),
        sa.Column("cumulative_reward", sa.Float(), nullable=False),
        sa.Column("cql_penalty", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_training_metrics_run_id", "training_metrics", ["run_id"])


def downgrade():
    op.drop_table("training_metrics")
    op.drop_table("training_runs")
    op.drop_table("offline_transitions")
    op.drop_table("items")
