"""add user_behavior_sequences, model_snapshots, shift_detection_records

Revision ID: 003
Revises: 002
Create Date: 2026-06-04
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "user_behavior_sequences",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer(), nullable=False, index=True),
        sa.Column("episode_id", sa.Integer(), nullable=False, index=True),
        sa.Column("step", sa.Integer(), nullable=False),
        sa.Column("action", sa.Integer(), nullable=False),
        sa.Column("clicked", sa.Boolean(), nullable=False),
        sa.Column("dwell_time", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("purchased", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("state", ARRAY(sa.Float()), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )

    op.create_table(
        "model_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("training_runs.id"), nullable=False, index=True),
        sa.Column("epoch", sa.Integer(), nullable=False),
        sa.Column("algorithm", sa.String(50), nullable=False),
        sa.Column("parameters_blob", sa.LargeBinary(), nullable=False),
        sa.Column("hyperparameters", JSONB(), nullable=False),
        sa.Column("state_dim", sa.Integer(), nullable=False),
        sa.Column("action_dim", sa.Integer(), nullable=False),
        sa.Column("performance_reward", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )

    op.create_table(
        "shift_detection_records",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("detection_time", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("shift_type", sa.String(50), nullable=False),
        sa.Column("metric_name", sa.String(100), nullable=False),
        sa.Column("metric_value", sa.Float(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("is_alert", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("details", JSONB(), nullable=True),
        sa.Column("triggered_retrain", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("retrain_run_id", sa.Integer(), sa.ForeignKey("training_runs.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )


def downgrade():
    op.drop_table("shift_detection_records")
    op.drop_table("model_snapshots")
    op.drop_table("user_behavior_sequences")
