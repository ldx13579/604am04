"""add ensemble_metrics, fqe_evaluations, fqe_metrics tables

Revision ID: 004
Revises: 003
Create Date: 2026-06-04
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "ensemble_metrics",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("training_runs.id"), nullable=False, index=True),
        sa.Column("epoch", sa.Integer(), nullable=False),
        sa.Column("uncertainty_mean", sa.Float(), nullable=False),
        sa.Column("uncertainty_max", sa.Float(), nullable=False),
        sa.Column("exploration_ratio", sa.Float(), nullable=False),
        sa.Column("per_model_losses", JSONB(), nullable=True),
        sa.Column("per_model_q_means", JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )

    op.create_table(
        "fqe_evaluations",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("source_run_id", sa.Integer(), sa.ForeignKey("training_runs.id"), nullable=False, index=True),
        sa.Column("algorithm", sa.String(50), nullable=False),
        sa.Column("hyperparameters", JSONB(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("estimated_value", sa.Float(), nullable=True),
        sa.Column("total_epochs", sa.Integer(), nullable=False, server_default="50"),
        sa.Column("current_epoch", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )

    op.create_table(
        "fqe_metrics",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("evaluation_id", sa.Integer(), sa.ForeignKey("fqe_evaluations.id"), nullable=False, index=True),
        sa.Column("epoch", sa.Integer(), nullable=False),
        sa.Column("estimated_value", sa.Float(), nullable=False),
        sa.Column("fqe_loss", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )


def downgrade():
    op.drop_table("fqe_metrics")
    op.drop_table("fqe_evaluations")
    op.drop_table("ensemble_metrics")
