"""add policy_versions table

Revision ID: 005
Revises: 004
Create Date: 2026-06-04
"""
from alembic import op
import sqlalchemy as sa

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "policy_versions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("training_runs.id"), nullable=False, index=True),
        sa.Column("snapshot_id", sa.Integer(), sa.ForeignKey("model_snapshots.id"), nullable=False),
        sa.Column("version_tag", sa.String(50), nullable=False),
        sa.Column("stage", sa.String(20), nullable=False, server_default="candidate"),
        sa.Column("fqe_evaluation_id", sa.Integer(), sa.ForeignKey("fqe_evaluations.id"), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )


def downgrade():
    op.drop_table("policy_versions")
