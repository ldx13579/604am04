"""add generation_status, error_detail, current_epoch

Revision ID: 002
Revises: 001
Create Date: 2026-06-04
"""
from alembic import op
import sqlalchemy as sa

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "generation_status",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("is_running", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("progress", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("total_generated", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("target_count", sa.Integer(), nullable=False, server_default="1000000"),
        sa.Column("last_episode_id", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
    )

    op.add_column("training_runs", sa.Column("current_epoch", sa.Integer(), nullable=True, server_default="0"))
    op.add_column("training_runs", sa.Column("error_detail", sa.Text(), nullable=True))


def downgrade():
    op.drop_column("training_runs", "error_detail")
    op.drop_column("training_runs", "current_epoch")
    op.drop_table("generation_status")
