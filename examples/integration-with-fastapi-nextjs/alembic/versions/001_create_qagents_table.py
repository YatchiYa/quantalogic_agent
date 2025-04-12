"""create qagents table

Revision ID: 001
Revises: 
Create Date: 2025-04-03 18:12:39.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('qagents',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('description', sa.String(), nullable=True),
        sa.Column('model_name', sa.String(), nullable=False),
        sa.Column('expertise', sa.String(), nullable=True),
        sa.Column('project', sa.String(), nullable=True),
        sa.Column('agent_mode', sa.String(), nullable=True, server_default='default'),
        sa.Column('tags', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('tools', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('agent_user_id_idx', 'qagents', ['user_id'])
    op.create_index('agent_organization_id_idx', 'qagents', ['organization_id'])


def downgrade() -> None:
    op.drop_index('agent_organization_id_idx')
    op.drop_index('agent_user_id_idx')
    op.drop_table('qagents')
