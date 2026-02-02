"""PostgreSQL state store for production use."""

import json
from typing import Any, Optional

from ai_navigator.models.capacity import CapacityPlan
from ai_navigator.models.deployment import DeploymentConfig
from ai_navigator.models.workflow import WorkflowState
from ai_navigator.state.manager import StateStore

try:
    import asyncpg
except ImportError:
    asyncpg = None  # type: ignore[assignment]


class PostgresStateStore(StateStore):
    """PostgreSQL implementation of state store for production."""

    def __init__(self, dsn: str) -> None:
        """Initialize with database connection string."""
        if asyncpg is None:
            raise ImportError("asyncpg is required for PostgreSQL state store")
        self._dsn = dsn
        self._pool: Optional[asyncpg.Pool] = None

    async def _get_pool(self) -> "asyncpg.Pool":
        """Get or create connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)
        return self._pool

    async def initialize(self) -> None:
        """Create database tables if they don't exist."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS workflows (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_workflows_user_id ON workflows(user_id);

                CREATE TABLE IF NOT EXISTS capacity_plans (
                    id TEXT PRIMARY KEY,
                    data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS deployment_configs (
                    id TEXT PRIMARY KEY,
                    data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """)

    async def save_workflow(self, state: WorkflowState) -> None:
        """Save or update workflow state."""
        pool = await self._get_pool()
        data = state.model_dump_json()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO workflows (id, user_id, data, updated_at)
                VALUES ($1, $2, $3, NOW())
                ON CONFLICT (id) DO UPDATE SET
                    data = EXCLUDED.data,
                    updated_at = NOW()
                """,
                state.id,
                state.user_id,
                data,
            )

    async def get_workflow(self, workflow_id: str) -> Optional[WorkflowState]:
        """Retrieve workflow state by ID."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM workflows WHERE id = $1", workflow_id
            )
            if row:
                return WorkflowState.model_validate_json(row["data"])
        return None

    async def get_workflow_by_user(self, user_id: str) -> Optional[WorkflowState]:
        """Get the most recent workflow for a user."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT data FROM workflows
                WHERE user_id = $1
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                user_id,
            )
            if row:
                return WorkflowState.model_validate_json(row["data"])
        return None

    async def delete_workflow(self, workflow_id: str) -> bool:
        """Delete workflow state."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM workflows WHERE id = $1", workflow_id
            )
            return result == "DELETE 1"

    async def save_capacity_plan(self, plan: CapacityPlan) -> None:
        """Save capacity plan."""
        pool = await self._get_pool()
        data = plan.model_dump_json()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO capacity_plans (id, data)
                VALUES ($1, $2)
                ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data
                """,
                plan.id,
                data,
            )

    async def get_capacity_plan(self, plan_id: str) -> Optional[CapacityPlan]:
        """Retrieve capacity plan by ID."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM capacity_plans WHERE id = $1", plan_id
            )
            if row:
                return CapacityPlan.model_validate_json(row["data"])
        return None

    async def save_deployment_config(self, config: DeploymentConfig) -> None:
        """Save deployment configuration."""
        pool = await self._get_pool()
        data = config.model_dump_json()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO deployment_configs (id, data, updated_at)
                VALUES ($1, $2, NOW())
                ON CONFLICT (id) DO UPDATE SET
                    data = EXCLUDED.data,
                    updated_at = NOW()
                """,
                config.id,
                data,
            )

    async def get_deployment_config(self, config_id: str) -> Optional[DeploymentConfig]:
        """Retrieve deployment configuration by ID."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM deployment_configs WHERE id = $1", config_id
            )
            if row:
                return DeploymentConfig.model_validate_json(row["data"])
        return None

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
