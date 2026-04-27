"""Mock tool endpoints used by LLM tool-calling flows."""

import asyncio
from typing import Any


async def _fetch_profile() -> dict[str, Any]:
    """Simulate fetching employee profile details."""
    await asyncio.sleep(1)
    return {"name": "Jane Doe", "grade": "Senior"}


async def _fetch_manager() -> dict[str, Any]:
    """Simulate fetching employee manager details."""
    await asyncio.sleep(1)
    return {"manager": "John Smith"}


async def _fetch_team() -> dict[str, Any]:
    """Simulate fetching employee team details."""
    await asyncio.sleep(1)
    return {"team_size": 8, "team_name": "Platform"}


async def get_employee_context(user_id: str) -> dict[str, Any]:
    """Fetch profile, manager, and team data in parallel for one user."""
    profile, manager_info, team_info = await asyncio.gather(
        _fetch_profile(), _fetch_manager(), _fetch_team()
    )
    return {"user_id": user_id, "profile": profile, "manager_info": manager_info, "team_info": team_info}

