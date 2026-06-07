"""Kill-switch control routes.

Exposes the process-wide :class:`~agentguard.control.KillSwitch` over HTTP so an
operator (or an on-call dashboard) can halt one or every guarded session without
touching the process running the agent.

Notes
-----
* In-process only — these endpoints affect sessions running in *this* API
  process. A multi-process deployment needs a shared switch (e.g. Redis-backed,
  see the roadmap's "Multi-process bus" item) for a single trip to halt every
  worker.
* Protected by ``require_api_key`` (see ``api/main.py``) — set
  ``AGENTGUARD_API_KEY`` so an unauthenticated kill switch on the public
  internet can't be used as a denial-of-service vector against your own agents.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from agentguard.control import get_default_kill_switch

from ..main import require_api_key

router = APIRouter(prefix="/control", tags=["control"], dependencies=[Depends(require_api_key)])


@router.post("/kill/{session_id}")
async def kill_session(session_id: str) -> dict:
    """Halt a single session — its next intercepted action raises/blocks immediately."""
    switch = get_default_kill_switch()
    switch.kill_session(session_id)
    return {"status": "killed", "session_id": session_id}


@router.post("/kill-all")
async def kill_all() -> dict:
    """Trip the global switch — halts every session in this process."""
    switch = get_default_kill_switch()
    switch.kill_all()
    return {"status": "killed", "scope": "all"}


@router.post("/revive/{session_id}")
async def revive_session(session_id: str) -> dict:
    """Untrip a single session, restoring normal operation for it."""
    switch = get_default_kill_switch()
    switch.revive_session(session_id)
    return {"status": "revived", "session_id": session_id}


@router.get("/status")
async def status() -> dict:
    """Return the current kill state — global flag and individually killed sessions."""
    switch = get_default_kill_switch()
    return switch.status()
