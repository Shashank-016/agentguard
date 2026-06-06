"""Async SQLAlchemy store — SQLite for dev, Postgres-compatible schema."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Index,
    String,
    Text,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from .events import SecurityEvent

logger = logging.getLogger(__name__)

_DEFAULT_URL = "sqlite+aiosqlite:///agentguard.db"


class Base(DeclarativeBase):
    pass


class EventRecord(Base):
    """ORM mapping for SecurityEvent rows."""

    __tablename__ = "security_events"

    event_id = Column(String(36), primary_key=True)
    session_id = Column(String(255), nullable=False, index=True)
    agent_id = Column(String(255), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    source = Column(String(32), nullable=False)
    event_type = Column(String(64), nullable=False, index=True)
    severity = Column(String(16), nullable=False, index=True)
    payload = Column(Text, default="{}")
    flags = Column(Text, default="[]")
    parent_event_id = Column(String(36), nullable=True)
    metadata_ = Column("metadata", Text, default="{}")

    __table_args__ = (
        Index("ix_severity_timestamp", "severity", "timestamp"),
        Index("ix_session_timestamp", "session_id", "timestamp"),
    )


class EventStore:
    """Async SQLAlchemy-backed event persistence layer.

    Parameters
    ----------
    database_url:
        SQLAlchemy async connection string. Defaults to a local SQLite file.
        Use ``sqlite+aiosqlite:///:memory:`` for in-process tests.
    """

    def __init__(self, database_url: str = _DEFAULT_URL) -> None:
        self._engine = create_async_engine(database_url, echo=False)
        self._session_factory = async_sessionmaker(
            self._engine, expire_on_commit=False
        )

    async def initialize(self) -> None:
        """Create tables if they do not already exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("EventStore initialized")

    async def save(self, event: SecurityEvent) -> None:
        """Persist a single SecurityEvent."""
        record = EventRecord(
            event_id=event.event_id,
            session_id=event.session_id,
            agent_id=event.agent_id,
            timestamp=event.timestamp,
            source=event.source,
            event_type=event.event_type,
            severity=event.severity,
            payload=json.dumps(event.payload),
            flags=json.dumps(event.flags),
            parent_event_id=event.parent_event_id,
            metadata_=json.dumps(event.metadata),
        )
        async with self._session_factory() as session:
            async with session.begin():
                session.add(record)

    async def get_event(self, event_id: str) -> Optional[SecurityEvent]:
        """Fetch a single event by ID, or None if not found."""
        async with self._session_factory() as session:
            row = await session.get(EventRecord, event_id)
            return _to_model(row) if row else None

    async def list_events(
        self,
        session_id: Optional[str] = None,
        severity: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[SecurityEvent]:
        """Query events with optional filters, ordered by timestamp descending."""
        stmt = select(EventRecord).order_by(EventRecord.timestamp.desc())
        if session_id:
            stmt = stmt.where(EventRecord.session_id == session_id)
        if severity:
            stmt = stmt.where(EventRecord.severity == severity)
        if event_type:
            stmt = stmt.where(EventRecord.event_type == event_type)
        stmt = stmt.limit(limit).offset(offset)

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [_to_model(r) for r in rows]

    async def list_sessions(self) -> list[dict]:
        """Return summary stats per session_id."""
        from sqlalchemy import func, case

        stmt = select(
            EventRecord.session_id,
            func.count(EventRecord.event_id).label("total"),
            func.sum(
                case((EventRecord.severity == "critical", 1), else_=0)
            ).label("critical_count"),
            func.sum(
                case((EventRecord.severity == "warning", 1), else_=0)
            ).label("warning_count"),
            func.min(EventRecord.timestamp).label("started_at"),
            func.max(EventRecord.timestamp).label("last_seen"),
        ).group_by(EventRecord.session_id).order_by(func.max(EventRecord.timestamp).desc())

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            rows = result.all()

        return [
            {
                "session_id": r.session_id,
                "total_events": r.total,
                "critical_count": int(r.critical_count or 0),
                "warning_count": int(r.warning_count or 0),
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "last_seen": r.last_seen.isoformat() if r.last_seen else None,
            }
            for r in rows
        ]

    async def count(self) -> int:
        """Total number of stored events."""
        from sqlalchemy import func

        async with self._session_factory() as session:
            result = await session.execute(select(func.count(EventRecord.event_id)))
            return result.scalar_one()

    async def close(self) -> None:
        await self._engine.dispose()


def _to_model(row: EventRecord) -> SecurityEvent:
    """Convert an ORM row back to a SecurityEvent Pydantic model."""
    return SecurityEvent(
        event_id=row.event_id,
        session_id=row.session_id,
        agent_id=row.agent_id,
        timestamp=row.timestamp,
        source=row.source,  # type: ignore[arg-type]
        event_type=row.event_type,  # type: ignore[arg-type]
        severity=row.severity,  # type: ignore[arg-type]
        payload=json.loads(row.payload or "{}"),
        flags=json.loads(row.flags or "[]"),
        parent_event_id=row.parent_event_id,
        metadata=json.loads(row.metadata_ or "{}"),
    )
