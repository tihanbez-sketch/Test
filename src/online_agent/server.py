"""HTTP frontend.

Exposes the agent over two endpoints:

    POST /chat         run a turn, return the final answer as JSON
    POST /chat/stream  run a turn, relay agent events as server-sent events

Conversations are keyed by an opaque session id and held in memory, so a single
process owns them; put a shared store behind :class:`SessionStore` before
running more than one replica.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .agent import Agent
from .config import AgentConfig
from .events import AgentError, Citation, TextDelta, TurnEnd

SESSION_TTL_SECONDS = 60 * 60
SWEEP_INTERVAL_SECONDS = 300


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, description="The user's message.")
    session_id: str | None = Field(
        default=None, description="Omit to start a new conversation."
    )


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    citations: list[dict[str, str | None]] = []
    stop_reason: str | None = None
    usage: dict[str, int] = {}
    error: str | None = None


class SessionStore:
    """In-memory agents keyed by session id, with idle expiry."""

    def __init__(self, config: AgentConfig, ttl: int = SESSION_TTL_SECONDS) -> None:
        self.config = config
        self.ttl = ttl
        self._agents: dict[str, Agent] = {}
        self._touched: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def get_or_create(self, session_id: str | None) -> tuple[str, Agent]:
        if session_id and session_id in self._agents:
            self._touched[session_id] = time.monotonic()
            return session_id, self._agents[session_id]
        if session_id:
            # An unknown id is a stale client, not an error — start fresh under
            # the id it already has so its next request lands in the same place.
            new_id = session_id
        else:
            new_id = uuid.uuid4().hex
        agent = Agent(self.config)
        self._agents[new_id] = agent
        self._touched[new_id] = time.monotonic()
        self._locks[new_id] = asyncio.Lock()
        return new_id, agent

    def lock(self, session_id: str) -> asyncio.Lock:
        return self._locks.setdefault(session_id, asyncio.Lock())

    def delete(self, session_id: str) -> bool:
        self._touched.pop(session_id, None)
        self._locks.pop(session_id, None)
        return self._agents.pop(session_id, None) is not None

    def sweep(self) -> int:
        cutoff = time.monotonic() - self.ttl
        stale = [sid for sid, seen in self._touched.items() if seen < cutoff]
        for sid in stale:
            # Never evict a session mid-turn.
            lock = self._locks.get(sid)
            if lock is not None and lock.locked():
                continue
            self.delete(sid)
        return len(stale)

    def __len__(self) -> int:
        return len(self._agents)


def create_app(config: AgentConfig | None = None) -> FastAPI:
    load_dotenv()
    store = SessionStore(config or AgentConfig.from_env())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(_sweeper(store))
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app = FastAPI(
        title="online-agent",
        description="A web-connected agent on the Claude API.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.store = store

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": store.config.model,
            "sessions": len(store),
            "web_enabled": store.config.web_search or store.config.web_fetch,
        }

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest = Body(...)) -> ChatResponse:
        session_id, agent = store.get_or_create(request.session_id)
        lock = store.lock(session_id)
        if lock.locked():
            raise HTTPException(
                status_code=409,
                detail="This session already has a turn in flight.",
            )

        parts: list[str] = []
        citations: list[dict[str, str | None]] = []
        stop_reason: str | None = None
        usage: dict[str, int] = {}
        error: str | None = None

        async with lock:
            async for event in agent.run(request.message):
                if isinstance(event, TextDelta):
                    parts.append(event.text)
                elif isinstance(event, Citation):
                    citations.append({"url": event.url, "title": event.title})
                elif isinstance(event, AgentError):
                    error = event.message
                elif isinstance(event, TurnEnd):
                    stop_reason = event.stop_reason
                    usage = {
                        "input_tokens": event.input_tokens,
                        "output_tokens": event.output_tokens,
                        "cache_read_tokens": event.cache_read_tokens,
                        "steps": event.steps,
                    }

        return ChatResponse(
            session_id=session_id,
            reply="".join(parts).strip(),
            citations=citations,
            stop_reason=stop_reason,
            usage=usage,
            error=error,
        )

    @app.post("/chat/stream")
    async def chat_stream(request: ChatRequest = Body(...)) -> StreamingResponse:
        session_id, agent = store.get_or_create(request.session_id)
        lock = store.lock(session_id)
        if lock.locked():
            raise HTTPException(
                status_code=409,
                detail="This session already has a turn in flight.",
            )

        async def events() -> AsyncIterator[str]:
            yield f'event: session\ndata: {{"session_id": "{session_id}"}}\n\n'
            async with lock:
                try:
                    async for event in agent.run(request.message):
                        yield f"event: {event.type}\ndata: {event.to_json()}\n\n"
                except asyncio.CancelledError:
                    # Client hung up mid-turn; stop cleanly rather than logging
                    # a traceback for a normal disconnect.
                    raise
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str) -> dict[str, bool]:
        return {"deleted": store.delete(session_id)}

    return app


async def _sweeper(store: SessionStore) -> None:
    while True:
        await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
        store.sweep()


app = create_app  # uvicorn --factory online_agent.server:app
