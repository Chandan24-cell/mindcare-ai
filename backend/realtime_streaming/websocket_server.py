from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.realtime_streaming.realtime_frame_pipeline import (
    run_realtime_frame_pipeline,
)
from backend.realtime_streaming.stream_session_manager import StreamSessionManager
from backend.realtime_streaming.websocket_models import StreamMetrics


logger = logging.getLogger(__name__)

router = APIRouter()

_session_manager = StreamSessionManager()


async def _safe_send_json(ws: WebSocket, payload: Dict[str, Any]) -> None:
    try:
        await ws.send_text(json.dumps(payload))
    except Exception:
        # never crash websocket server
        pass


@router.websocket("/ws/realtime-monitor")
async def realtime_monitor(ws: WebSocket) -> None:
    """Realtime websocket endpoint (Phase 8 scaffolding).

    Accepts JSON messages. For now, expects:
    - {"session_id": "...", "mode": "mock|real", "frame": {...} }

    Replies with a deterministic placeholder structure matching
    the required response format.
    """

    await ws.accept()

    session_id = str(uuid.uuid4())
    mode = "mock"

    await _session_manager.create(session_id, created_at=time.time())

    # Heartbeat loop (ping-like messages)
    async def _heartbeat() -> None:
        while True:
            try:
                await _safe_send_json(
                    ws,
                    {
                        "type": "heartbeat",
                        "timestamp": time.time(),
                        "session_id": session_id,
                    },
                )
                await asyncio.sleep(5)
            except Exception:
                return

    hb_task = asyncio.create_task(_heartbeat())

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                await _safe_send_json(
                    ws,
                    {
                        "error": "Invalid JSON",
                        "session_id": session_id,
                        "type": "error",
                    },
                )
                continue

            session_id = str(msg.get("session_id") or session_id)
            mode = str(msg.get("mode") or mode)

            frame = msg.get("frame")

            payload = await run_realtime_frame_pipeline(
                frame=frame,
                mode=mode,
            )

            payload.update(
                {
                    "timestamp": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                    "session_id": session_id,
                    "stream_metrics": payload.get("stream_metrics")
                    or StreamMetrics(
                        stream_fps=0,
                        avg_latency_ms=0,
                        dropped_frames=0,
                        stream_quality="init",
                    ).to_dict(),
                }
            )

            await _safe_send_json(ws, payload)

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Realtime websocket error")
    finally:
        try:
            hb_task.cancel()
        except Exception:
            pass
        try:
            await _session_manager.delete(session_id)
        except Exception:
            pass

