"""JARVIS Lite — WebSocket Real-Time Chat & Streaming Endpoint.

Streams reasoning tokens, tool invocation status, and audio chunk events
to the frontend over localhost WebSocket.
"""

from __future__ import annotations

import json

from fastapi import WebSocket, WebSocketDisconnect

from ai.model_router import router
from core.logging import get_logger
from memory.conversation_store import save_message
from security.secrets import verify_api_token
from voice.pipeline import pipeline

logger = get_logger("api.ws")


async def handle_chat_websocket(websocket: WebSocket) -> None:
    """Handle bidirectional WebSocket communication for real-time streaming."""
    await websocket.accept()
    logger.info("websocket_client_connected")

    # Verify initial auth handshake
    auth_header = websocket.headers.get("Authorization") or websocket.query_params.get("token")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
    else:
        token = auth_header or ""

    if not verify_api_token(token):
        logger.warn("websocket_unauthorized_token")
        await websocket.send_json({"type": "error", "message": "Unauthorized: Invalid or missing Bearer token."})
        await websocket.close(code=1008)
        return

    await websocket.send_json({"type": "ready", "message": "JARVIS Lite WebSocket active."})

    try:
        while True:
            raw_msg = await websocket.receive_text()
            try:
                data = json.loads(raw_msg)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON format."})
                continue

            msg_type = data.get("type", "chat")
            user_text = data.get("text", "").strip()

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if not user_text:
                continue

            # Stream response back
            await websocket.send_json({"type": "stream_start", "query": user_text})

            # Check if command routes to a tool
            from intent.classifier import classifier
            from intent.schemas import IntentType
            intent = await classifier.classify(user_text)

            tool_intents = (
                IntentType.GET_TIME,
                IntentType.OPEN_APPLICATION,
                IntentType.SYSTEM_STATS,
                IntentType.EMERGENCY_STOP,
            )
            if intent.type in tool_intents:
                response_text = await pipeline.process_command(user_text, play_speech=False)
                await websocket.send_json({"type": "token", "content": response_text})
                await websocket.send_json({"type": "stream_end", "full_response": response_text})
            else:
                # Conversational streaming via LLM
                full_tokens = []
                messages = [{"role": "user", "content": user_text}]
                try:
                    async for token_chunk in router.stream_chat(messages=messages, task_type="chat"):
                        full_tokens.append(token_chunk)
                        await websocket.send_json({"type": "token", "content": token_chunk})

                    final_response = "".join(full_tokens)
                    save_message("user", user_text, intent="chat")
                    save_message("assistant", final_response, intent="chat")
                    await websocket.send_json({"type": "stream_end", "full_response": final_response})
                except Exception as exc:
                    err_msg = f"Model streaming failed: {exc}"
                    await websocket.send_json({"type": "error", "message": err_msg})

    except WebSocketDisconnect:
        logger.info("websocket_client_disconnected")
    except Exception as exc:
        logger.error("websocket_unexpected_error", error=str(exc))
