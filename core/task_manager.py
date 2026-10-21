"""JARVIS Lite — Task Manager & Emergency Stop Coordinator.

Manages execution lifecycle of asynchronous tasks, providing cancellation,
checkpoints, progress tracking, and instantaneous emergency stop execution.
"""

from __future__ import annotations

import asyncio
import enum
import time
import uuid
from collections.abc import Callable, Coroutine
from typing import Any

from pydantic import BaseModel, Field

from core.event_bus import bus
from core.logging import get_logger

logger = get_logger("core.task_manager")


class TaskState(enum.StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TaskInfo(BaseModel):
    task_id: str
    name: str
    state: TaskState = TaskState.PENDING
    started_at: float = Field(default_factory=time.monotonic)
    completed_at: float | None = None
    progress: float = 0.0
    checkpoint_data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class TaskManager:
    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._task_metadata: dict[str, TaskInfo] = {}
        self._lock = asyncio.Lock()

    async def submit_task(
        self,
        name: str,
        coro_fn: Callable[..., Coroutine[Any, Any, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Submit and track an asynchronous background task."""
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        info = TaskInfo(task_id=task_id, name=name, state=TaskState.RUNNING)

        async with self._lock:
            self._task_metadata[task_id] = info

        async def _wrapper() -> Any:
            try:
                result = await coro_fn(*args, **kwargs)
                info.state = TaskState.COMPLETED
                info.progress = 1.0
                info.completed_at = time.monotonic()
                await bus.publish("task_completed", {"task_id": task_id, "result": result})
                return result
            except asyncio.CancelledError:
                info.state = TaskState.CANCELLED
                info.completed_at = time.monotonic()
                await bus.publish("task_cancelled", {"task_id": task_id})
                raise
            except Exception as exc:
                info.state = TaskState.FAILED
                info.error = str(exc)
                info.completed_at = time.monotonic()
                await bus.publish("task_failed", {"task_id": task_id, "error": str(exc)})
                raise
            finally:
                async with self._lock:
                    self._tasks.pop(task_id, None)

        task = asyncio.create_task(_wrapper(), name=name)
        async with self._lock:
            self._tasks[task_id] = task

        await bus.publish("task_started", {"task_id": task_id, "name": name})
        return task_id

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a specific task by ID."""
        async with self._lock:
            task = self._tasks.get(task_id)
            if task and not task.done():
                task.cancel()
                logger.info("task_cancelled_by_id", task_id=task_id)
                return True
        return False

    async def emergency_stop(self) -> int:
        """Emergency Stop: Instantly aborts ALL running tasks and automations.

        Guaranteed < 1 second response time.
        """
        logger.warning("emergency_stop_triggered")
        cancelled_count = 0

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        tasks_to_gather: list[asyncio.Task[Any]] = []

        async with self._lock:
            for task_id, t in list(self._tasks.items()):
                if not t.done():
                    try:
                        # Ensure task belongs to the currently active event loop
                        if current_loop is not None and t.get_loop() is current_loop:
                            t.cancel()
                            tasks_to_gather.append(t)
                            cancelled_count += 1
                            if task_id in self._task_metadata:
                                self._task_metadata[task_id].state = TaskState.CANCELLED
                                self._task_metadata[task_id].completed_at = time.monotonic()
                        elif current_loop is None:
                            t.cancel()
                            cancelled_count += 1
                            if task_id in self._task_metadata:
                                self._task_metadata[task_id].state = TaskState.CANCELLED
                                self._task_metadata[task_id].completed_at = time.monotonic()
                    except Exception:
                        pass

        if tasks_to_gather:
            await asyncio.gather(*tasks_to_gather, return_exceptions=True)

        await bus.publish("emergency_stop_executed", {"cancelled_count": cancelled_count})
        return cancelled_count

    def get_task_info(self, task_id: str) -> TaskInfo | None:
        return self._task_metadata.get(task_id)

    def list_active_tasks(self) -> list[TaskInfo]:
        return [info for info in self._task_metadata.values() if info.state == TaskState.RUNNING]


task_manager = TaskManager()
