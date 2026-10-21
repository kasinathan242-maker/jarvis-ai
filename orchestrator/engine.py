"""JARVIS Lite — Multi-Agent DAG Workflow Execution & Checkpointing Engine.

Executes DAG workflows with parallel step execution, fault-tolerant checkpointing,
pause/resume/cancel controls, and post-restart recovery.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.event_bus import bus
from core.logging import get_logger
from orchestrator.dag import DAGNode, NodeStatus, TaskDAG
from tools.registry import registry

logger = get_logger("orchestrator.engine")
CHECKPOINTS_DIR = Path("data/checkpoints")


class WorkflowEngine:
    """Async workflow execution engine with checkpointing and fault recovery."""

    def __init__(self) -> None:
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        self._active_workflows: dict[str, TaskDAG] = {}
        self._paused_workflows: set[str] = set()
        self._cancelled_workflows: set[str] = set()
        self._custom_handlers: dict[str, Callable[..., Any]] = {}

    def register_handler(self, action_name: str, handler: Callable[..., Any]) -> None:
        """Register a custom agent action handler."""
        self._custom_handlers[action_name] = handler

    def _get_checkpoint_path(self, workflow_id: str) -> Path:
        return CHECKPOINTS_DIR / f"{workflow_id}.json"

    def save_checkpoint(self, dag: TaskDAG, state_label: str = "running") -> Path:
        """Persist workflow DAG state to disk."""
        path = self._get_checkpoint_path(dag.workflow_id)
        payload = {
            "workflow_id": dag.workflow_id,
            "name": dag.name,
            "state_label": state_label,
            "saved_at": time.time(),
            "saved_at_iso": datetime.now(UTC).isoformat(),
            "dag": dag.to_dict(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def load_checkpoint(self, workflow_id: str) -> TaskDAG | None:
        """Load workflow DAG from disk checkpoint."""
        path = self._get_checkpoint_path(workflow_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return TaskDAG.from_dict(data["dag"])
        except Exception as exc:
            logger.error("failed_to_load_checkpoint", workflow_id=workflow_id, error=str(exc))
            return None

    def recover_interrupted_workflows(self) -> list[dict[str, Any]]:
        """Scan checkpoint directory on startup and mark or resume interrupted workflows."""
        interrupted: list[dict[str, Any]] = []
        for p in CHECKPOINTS_DIR.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                label = data.get("state_label", "")
                if label == "running":
                    wid = data["workflow_id"]
                    # Mark interrupted state clearly
                    data["state_label"] = "interrupted_after_restart"
                    data["interrupted_at"] = time.time()
                    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
                    interrupted.append({
                        "workflow_id": wid,
                        "name": data.get("name", ""),
                        "status": "marked_interrupted",
                    })
                    logger.warn("interrupted_workflow_marked_on_startup", workflow_id=wid)
            except Exception as exc:
                logger.error("failed_to_inspect_checkpoint_file", path=str(p), error=str(exc))
        return interrupted

    async def execute_node(self, dag: TaskDAG, node: DAGNode) -> Any:
        """Execute a single DAG node, passing upstream results."""
        node.status = NodeStatus.RUNNING
        node.started_at = time.time()
        self.save_checkpoint(dag, state_label="running")

        await bus.publish(
            "workflow_node_started",
            {"workflow_id": dag.workflow_id, "node_id": node.node_id, "name": node.name},
        )

        # Assemble dependency outputs for structured handoff
        parent_outputs = {
            dep_id: dag.nodes[dep_id].result
            for dep_id in node.dependencies
            if dep_id in dag.nodes
        }
        params = dict(node.parameters)
        params["_upstream_context"] = parent_outputs

        try:
            # 1. Custom handler dispatch
            if node.action in self._custom_handlers:
                handler = self._custom_handlers[node.action]
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(**params)
                else:
                    result = handler(**params)
            # 2. Tool registry dispatch
            elif node.action in registry:
                tool_res = await registry.dispatch(
                    name=node.action,
                    raw_params=node.parameters,
                    user_approved=True,
                )
                if not tool_res.success:
                    raise RuntimeError(f"Tool execution failed: {tool_res.error}")
                if tool_res.data is not None and hasattr(tool_res.data, "model_dump"):
                    result = tool_res.data.model_dump()
                elif tool_res.data is not None and hasattr(tool_res.data, "dict"):
                    result = tool_res.data.dict()
                else:
                    result = tool_res.data
            else:
                # Built-in simulation / fallback
                result = {"output": f"Executed action {node.action} on node {node.node_id}"}

            node.status = NodeStatus.COMPLETED
            node.result = result
            node.completed_at = time.time()

            await bus.publish(
                "workflow_node_completed",
                {"workflow_id": dag.workflow_id, "node_id": node.node_id, "result": str(result)[:200]},
            )
            return result

        except Exception as exc:
            logger.error("workflow_node_failed", node_id=node.node_id, error=str(exc))
            node.status = NodeStatus.FAILED
            node.error = str(exc)
            node.completed_at = time.time()
            await bus.publish(
                "workflow_node_failed",
                {"workflow_id": dag.workflow_id, "node_id": node.node_id, "error": str(exc)},
            )
            raise

    async def run(self, dag: TaskDAG) -> TaskDAG:
        """Execute a full DAG workflow to completion with wave-based parallelism."""
        dag.validate()
        self._active_workflows[dag.workflow_id] = dag
        self.save_checkpoint(dag, state_label="running")

        logger.info("workflow_execution_started", workflow_id=dag.workflow_id, nodes=len(dag.nodes))

        while not dag.is_finished():
            if dag.workflow_id in self._cancelled_workflows:
                for n in dag.nodes.values():
                    if n.status == NodeStatus.PENDING:
                        n.status = NodeStatus.CANCELLED
                self.save_checkpoint(dag, state_label="cancelled")
                logger.warn("workflow_execution_cancelled", workflow_id=dag.workflow_id)
                break

            while dag.workflow_id in self._paused_workflows:
                await asyncio.sleep(0.2)

            ready_nodes = dag.get_ready_nodes()
            if not ready_nodes:
                # If there are no ready nodes but not finished, some dependency failed
                for n in dag.nodes.values():
                    if n.status == NodeStatus.PENDING:
                        n.status = NodeStatus.CANCELLED
                        n.error = "Upstream dependency failed"
                break

            # Execute wave of independent ready nodes concurrently
            tasks = [self.execute_node(dag, node) for node in ready_nodes]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            self.save_checkpoint(dag, state_label="running")

            # Check if any tasks failed
            for res in results:
                if isinstance(res, Exception):
                    logger.error("error_in_node_wave", error=str(res))

        final_label = "completed" if not dag.has_failures() else "failed"
        self.save_checkpoint(dag, state_label=final_label)
        logger.info("workflow_execution_finished", workflow_id=dag.workflow_id, status=final_label)
        return dag

    def pause(self, workflow_id: str) -> bool:
        if workflow_id in self._active_workflows:
            self._paused_workflows.add(workflow_id)
            return True
        return False

    def resume(self, workflow_id: str) -> bool:
        if workflow_id in self._paused_workflows:
            self._paused_workflows.discard(workflow_id)
            return True
        return False

    def cancel(self, workflow_id: str) -> bool:
        if workflow_id in self._active_workflows:
            self._cancelled_workflows.add(workflow_id)
            return True
        return False


workflow_engine = WorkflowEngine()
