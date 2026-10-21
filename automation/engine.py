"""JARVIS Lite — Automation Engine (YAML Workflows, Triggers, Retries, Dry-Run).

Parses user automation recipes, binds them to voice, hotkey, schedule, or folder change triggers,
and executes them with retry logic and dry-run preview capabilities.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import yaml

from automation.schema import WorkflowDefinition
from core.logging import get_logger
from tools.registry import registry

logger = get_logger("automation.engine")
WORKFLOWS_DIR = Path("workflows")


class AutomationEngine:
    """Manages declarative automation workflows."""

    def __init__(self) -> None:
        WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)
        self._workflows: dict[str, WorkflowDefinition] = {}
        self.load_all()

    def load_all(self) -> int:
        """Scan workflows/ directory and load all .yaml / .yml files."""
        self._workflows.clear()
        count = 0
        for path in WORKFLOWS_DIR.glob("*.y*ml"):
            try:
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
                if raw and "name" in raw and "steps" in raw:
                    wf = WorkflowDefinition(**raw)
                    self._workflows[wf.name.lower()] = wf
                    count += 1
            except Exception as exc:
                logger.error("failed_to_load_workflow_file", path=str(path), error=str(exc))
        logger.info("automation_workflows_loaded", count=count)
        return count

    def get_workflow(self, name: str) -> WorkflowDefinition | None:
        return self._workflows.get(name.lower())

    def list_workflows(self) -> list[dict[str, Any]]:
        return [
            {
                "name": wf.name,
                "description": wf.description,
                "enabled": wf.enabled,
                "steps_count": len(wf.steps),
                "triggers_count": len(wf.triggers),
            }
            for wf in self._workflows.values()
        ]

    def set_enabled(self, name: str, enabled: bool) -> bool:
        wf = self.get_workflow(name)
        if wf:
            wf.enabled = enabled
            logger.info("workflow_toggled", name=name, enabled=enabled)
            return True
        return False

    async def execute_workflow(
        self,
        name: str,
        dry_run: bool = False,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a declared workflow with loop, condition, and retry support."""
        wf = self.get_workflow(name)
        if not wf:
            raise ValueError(f"Workflow '{name}' not found.")
        if not wf.enabled:
            return {"workflow": name, "status": "skipped", "reason": "Workflow is disabled."}

        logger.info("executing_workflow", name=name, dry_run=dry_run)
        step_results: list[dict[str, Any]] = []

        for step in wf.steps:
            # Check condition if provided
            if step.condition and context:
                # Safe evaluation against context variables
                if not bool(context.get(step.condition)):
                    step_results.append({
                        "step": step.name,
                        "status": "skipped_by_condition",
                        "condition": step.condition,
                    })
                    continue

            # Dry run preview
            if dry_run:
                step_results.append({
                    "step": step.name,
                    "action": step.action,
                    "parameters": step.parameters,
                    "status": "dry_run_preview",
                })
                continue

            # Execute with loops and retries
            for loop_idx in range(step.loop_count):
                attempt = 0
                max_attempts = step.retries + 1
                success = False
                last_error: str | None = None
                res_data: Any = None

                while attempt < max_attempts:
                    attempt += 1
                    try:
                        if step.action in registry:
                            tool_res = await registry.dispatch(
                                name=step.action,
                                raw_params=step.parameters,
                                user_approved=True,
                            )
                            if not tool_res.success:
                                raise RuntimeError(f"Tool error: {tool_res.error}")
                            if tool_res.data is not None and hasattr(tool_res.data, "model_dump"):
                                res_data = tool_res.data.model_dump()
                            elif tool_res.data is not None and hasattr(tool_res.data, "dict"):
                                res_data = tool_res.data.dict()
                            else:
                                res_data = tool_res.data
                        else:
                            res_data = {"executed": step.action}

                        success = True
                        break
                    except Exception as exc:
                        last_error = str(exc)
                        if attempt < max_attempts:
                            await asyncio.sleep(0.5 * attempt)

                step_results.append({
                    "step": step.name,
                    "action": step.action,
                    "loop": loop_idx + 1,
                    "attempts": attempt,
                    "status": "success" if success else "failed",
                    "result": res_data,
                    "error": last_error,
                })

                if not success:
                    # Halt on fatal step failure
                    return {
                        "workflow": name,
                        "status": "failed",
                        "steps": step_results,
                        "failed_step": step.name,
                    }

        return {
            "workflow": name,
            "status": "completed",
            "dry_run": dry_run,
            "steps": step_results,
        }


automation_engine = AutomationEngine()
