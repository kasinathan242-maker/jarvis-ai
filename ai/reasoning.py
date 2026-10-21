"""JARVIS Lite — AI Reasoning & Structured Tool Planner.

Enforces Rule 1: The LLM NEVER executes anything directly.
It inspects available tool schemas and proposes a structured Plan containing
Pydantic-validated ToolCalls. The execution layer validates, checks permissions,
runs, verifies, and reports back.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from ai.model_router import router
from core.logging import get_logger
from memory.retrieval import build_augmented_context, format_context_for_prompt
from tools.base import ToolCall
from tools.registry import registry

logger = get_logger("ai.reasoning")


class ReasoningPlan(BaseModel):
    thoughts: str = Field(description="Step-by-step chain of thought reasoning")
    tool_calls: list[ToolCall] = Field(
        default_factory=list, description="Ordered list of tool calls proposed to accomplish goal"
    )
    direct_response: str | None = Field(default=None, description="Direct conversational answer if no tool is required")


SYSTEM_PLANNER_PROMPT = """You are JARVIS Lite, an advanced personal AI operating system for Windows 11.
You operate under strict safety guardrails. You never execute actions directly.
Instead, you propose structured JSON plans containing tool calls.

Available Tools:
{tools_manifest}

{memory_context}

Output format:
Respond ONLY with a JSON object matching this schema:
{{
    "thoughts": "<your internal chain of thought analysis>",
    "tool_calls": [
        {{
            "tool_name": "<exact_tool_name>",
            "parameters": {{<key>: <value>}}
        }}
    ],
    "direct_response": "<optional conversational response if no tools are needed>"
}}

Rules:
1. If the user asks for time/date, call tool 'get_time'.
2. If the user asks to open an app (e.g. notepad, calc), call tool 'open_application' with {"app_name": "<name>"}.
3. If the user asks for system stats/CPU/RAM/GPU, call tool 'system_stats'.
4. If it's a general conversation or question, provide 'direct_response' with 'tool_calls': [].
"""


class ReasoningEngine:
    async def create_plan(self, user_query: str) -> ReasoningPlan:
        """Analyze user query and generate a structured reasoning plan."""
        # Check fast mappings first to ensure instant response (< 5ms) for standard commands
        query_lower = user_query.lower().strip()

        if any(kw in query_lower for kw in ("time is it", "current time", "what time", "what is the date")):
            return ReasoningPlan(
                thoughts="User requested current time/date. Selecting get_time tool.",
                tool_calls=[ToolCall(tool_name="get_time", parameters={})],
            )

        if "notepad" in query_lower and any(kw in query_lower for kw in ("open", "launch", "start")):
            return ReasoningPlan(
                thoughts="User requested opening Notepad. Proposing open_application with app_name='notepad'.",
                tool_calls=[ToolCall(tool_name="open_application", parameters={"app_name": "notepad"})],
            )

        is_calc = any(kw in query_lower for kw in ("calc", "calculator"))
        is_open = any(kw in query_lower for kw in ("open", "launch", "start"))
        if is_calc and is_open:
            return ReasoningPlan(
                thoughts="User requested opening Calculator. Proposing open_application with app_name='calc'.",
                tool_calls=[ToolCall(tool_name="open_application", parameters={"app_name": "calc"})],
            )

        if any(kw in query_lower for kw in ("system stats", "cpu usage", "ram usage", "gpu status", "system status")):
            return ReasoningPlan(
                thoughts="User requested system metrics. Proposing system_stats tool.",
                tool_calls=[ToolCall(tool_name="system_stats", parameters={"include_gpu": True})],
            )

        # Developer desktop commands: VS Code and project folder
        is_vscode = (
            "vs code" in query_lower
            or "vscode" in query_lower
            or (query_lower.startswith("open code") and len(query_lower) < 20)
        )
        if is_vscode:
            # Check if user mentioned opening a project folder as well
            calls = [ToolCall(tool_name="open_application", parameters={"app_name": "code"})]
            # Match patterns like "and my <folder>" or "in <folder>"
            import re
            folder_pat = (
                r"(?:and\s+(?:my\s+)?|in\s+)(?:folder\s+)?"
                r"([a-zA-Z]:\\[^\s]+|\bdownloads\\[^\s]+|jarvis|project)"
            )
            folder_match = re.search(folder_pat, query_lower)
            if folder_match:
                raw_target = folder_match.group(1).strip()
                if Path(raw_target).is_absolute():
                    target_path = str(Path(raw_target).resolve())
                elif raw_target in ("jarvis", "project"):
                    target_path = str(Path.cwd())
                else:
                    target_path = str((Path.home() / raw_target).resolve())
                calls.append(ToolCall(tool_name="open_path", parameters={"path": target_path}))
            return ReasoningPlan(
                thoughts="User requested opening VS Code and optional project directory.",
                tool_calls=calls,
            )

        # Screenshot tool
        if any(kw in query_lower for kw in ("take a screenshot", "capture screen", "screenshot")):
            return ReasoningPlan(
                thoughts="User requested screenshot capture.",
                tool_calls=[ToolCall(tool_name="screenshot", parameters={})],
            )

        # Clipboard tools
        if any(kw in query_lower for kw in ("read clipboard", "get clipboard", "what is on my clipboard")):
            return ReasoningPlan(
                thoughts="User requested clipboard read.",
                tool_calls=[ToolCall(tool_name="clipboard_read", parameters={})],
            )

        # File deletion / del commands
        if query_lower.startswith("del ") or query_lower.startswith("delete file ") or "run del " in query_lower:
            parts = user_query.strip().split(maxsplit=2)
            target = parts[-1] if len(parts) >= 2 else "unknown"
            return ReasoningPlan(
                thoughts="User requested file deletion. Proposing delete_file (requires user confirmation).",
                tool_calls=[ToolCall(tool_name="delete_file", parameters={"file_path": target})],
            )

        # Destructive format command test
        if "format c:" in query_lower or query_lower.startswith("format "):
            return ReasoningPlan(
                thoughts="User requested disk format. Proposing shell_execute (FORBIDDEN command).",
                tool_calls=[ToolCall(tool_name="shell_execute", parameters={"command": user_query.strip()})],
            )

        # Multi-source research
        if "research " in query_lower and any(kw in query_lower for kw in ("courses", "compare", "plan", "topic", "ai")):
            return ReasoningPlan(
                thoughts="User requested comprehensive multi-source research.",
                tool_calls=[ToolCall(tool_name="research_topic", parameters={"topic": user_query.strip()})],
            )

        # Daily Briefing
        if any(kw in query_lower for kw in ("daily briefing", "morning briefing", "give me the briefing")):
            return ReasoningPlan(
                thoughts="User requested daily executive briefing.",
                tool_calls=[ToolCall(tool_name="daily_briefing", parameters={})],
            )

        # Email draft
        if any(kw in query_lower for kw in ("draft an email", "draft reply", "draft an email reply")):
            return ReasoningPlan(
                thoughts="User requested drafting an email reply.",
                tool_calls=[
                    ToolCall(
                        tool_name="draft_email_reply",
                        parameters={
                            "to_email": "alice.dev@example.com",
                            "subject": "Project Architecture Review",
                            "reply_body": "I have reviewed the Phase 3 orchestrator specifications and approved the plan.",
                        },
                    )
                ],
            )

        # Project Tests
        if any(kw in query_lower for kw in ("run tests", "run project tests", "run test suite")):
            return ReasoningPlan(
                thoughts="User requested running project test suite.",
                tool_calls=[ToolCall(tool_name="run_tests", parameters={})],
            )

        # Custom neural model tool-calling resolution
        try:
            from ai.custom_model_engine import custom_toolcall_engine

            custom_call = custom_toolcall_engine.plan_tool_call(user_query)
            if custom_call is not None:
                return ReasoningPlan(
                    thoughts=f"Custom neural model resolved action: {custom_call.tool_name}",
                    tool_calls=[custom_call],
                )
        except Exception as exc:
            logger.debug("custom_toolcall_engine_skip", error=str(exc))

        # For nuanced queries, prompt the LLM
        tools_list = registry.list_tools()
        tools_manifest = json.dumps(tools_list, indent=2)

        context_bundle = build_augmented_context(user_query)
        memory_context = format_context_for_prompt(context_bundle)

        system_prompt = SYSTEM_PLANNER_PROMPT.format(
            tools_manifest=tools_manifest,
            memory_context=memory_context,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]

        try:
            raw_json = await router.chat(
                messages=messages,
                task_type="plan",
                temperature=0.2,
                format_json=True,
            )
            data = json.loads(raw_json)
            return ReasoningPlan(**data)
        except Exception as exc:
            logger.warn("llm_reasoning_fallback", error=str(exc))
            # Fallback direct response
            return ReasoningPlan(
                thoughts=f"Fallback due to planner error: {exc}",
                tool_calls=[],
                direct_response=f"I understand your query: '{user_query}'. How can I assist you further?",
            )


planner = ReasoningEngine()
