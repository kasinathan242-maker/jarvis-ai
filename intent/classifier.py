"""JARVIS Lite — Intent Classifier.

Two-stage classification:
1. Fast Path: Low-latency regex and keyword pattern matching (< 1ms).
2. Slow Path: Small-LLM semantic routing (Qwen 1.5B/3B) returning structured JSON Intent.
"""

from __future__ import annotations

import json
import re

from ai.model_router import router
from intent.schemas import Intent, IntentType

FAST_PATTERNS: list[tuple[re.Pattern[str], IntentType, dict[str, str]]] = [
    # Emergency Stop
    (re.compile(r"\b(?:jarvis\s+)?stop\s+everything\b", re.IGNORECASE), IntentType.EMERGENCY_STOP, {}),
    (re.compile(r"\bemergency\s+stop\b", re.IGNORECASE), IntentType.EMERGENCY_STOP, {}),

    # Get Time
    (
        re.compile(
            r"\b(?:what\s+time\s+is\s+it|current\s+time|tell\s+me\s+the\s+time|what\s+is\s+the\s+date|today'?s\s+date)\b",
            re.IGNORECASE,
        ),
        IntentType.GET_TIME,
        {},
    ),

    # Open Application
    (
        re.compile(
            r"\b(?:open|launch|start)\s+(notepad|calc|calculator|code|vscode|explorer|terminal)\b",
            re.IGNORECASE,
        ),
        IntentType.OPEN_APPLICATION,
        {"slot_target": "app_name"},
    ),

    # System Stats
    (
        re.compile(
            r"\b(?:system\s+stats|system\s+status|cpu\s+usage|ram\s+usage|gpu\s+status|vram|hardware\s+status)\b",
            re.IGNORECASE,
        ),
        IntentType.SYSTEM_STATS,
        {},
    ),

    # Greetings
    (
        re.compile(r"^(?:hi|hello|hey|good\s+morning|good\s+afternoon|good\s+evening)\b", re.IGNORECASE),
        IntentType.GREETING,
        {},
    ),

    # File Deletion and Formatting (Reasoning / Tools)
    (re.compile(r"^(?:del\s+|delete\s+|format\s+)", re.IGNORECASE), IntentType.REASONING, {}),

    # Screen Understanding
    (
        re.compile(
            r"\b(?:what(?:'s|\s+is)\s+on\s+my\s+screen|look\s+at\s+my\s+screen|read\s+my\s+screen|describe\s+my\s+screen|analyze\s+screen)\b",
            re.IGNORECASE,
        ),
        IntentType.ANALYZE_SCREEN,
        {},
    ),

    # Exit
    (re.compile(r"\b(?:exit|quit|shutdown\s+jarvis|goodbye|bye)\b", re.IGNORECASE), IntentType.EXIT, {}),
]


class IntentClassifier:
    def classify_fast(self, text: str) -> Intent | None:
        """Evaluate fast-path regex rules. Returns Intent if pattern matches with high confidence."""
        clean = text.strip()
        for pattern, itype, slot_rule in FAST_PATTERNS:
            match = pattern.search(clean)
            if match:
                slots = {}
                if "slot_target" in slot_rule and match.groups():
                    slots[slot_rule["slot_target"]] = match.group(1).lower()
                return Intent(
                    type=itype,
                    confidence=0.98,
                    slots=slots,
                    raw_query=text,
                )
        return None

    async def classify(self, text: str) -> Intent:
        """Classify user intent using fast path regex, then custom neural model, then LLM."""
        # 1. Fast Path (< 1ms Regex)
        fast_result = self.classify_fast(text)
        if fast_result:
            return fast_result

        # 2. Custom Trained Neural Intent Classifier (< 2ms ONNX Runtime, 0 VRAM)
        try:
            from ai.custom_model_engine import custom_intent_engine

            pred_intent, confidence = custom_intent_engine.predict(text)
            if confidence >= 0.55:
                slots: dict[str, str] = {}
                itype = IntentType.CHAT

                if pred_intent == "open_application":
                    itype = IntentType.OPEN_APPLICATION
                    m = re.search(r"(?:open|launch|start)\s+([a-zA-Z0-9_\-\s]+)", text.lower())
                    if m:
                        slots["app_name"] = m.group(1).strip()
                elif pred_intent == "system_stats":
                    itype = IntentType.SYSTEM_STATS
                elif pred_intent == "screen_analysis":
                    itype = IntentType.ANALYZE_SCREEN
                elif pred_intent == "emergency_stop":
                    itype = IntentType.EMERGENCY_STOP
                elif pred_intent in [
                    "search_files",
                    "delete_file",
                    "run_shell",
                    "clipboard_action",
                    "window_action",
                    "browse_web",
                    "email_calendar",
                    "document_qa",
                ]:
                    itype = IntentType.REASONING
                elif pred_intent == "general_chat":
                    itype = IntentType.CHAT

                return Intent(type=itype, confidence=confidence, slots=slots, raw_query=text)
        except Exception:
            pass

        # 3. LLM Router Path
        prompt = (
            "You are an intent classifier for JARVIS Lite. Categorize the user's input into exactly ONE intent:\n"
            "Options: ['get_time', 'open_application', 'system_stats', 'chat', 'reasoning', "
            "'memory_store', 'memory_query', 'emergency_stop', 'exit', 'unknown']\n\n"
            "Reply strictly with a valid JSON object in this format:\n"
            '{"type": "<intent_type>", "confidence": 0.85, "slots": {"app_name": "<app_if_applicable>"}}\n\n'
            f'User input: "{text}"'
        )

        try:
            raw_response = await router.chat(
                messages=[{"role": "user", "content": prompt}],
                task_type="route",
                temperature=0.1,
                format_json=True,
            )
            data = json.loads(raw_response)
            itype = IntentType(data.get("type", "chat"))
            confidence = float(data.get("confidence", 0.7))
            slots = data.get("slots", {})
            return Intent(type=itype, confidence=confidence, slots=slots, raw_query=text)
        except Exception:
            # Fallback to chat intent on routing failure or offline model
            return Intent(
                type=IntentType.CHAT,
                confidence=0.5,
                slots={},
                raw_query=text,
            )


classifier = IntentClassifier()

