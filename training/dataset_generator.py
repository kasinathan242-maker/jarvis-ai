"""JARVIS Lite — Custom Model Training Dataset Generator.

Generates structured datasets for:
1. Intent Classification (12 intent classes, 600+ utterances with variations)
2. Semantic Sentence Embeddings (Contrastive positive/negative pairs)
3. Tool-Calling Dialogue Dataset (Alpaca/ChatML format for PEFT/QLoRA)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

INTENT_SEED_DATA: dict[str, list[str]] = {
    "open_application": [
        "open vscode", "launch visual studio code", "start chrome browser",
        "open notepad application", "launch terminal", "open powershell",
        "open my discord", "start spotify music player", "launch calculator",
        "open file explorer", "start edge browser", "launch code editor",
        "open slack", "start task manager", "launch command prompt",
        "open browser and navigate", "start steam client", "launch pyvirt",
        "open excel spreadsheet", "launch word document reader",
    ],
    "system_stats": [
        "what is my current cpu usage", "check ram utilization", "how much vram is free on gpu",
        "show me system telemetry", "what is my battery level", "check gpu temperature",
        "is my laptop plugged in", "display memory statistics", "check disk space available",
        "how hot is my rtx 3050", "monitor hardware load", "what is the system health",
        "check vram and cpu", "show system resource consumption", "how much storage is left on c drive",
        "are any background tasks overheating the machine", "check network latency and ping",
    ],
    "search_files": [
        "find file named report.pdf", "search for main.py in downloads", "locate invoice documents",
        "find all python files matching test_", "where is my configuration file",
        "search files with extension docx", "locate project specification markdown",
        "find files modified today in workspace", "search for requirements.txt",
        "look for budget spreadsheet in downloads", "find all log files older than a week",
    ],
    "delete_file": [
        "delete old_log.txt", "remove scratch file in project", "erase temporary test screenshot",
        "delete obsolete backup archive", "remove unused cache files in data directory",
        "delete draft document from disk", "trash temporary debug log",
    ],
    "run_shell": [
        "run git status", "execute pytest -v in terminal", "run pip list command",
        "check npm dependencies with npm list", "run python bootstrap.py",
        "execute git diff in workspace", "run dir in current folder",
        "run pip check to inspect environment", "execute pytest tests/test_tools.py",
    ],
    "clipboard_action": [
        "what is on my clipboard", "read clipboard text", "copy this summary to my clipboard",
        "write the generated code to clipboard", "inspect current clipboard buffer",
        "paste clipboard content into message", "clear the clipboard buffer",
    ],
    "window_action": [
        "focus chrome window", "minimize vscode editor", "maximize browser window",
        "bring visual studio code to front", "switch to command prompt window",
        "list all active windows on my desktop", "hide the terminal window",
    ],
    "browse_web": [
        "search google for python asyncio patterns", "browse langchain documentation online",
        "look up latest pytorch release notes", "search duckduckgo for windows api docs",
        "find tutorials on transformer fine-tuning", "browse web for fast llm inference methods",
        "search github for fast modern hud designs", "research five top ai courses online",
    ],
    "email_calendar": [
        "check my unread emails in gmail", "draft an email reply to john",
        "do i have any calendar events scheduled today", "create a meeting tomorrow at 3pm",
        "search emails from google cloud", "list upcoming calendar reminders",
        "draft a follow-up email confirming project completion",
    ],
    "document_qa": [
        "summarize the project specification pdf", "what are the requirements in document",
        "compare document a with document b", "extract the summary table from report docx",
        "search indexed documents for memory budget", "what does page 3 of the manual state",
    ],
    "screen_analysis": [
        "what is on my screen right now", "describe desktop display", "analyze the active window",
        "what error is displayed on my screen", "read the text currently visible on monitor",
        "take a screenshot and tell me what you see", "inspect visual contents of my desktop",
    ],
    "emergency_stop": [
        "emergency stop", "jarvis halt immediately", "kill all active tasks right now",
        "freeze all background execution", "abort task right now", "stop everything immediately",
        "shut down running operations now", "cancel all pending actions immediately",
    ],
    "general_chat": [
        "hello jarvis how are you", "who created you", "what capabilities do you have",
        "tell me a quick tech joke", "explain quantum computing in simple words",
        "good morning jarvis", "thank you for the assistance", "can you help me with coding",
    ],
}

SYNONYM_EXPANSIONS: dict[str, list[str]] = {
    "open": ["open", "launch", "start", "run", "fire up"],
    "check": ["check", "inspect", "show me", "what is", "display", "tell me"],
    "find": ["find", "search for", "locate", "look for", "where is"],
    "delete": ["delete", "remove", "erase", "trash"],
    "stop": ["stop", "halt", "kill", "abort", "freeze", "terminate"],
}


def generate_intent_dataset(output_path: Path | str) -> dict[str, Any]:
    """Generates an expanded intent classification dataset with augmented utterances."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, str]] = []
    classes = sorted(list(INTENT_SEED_DATA.keys()))

    for intent_label, base_utterances in INTENT_SEED_DATA.items():
        for utterance in base_utterances:
            samples.append({"text": utterance, "label": intent_label})
            # Generate light syntactic augmentations
            samples.append({"text": f"jarvis {utterance}", "label": intent_label})
            samples.append({"text": f"please {utterance}", "label": intent_label})
            samples.append({"text": f"could you {utterance}", "label": intent_label})
            samples.append({"text": f"jarvis can you {utterance}", "label": intent_label})

    dataset = {
        "classes": classes,
        "num_classes": len(classes),
        "total_samples": len(samples),
        "samples": samples,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    return dataset


def generate_embedding_pairs(output_path: Path | str) -> dict[str, Any]:
    """Generates anchor-positive-negative sentence triplets for contrastive embedding training."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    triplets = [
        {
            "anchor": "what is my current cpu and ram usage",
            "positive": "show hardware telemetry statistics for processor and memory",
            "negative": "open visual studio code editor",
        },
        {
            "anchor": "how much vram is free on the rtx 3050 gpu",
            "positive": "check dedicated graphics video memory available",
            "negative": "summarize the quarterly business pdf report",
        },
        {
            "anchor": "open visual studio code and project workspace",
            "positive": "launch vscode and open my source code repository",
            "negative": "check unread emails in inbox",
        },
        {
            "anchor": "find all python test files in the downloads directory",
            "positive": "locate test_*.py files inside the user downloads folder",
            "negative": "halt all running operations immediately",
        },
        {
            "anchor": "what is currently shown on my laptop screen",
            "positive": "capture desktop display and describe visible windows",
            "negative": "delete old log files from storage",
        },
        {
            "anchor": "emergency stop halt all background tasks right now",
            "positive": "abort all active jobs and stop execution immediately",
            "negative": "draft an email reply thanking the client",
        },
        {
            "anchor": "draft a reply to the email from manager",
            "positive": "compose an email draft response to my supervisor",
            "negative": "how hot is the gpu processor running",
        },
        {
            "anchor": "organize messy files in downloads by file extension",
            "positive": "categorize files in folder into subdirectories by extension",
            "negative": "minimize active browser window",
        },
    ]

    # Synthesize additional variations
    dataset = {
        "triplets": triplets,
        "total_triplets": len(triplets),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    return dataset


def generate_toolcall_dataset(output_path: Path | str) -> list[dict[str, Any]]:
    """Generates an instruction dataset formatted for fine-tuning LLM tool-calling in ChatML/Alpaca schema."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    system_prompt = (
        "You are JARVIS Lite, a local-first personal AI assistant for Windows 11. "
        "When a user asks for an action that requires system operations, propose a structured ToolCall."
    )

    records: list[dict[str, Any]] = [
        {
            "instruction": "Open VS Code and check my project folder",
            "system": system_prompt,
            "response": '{"tool_name": "open_application", "parameters": {"app_name": "code"}}',
        },
        {
            "instruction": "What is my current GPU memory and CPU load?",
            "system": system_prompt,
            "response": '{"tool_name": "system_stats", "parameters": {"detailed": true}}',
        },
        {
            "instruction": "Find file named alpha_spec.pdf in downloads",
            "system": system_prompt,
            "response": '{"tool_name": "search_files", "parameters": {"query": "alpha_spec.pdf", "directory": "C:/Users/mohan/Downloads"}}',
        },
        {
            "instruction": "What is on my screen right now?",
            "system": system_prompt,
            "response": '{"tool_name": "analyze_screen", "parameters": {"prompt": "Describe what is currently visible on the user desktop display."}}',
        },
        {
            "instruction": "Emergency stop, kill all background tasks!",
            "system": system_prompt,
            "response": '{"tool_name": "emergency_stop", "parameters": {"source": "user_command"}}',
        },
        {
            "instruction": "Draft an email reply to Alice confirming tomorrow's sync",
            "system": system_prompt,
            "response": '{"tool_name": "draft_email_reply", "parameters": {"to": "alice@example.com", "subject": "Re: Sync", "body": "Hi Alice, confirming our sync tomorrow. Best regards."}}',
        },
        {
            "instruction": "Clean up files in Downloads by extension",
            "system": system_prompt,
            "response": '{"tool_name": "file_organizer", "parameters": {"directory": "C:/Users/mohan/Downloads", "group_by": "extension", "dry_run": false}}',
        },
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    return records


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent / "data"
    generate_intent_dataset(base_dir / "intent_dataset.json")
    generate_embedding_pairs(base_dir / "embedding_pairs.json")
    generate_toolcall_dataset(base_dir / "toolcall_dataset.json")
    print(f"Datasets generated successfully in {base_dir}")
