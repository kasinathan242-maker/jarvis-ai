"""JARVIS Lite — Voice & Command Processing Pipeline.

Orchestrates end-to-end execution:
Audio / Text Input -> Faster-Whisper STT -> Intent Classification ->
AI Reasoning Plan -> Tool Execution with Permissions & Approvals ->
Memory Persistence -> Interruptible TTS Spoken Response.
"""

from __future__ import annotations

import time

import numpy as np

from ai.reasoning import planner
from core.emergency_stop import emergency_stop
from core.errors import SecurityViolationError
from core.event_bus import bus
from core.logging import get_logger
from intent.classifier import classifier
from intent.schemas import IntentType
from memory.conversation_store import save_message
from memory.extractor import extract_memory_candidates
from memory.long_term_memory import save_long_term_memory
from security.approval import approval_manager
from tools.registry import registry
from voice.speech_to_text import transcribe_audio_array
from voice.text_to_speech import tts

logger = get_logger("voice.pipeline")


class VoicePipeline:
    async def process_command(
        self,
        user_text: str,
        play_speech: bool = False,
        user_approved: bool | None = None,
    ) -> str:
        """Process a text command through the full pipeline."""
        start_time = time.monotonic()
        clean_text = user_text.strip()
        if not clean_text:
            return "No command detected."

        logger.info("processing_command", input=clean_text)
        await bus.publish("command_received", {"text": clean_text})

        # 0. Check for pending approval voice answers first
        pending = approval_manager.get_pending()
        if pending:
            lower = clean_text.lower()
            if any(w in lower for w in ("confirm", "yes", "proceed", "approve", "do it")):
                resolved = approval_manager.resolve_latest_by_voice(approved=True)
                if resolved:
                    msg = f"Action '{resolved.tool_name}' approved by voice. Proceeding."
                    save_message("user", clean_text, intent="approval")
                    save_message("assistant", msg, intent="approval")
                    if play_speech:
                        await tts.speak(msg, play_locally=True)
                    return msg

            if any(w in lower for w in ("cancel", "no", "deny", "abort", "reject", "stop")):
                resolved = approval_manager.resolve_latest_by_voice(approved=False)
                if resolved:
                    msg = f"Action '{resolved.tool_name}' cancelled by voice. Nothing was executed."
                    save_message("user", clean_text, intent="approval")
                    save_message("assistant", msg, intent="approval")
                    if play_speech:
                        await tts.speak(msg, play_locally=True)
                    return msg

        # 1. Classify Intent
        intent = await classifier.classify(clean_text)
        logger.info("intent_classified", type=intent.type.value, confidence=intent.confidence)

        # 2. Check Emergency Stop
        if intent.type == IntentType.EMERGENCY_STOP:
            halt_info = await emergency_stop.trigger_halt(source="voice" if play_speech else "text")
            response = (
                f"Emergency stop executed immediately. Cancelled {halt_info['tasks_cancelled']} active task(s) "
                f"and {halt_info['approvals_cancelled']} pending approval(s) in {halt_info['duration_ms']} ms."
            )
            save_message("user", clean_text, intent=intent.type.value)
            save_message("assistant", response, intent=intent.type.value)
            if play_speech:
                await tts.speak(response, play_locally=True)
            return response

        # 3. Handle Greeting / Exit
        if intent.type == IntentType.GREETING:
            response = "Hello! I am JARVIS Lite, your personal AI operating system. All systems are operational."
            save_message("user", clean_text, intent=intent.type.value)
            save_message("assistant", response, intent=intent.type.value)
            if play_speech:
                await tts.speak(response, play_locally=True)
            return response

        if intent.type == IntentType.EXIT:
            response = "Shutting down JARVIS Lite. Have a productive day!"
            save_message("user", clean_text, intent=intent.type.value)
            save_message("assistant", response, intent=intent.type.value)
            if play_speech:
                await tts.speak(response, play_locally=True)
            return response

        if intent.type == IntentType.ANALYZE_SCREEN:
            tool_res = await registry.dispatch("analyze_screen", {"question": clean_text}, user_approved=True)
            if tool_res.success and tool_res.data:
                desc = getattr(tool_res.data, "description", "")
                if not desc and isinstance(tool_res.data, dict):
                    desc = tool_res.data.get("description", "")
                response = f"Here is what is on your screen: {desc}"
            else:
                response = f"Unable to analyze the screen: {tool_res.error}"
            save_message("user", clean_text, intent=intent.type.value)
            save_message("assistant", response, intent=intent.type.value)
            if play_speech:
                await tts.speak(response, play_locally=True)
            return response

        # 4. Propose Reasoning Plan & Tool Calls
        plan = await planner.create_plan(clean_text)
        logger.info("reasoning_plan_created", thoughts=plan.thoughts, tool_calls_count=len(plan.tool_calls))

        response = ""

        # 5. Execute Tool Calls if proposed
        if plan.tool_calls:
            results_summary = []
            for call in plan.tool_calls:
                try:
                    result = await registry.execute(call, user_approved=user_approved)
                    if result.success and result.data:
                        # Format human response based on tool
                        if call.tool_name == "get_time":
                            formatted_time = getattr(result.data, "formatted_time", str(result.data))
                            results_summary.append(f"The current time is {formatted_time}.")
                        elif call.tool_name == "open_application":
                            app_name = getattr(result.data, "app_name", "application")
                            pid = getattr(result.data, "pid", "")
                            msg = f"I have opened {app_name.capitalize()} (PID: {pid}). Window verified."
                            results_summary.append(msg)
                        elif call.tool_name == "close_application":
                            app_name = getattr(result.data, "app_name", "application")
                            closed = getattr(result.data, "closed_processes", 0)
                            results_summary.append(f"Closed {app_name.capitalize()} ({closed} process(es) terminated).")
                        elif call.tool_name == "system_stats":
                            cpu = getattr(result.data, "cpu_percent", 0.0)
                            ram_used = getattr(result.data, "ram_used_gb", 0.0)
                            ram_total = getattr(result.data, "ram_total_gb", 0.0)
                            gpu = getattr(result.data, "gpu_name", "N/A")
                            results_summary.append(
                                f"System status: CPU at {cpu}%, RAM using {ram_used} GB of {ram_total} GB. GPU: {gpu}."
                            )
                        elif call.tool_name == "screenshot":
                            p = getattr(result.data, "file_path", "data/screenshots")
                            results_summary.append(f"Screenshot captured and saved to {p}.")
                        elif call.tool_name == "clipboard_read":
                            content = getattr(result.data, "content", "")
                            results_summary.append(f"Clipboard content: '{content}'.")
                        elif call.tool_name == "clipboard_write":
                            results_summary.append("Text successfully copied to clipboard.")
                        elif call.tool_name == "open_path":
                            target = getattr(result.data, "resolved_path", "")
                            results_summary.append(f"Opened path: {target}.")
                        elif call.tool_name == "delete_file":
                            target = getattr(result.data, "target_path", "")
                            results_summary.append(f"Deleted file '{target}'. Verified removal from disk.")
                        elif call.tool_name == "shell_execute":
                            out_val = getattr(result.data, "stdout", "").strip()
                            err_val = getattr(result.data, "stderr", "").strip()
                            out = out_val or err_val
                            results_summary.append(f"Command output: {out[:200]}")
                        elif call.tool_name == "window_control":
                            action = getattr(result.data, "action_performed", "")
                            title = getattr(result.data, "window_title", "")
                            results_summary.append(f"Window '{title}' {action} performed successfully.")
                        elif call.tool_name == "research_topic":
                            p = getattr(result.data, "report_path", "")
                            results_summary.append(f"Research complete. Detailed dossier saved to {p}.")
                        elif call.tool_name == "daily_briefing":
                            text = getattr(result.data, "briefing_text", "")
                            results_summary.append(text)
                        elif call.tool_name == "draft_email_reply":
                            to_e = getattr(result.data, "to_email", "")
                            sub = getattr(result.data, "subject", "")
                            results_summary.append(f"Drafted email reply to {to_e} regarding '{sub}'. Ready for approval.")
                        elif call.tool_name == "send_email":
                            to_e = getattr(result.data, "to_email", "")
                            results_summary.append(f"Email successfully sent to {to_e}.")
                        elif call.tool_name == "set_reminder":
                            msg = getattr(result.data, "message", "")
                            results_summary.append(f"Reminder set for: '{msg}'.")
                        elif call.tool_name == "organize_files":
                            count = getattr(result.data, "total_files_evaluated", 0)
                            dry = getattr(result.data, "dry_run", True)
                            results_summary.append(f"Organized files ({count} files evaluated, dry_run={dry}).")
                        elif call.tool_name == "run_tests":
                            passed = getattr(result.data, "all_passed", False)
                            t_cnt = getattr(result.data, "total_tests", 0)
                            status_str = "All tests passed" if passed else "Tests failed"
                            results_summary.append(f"Test suite execution finished: {status_str} ({t_cnt} tests).")
                        else:
                            results_summary.append(f"Tool {call.tool_name} completed.")
                    else:
                        results_summary.append(f"Failed to execute {call.tool_name}: {result.error}")
                except SecurityViolationError as sec_err:
                    results_summary.append(f"Security block: {sec_err}")
                except Exception as exc:
                    results_summary.append(f"Error executing {call.tool_name}: {exc}")

            response = " ".join(results_summary)
        elif plan.direct_response:
            response = plan.direct_response
        else:
            response = "Understood."

        # 6. Extract candidate memories
        candidates = extract_memory_candidates(clean_text)
        for cand in candidates:
            save_long_term_memory(
                content=cand.content,
                category=cand.category,
                source=f"conversation:{clean_text[:30]}",
                approved=True,
            )

        # 7. Persist to SQLite Conversation Store
        save_message("user", clean_text, intent=intent.type.value)
        save_message("assistant", response, intent=intent.type.value)

        elapsed = round(time.monotonic() - start_time, 2)
        logger.info("command_completed", duration_seconds=elapsed, response=response)

        # 8. Speak response if voice requested (interruptible)
        if play_speech and response:
            await tts.speak(response, play_locally=True)

        return response

    async def process_audio(self, audio_array: np.ndarray, play_speech: bool = True) -> str:
        """Transcribe audio from microphone and execute through command pipeline."""
        # Interrupt any speech in progress the moment user speaks
        if tts.is_playing:
            tts.stop()

        transcribed_text = transcribe_audio_array(audio_array)
        if not transcribed_text:
            return "Could not understand audio."
        return await self.process_command(transcribed_text, play_speech=play_speech)


pipeline = VoicePipeline()
