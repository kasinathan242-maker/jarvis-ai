# JARVIS Lite — Local-First Personal AI Operating System for Windows 11

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-CPU_Optimized-005CED?style=flat-square&logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6F00?style=flat-square)](https://trychroma.com)
[![Mypy Strict](https://img.shields.io/badge/Mypy-Strict_100%25-2F5A88?style=flat-square)](https://mypy.readthedocs.io)
[![Ruff](https://img.shields.io/badge/Ruff-Checked-D7FF64?style=flat-square&logo=ruff&logoColor=black)](https://astral.sh/ruff)
[![Tests Passing](https://img.shields.io/badge/Tests-72%2F72_Passing-brightgreen?style=flat-square)](https://pytest.org)

**JARVIS Lite** is a production-grade, local-first personal AI operating system custom-engineered for Windows 11 on the RTX 3050 Laptop GPU (6 GB VRAM). 

It operates completely **zero-daemon** and **zero-Ollama** by default, running custom neural intent classification (< 2ms) and 768-dimensional contrastive semantic embeddings on CPU with 0 MB VRAM footprint, reserving 100% of GPU resources for real-time speech transcription and local vision analysis.

---

## Key Highlights

- **Native Custom Neural Models (No Ollama Daemon Needed):**
  - **Intent Classifier:** Custom PyTorch MLP trained on domain dialogue datasets, exported to ONNX (`models/intent_classifier/model.onnx`). Runs in **< 2ms** on CPU with **0 MB VRAM** usage.
  - **Semantic Embeddings:** Custom Triplet Margin Contrastive encoder projecting text to 768-dimensional L2-normalized unit vectors, natively synced with ChromaDB vector collections.
  - **LoRA QLoRA Fine-Tuning Pipeline:** Full PEFT training pipeline with automated VRAM budget guards (< 5.2 GB) for causal language models on consumer GPUs.
- **Strict Quality Bar & Testing:**
  - **72 / 72 Passing Tests** in Pytest across all 4 phases.
  - **100% Strict Type Safety** via `mypy --strict .` across 119 source files.
  - **0 Linter Errors** via `ruff check .`.
- **Zero-Trust Security Architecture:**
  - Mandatory post-execution `verify()` contract on every tool call.
  - Path sandboxing restricting execution strictly within the user workspace.
  - Shell command risk classifier (Allowlist, Confirmation Required, Forbidden blocklist).
  - Prompt injection defense using `<untrusted_content>` boundary isolation.
  - Windows Credential Manager (`keyring`) bearer tokens for LAN dashboard access.
- **Voice & Multimodal Pipeline:**
  - Streaming `openWakeWord` ("Hey Jarvis") on CPU (< 5% idle CPU).
  - Silero Voice Activity Detection (VAD).
  - `faster-whisper` quantized int8 STT on CPU to protect GPU VRAM.
  - Interruptible Edge-TTS with instant cancellation upon user barge-in.
- **Multitasking & System HUD:**
  - Dark Cyberpunk Agency HUD with real-time SVG CPU/VRAM telemetry gauges.
  - Global Emergency Stop hotkey (`Ctrl + Alt + Shift + X`) halting any active task within 1 second.
  - Multi-step workflow orchestration engine with atomic JSON checkpoint recovery.

---

## Architecture Overview

```
                          ┌───────────────────────────┐
                          │   Voice / Text / Remote   │
                          │   Wake Word / CLI / HUD   │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │   Custom Intent Engine    │
                          │   ONNX CPU Neural Model   │
                          │     (< 2ms, 0 MB VRAM)    │
                          └─────────────┬─────────────┘
                                        │
                 ┌──────────────────────┼──────────────────────┐
                 ▼                      ▼                      ▼
        ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
        │  System & Apps  │    │  ChromaDB RAG   │    │ Autonomous DAG  │
        │ Stats, Windows, │    │ 768-dim Custom  │    │ Multi-Step Flow │
        │ Shell, Apps, UI │    │ ONNX Embeddings │    │ & Checkpoints   │
        └────────┬────────┘    └────────┬────────┘    └────────┬────────┘
                 │                      │                      │
                 └──────────────────────┼──────────────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │  Zero-Trust Safety Engine │
                          │  Verify() Post-Condition  │
                          │  Keyring & Injection Wall │
                          └───────────────────────────┘
```

---

## Repository Structure

```
jarvis/
├── main.py                    # Master CLI, server entrypoint, & voice loop
├── bootstrap.py               # Pre-flight hardware, VRAM, and model verification
├── shutdown.py                # OS signal trap and graceful resource teardown
├── pyproject.toml             # Ruff, Pytest, and strict Mypy configurations
├── requirements.txt           # Production dependency specifications
├── config/
│   ├── settings.py            # Pydantic Settings V2 configuration engine
│   └── settings.yaml          # System, model, audio, and safety thresholds
├── ai/
│   ├── custom_model_engine.py # Zero-daemon ONNX runtime (Intent + Embeddings)
│   ├── circuit_breaker.py     # Resilient service circuit breaker (CLOSED/OPEN/HALF_OPEN)
│   ├── reasoning.py           # Structured ToolCall synthesis and reasoning
│   ├── ollama_client.py       # Optional secondary Ollama fallback client
│   └── lmstudio_client.py     # Optional secondary LM Studio fallback client
├── intent/
│   ├── classifier.py          # Fast neural ONNX classifier + regex router
│   └── schemas.py             # Pydantic Intent schemas
├── training/
│   ├── dataset_generator.py   # Synthetic intent, triplet, & tool-calling dataset generator
│   ├── train_intent.py        # PyTorch classifier training & ONNX export
│   ├── train_embeddings.py    # Contrastive triplet loss training & 768-dim ONNX export
│   ├── lora_finetune.py       # PEFT QLoRA fine-tuning with VRAM budget guards
│   └── data/                  # Curated training corpora and dialogue samples
├── models/
│   ├── intent_classifier/     # Production ONNX model, vocab, and class configs
│   └── embeddings/            # Production 768-dim ONNX vector model and vocab
├── core/
│   ├── emergency_stop.py      # Global emergency halt (< 1s kill switch)
│   ├── lifecycle.py           # Async component startup & lifecycle manager
│   ├── event_bus.py           # Pub/Sub event bus
│   ├── task_manager.py        # Background task registry
│   ├── logging.py             # Structlog JSON logging with secret scrubbing
│   └── errors.py              # Typed domain exceptions
├── security/
│   ├── permissions.py         # Shell command allowlist/blocklist & path sandboxing
│   ├── secrets.py             # Windows Keyring API token management
│   ├── approval.py            # Async human-in-the-loop approval tickets
│   ├── audit_log.py           # Immutable SQLite audit log for all tool calls
│   └── injection_guard.py     # Prompt injection detection and XML tag isolation
├── tools/
│   ├── base.py                # BaseTool contract requiring verify() implementation
│   ├── registry.py            # Discoverable tool registry & dispatcher
│   ├── time_tool.py           # Current time and timezone tool
│   ├── system_stats.py        # CPU, RAM, Disk, and NVIDIA VRAM telemetry
│   ├── applications.py        # Safe application launcher with PID verification
│   ├── clipboard.py           # Clipboard reader/writer with pasteback verification
│   ├── screenshot.py          # Screenshot capture tool
│   ├── windows.py             # Windows focus, minimize, and restore
│   ├── files.py               # Sandboxed search, open, and verified delete
│   ├── shell.py               # Sandboxed shell execution with permission gating
│   ├── browser.py             # Playwright web navigation and scraper
│   ├── organizer.py           # File organizer & regex batch rename
│   ├── reminders.py           # Reminders and scheduler
│   ├── google.py              # Gmail and Google Calendar tools
│   ├── developer.py           # Code reading, symbol search, diff edit, and test execution
│   └── vision.py              # Screen analyzer tool
├── memory/
│   ├── conversation_store.py  # SQLite conversation history
│   ├── long_term_memory.py    # ChromaDB memory store with user approval
│   ├── retrieval.py           # Hybrid reciprocal-rank memory search
│   └── extractor.py           # Fact extractor from user dialogues
├── orchestrator/
│   ├── dag.py                 # TaskDAG dependency resolver with cycle detection
│   └── engine.py              # Multi-agent DAG execution with atomic checkpoints
├── automation/
│   ├── schema.py              # Declarative YAML workflow definitions
│   └── engine.py              # Trigger and workflow automation engine
├── documents/
│   ├── loader.py              # PDF, DOCX, and text document parsers
│   ├── indexer.py             # ChromaDB vector indexer with exact page citations
│   ├── operations.py          # Summarizer and table extractor
│   └── watcher.py             # Directory observer for auto-indexing
├── proactive/
│   └── engine.py              # Proactive suggestions with quiet hours
├── personalization/
│   └── manager.py             # Adaptive preference learning
├── maintenance/
│   ├── backup.py              # Gzip tarball backup & SHA256 restore manager
│   └── crash_reporter.py      # Secret-redacted crash reports
├── vision/
│   ├── screen_reader.py       # Desktop screenshot capture & local vision reasoning
│   └── gestures.py            # Webcam hand gesture controller (opt-in)
├── voice/
│   ├── speech_to_text.py      # Faster-Whisper int8 STT
│   ├── text_to_speech.py      # Edge-TTS / Piper playback
│   ├── vad.py                 # Silero voice activity detection
│   ├── wake_word.py           # openWakeWord listener
│   └── pipeline.py            # End-to-end voice processing loop
├── api/
│   ├── server.py              # FastAPI server & routes
│   ├── ws.py                  # Real-time WebSocket telemetry stream
│   └── remote.py              # LAN companion REST endpoints
├── ui/
│   ├── index.html             # Cyberpunk Command HUD interface
│   ├── app.css                # Visual theme & animations
│   ├── app.js                 # HUD frontend controller
│   └── remote.html            # Mobile-friendly companion dashboard
└── tests/                     # 72 Comprehensive unit, integration, and security tests
```

---

## Quickstart Guide

### 1. Environment Setup

Prerequisites: **Windows 11**, **Python 3.11**, and an NVIDIA GPU (RTX 3050 or higher recommended).

```powershell
# Create Python 3.11 virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install locked dependencies
pip install -r requirements.txt
```

### 2. Pre-Flight Diagnostic Check

Run the pre-flight verification script to confirm GPU VRAM, audio hardware, disk space, and models:

```powershell
python bootstrap.py
```

Expected output:
```
==========================================================================================
                         JARVIS LITE — PRE-FLIGHT BOOTSTRAP CHECK                         
==========================================================================================
CATEGORY        | COMPONENT                 | STATUS       | DETAILS
------------------------------------------------------------------------------------------
Runtime         | Python Version            | [PASS]       | Detected Python 3.11.16 (Required >= 3.11)
Dependencies    | FastAPI Framework         | [PASS]       | Module 'fastapi' loaded
Dependencies    | ChromaDB Vector Store     | [PASS]       | Module 'chromadb' loaded
Dependencies    | Faster-Whisper STT        | [PASS]       | Module 'faster_whisper' loaded
Hardware        | NVIDIA GPU & VRAM         | [PASS]       | NVIDIA GeForce RTX 3050 (6.0 GB VRAM)
Custom Models   | ONNX Intent Classifier    | [PASS]       | Active: model.onnx (< 2ms zero-daemon inference)
Custom Models   | ONNX Embeddings Model     | [PASS]       | Active: model.onnx (768-dim vector encoder)
==========================================================================================
>> BOOTSTRAP PASSED: All critical systems are operational.
```

### 3. Launching JARVIS Lite

#### Option A: Start Server & Web HUD
```powershell
python main.py
```
* **Command HUD:** Open your browser to `http://127.0.0.1:8001/hud`
* **Mobile Companion (LAN):** Open `http://<YOUR_LAN_IP>:8001/remote` from your phone or tablet.

#### Option B: Interactive Voice / Command Shell
```powershell
python main.py --cli
```

---

## Custom Model Training Suite

JARVIS Lite comes with a built-in end-to-end model training suite in the `training/` package:

```powershell
# Step 1: Generate synthetic training datasets for intents, triplets, & tool dialogues
python -m training.dataset_generator

# Step 2: Train custom neural intent classifier & export to ONNX
python -m training.train_intent

# Step 3: Train custom semantic embedding encoder & export to ONNX (768-dim)
python -m training.train_embeddings

# Step 4: (Optional) Run PEFT/QLoRA fine-tuning for causal LLMs on RTX 3050
python -m training.lora_finetune
```

---

## Running Quality Gates & Tests

To execute the full verification suite:

```powershell
# 1. Run all 72 unit, integration, security, and custom model tests
pytest -v

# 2. Run Ruff linter (0 errors)
ruff check .

# 3. Run strict type checking across all source files (0 errors)
mypy --strict .
```

---

## Safety & Emergency Controls

* **Global Emergency Stop Hotkey:** Press `Ctrl + Alt + Shift + X` from anywhere in Windows to immediately terminate active workflows, shell processes, and audio streams.
* **CLI Emergency Command:** Say or type `"jarvis stop everything"` or `"emergency stop"`.
* **Zero-Trust Verification:** Tools will report honest failures if system state modifications cannot be verified post-execution.

---

## License

This project is licensed under the MIT License.
