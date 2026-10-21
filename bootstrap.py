"""JARVIS Lite — System Bootstrap & Environment Verification.

Performs strict pre-flight checks on runtime, dependencies, hardware,
audio devices, local model servers, and pulled models.
Outputs a clean PASS/FAIL status matrix.
"""

from __future__ import annotations

import shutil
import sys
import warnings
from pathlib import Path
from typing import NamedTuple

# Suppress NVML deprecation warning in output
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent


class CheckResult(NamedTuple):
    category: str
    component: str
    passed: bool
    status_label: str
    details: str
    critical: bool


def check_python_version() -> CheckResult:
    major, minor, micro = sys.version_info[:3]
    version_str = f"{major}.{minor}.{micro}"
    passed = (major == 3 and minor >= 11)
    status = "PASS" if passed else "FAIL"
    details = f"Detected Python {version_str} (Required >= 3.11)"
    return CheckResult("Runtime", "Python Version", passed, status, details, critical=True)


def check_core_dependencies() -> list[CheckResult]:
    required_packages = [
        ("fastapi", "FastAPI Framework"),
        ("uvicorn", "Uvicorn ASGI Server"),
        ("pydantic", "Pydantic V2"),
        ("pydantic_settings", "Pydantic Settings"),
        ("sqlalchemy", "SQLAlchemy 2.0 ORM"),
        ("alembic", "Alembic Migrations"),
        ("chromadb", "ChromaDB Vector Store"),
        ("faster_whisper", "Faster-Whisper STT"),
        ("edge_tts", "Edge-TTS Engine"),
        ("keyring", "Windows Keyring Secrets"),
        ("psutil", "System Telemetry"),
        ("structlog", "Structured JSON Logging"),
        ("pytest", "Pytest Suite"),
    ]

    results: list[CheckResult] = []
    for pkg, name in required_packages:
        try:
            __import__(pkg)
            results.append(
                CheckResult("Dependencies", name, True, "PASS", f"Module '{pkg}' loaded", critical=True)
            )
        except ImportError as exc:
            results.append(
                CheckResult("Dependencies", name, False, "FAIL", f"Missing module: {exc}", critical=True)
            )
    return results


def check_gpu_and_vram() -> CheckResult:
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            return CheckResult("Hardware", "NVIDIA GPU", False, "WARN", "No NVIDIA GPU detected", critical=False)

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode("utf-8")
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_vram_gb = round(mem_info.total / (1024**3), 2)
        free_vram_gb = round(mem_info.free / (1024**3), 2)
        pynvml.nvmlShutdown()

        details = f"{gpu_name} — Total: {total_vram_gb} GB VRAM (Free: {free_vram_gb} GB)"
        passed = total_vram_gb >= 5.5  # RTX 3050 6GB target
        status = "PASS" if passed else "WARN"
        return CheckResult("Hardware", "NVIDIA GPU & VRAM", passed, status, details, critical=False)
    except Exception as exc:
        return CheckResult("Hardware", "NVIDIA GPU", False, "WARN", f"NVML query skipped: {exc}", critical=False)


def check_disk_space() -> CheckResult:
    try:
        usage = shutil.disk_usage(PROJECT_ROOT)
        free_gb = round(usage.free / (1024**3), 2)
        total_gb = round(usage.total / (1024**3), 2)
        passed = free_gb >= 5.0
        status = "PASS" if passed else "FAIL"
        details = f"{free_gb} GB free of {total_gb} GB total (Minimum required: 5.0 GB)"
        return CheckResult("Hardware", "Disk Storage", passed, status, details, critical=True)
    except Exception as exc:
        return CheckResult("Hardware", "Disk Storage", False, "FAIL", f"Failed to check disk: {exc}", critical=True)


def check_audio_devices() -> tuple[CheckResult, CheckResult]:
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_count = sum(1 for d in devices if d.get("max_input_channels", 0) > 0)
        output_count = sum(1 for d in devices if d.get("max_output_channels", 0) > 0)

        in_result = CheckResult(
            "Hardware",
            "Microphone (Input)",
            input_count > 0,
            "PASS" if input_count > 0 else "WARN",
            f"{input_count} recording device(s) detected",
            critical=False,
        )
        out_result = CheckResult(
            "Hardware",
            "Speaker (Output)",
            output_count > 0,
            "PASS" if output_count > 0 else "WARN",
            f"{output_count} playback device(s) detected",
            critical=False,
        )
        return in_result, out_result
    except Exception as exc:
        err = CheckResult("Hardware", "Audio Subsystem", False, "WARN", f"Audio device query failed: {exc}", critical=False)
        return err, err


def check_ollama_server() -> tuple[CheckResult, list[CheckResult]]:
    import httpx

    from config.settings import settings

    base_url = settings.models.ollama.base_url.rstrip("/")
    tags_url = f"{base_url}/api/tags"

    server_check: CheckResult
    model_checks: list[CheckResult] = []

    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(tags_url)
            if resp.status_code == 200:
                data = resp.json()
                installed_models = [m.get("name", "") for m in data.get("models", [])]
                server_check = CheckResult(
                    "Model Server",
                    "Ollama Service",
                    True,
                    "PASS",
                    f"Listening on {base_url} ({len(installed_models)} models loaded)",
                    critical=True,
                )

                required_targets = [
                    ("Chat/Plan LLM", settings.models.routing.get("chat", "qwen2.5:7b-instruct-q4_K_M")),
                    ("Fast Router LLM", settings.models.routing.get("route", "qwen2.5:1.5b-instruct-q4_K_M")),
                    ("Embeddings", settings.models.routing.get("embed", "nomic-embed-text")),
                ]

                for label, target_model in required_targets:
                    # Match exact or base prefix (e.g. qwen2.5:7b matches qwen2.5:7b-instruct-q4_K_M or qwen2.5:7b)
                    base_name = target_model.split(":")[0]
                    found = any(
                        target_model in m or (base_name in m and "qwen" in m)
                        for m in installed_models
                    )
                    status = "PASS" if found else "PULL_NEEDED"
                    details = f"Installed: {target_model}" if found else f"Not pulled yet. Run: 'ollama pull {target_model}'"
                    model_checks.append(
                        CheckResult("Models", label, found, status, details, critical=False)
                    )
            else:
                server_check = CheckResult(
                    "Model Server",
                    "Ollama Service (Optional)",
                    False,
                    "INFO",
                    f"HTTP error {resp.status_code} from {tags_url}. Operating on Custom ONNX models.",
                    critical=False,
                )
    except Exception as exc:
        server_check = CheckResult(
            "Model Server",
            "Ollama Service (Optional)",
            False,
            "INFO",
            f"Offline ({type(exc).__name__}). Operating zero-daemon on native Custom ONNX models.",
            critical=False,
        )

    return server_check, model_checks


def check_custom_models() -> list[CheckResult]:
    results = []
    intent_onnx = PROJECT_ROOT / "models" / "intent_classifier" / "model.onnx"
    embed_onnx = PROJECT_ROOT / "models" / "embeddings" / "model.onnx"
    lora_dir = PROJECT_ROOT / "models" / "lora_adapters" / "jarvis_toolcall"

    results.append(
        CheckResult(
            "Custom Models",
            "ONNX Intent Classifier",
            intent_onnx.exists(),
            "PASS" if intent_onnx.exists() else "WARN",
            f"Active: {intent_onnx.name} (< 2ms zero-daemon inference)"
            if intent_onnx.exists()
            else "Not trained yet. Run 'python -m training.train_intent'",
            critical=False,
        )
    )
    results.append(
        CheckResult(
            "Custom Models",
            "ONNX Embeddings Model",
            embed_onnx.exists(),
            "PASS" if embed_onnx.exists() else "WARN",
            f"Active: {embed_onnx.name} (768-dim vector encoder)"
            if embed_onnx.exists()
            else "Not trained yet. Run 'python -m training.train_embeddings'",
            critical=False,
        )
    )
    results.append(
        CheckResult(
            "Custom Models",
            "LoRA Toolcall Adapter",
            lora_dir.exists(),
            "PASS" if lora_dir.exists() else "INFO",
            "Adapter weights present in models/lora_adapters"
            if lora_dir.exists()
            else "Optional. Run 'python -m training.lora_finetune'",
            critical=False,
        )
    )
    return results


def print_bootstrap_report(results: list[CheckResult]) -> bool:
    # Color codes for terminal
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    title = "JARVIS LITE — PRE-FLIGHT BOOTSTRAP CHECK"
    print("\n" + "=" * 90)
    print(f"{BOLD}{CYAN}{title:^90}{RESET}")
    print("=" * 90)
    print(f"{'CATEGORY':<15} | {'COMPONENT':<25} | {'STATUS':<12} | {'DETAILS'}")
    print("-" * 90)

    critical_failures = 0

    for r in results:
        if r.status_label == "PASS":
            status_formatted = f"{GREEN}[PASS]{RESET}"
        elif r.status_label in ("WARN", "PULL_NEEDED"):
            status_formatted = f"{YELLOW}[{r.status_label}]{RESET}"
        elif r.status_label == "INFO":
            status_formatted = f"{CYAN}[INFO]{RESET}"
        else:
            status_formatted = f"{RED}[FAIL]{RESET}"
            if r.critical:
                critical_failures += 1

        print(f"{r.category:<15} | {r.component:<25} | {status_formatted:<21} | {r.details}")

    print("=" * 90)
    if critical_failures == 0:
        print(f"{GREEN}{BOLD}>> BOOTSTRAP PASSED: All critical systems are operational.{RESET}\n")
        return True
    else:
        print(f"{RED}{BOLD}>> BOOTSTRAP FAILED: {critical_failures} critical requirement(s) missing.{RESET}\n")
        return False


def run_bootstrap() -> bool:
    checks: list[CheckResult] = []
    checks.append(check_python_version())
    checks.extend(check_core_dependencies())
    checks.append(check_gpu_and_vram())
    checks.append(check_disk_space())

    in_res, out_res = check_audio_devices()
    checks.append(in_res)
    checks.append(out_res)

    # Custom Native Models
    checks.extend(check_custom_models())

    srv_check, model_checks = check_ollama_server()
    checks.append(srv_check)
    checks.extend(model_checks)

    passed = print_bootstrap_report(checks)
    return passed


if __name__ == "__main__":
    success = run_bootstrap()
    sys.exit(0 if success else 1)
