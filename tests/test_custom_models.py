"""Tests for Custom Models Architecture, Training Pipelines, and Zero-Ollama Inference."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from ai.custom_model_engine import (
    custom_chroma_embed_fn,
    custom_embedding_engine,
    custom_intent_engine,
    custom_toolcall_engine,
)
from intent.classifier import classifier
from intent.schemas import IntentType
from training.dataset_generator import (
    generate_embedding_pairs,
    generate_intent_dataset,
    generate_toolcall_dataset,
)
from training.lora_finetune import check_gpu_vram_budget, train_lora_adapter


def test_dataset_generators(tmp_path: Path) -> None:
    intent_data = generate_intent_dataset(tmp_path / "intents.json")
    assert intent_data["num_classes"] >= 10
    assert intent_data["total_samples"] > 50

    embed_data = generate_embedding_pairs(tmp_path / "triplets.json")
    assert embed_data["total_triplets"] >= 5

    tool_data = generate_toolcall_dataset(tmp_path / "tools.json")
    assert len(tool_data) >= 5


def test_custom_intent_engine_predictions() -> None:
    intent, conf = custom_intent_engine.predict("open visual studio code editor")
    assert intent == "open_application"
    assert conf > 0.40

    intent, conf = custom_intent_engine.predict("what is my current cpu and ram load")
    assert intent == "system_stats"
    assert conf > 0.40

    intent, conf = custom_intent_engine.predict("emergency stop halt everything immediately")
    assert intent == "emergency_stop"
    assert conf > 0.40


def test_custom_embedding_engine_properties() -> None:
    vec = custom_embedding_engine.embed_text("JARVIS Lite local operating system")
    assert len(vec) == 768
    # Assert vector is L2 normalized: sum(v^2) == 1.0
    l2_norm = math.sqrt(sum(v * v for v in vec))
    assert math.isclose(l2_norm, 1.0, rel_tol=1e-3)

    docs = ["System memory status", "Emergency halt running tasks"]
    multi_vecs = custom_embedding_engine.embed_documents(docs)
    assert len(multi_vecs) == 2
    assert len(multi_vecs[0]) == 768


def test_custom_toolcall_engine_resolution() -> None:
    call = custom_toolcall_engine.plan_tool_call("check current cpu and vram telemetry")
    assert call is not None
    assert call.tool_name == "system_stats"

    call = custom_toolcall_engine.plan_tool_call("open visual studio code")
    assert call is not None
    assert call.tool_name == "open_application"
    assert call.parameters["app_name"] == "code"

    call = custom_toolcall_engine.plan_tool_call("emergency stop kill active tasks")
    assert call is not None
    assert call.tool_name == "emergency_stop"


@pytest.mark.asyncio
async def test_intent_classifier_uses_custom_model_fast_path() -> None:
    # Query not in fast regex list to test custom neural classifier path
    intent = await classifier.classify("how hot is my rtx 3050 processor running")
    assert intent.type == IntentType.SYSTEM_STATS
    assert intent.confidence > 0.50


def test_chroma_embed_fn_integration() -> None:
    res = custom_chroma_embed_fn(["User preference: dark mode", "Meeting tomorrow at 3pm"])
    assert len(res) == 2
    assert len(res[0]) == 768


def test_lora_hardware_budget_guard() -> None:
    vram_info = check_gpu_vram_budget()
    assert "status" in vram_info
    assert "vram_total_gb" in vram_info


def test_lora_training_pipeline_dry_run(tmp_path: Path) -> None:
    dataset_file = tmp_path / "toolcall_dataset.json"
    generate_toolcall_dataset(dataset_file)
    output_dir = tmp_path / "lora_output"

    res = train_lora_adapter(
        dataset_path=dataset_file,
        output_dir=output_dir,
        model_name="unsupported_offline_mock",
    )
    assert res["status"] in ("OFFLINE_CONFIG_READY", "SUCCESS")
    assert Path(res["output_dir"]).exists()
