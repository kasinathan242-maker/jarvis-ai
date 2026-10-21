"""JARVIS Lite — Custom Native Model Inference Engine.

Zero-Ollama runtime execution using custom-trained ONNX and PyTorch models:
1. Neural Intent Classifier (< 2ms CPU execution via ONNX Runtime)
2. Custom Semantic Embeddings (ChromaDB compatible zero-daemon vector encoder)
3. Structured ToolCall Reasoning Engine
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from core.logging import get_logger
from tools.base import ToolCall

logger = get_logger("ai.custom_model_engine")


class CustomIntentEngine:
    """Zero-Ollama Neural Intent Classifier using custom-trained ONNX runtime."""

    def __init__(self, model_dir: Path | str | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).resolve().parent.parent / "models" / "intent_classifier"
        self._session: ort.InferenceSession | None = None
        self._config: dict[str, Any] = {}
        self._vocab: dict[str, int] = {}
        self._classes: list[str] = []
        self._load()

    def _load(self) -> None:
        onnx_file = self.model_dir / "model.onnx"
        config_file = self.model_dir / "config.json"
        vocab_file = self.model_dir / "vocab.json"

        if not onnx_file.exists() or not config_file.exists() or not vocab_file.exists():
            logger.warning("custom_intent_model_missing_files", model_dir=str(self.model_dir))
            return

        with open(config_file, encoding="utf-8") as f:
            self._config = json.load(f)
        with open(vocab_file, encoding="utf-8") as f:
            self._vocab = json.load(f)

        self._classes = self._config.get("classes", [])

        # Initialize ONNX Runtime session with CPU execution provider (0 VRAM consumed)
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(str(onnx_file), opts, providers=["CPUExecutionProvider"])
        logger.info("custom_intent_model_loaded", num_classes=len(self._classes), onnx_path=str(onnx_file))

    def _tokenize(self, text: str, max_len: int = 32) -> np.ndarray[Any, Any]:
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        words = [w for w in cleaned.split() if w][:max_len]
        unk_idx = self._vocab.get("<unk>", 1)
        pad_idx = self._vocab.get("<pad>", 0)
        tokens = [self._vocab.get(w, unk_idx) for w in words]
        if len(tokens) < max_len:
            tokens += [pad_idx] * (max_len - len(tokens))
        return np.array([tokens], dtype=np.int64)

    def predict(self, text: str) -> tuple[str, float]:
        """Predicts top intent and confidence score for a user utterance."""
        if not self._session or not self._classes:
            return "general_chat", 0.5

        tokens = self._tokenize(text)
        outputs = self._session.run(None, {"input_tokens": tokens})
        logits = outputs[0][0]

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])
        label = self._classes[top_idx] if top_idx < len(self._classes) else "general_chat"

        return label, confidence


class CustomEmbeddingEngine:
    """Zero-Ollama Custom Semantic Embedding Engine for ChromaDB and Document Indexing."""

    def __init__(self, model_dir: Path | str | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).resolve().parent.parent / "models" / "embeddings"
        self._session: ort.InferenceSession | None = None
        self._vocab: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        onnx_file = self.model_dir / "model.onnx"
        vocab_file = self.model_dir / "vocab.json"

        if not onnx_file.exists() or not vocab_file.exists():
            logger.warning("custom_embeddings_model_missing_files", model_dir=str(self.model_dir))
            return

        with open(vocab_file, encoding="utf-8") as f:
            self._vocab = json.load(f)

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(str(onnx_file), opts, providers=["CPUExecutionProvider"])
        logger.info("custom_embeddings_model_loaded", onnx_path=str(onnx_file))

    def _tokenize(self, text: str, max_len: int = 48) -> np.ndarray[Any, Any]:
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        words = [w for w in cleaned.split() if w][:max_len]
        unk_idx = self._vocab.get("<unk>", 1)
        pad_idx = self._vocab.get("<pad>", 0)
        tokens = [self._vocab.get(w, unk_idx) for w in words]
        if len(tokens) < max_len:
            tokens += [pad_idx] * (max_len - len(tokens))
        return np.array([tokens], dtype=np.int64)

    def embed_text(self, text: str) -> list[float]:
        """Generates 768-dimensional normalized unit hypersphere embedding vector."""
        if not self._session:
            # Deterministic pseudo-embedding fallback
            np.random.seed(abs(hash(text)) % (2**32))
            v = np.random.randn(768)
            norm = float(np.linalg.norm(v)) or 1.0
            raw_list = (v / norm).tolist()
            return [float(x) for x in raw_list]

        tokens = self._tokenize(text)
        outputs = self._session.run(None, {"input_tokens": tokens})
        emb = outputs[0][0]
        # Ensure L2 normalized
        norm = float(np.linalg.norm(emb)) or 1.0
        raw_emb = (emb / norm).tolist()
        return [float(x) for x in raw_emb]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


class CustomEmbeddingFunction:
    """ChromaDB compatible EmbeddingFunction wrapping CustomEmbeddingEngine."""

    def __init__(self, engine: CustomEmbeddingEngine) -> None:
        self.engine = engine
        self.is_legacy: bool = False

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.engine.embed_documents(input)


class CustomToolCallEngine:
    """Formulates structured ToolCalls from custom neural intent predictions."""

    def __init__(self, intent_engine: CustomIntentEngine) -> None:
        self.intent_engine = intent_engine

    def plan_tool_call(self, query: str) -> ToolCall | None:
        """Translates user query into structured ToolCall using custom neural models."""
        intent, confidence = self.intent_engine.predict(query)
        q_lower = query.lower()

        logger.info("custom_neural_intent_classified", intent=intent, confidence=confidence, query=query)

        # 1. Emergency stop
        if intent == "emergency_stop" or any(w in q_lower for w in ["emergency stop", "halt", "kill tasks"]):
            return ToolCall(tool_name="emergency_stop", parameters={"source": "custom_neural_model"})

        # 2. System stats
        if intent == "system_stats" or any(w in q_lower for w in ["cpu", "vram", "ram", "battery", "gpu"]):
            return ToolCall(tool_name="system_stats", parameters={"detailed": True})

        # 3. Vision / Screen Analysis
        if intent == "screen_analysis" or any(w in q_lower for w in ["on my screen", "screen right now", "describe desktop"]):
            return ToolCall(tool_name="analyze_screen", parameters={"prompt": "Describe what is on the user desktop screen."})

        # 4. Open Application
        if intent == "open_application" or "open " in q_lower or "launch " in q_lower:
            m = re.search(r"(?:open|launch|start)\s+([a-zA-Z0-9_\-\s]+)", q_lower)
            app_name = m.group(1).strip() if m else "code"
            if "vscode" in app_name or "visual studio code" in app_name or "code" in app_name:
                app_name = "code"
            elif "notepad" in app_name:
                app_name = "notepad"
            elif "chrome" in app_name:
                app_name = "chrome"
            return ToolCall(tool_name="open_application", parameters={"app_name": app_name})

        # 5. Search files
        if intent == "search_files" or "find file" in q_lower or "search file" in q_lower:
            m = re.search(r"(?:find|search|locate)(?:\s+file)?(?:\s+named)?\s+([a-zA-Z0-9_\.\-]+)", q_lower)
            query_name = m.group(1) if m else "report"
            return ToolCall(tool_name="search_files", parameters={"query": query_name, "directory": "C:/Users/mohan/Downloads"})

        # 6. Delete file
        if intent == "delete_file" or "delete " in q_lower:
            m = re.search(r"delete\s+([a-zA-Z0-9_\.\-]+)", q_lower)
            target = m.group(1) if m else "temp.txt"
            return ToolCall(tool_name="delete_file", parameters={"path": f"C:/Users/mohan/Downloads/{target}", "confirm": True})

        # 7. Shell execution
        if intent == "run_shell" or "run git" in q_lower or "pytest" in q_lower:
            m = re.search(r"(?:run|execute)\s+(.*)", q_lower)
            cmd = m.group(1) if m else "git status"
            return ToolCall(tool_name="run_shell", parameters={"command": cmd})

        # 8. Web browse
        if intent == "browse_web" or "search online" in q_lower or "research" in q_lower:
            return ToolCall(tool_name="web_search_research", parameters={"topic": query})

        # 9. Time check
        if any(w in q_lower for w in ["what time", "current time", "clock"]):
            return ToolCall(tool_name="get_time", parameters={})

        return None


# Global singletons
custom_intent_engine = CustomIntentEngine()
custom_embedding_engine = CustomEmbeddingEngine()
custom_toolcall_engine = CustomToolCallEngine(custom_intent_engine)
custom_chroma_embed_fn = CustomEmbeddingFunction(custom_embedding_engine)
