from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class SystemConfig(BaseModel):
    name: str = "JARVIS Lite"
    version: str = "1.0.0"
    env: str = "development"
    log_level: str = "INFO"
    logs_dir: str = "logs"
    data_dir: str = "data"

class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8001
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(default_factory=lambda: [
        "http://localhost:8001",
        "http://127.0.0.1:8001",
        "http://localhost:1420",
        "tauri://localhost"
    ])

class ModelClientConfig(BaseModel):
    base_url: str
    connect_timeout: float = 5.0
    read_timeout: float = 120.0
    max_retries: int = 3
    circuit_breaker_failure_threshold: int = 3
    circuit_breaker_recovery_time: float = 30.0

class CustomModelsConfig(BaseModel):
    enabled: bool = True
    models_dir: str = "models"
    intent_model_path: str = "models/intent_classifier/model.onnx"
    embeddings_model_path: str = "models/embeddings/model.onnx"
    lora_adapter_path: str = "models/lora_adapters/jarvis_toolcall"

class ModelsConfig(BaseModel):
    custom: CustomModelsConfig = Field(default_factory=CustomModelsConfig)
    ollama: ModelClientConfig = Field(
        default_factory=lambda: ModelClientConfig(base_url="http://127.0.0.1:11434")
    )
    lmstudio: ModelClientConfig = Field(
        default_factory=lambda: ModelClientConfig(base_url="http://127.0.0.1:1234/v1")
    )
    routing: dict[str, str] = Field(
        default_factory=lambda: {
            "chat": "qwen2.5:7b-instruct-q4_K_M",
            "route": "qwen2.5:1.5b-instruct-q4_K_M",
            "plan": "qwen2.5:7b-instruct-q4_K_M",
            "code": "qwen2.5:7b-instruct-q4_K_M",
            "summarize": "qwen2.5:7b-instruct-q4_K_M",
            "embed": "nomic-embed-text"
        }
    )

class STTConfig(BaseModel):
    model: str = "base.en"
    device: str = "cpu"
    compute_type: str = "int8"

class TTSConfig(BaseModel):
    engine: str = "edge-tts"
    voice: str = "en-US-ChristopherNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"

class VoiceConfig(BaseModel):
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)

class DatabaseConfig(BaseModel):
    sqlite_url: str = "sqlite:///data/jarvis.db"
    chroma_persist_dir: str = "data/chroma"

class SecurityConfig(BaseModel):
    emergency_hotkey: str = "Ctrl+Alt+Shift+X"
    emergency_phrase: str = "jarvis stop everything"
    allow_shell_commands: list[str] = Field(
        default_factory=lambda: [
            "git", "python", "pip", "npm", "node", "pytest", "dir", "where", "code"
        ]
    )
    block_shell_commands: list[str] = Field(
        default_factory=lambda: [
            "format", "diskpart", "reg delete", "bcdedit", "del /s", "rm -rf"
        ]
    )

class AppSettings(BaseSettings):
    system: SystemConfig = Field(default_factory=SystemConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    # Environment variable overrides
    OLLAMA_BASE_URL: str | None = None
    API_PORT: int | None = None
    LOG_LEVEL: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

def load_settings() -> AppSettings:
    yaml_path = PROJECT_ROOT / "config" / "settings.yaml"
    yaml_data: dict[str, Any] = {}

    if yaml_path.exists():
        try:
            with open(yaml_path, encoding="utf-8") as f:
                content = yaml.safe_load(f)
                if isinstance(content, dict):
                    yaml_data = content
        except Exception as e:
            print(f"[CONFIG ERROR] Failed to parse {yaml_path}: {e}")
            raise SystemExit(1) from e

    try:
        settings_obj = AppSettings(**yaml_data)
        # Apply direct environment variable overrides if present
        if settings_obj.OLLAMA_BASE_URL:
            settings_obj.models.ollama.base_url = settings_obj.OLLAMA_BASE_URL
        if settings_obj.API_PORT:
            settings_obj.server.port = settings_obj.API_PORT
        if settings_obj.LOG_LEVEL:
            settings_obj.system.log_level = settings_obj.LOG_LEVEL
        return settings_obj
    except ValidationError as err:
        print("\n[CRITICAL CONFIGURATION ERROR] The system configuration is invalid:")
        for error in err.errors():
            loc = " -> ".join(str(part) for part in error.get("loc", []))
            print(f"  • Field [{loc}]: {error.get('msg')} (input: {error.get('input')})")
        raise SystemExit(1) from err

settings = load_settings()
