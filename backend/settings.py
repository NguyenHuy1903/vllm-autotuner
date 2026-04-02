"""
Application settings cho vLLM Auto-Tuner.
Đọc cấu hình từ file YAML ở root project và cho phép override bằng
environment variables.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / "config.yaml"
ENV_FILE = PROJECT_ROOT / ".env"


def _load_env_file() -> None:
    """Load biến môi trường từ file .env nếu có (format KEY=VALUE)."""
    if not ENV_FILE.exists():
        return

    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Không override biến đã được export từ shell/CI.
        os.environ.setdefault(key, value)


def _load_yaml_config() -> dict[str, Any]:
    """Load YAML config từ config.yaml nếu có."""
    if not CONFIG_FILE.exists():
        return {}

    data = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return data


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_env_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None:
        return default
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return items or default


def _cfg_get(cfg: dict[str, Any], path: str, default: Any) -> Any:
    """Lấy giá trị nested theo path dạng section.key.subkey từ dict config."""
    cur: Any = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _get_str(cfg: dict[str, Any], env_name: str, yaml_path: str, default: str) -> str:
    if env_name in os.environ:
        return os.environ[env_name]
    value = _cfg_get(cfg, yaml_path, default)
    return str(value)


def _get_int(cfg: dict[str, Any], env_name: str, yaml_path: str, default: int) -> int:
    if env_name in os.environ:
        return _get_env_int(env_name, default)
    value = _cfg_get(cfg, yaml_path, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_list(cfg: dict[str, Any], env_name: str, yaml_path: str, default: list[str]) -> list[str]:
    if env_name in os.environ:
        return _get_env_list(env_name, default)

    value = _cfg_get(cfg, yaml_path, default)
    if isinstance(value, list):
        items = [str(v).strip() for v in value if str(v).strip()]
        return items or default
    if isinstance(value, str):
        items = [x.strip() for x in value.split(",") if x.strip()]
        return items or default
    return default


_load_env_file()
_raw_cfg = _load_yaml_config()


@dataclass(frozen=True)
class Settings:
    models_dir: str = _get_str(
        _raw_cfg,
        "AUTOTUNER_MODELS_DIR",
        "paths.models_dir",
        "/projects/MedTrivita/common/models",
    )
    container_models_dir: str = _get_str(
        _raw_cfg,
        "AUTOTUNER_CONTAINER_MODELS_DIR",
        "paths.container_models_dir",
        "/models",
    )
    default_docker_image: str = _get_str(
        _raw_cfg,
        "AUTOTUNER_DEFAULT_DOCKER_IMAGE",
        "docker.default_image",
        "vllm/vllm-openai:v0.18.1",
    )

    port_start: int = _get_int(_raw_cfg, "AUTOTUNER_PORT_START", "docker.port_start", 9200)
    port_end: int = _get_int(_raw_cfg, "AUTOTUNER_PORT_END", "docker.port_end", 9300)
    container_prefix: str = _get_str(_raw_cfg, "AUTOTUNER_CONTAINER_PREFIX", "docker.container_prefix", "autotuner-")

    db_path: str = _get_str(
        _raw_cfg,
        "AUTOTUNER_DB_PATH",
        "paths.db_path",
        str(PROJECT_ROOT / "data" / "results.db"),
    )
    models_cache_ttl_s: int = _get_int(
        _raw_cfg,
        "AUTOTUNER_MODELS_CACHE_TTL_S",
        "runtime.models_cache_ttl_s",
        60,
    )

    api_host: str = _get_str(_raw_cfg, "AUTOTUNER_API_HOST", "api.host", "0.0.0.0")
    api_port: int = _get_int(_raw_cfg, "AUTOTUNER_API_PORT", "api.port", 9100)
    cors_allow_origins: list[str] = None  # type: ignore[assignment]

    frontend_build_dir: str = _get_str(
        _raw_cfg,
        "AUTOTUNER_FRONTEND_BUILD_DIR",
        "paths.frontend_build_dir",
        str(PROJECT_ROOT / "frontend" / "build"),
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cors_allow_origins",
            _get_list(_raw_cfg, "AUTOTUNER_CORS_ALLOW_ORIGINS", "api.cors_allow_origins", ["*"]),
        )


settings = Settings()
