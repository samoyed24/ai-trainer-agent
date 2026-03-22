"""配置管理与 dashboard 协议适配。"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests


def _default_device() -> str:
    try:
        torch = __import__("torch")
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _platform_user_config_path(app_name: str) -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / app_name / "config.json"

    if os.name == "nt":
        base = Path(os.getenv("APPDATA", str(Path.home())))
        return base / app_name / "config.json"

    return Path.home() / ".config" / app_name / "config.json"


def _default_user_config_path() -> Path:
    env_path = os.getenv("TRAIN_GUARD_CONFIG_PATH")
    if env_path:
        return Path(env_path).expanduser()
    return _platform_user_config_path("train-guard")


CONFIG_PATH = _default_user_config_path()

DEFAULT_CONFIG: dict[str, Any] = {
    "backend": {
        "base_url": "",
        "access_key_id": "",
        "secret_key": "",
        "project_id": "",
        "timeout": 10,
    },
    "server": {
        "url": "",
        "timeout": 10,
        "retry_count": 3,
    },
    "agent": {
        "upload_frequency": "epoch",
        "upload_interval": 1,
        "enable_async": True,
    },
    "metrics": {
        "monitor": {
            "loss": True,
            "accuracy": True,
            "learning_rate": True,
            "batch_time": True,
            "epoch_time": True,
        },
        "include_system_info": True,
        "system_info_prefix": "system",
    },
    "model": {
        "model_name": "ResNet18",
        "num_classes": 10,
        "pretrained": False,
    },
    "training": {
        "num_epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "device": _default_device(),
    },
}


def _warn(message: str) -> None:
    print(f"[train-guard] {message}", file=sys.stderr)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    if override:
        _deep_merge(merged, override)
    return merged


def load_user_override(path: Path | None = None) -> dict[str, Any]:
    target = Path(path) if path else CONFIG_PATH
    if not target.exists():
        return {}

    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _warn(f"用户配置 JSON 解析失败，已忽略: {target} ({exc})")
        return {}
    except OSError as exc:
        _warn(f"读取用户配置失败，已忽略: {target} ({exc})")
        return {}

    if not isinstance(data, dict):
        _warn(f"用户配置格式非法（应为对象），已忽略: {target}")
        return {}

    return data


def load_config(path: Path | None = None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    override = load_user_override(path)
    return merge_config(config, override)


def _normalize_backend_base_url(base_url: str) -> str:
    cleaned = (base_url or "").strip().rstrip("/")
    if not cleaned:
        return ""
    if cleaned.endswith("/api/agent/config"):
        return cleaned[: -len("/api/agent/config")]
    if cleaned.endswith("/api"):
        return cleaned[: -len("/api")]
    return cleaned


def _build_agent_config_url(base_url: str) -> str:
    normalized = _normalize_backend_base_url(base_url)
    if not normalized:
        raise ValueError("backend.base_url 不能为空")
    return f"{normalized}/api/agent/config"


def _build_agent_headers(access_key_id: str, secret_key: str, project_id: str) -> dict[str, str]:
    return {
        "X-Access-Key-Id": access_key_id,
        "X-Secret-Key": secret_key,
        "X-Project-Id": project_id,
    }


def _extract_remote_config_payload(response_data: Any) -> dict[str, Any]:
    if not isinstance(response_data, dict):
        raise RuntimeError("后端配置响应格式非法（应为 JSON 对象）")

    candidates: list[Any] = []
    data = response_data.get("data")
    if isinstance(data, dict):
        candidates.append(data.get("config"))
        candidates.append(data)
    candidates.append(response_data.get("config"))
    candidates.append(response_data)

    for candidate in candidates:
        if isinstance(candidate, dict):
            return candidate

    raise RuntimeError("后端响应中未找到可用配置")


def _normalize_remote_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    server = normalized.get("server")
    if not isinstance(server, dict):
        normalized["server"] = {}
    return normalized


def fetch_remote_config(
    base_url: str,
    access_key_id: str,
    secret_key: str,
    project_id: str,
    timeout: int = 10,
) -> dict[str, Any]:
    if not all([base_url, access_key_id, secret_key, project_id]):
        raise ValueError("base_url、access_key_id、secret_key、project_id 不能为空")

    config_url = _build_agent_config_url(base_url)
    headers = _build_agent_headers(access_key_id, secret_key, project_id)

    try:
        response = requests.get(config_url, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"请求后端配置失败: {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(f"请求后端配置失败: status={response.status_code}")

    try:
        response_data = response.json()
    except ValueError as exc:
        raise RuntimeError("后端配置响应不是合法 JSON") from exc

    raw_config = _extract_remote_config_payload(response_data)
    return _normalize_remote_config(raw_config)


def save_config(config: dict[str, Any], path: Path | None = None) -> Path:
    target = Path(path) if path else CONFIG_PATH
    target = target.expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def get_config_value(key_path: str | None = None) -> Any:
    config = load_config()
    if not key_path:
        return config

    current: Any = config
    for part in key_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"配置键不存在: {key_path}")
        current = current[part]
    return current


def set_config_value(key_path: str, value: Any) -> dict[str, Any]:
    if not key_path:
        raise ValueError("key_path 不能为空")

    parts = key_path.split(".")
    root_key = parts[0]
    if root_key not in DEFAULT_CONFIG:
        raise KeyError(f"顶层配置键不存在: {root_key}")

    config = load_config()
    current: dict[str, Any] = config
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value

    current[parts[-1]] = value
    save_config(config)
    _refresh_exported_constants(config)
    return config


def unset_config_value(key_path: str) -> dict[str, Any]:
    if not key_path:
        raise ValueError("key_path 不能为空")

    parts = key_path.split(".")
    config = load_config()
    current: Any = config

    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"配置键不存在: {key_path}")
        current = current[part]

    leaf = parts[-1]
    if not isinstance(current, dict) or leaf not in current:
        raise KeyError(f"配置键不存在: {key_path}")

    del current[leaf]
    save_config(config)
    _refresh_exported_constants(config)
    return config


def reset_config() -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    save_config(config)
    _refresh_exported_constants(config)
    return config


def init_user_config(force: bool = False) -> Path:
    if CONFIG_PATH.exists() and not force:
        return CONFIG_PATH
    return save_config(load_config())


def _refresh_exported_constants(config: dict[str, Any]) -> None:
    global BACKEND_CONFIG
    global SERVER_CONFIG
    global AGENT_CONFIG
    global METRICS_CONFIG
    global MODEL_CONFIG
    global TRAINING_CONFIG

    BACKEND_CONFIG = config["backend"]
    SERVER_CONFIG = config["server"]
    AGENT_CONFIG = config["agent"]
    METRICS_CONFIG = config["metrics"]
    MODEL_CONFIG = config["model"]
    TRAINING_CONFIG = config["training"]


_refresh_exported_constants(load_config())
