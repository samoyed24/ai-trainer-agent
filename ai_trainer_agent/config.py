"""配置管理模块。

该模块提供两类能力：
1. 向外暴露与历史兼容的常量：BACKEND_CONFIG / SERVER_CONFIG / AGENT_CONFIG / ...
2. 提供持久化读写接口，支持通过 CLI 修改用户配置。
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import requests


def _default_device() -> str:
    """推断默认训练设备。"""
    try:
        torch = __import__("torch")
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _default_user_config_path() -> Path:
    """返回用户配置文件路径。"""
    env_path = os.getenv("AI_TRAINER_CONFIG_PATH")
    if env_path:
        return Path(env_path).expanduser()

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "ai-trainer-agent" / "config.json"

    if os.name == "nt":
        base = Path(os.getenv("APPDATA", str(Path.home())))
        return base / "ai-trainer-agent" / "config.json"

    return Path.home() / ".config" / "ai-trainer-agent" / "config.json"


CONFIG_PATH = _default_user_config_path()

DEFAULT_CONFIG: Dict[str, Any] = {
    "backend": {
        "api": "",
        "app_id": "",
        "app_secret": "",
        "timeout": 10,
    },
    "server": {
        "url": "https://webhook.site/1b61dd98-e63b-45d7-9087-49b4bcd70ca6",
        "timeout": 10,
        "retry_count": 3,
    },
    "agent": {
        "upload_frequency": "epoch",
        "upload_interval": 1,
        "buffer_size": 100,
        "enable_async": True,
    },
    "metrics": {
        "monitor": {
            "loss": True,
            "accuracy": True,
            "learning_rate": True,
            "batch_time": True,
            "epoch_time": True,
            "model_size": True,
        },
        "include_system_info": True,
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
    print(f"[ai-trainer-agent] {message}", file=sys.stderr)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并字典，override 优先。"""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """返回合并后的新配置，不修改输入。"""
    merged = copy.deepcopy(base)
    if override:
        _deep_merge(merged, override)
    return merged


def load_user_override(path: Optional[Path] = None) -> Dict[str, Any]:
    """加载用户配置覆盖项。"""
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


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """加载生效配置（默认配置 + 用户覆盖）。"""
    config = copy.deepcopy(DEFAULT_CONFIG)
    override = load_user_override(path)
    return merge_config(config, override)


def _is_remote_config_like(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False

    known_top_level_keys = {
        "backend",
        "server",
        "agent",
        "metrics",
        "model",
        "training",
        "server_url",
    }
    return any(key in payload for key in known_top_level_keys)


def _extract_remote_config_payload(response_data: Any) -> Dict[str, Any]:
    """从后端响应中提取配置对象，兼容常见包装格式。"""
    if not isinstance(response_data, dict):
        raise RuntimeError("后端配置响应格式非法（应为 JSON 对象）")

    candidates = []
    for key in ("config", "data", "result"):
        value = response_data.get(key)
        if isinstance(value, dict):
            # 支持 {"data": {"config": {...}}}
            candidates.append(value.get("config"))
            candidates.append(value)

    candidates.append(response_data.get("config"))
    candidates.append(response_data)

    for candidate in candidates:
        if _is_remote_config_like(candidate):
            return candidate

    raise RuntimeError("后端响应中未找到可用配置")


def _normalize_remote_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """标准化后端配置键，转换为本地统一结构。"""
    normalized = copy.deepcopy(config)

    # 兼容后端只返回 server_url 的简化格式
    server_url = normalized.pop("server_url", None)
    if server_url:
        server = normalized.get("server")
        if not isinstance(server, dict):
            server = {}
            normalized["server"] = server
        server.setdefault("url", server_url)

    return normalized


def fetch_remote_config(
    backend_api: str,
    app_id: str,
    app_secret: str,
    timeout: int = 10,
) -> Dict[str, Any]:
    """通过后端 API 拉取远程配置。"""
    if not backend_api or not app_id or not app_secret:
        raise ValueError("backend_api、app_id、app_secret 不能为空")

    headers = {
        "X-App-Id": app_id,
        "X-App-Secret": app_secret,
    }
    payload = {
        "app_id": app_id,
        "app_secret": app_secret,
    }

    try:
        response = requests.post(
            backend_api,
            json=payload,
            headers=headers,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"请求后端配置失败: {exc}") from exc

    if response.status_code not in (200, 201):
        raise RuntimeError(f"请求后端配置失败: status={response.status_code}")

    try:
        response_data = response.json()
    except ValueError as exc:
        raise RuntimeError("后端配置响应不是合法 JSON") from exc

    raw_config = _extract_remote_config_payload(response_data)
    return _normalize_remote_config(raw_config)


def save_config(config: Dict[str, Any], path: Optional[Path] = None) -> Path:
    """保存完整配置。"""
    target = Path(path) if path else CONFIG_PATH
    target = target.expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def get_config_value(key_path: Optional[str] = None) -> Any:
    """按点路径读取配置值，如 server.url。"""
    config = load_config()
    if not key_path:
        return config

    current: Any = config
    for part in key_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"配置键不存在: {key_path}")
        current = current[part]
    return current


def set_config_value(key_path: str, value: Any) -> Dict[str, Any]:
    """按点路径设置配置值，并持久化。"""
    if not key_path:
        raise ValueError("key_path 不能为空")

    parts = key_path.split(".")
    root_key = parts[0]
    if root_key not in DEFAULT_CONFIG:
        raise KeyError(f"顶层配置键不存在: {root_key}")

    config = load_config()
    current: Dict[str, Any] = config
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


def unset_config_value(key_path: str) -> Dict[str, Any]:
    """按点路径删除配置值，并持久化。"""
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


def reset_config() -> Dict[str, Any]:
    """重置为默认配置并持久化。"""
    config = copy.deepcopy(DEFAULT_CONFIG)
    save_config(config)
    _refresh_exported_constants(config)
    return config


def init_user_config(force: bool = False) -> Path:
    """初始化用户配置文件。"""
    if CONFIG_PATH.exists() and not force:
        return CONFIG_PATH
    return save_config(load_config())


def _refresh_exported_constants(config: Dict[str, Any]) -> None:
    """刷新兼容常量导出。"""
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


# 兼容旧代码：import 后可直接使用这些常量。
_refresh_exported_constants(load_config())
