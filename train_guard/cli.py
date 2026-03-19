"""命令行配置管理工具。"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .config import (
    CONFIG_PATH,
    get_config_value,
    init_user_config,
    load_config,
    reset_config,
    set_config_value,
    unset_config_value,
)


def _format_output(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    return str(value)


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    truthy = {"1", "true", "yes", "on", "y"}
    falsy = {"0", "false", "no", "off", "n"}

    if normalized in truthy:
        return True
    if normalized in falsy:
        return False
    raise ValueError(f"无法解析为布尔值: {value}")


def _parse_value(raw: str, value_type: str) -> Any:
    if value_type == "str":
        return raw
    if value_type == "int":
        return int(raw)
    if value_type == "float":
        return float(raw)
    if value_type == "bool":
        return _parse_bool(raw)
    if value_type == "json":
        return json.loads(raw)

    # auto: 尝试 JSON，失败则按字符串处理
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train-guard-config",
        description="查看和修改 train-guard 的用户配置",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("path", help="显示用户配置文件路径")

    show_parser = subparsers.add_parser("show", help="显示完整配置或某个键")
    show_parser.add_argument("--key", help="点路径键，例如 server.url")

    get_parser = subparsers.add_parser("get", help="读取某个配置键")
    get_parser.add_argument("key", help="点路径键，例如 training.batch_size")

    set_parser = subparsers.add_parser("set", help="设置某个配置键")
    set_parser.add_argument("key", help="点路径键，例如 server.url")
    set_parser.add_argument("value", help="要写入的值")
    set_parser.add_argument(
        "--type",
        default="auto",
        choices=["auto", "str", "int", "float", "bool", "json"],
        help="值类型，默认 auto",
    )

    unset_parser = subparsers.add_parser("unset", help="删除某个配置键")
    unset_parser.add_argument("key", help="点路径键，例如 metrics.monitor.loss")

    reset_parser = subparsers.add_parser("reset", help="重置为默认配置")
    reset_parser.add_argument(
        "--yes",
        action="store_true",
        help="确认重置（无该参数将不会执行）",
    )

    init_parser = subparsers.add_parser("init", help="初始化用户配置文件")
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="覆盖已存在的用户配置文件",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "path":
            print(str(CONFIG_PATH))
            return 0

        if args.command == "show":
            if args.key:
                print(_format_output(get_config_value(args.key)))
            else:
                print(_format_output(load_config()))
            return 0

        if args.command == "get":
            print(_format_output(get_config_value(args.key)))
            return 0

        if args.command == "set":
            parsed_value = _parse_value(args.value, args.type)
            updated = set_config_value(args.key, parsed_value)
            print(_format_output(updated))
            return 0

        if args.command == "unset":
            updated = unset_config_value(args.key)
            print(_format_output(updated))
            return 0

        if args.command == "reset":
            if not args.yes:
                print("未执行重置。请使用 --yes 确认。", file=sys.stderr)
                return 2
            updated = reset_config()
            print(_format_output(updated))
            return 0

        if args.command == "init":
            path = init_user_config(force=args.force)
            print(str(path))
            return 0

        parser.print_help()
        return 0

    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
