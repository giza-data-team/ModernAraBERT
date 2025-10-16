import argparse
from typing import Callable, Tuple, Dict, Any

import yaml


def _build_minimal_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--print-default-config",
        action="store_true",
        help="Print a YAML template of default arguments and exit",
    )
    return parser


def _collect_defaults(parser: argparse.ArgumentParser) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    for action in parser._actions:
        if not action.dest or action.dest == "help":
            continue
        defaults[action.dest] = action.default
    return defaults


def _allowed_keys(parser: argparse.ArgumentParser) -> set[str]:
    keys: set[str] = set()
    for action in parser._actions:
        if action.dest and action.dest != "help":
            keys.add(action.dest)
    return keys


def _dump_yaml(data: Dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=True, allow_unicode=True)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit("Config file must contain a mapping of keys to values")
    return data


def parse_args_with_optional_config(
    build_parser: Callable[[], argparse.ArgumentParser],
) -> Tuple[argparse.Namespace, bool]:
    """Parse CLI arguments with optional mutually exclusive YAML config.

    Returns (args, from_config) where from_config indicates YAML was used.
    Enforces: when --config is supplied, no other user arguments are allowed.
    Supports: --print-default-config to emit defaults as YAML and exit(0).
    """
    minimal = _build_minimal_parser()
    minimal_args, remaining = minimal.parse_known_args()

    full_parser = build_parser()

    if minimal_args.print_default_config:
        defaults = _collect_defaults(full_parser)
        print(_dump_yaml(defaults))
        raise SystemExit(0)

    if minimal_args.config:
        if remaining:
            raise SystemExit(
                "Cannot combine --config with other arguments. Provide only --config."
            )

        yaml_data = _load_yaml(minimal_args.config)
        allowed = _allowed_keys(full_parser)
        unknown_keys = [k for k in yaml_data.keys() if k not in allowed]
        if unknown_keys:
            raise SystemExit(
                f"Unknown keys in config: {unknown_keys}. Allowed keys: {sorted(allowed)}"
            )

        defaults = _collect_defaults(full_parser)
        merged: Dict[str, Any] = {**defaults, **yaml_data}
        return argparse.Namespace(**merged), True

    return full_parser.parse_args(), False


