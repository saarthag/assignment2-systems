"""Introspect the benchmark CLI without importing it.

``cs336_systems/benchmark.py`` imports torch at module import time, which is
heavy (and unavailable on machines that only drive the benchmark remotely).
Instead we parse the ``parse_args`` function with :mod:`ast` and reconstruct the
argparse schema: flags, type, default, choices and help text.

This keeps the dashboard form automatically in sync with the CLI: add a
``parser.add_argument(...)`` on the remote side and it shows up in the UI.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

BENCHMARK_PATH = Path(__file__).resolve().parent.parent / "cs336_systems" / "benchmark.py"

_TYPE_NAMES = {"int": "int", "float": "float", "str": "str"}


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):  # pragma: no cover - defensive
        return None


def _flag_to_name(flags: list[str]) -> str:
    """Pick the most descriptive flag and turn it into a python identifier."""
    long_flags = [f for f in flags if f.startswith("--")]
    chosen = max(long_flags or flags, key=len)
    return chosen.lstrip("-").replace("-", "_")


def _parse_add_argument(call: ast.Call) -> dict[str, Any] | None:
    flags = [a.value for a in call.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
    if not flags:
        return None

    entry: dict[str, Any] = {
        "name": _flag_to_name(flags),
        "flags": flags,
        "flag": max([f for f in flags if f.startswith("--")] or flags, key=len),
        "type": "str",
        "default": None,
        "help": "",
        "choices": None,
        "required": False,
        "action": None,
    }

    for kw in call.keywords:
        key, value = kw.arg, kw.value
        if key == "type" and isinstance(value, ast.Name):
            entry["type"] = _TYPE_NAMES.get(value.id, "str")
        elif key == "default":
            entry["default"] = _literal(value)
        elif key == "help":
            entry["help"] = _literal(value) or ""
        elif key == "choices" and isinstance(value, (ast.List, ast.Tuple)):
            entry["choices"] = [el.value for el in value.elts if isinstance(el, ast.Constant)]
        elif key == "required":
            entry["required"] = bool(_literal(value))
        elif key == "action" and isinstance(value, ast.Constant):
            entry["action"] = value.value

    if entry["action"] == "store_true":
        entry["type"] = "bool"
        entry["default"] = bool(entry["default"])

    return entry


def extract_cli_schema(path: Path = BENCHMARK_PATH) -> list[dict[str, Any]]:
    """Return the list of argparse arguments defined in ``parse_args``."""
    if not path.exists():
        return []

    tree = ast.parse(path.read_text(), filename=str(path))

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "parse_args":
            entries: list[dict[str, Any]] = []
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and child.func.attr == "add_argument":
                    parsed = _parse_add_argument(child)
                    if parsed:
                        entries.append(parsed)
            return entries

    return []


if __name__ == "__main__":  # manual smoke test
    import json

    print(json.dumps(extract_cli_schema(), indent=2))
