"""Minimal YAML config loading + CLI overrides.

We keep a soft dependency on PyYAML: if it is missing we fall back to JSON so
the scripts still run.  Config files are plain dicts with ``data`` / ``model`` /
``train`` sections (see ``configs/round1.yaml``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore

            return yaml.safe_load(text) or {}
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "PyYAML is required to read .yaml configs "
                "(pip install pyyaml), or supply a .json config.") from exc
    import json

    return json.loads(text)


def merge_overrides(config: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply ``section.key=value`` CLI overrides onto a nested config dict."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got: {item}")
        key, raw = item.split("=", 1)
        node = config
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(raw)
    return config


def _coerce(value: str) -> Any:
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value
