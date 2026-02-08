from __future__ import annotations

from importlib import import_module
from typing import Any, Optional, Tuple


def optional_import(module_name: str) -> Tuple[Optional[Any], Optional[Exception]]:
    try:
        return import_module(module_name), None
    except Exception as exc:  # pragma: no cover - defensive
        return None, exc
