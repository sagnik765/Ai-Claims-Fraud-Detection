from __future__ import annotations

import re
from typing import Iterable


_whitespace = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.strip().lower()
    text = _whitespace.sub(" ", text)
    return text


def join_fields(values: Iterable[str]) -> str:
    return " ".join([normalize_text(v) for v in values if v])
