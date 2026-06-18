from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class AgentResult:
    name: str
    outputs: Dict[str, Any]


class BaseAgent:
    name: str = "base"

    def run(self, payload: Dict[str, Any]) -> AgentResult:
        raise NotImplementedError
