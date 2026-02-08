from __future__ import annotations

from typing import Any, Dict, Optional
import os

from src.utils.optional import optional_import


class OpenAIRationaleGenerator:
    def __init__(self, model: str):
        self.model = model
        self._client = None
        self._available = False

        if not os.getenv("OPENAI_API_KEY"):
            return

        openai_mod, _ = optional_import("openai")
        if openai_mod is None:
            return

        try:
            from openai import OpenAI  # type: ignore

            self._client = OpenAI()
            self._available = True
        except Exception:
            self._client = None
            self._available = False

    def available(self) -> bool:
        return self._available and self._client is not None

    def summarize(self, facts: Dict[str, Any]) -> Optional[str]:
        if not self.available():
            return None

        prompt = (
            "You are assisting a claims investigator. "
            "Using ONLY the provided facts, write 2-3 concise sentences explaining "
            "why a claim may require review/decline and why the claim amount is high/typical. "
            "Do not add new facts. If evidence is insufficient, say so.\n\n"
            f"Facts:\n{facts}\n\n"
            "Return plain text only."
        )

        try:
            response = self._client.responses.create(
                model=self.model,
                input=prompt,
            )
            text = getattr(response, "output_text", None)
            if text:
                return str(text).strip()
        except Exception:
            return None
        return None
