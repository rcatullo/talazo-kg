from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from openai import OpenAI

from pipeline.utils.settings import Settings, load_settings

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or load_settings()
        self.client = OpenAI()

    def _request(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        json_mode: bool = False,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": "You are a biomedical relation extraction assistant."},
                {"role": "user", "content": prompt},
            ],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            logger.error("LLM request failed: %s", exc, exc_info=True)
            return {"text": "", "json": None}
        message = response.choices[0].message
        content = message.content or ""
        data = None
        if json_mode and content:
            try:
                data = json.loads(content)
            except json.JSONDecodeError as exc:
                logger.error("Failed to decode LLM JSON response: %s", exc)
                data = None
        return {"text": content, "json": data}

    def complete(self, prompt: str, temperature: Optional[float] = None) -> Dict[str, Any]:
        return self._request(prompt, temperature, json_mode=False)

    def json_complete(self, prompt: str) -> Dict[str, Any]:
        return self._request(prompt, temperature=None, json_mode=True)

