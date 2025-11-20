from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = OpenAI()

    def build_chat_completion_kwargs(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        json_mode: bool = False,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.config["llm"]["model"],
            "messages": [
                {"role": "system", "content": "You are a biomedical relation extraction assistant."},
                {"role": "user", "content": prompt},
            ],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def _request(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        json_mode: bool = False,
    ) -> Dict[str, Any]:
        kwargs = self.build_chat_completion_kwargs(
            prompt=prompt,
            temperature=temperature,
            json_mode=json_mode,
        )
        logger.debug(
            "LLM request model=%s json_mode=%s temperature=%s prompt=%s",
            self.config["llm"]["model"],
            json_mode,
            temperature,
            prompt,
        )
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            logger.error("LLM request failed: %s", exc, exc_info=True)
            return {"text": "", "json": None}
        message = response.choices[0].message
        content = message.content or ""
        logger.debug(
            "LLM response model=%s json_mode=%s text=%s",
            self.config["llm"]["model"],
            json_mode,
            content,
        )
        data = None
        if json_mode and content:
            try:
                data = json.loads(content)
            except json.JSONDecodeError as exc:
                logger.error("Failed to decode LLM JSON response: %s", exc)
                data = None
            else:
                logger.debug("LLM parsed JSON response: %s", data)
        return {"text": content, "json": data}

    def complete(self, prompt: str, temperature: Optional[float] = None) -> Dict[str, Any]:
        return self._request(prompt, temperature, json_mode=False)

    def json_complete(self, prompt: str) -> Dict[str, Any]:
        return self._request(prompt, temperature=None, json_mode=True)

