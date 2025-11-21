import logging
import re
from typing import Dict

from .loader import SchemaLoader

logger = logging.getLogger(__name__)


class Normalizer:
    def __init__(self, schema: SchemaLoader):
        self.policy = schema.normalization_policy()

    @staticmethod
    def _slug(text: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip().lower())
        return cleaned.strip("_") or "unknown"

    @staticmethod
    def _coerce_ids(raw_ids):
        if isinstance(raw_ids, dict):
            return raw_ids
        if isinstance(raw_ids, list):
            converted = {}
            for item in raw_ids:
                if not isinstance(item, dict):
                    continue
                key = item.get("type") or item.get("namespace") or item.get("name")
                value = item.get("id") or item.get("value")
                if key and value:
                    converted[key] = value
            return converted
        if isinstance(raw_ids, str):
            return {"id": raw_ids}
        if raw_ids is None:
            return {}
        logger.warning("Unsupported ids payload %s", raw_ids)
        return {}

    def normalize(self, entity: Dict) -> Dict:
        cls = entity.get("class")
        policy = self.policy.get(cls, {})
        ids = self._coerce_ids(entity.get("ids"))
        chosen = None
        if policy:
            primary = policy.get("primary")
            alternates = policy.get("alternates", [])
            if primary and ids.get(primary):
                chosen = ids[primary]
            else:
                for alt in alternates:
                    if ids.get(alt):
                        chosen = ids[alt]
                        break
        if not chosen:
            chosen = f"{cls}:{self._slug(entity.get('text', 'unknown'))}"
        normalized = entity.copy()
        normalized["id"] = chosen
        return normalized

