from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

PIPELINE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = PIPELINE_DIR / "config.yaml"


@dataclass
class Settings:
    llm_model: str
    prompt_version: str = "v1"
    model_version: str = "v1"


_SETTINGS: Settings | None = None


def load_settings() -> Settings:
    global _SETTINGS
    if _SETTINGS is None:
        raw: Dict[str, Any] = {}
        if CONFIG_PATH.exists():
            with CONFIG_PATH.open("r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
        llm_cfg = raw.get("llm", {})
        _SETTINGS = Settings(
            llm_model=llm_cfg.get("model", "gpt-4o-mini"),
            prompt_version=str(raw.get("prompt_version", "v1")),
            model_version=str(raw.get("model_version", "v1")),
        )
    return _SETTINGS


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

