from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

PIPELINE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = PIPELINE_DIR / "config.yaml"

_CONFIG: Dict[str, Any] | None = None


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml and return as a dictionary."""
    global _CONFIG
    if _CONFIG is None:
        if CONFIG_PATH.exists():
            with CONFIG_PATH.open("r", encoding="utf-8") as fh:
                _CONFIG = yaml.safe_load(fh) or {}
        else:
            _CONFIG = {}
        # Resolve relative paths
        if "data" in _CONFIG:
            data = _CONFIG["data"]
            for key in ["input_file", "output_file"]:
                path = Path(data[key])
                if not path.is_absolute():
                    data[key] = str(PIPELINE_DIR / path)
        if "logging" in _CONFIG:
            logging = _CONFIG["logging"]
            for key in ["log_file", "relation_log_file"]:
                path = Path(logging[key])
                if not path.is_absolute():
                    logging[key] = str(PIPELINE_DIR / path)
    return _CONFIG


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("//") or line.startswith("#"):
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def log_result(result: Dict, log_path: Path) -> None:
    ensure_dir(log_path)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(result) + "\n")


class PostProcessor:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def filter(self, results: Iterable[Dict]) -> List[Dict]:
        return [res for res in results if res.get("confidence", 0.0) >= self.threshold]

    def aggregate(self, results: Iterable[Dict]) -> List[Dict]:
        grouped: Dict[Tuple[str, str, str], Dict] = {}
        for res in results:
            subj = res["subject"]["id"]
            obj = res["object"]["id"]
            predicate = res["predicate"]
            key = (subj, predicate, obj)
            entry = grouped.setdefault(
                key,
                {
                    "subject": res["subject"],
                    "object": res["object"],
                    "predicate": predicate,
                    "confidence": res["confidence"],
                    "pmids": set(),
                    "sentences": [],
                    "model_name": res.get("model_name"),
                    "model_version": res.get("model_version"),
                    "prompt_version": res.get("prompt_version"),
                    "timestamp": timestamp(),
                },
            )
            entry["confidence"] = max(entry["confidence"], res["confidence"])
            entry["pmids"].add(res["pmid"])
            entry["sentences"].append(
                {
                    "pmid": res["pmid"],
                    "sentence_id": res["sentence_id"],
                    "sentence": res["sentence"],
                    "explanation": res.get("explanation", ""),
                }
            )
        aggregated = []
        for entry in grouped.values():
            entry["pmids"] = sorted(entry["pmids"])
            aggregated.append(entry)
        return aggregated

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Sentence:
    pmid: str
    sentence_id: int
    text: str


def split_text(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    return [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]


def load_sentences(jsonl_path: Path) -> Iterable[Sentence]:
    for record in read_jsonl(jsonl_path):
        pmid = str(record.get("pmid") or "")
        abstract = record.get("abstract") or ""
        for idx, sentence in enumerate(split_text(abstract)):
            yield Sentence(pmid=pmid, sentence_id=idx, text=sentence)