import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from .settings import read_jsonl

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

