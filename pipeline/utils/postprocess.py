import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .settings import ensure_dir, timestamp


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

