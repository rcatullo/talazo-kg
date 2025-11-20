from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pipeline.model.llm_client import LLMClient
from pipeline.utils.api_req_parallel import process_api_requests_from_file
from pipeline.utils import ensure_dir, timestamp, CandidatePair

PIPELINE_DIR = Path(__file__).resolve().parent.parent
RELATION_REQUESTS_FILE = PIPELINE_DIR / "relation_extraction" / "tmp" / "requests.jsonl"
RELATION_RESULTS_FILE = PIPELINE_DIR / "relation_extraction" / "tmp" / "results.jsonl"

logger = logging.getLogger(__name__)


class RelationExtraction:
    def __init__(
        self,
        llm_client: LLMClient,
        config: Dict[str, Any],
    ) -> None:
        self.llm = llm_client
        self.config = config
        self.total_pairs = 0
        self._requests_handle = None
        self._prepare_request_file()

    def _prepare_request_file(self) -> None:
        ensure_dir(RELATION_REQUESTS_FILE)
        if RELATION_REQUESTS_FILE.exists():
            RELATION_REQUESTS_FILE.unlink()
        self._requests_handle = RELATION_REQUESTS_FILE.open("w", encoding="utf-8")
    
    def _build_prompt(self, pair: CandidatePair) -> str:
        subject = pair.subject.get("text")
        obj = pair.obj.get("text")
        allowed = "\n".join(
            f"- {pred.name}: {pred.description[:140]}"
            for pred in pair.predicates
        )
        sentence = pair.sentence.replace(subject, f"[SUBJ]{subject}[/SUBJ]", 1)
        sentence = sentence.replace(obj, f"[OBJ]{obj}[/OBJ]", 1)
        return (
            "Determine which predicate (if any) fits the sentence and give a concise explanation.\n"
            f"Sentence: {sentence}\n"
            f"Allowed predicates:\n{allowed}\n"
            "Respond as JSON {predicate: str, confidence: float, explanation: str}."
        )

    def add_pairs(self, pairs: Iterable[CandidatePair]) -> None:
        if not pairs:
            return
        if self._requests_handle is None:
            self._prepare_request_file()
        for pair in pairs:
            prompt = self._build_prompt(pair)
            payload = self.llm.build_chat_completion_kwargs(
                prompt=prompt,
                json_mode=True,
            )
            payload["metadata"] = self._metadata_from_pair(pair)
            self._requests_handle.write(json.dumps(payload) + "\n")
            self.total_pairs += 1

    def _metadata_from_pair(self, pair: CandidatePair) -> Dict:
        return {
            "pmid": pair.pmid,
            "sentence_id": pair.sentence_id,
            "sentence": pair.sentence,
            "subject": pair.subject,
            "object": pair.obj,
            "predicate_names": [pred.name for pred in pair.predicates],
            "model_name": self.config["llm"]["model"],
            "model_version": self.config.get("model_version", "v1"),
            "prompt_version": self.config.get("prompt_version", "v1"),
        }

    def run(self) -> List[Dict]:
        self._close_request_file()
        if self.total_pairs == 0:
            logger.info("No candidate pairs queued for relation extraction; skipping API call.")
            self._clear_results_file()
            return []
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Relation extraction requires an API key. Set OPENAI_API_KEY environment variable."
            )
        logger.info(
            "Starting relation extraction for %d pairs using %s",
            self.total_pairs,
            self.config["llm"]["request_url"],
        )
        self._clear_results_file()
        ensure_dir(RELATION_RESULTS_FILE)
        asyncio.run(
            process_api_requests_from_file(
                requests_filepath=str(RELATION_REQUESTS_FILE),
                save_filepath=str(RELATION_RESULTS_FILE),
                request_url=self.config["llm"]["request_url"],
                api_key=api_key,
                max_requests_per_minute=float(self.config["llm"]["max_requests_per_minute"]),
                max_tokens_per_minute=float(self.config["llm"]["max_tokens_per_minute"]),
                token_encoding_name=self.config["llm"]["token_encoding_name"],
                max_attempts=int(self.config["relation_extraction"]["max_attempts"]),
                logging_level=int(self.config["logging"]["logging_level"]),
            )
        )
        return self._read_results()

    def _read_results(self) -> List[Dict]:
        results: List[Dict] = []
        if not RELATION_RESULTS_FILE.exists():
            logger.warning("Relation extraction results file %s not found.", RELATION_RESULTS_FILE)
            return results
        with RELATION_RESULTS_FILE.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed relation extraction response line.")
                    continue
                if not isinstance(payload, list) or len(payload) < 2:
                    logger.warning("Unexpected relation extraction payload: %s", payload)
                    continue
                response = payload[1]
                metadata = payload[2] if len(payload) > 2 else None
                relation = self._build_relation(metadata, response)
                if relation:
                    results.append(relation)
        return results

    def _build_relation(self, metadata: Optional[Dict], response: Dict) -> Optional[Dict]:
        if metadata is None:
            logger.warning("Relation extraction response missing metadata.")
            return None
        if isinstance(response, list):
            logger.error(
                "Relation extraction for pmid=%s sentence_id=%s failed after retries: %s",
                metadata.get("pmid"),
                metadata.get("sentence_id"),
                response,
            )
            return None
        if "error" in response:
            logger.error(
                "Relation extraction for pmid=%s sentence_id=%s returned API error: %s",
                metadata.get("pmid"),
                metadata.get("sentence_id"),
                response["error"],
            )
            return None
        choices = response.get("choices") or []
        if not choices:
            logger.warning(
                "Relation extraction response missing choices for pmid=%s sentence_id=%s",
                metadata.get("pmid"),
                metadata.get("sentence_id"),
            )
            return None
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        if not content:
            logger.warning(
                "Relation extraction response missing content for pmid=%s sentence_id=%s",
                metadata.get("pmid"),
                metadata.get("sentence_id"),
            )
            return None
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            logger.warning(
                "Failed to decode relation extraction JSON for pmid=%s sentence_id=%s: %s",
                metadata.get("pmid"),
                metadata.get("sentence_id"),
                content,
            )
            return None
        predicate = result.get("predicate")
        allowed = metadata.get("predicate_names", [])
        if predicate not in allowed:
            logger.debug(
                "Predicate %s not allowed for pmid=%s sentence_id=%s; skipping.",
                predicate,
                metadata.get("pmid"),
                metadata.get("sentence_id"),
            )
            return None
        confidence = float(result.get("confidence", 0.0))
        return {
            "pmid": metadata.get("pmid"),
            "sentence_id": metadata.get("sentence_id"),
            "sentence": metadata.get("sentence"),
            "subject": metadata.get("subject"),
            "object": metadata.get("object"),
            "predicate": predicate,
            "confidence": confidence,
            "model_name": metadata.get("model_name"),
            "model_version": metadata.get("model_version"),
            "prompt_version": metadata.get("prompt_version"),
            "timestamp": timestamp(),
            "explanation": result.get("explanation", ""),
        }

    def _close_request_file(self) -> None:
        if self._requests_handle is not None:
            self._requests_handle.close()
            self._requests_handle = None

    def _clear_results_file(self) -> None:
        ensure_dir(RELATION_RESULTS_FILE)
        if RELATION_RESULTS_FILE.exists():
            RELATION_RESULTS_FILE.unlink()

