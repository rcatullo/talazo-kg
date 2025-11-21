from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from model.llm_client import LLMClient
from schema import SchemaLoader, Normalizer
from utils.api_req_parallel import process_api_requests_from_file
from utils import ensure_dir, Sentence

ROOT_DIR = Path(__file__).resolve().parent.parent
NER_REQUESTS_FILE = ROOT_DIR / "named_entity_recognition" / "tmp" / "requests.jsonl"
NER_RESULTS_FILE = ROOT_DIR / "named_entity_recognition" / "tmp" / "results.jsonl"

logger = logging.getLogger(__name__)


class NamedEntityRecognition:
    def __init__(
        self,
        schema: SchemaLoader,
        normalizer: Normalizer,
        llm_client: LLMClient,
        config: Dict[str, Any],
    ) -> None:
        self.schema = schema
        self.normalizer = normalizer
        self.llm = llm_client
        self.classes = list(schema.entity_classes().keys())
        self.config = config
        self.total_sentences = 0
        self._requests_handle = None
        self._sentence_lookup: Dict[Tuple[str, int], Sentence] = {}
        self._prepare_request_file()

    def _prepare_request_file(self) -> None:
        ensure_dir(NER_REQUESTS_FILE)
        if NER_REQUESTS_FILE.exists():
            NER_REQUESTS_FILE.unlink()
        self._requests_handle = NER_REQUESTS_FILE.open("w", encoding="utf-8")

    def add_sentences(self, sentences: Iterable[Sentence]) -> None:
        for sentence in sentences:
            self.add_sentence(sentence)

    def add_sentence(self, sentence: Sentence) -> None:
        key = (sentence.pmid, sentence.sentence_id)
        self._sentence_lookup[key] = sentence
        self.total_sentences += 1
        if self._requests_handle is None:
            self._prepare_request_file()
        prompt = self._build_prompt(sentence)
        payload = self.llm.build_chat_completion_kwargs(prompt=prompt, json_mode=True)
        payload["metadata"] = {
            "pmid": sentence.pmid,
            "sentence_id": sentence.sentence_id,
            "text": sentence.text,
        }
        self._requests_handle.write(json.dumps(payload) + "\n")
    
    def _build_prompt(self, sentence: Sentence) -> str:
        class_list = ", ".join(self.classes)
        return (
            "Identify biomedical entities for the sentence below (if any). If the entity doesn't fit confidently into one of the given classes, even if it is biomedical in nature, omit it.\n"
            f"Classes: {class_list}.\n"
            "Return JSON with `entities`: [{text,class,start,end,ids}].\n"
            f"Sentence: {sentence.text}"
        )

    def run(self) -> Dict[Tuple[str, int], List[Dict]]:
        self._close_request_file()
        if self.total_sentences == 0:
            logger.info("No sentences queued for named entity recognition; skipping API call.")
            self._clear_results_file()
            return {}
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Named entity recognition requires an API key. Set OPENAI_API_KEY environment variable."
            )
        logger.info(
            "Starting named entity recognition for %d sentences using %s",
            self.total_sentences,
            self.config["llm"]["request_url"],
        )
        self._clear_results_file()
        ensure_dir(NER_RESULTS_FILE)
        asyncio.run(
            process_api_requests_from_file(
                requests_filepath=str(NER_REQUESTS_FILE),
                save_filepath=str(NER_RESULTS_FILE),
                request_url=self.config["llm"]["request_url"],
                api_key=api_key,
                max_requests_per_minute=float(self.config["llm"]["max_requests_per_minute"]),
                max_tokens_per_minute=float(self.config["llm"]["max_tokens_per_minute"]),
                token_encoding_name=self.config["llm"]["token_encoding_name"],
                max_attempts=int(self.config["named_entity_recognition"]["max_attempts"]),
                logging_level=int(self.config["logging"]["logging_level"]),
            )
        )
        return self._collect_entities()

    def _collect_entities(self) -> Dict[Tuple[str, int], List[Dict]]:
        mapping: Dict[Tuple[str, int], List[Dict]] = {}
        if not NER_RESULTS_FILE.exists():
            logger.warning(
                "Named entity recognition results file %s not found.", NER_RESULTS_FILE
            )
            return mapping
        with NER_RESULTS_FILE.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    _, response_payload, metadata = self._decode_line(line)
                except ValueError as err:
                    logger.warning("Skipping malformed named entity recognition response: %s", err)
                    continue
                key = (metadata.get("pmid"), metadata.get("sentence_id"))
                sentence = self._sentence_lookup.get(key)
                if sentence is None:
                    logger.warning("Unknown sentence metadata for key=%s", key)
                    continue
                entities = self._parse_response(sentence, metadata, response_payload)
                mapping[key] = entities
        return mapping

    def _decode_line(self, line: str):
        payload = json.loads(line)
        if not isinstance(payload, list) or len(payload) < 2:
            raise ValueError("Unexpected payload format")
        request = payload[0]
        response = payload[1]
        metadata = payload[2] if len(payload) > 2 else {}
        return request, response, metadata

    def _parse_response(
        self,
        sentence: Sentence,
        metadata: Dict,
        response,
    ) -> List[Dict]:
        key = (sentence.pmid, sentence.sentence_id)
        if isinstance(response, list):
            logger.error(
                "Named entity recognition failed after retries for %s/%s -> %s",
                *key,
                response,
            )
            return []
        if "error" in response:
            logger.error(
                "Named entity recognition API error for %s/%s: %s",
                *key,
                response["error"],
            )
            return []
        choices = response.get("choices") or []
        if not choices:
            logger.warning("Named entity recognition missing choices for %s/%s", *key)
            return []
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        if not content:
            logger.warning("Named entity recognition missing content for %s/%s", *key)
            return []
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            logger.warning(
                "Failed to decode named entity recognition JSON for %s/%s: %s",
                *key,
                content,
            )
            return []
        return [self.normalizer.normalize(e) for e in payload.get("entities", [])]

    def _close_request_file(self) -> None:
        if self._requests_handle is not None:
            self._requests_handle.close()
            self._requests_handle = None

    def _clear_results_file(self) -> None:
        ensure_dir(NER_RESULTS_FILE)
        if NER_RESULTS_FILE.exists():
            NER_RESULTS_FILE.unlink()

