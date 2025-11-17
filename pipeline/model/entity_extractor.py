import json
import logging
from typing import Dict, Iterable, List, Optional, Tuple

from pipeline.model.llm_client import LLMClient
from pipeline.schema.loader import SchemaLoader
from pipeline.schema.normalizer import Normalizer
from pipeline.utils.sentence_splitter import Sentence

logger = logging.getLogger(__name__)


class EntityExtractor:
    def __init__(
        self,
        schema: SchemaLoader,
        normalizer: Normalizer,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        self.schema = schema
        self.normalizer = normalizer
        self.llm = llm_client or LLMClient()
        self.classes = list(schema.entity_classes().keys())

    def extract_batch(self, sentences: List[Sentence]) -> Dict[Tuple[str, int], List[Dict]]:
        if not sentences:
            return {}
        prompt = self._batch_prompt(sentences)
        response = self.llm.json_complete(prompt)
        payload = response.get("json") or {}
        results = payload.get("results", [])
        mapping: Dict[Tuple[str, int], List[Dict]] = {}
        for result in results:
            pmid = str(result.get("pmid"))
            sentence_id = int(result.get("sentence_id", 0))
            entities = result.get("entities", [])
            key = (pmid, sentence_id)
            mapping[key] = [self.normalizer.normalize(e) for e in entities]
        missing = [
            (s.pmid, s.sentence_id)
            for s in sentences
            if (s.pmid, s.sentence_id) not in mapping
        ]
        for pmid, sentence_id in missing:
            logger.warning("No entities returned for pmid=%s sentence_id=%s", pmid, sentence_id)
            mapping[(pmid, sentence_id)] = []
        return mapping

    def extract(self, sentence: Sentence) -> List[Dict]:
        return self.extract_batch([sentence])[(sentence.pmid, sentence.sentence_id)]

    def _prompt(self, text: str) -> str:
        class_list = ", ".join(self.classes)
        return (
            "Identify biomedical entities in the sentence.\n"
            f"Classes: {class_list}\n"
            "Return JSON with `entities` list of {text, class, start, end, ids}."
            f"\nSentence: {text}"
        )

    def _batch_prompt(self, sentences: Iterable[Sentence]) -> str:
        class_list = ", ".join(self.classes)
        entries = []
        for sentence in sentences:
            entries.append(
                {
                    "pmid": sentence.pmid,
                    "sentence_id": sentence.sentence_id,
                    "text": sentence.text,
                }
            )
        return (
            "Identify biomedical entities for each sentence below.\n"
            f"Classes: {class_list}.\n"
            "Return JSON with `results`: [{pmid, sentence_id, entities:[{text,class,start,end,ids}]}].\n"
            f"Sentences JSON:\n{json.dumps(entries, ensure_ascii=False)}"
        )

