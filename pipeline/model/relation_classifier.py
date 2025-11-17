import logging
from typing import Dict, Optional

from pipeline.model.llm_client import LLMClient
from pipeline.model.pairing import CandidatePair
from pipeline.schema.loader import SchemaLoader
from pipeline.utils.settings import Settings, load_settings, timestamp

logger = logging.getLogger(__name__)


class RelationClassifier:
    def __init__(
        self,
        schema: SchemaLoader,
        llm_client: Optional[LLMClient] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.schema = schema
        self.llm = llm_client or LLMClient()
        self.settings = settings or load_settings()

    def classify(self, pair: CandidatePair) -> Optional[Dict]:
        prompt = self._prompt(pair)
        response = self.llm.json_complete(prompt)
        result = response.get("json") or {}
        if not result:
            logger.warning(
                "No predicate returned for pmid=%s sentence_id=%s pair=%s/%s",
                pair.pmid,
                pair.sentence_id,
                pair.subject.get("text"),
                pair.obj.get("text"),
            )
            return None
        predicate = result.get("predicate")
        confidence = float(result.get("confidence", 0.0))
        if predicate not in [p.name for p in pair.predicates]:
            return None
        return {
            "pmid": pair.pmid,
            "sentence_id": pair.sentence_id,
            "sentence": pair.sentence,
            "subject": pair.subject,
            "object": pair.obj,
            "predicate": predicate,
            "confidence": confidence,
            "model_name": self.settings.llm_model,
            "model_version": self.settings.model_version,
            "prompt_version": self.settings.prompt_version,
            "timestamp": timestamp(),
            "explanation": result.get("explanation", ""),
        }

    def _prompt(self, pair: CandidatePair) -> str:
        subject = pair.subject.get("text")
        obj = pair.obj.get("text")
        allowed = "\n".join(
            f"- {pred.name}: {pred.description[:140]}"
            for pred in pair.predicates
        )
        sentence = pair.sentence.replace(subject, f"[SUBJ]{subject}[/SUBJ]", 1)
        sentence = sentence.replace(obj, f"[OBJ]{obj}[/OBJ]", 1)
        return (
            "Determine which predicate (if any) fits the sentence.\n"
            f"Sentence: {sentence}\n"
            f"Allowed predicates:\n{allowed}\n"
            "Respond as JSON {predicate: str, confidence: float, explanation: str}."
        )


