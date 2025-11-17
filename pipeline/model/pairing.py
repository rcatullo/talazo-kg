from dataclasses import dataclass
from typing import Dict, List

from pipeline.schema.loader import Predicate, SchemaLoader
from pipeline.utils.sentence_splitter import Sentence


@dataclass
class CandidatePair:
    pmid: str
    sentence_id: int
    sentence: str
    subject: Dict
    obj: Dict
    predicates: List[Predicate]


class PairGenerator:
    def __init__(self, schema: SchemaLoader, max_char_distance: int = 120):
        self.predicates = schema.predicates()
        self.max_char_distance = max_char_distance

    def generate(self, sentence: Sentence, entities: List[Dict]) -> List[CandidatePair]:
        pairs: List[CandidatePair] = []
        for subj in entities:
            for obj in entities:
                if subj is obj:
                    continue
                distance = self._distance(subj, obj)
                if distance > self.max_char_distance:
                    continue
                allowed = self._allowed_predicates(subj, obj)
                if not allowed:
                    continue
                pairs.append(
                    CandidatePair(
                        pmid=sentence.pmid,
                        sentence_id=sentence.sentence_id,
                        sentence=sentence.text,
                        subject=subj,
                        obj=obj,
                        predicates=allowed,
                    )
                )
        return pairs

    def _allowed_predicates(self, subj: Dict, obj: Dict) -> List[Predicate]:
        allowed = []
        subj_cls = subj.get("class")
        obj_cls = obj.get("class")
        for pred in self.predicates.values():
            if subj_cls in pred.domain and obj_cls in pred.range:
                allowed.append(pred)
        return allowed

    def _distance(self, subj: Dict, obj: Dict) -> int:
        span_a = subj.get("span") or [0, 0]
        span_b = obj.get("span") or [0, 0]
        start = min(span_a[0], span_b[0])
        end = max(span_a[1], span_b[1])
        return end - start

