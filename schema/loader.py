from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_DIR = REPO_ROOT / "schema"


@dataclass
class Predicate:
    name: str
    domain: List[str]
    range: List[str]
    description: str
    guideline: str


class SchemaLoader:
    def __init__(
        self,
        model_path: Path | None = None,
        guideline_path: Path | None = None,
        idpolicy_path: Path | None = None,
    ) -> None:
        self.model_path = model_path or (SCHEMA_DIR / "model.yaml")
        self.guideline_path = guideline_path or (SCHEMA_DIR / "annotation_guideline.yaml")
        self.idpolicy_path = idpolicy_path or (SCHEMA_DIR / "idpolicy.yaml")
        self._model: Dict[str, Any] | None = None
        self._guidelines: Dict[str, Any] | None = None
        self._idpolicy: Dict[str, Any] | None = None

    @property
    def model(self) -> Dict[str, Any]:
        if self._model is None:
            with self.model_path.open("r", encoding="utf-8") as fh:
                self._model = yaml.safe_load(fh) or {}
        return self._model

    @property
    def guidelines(self) -> Dict[str, Any]:
        if self._guidelines is None:
            with self.guideline_path.open("r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
            self._guidelines = raw.get("annotation_guideline", {})
        return self._guidelines

    @property
    def idpolicy(self) -> Dict[str, Any]:
        if self._idpolicy is None:
            with self.idpolicy_path.open("r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
            self._idpolicy = raw.get("id_policy", {})
        return self._idpolicy

    def entity_classes(self) -> Dict[str, Any]:
        return self.model.get("classes", {})

    def predicates(self) -> Dict[str, Predicate]:
        slots = self.model.get("slots", {})
        predicates: Dict[str, Predicate] = {}
        for name, slot in slots.items():
            domain = slot.get("domain", [])
            rng = slot.get("range", [])
            guideline = self.guidelines.get(name, {})
            predicates[name] = Predicate(
                name=name,
                domain=domain,
                range=rng,
                description=(guideline.get("definition", "") or "")[:280],
                guideline="\n".join(guideline.get("decision_rule", {}).get("accept_if", [])[:3]),
            )
        return predicates

    def normalization_policy(self) -> Dict[str, Dict[str, List[str]]]:
        policy = {}
        for cls, rule in self.idpolicy.items():
            primary = rule.get("primary")
            alternates = rule.get("alternates", [])
            policy[cls] = {"primary": primary, "alternates": alternates}
        return policy

