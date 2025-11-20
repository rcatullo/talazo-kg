import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

if __package__ is None or __package__ == "":
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))

from pipeline.model.llm_client import LLMClient
from pipeline.utils.pairing import PairGenerator
from pipeline.named_entity_recognition import NamedEntityRecognition
from pipeline.relation_extraction import RelationExtraction
from pipeline.schema.loader import SchemaLoader
from pipeline.schema.normalizer import Normalizer
from pipeline.utils.utils import ensure_dir, load_config, write_jsonl, PostProcessor, log_result, Sentence, load_sentences

PIPELINE_DIR = Path(__file__).resolve().parent
PIPELINE_LOG_FILE = PIPELINE_DIR / "logs" / "pipeline.log"

logger = logging.getLogger("pipeline.run")


def configure_logging(log_level: str) -> None:
    ensure_dir(PIPELINE_LOG_FILE)
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = logging.FileHandler(PIPELINE_LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)


def log_stage(stage: str, **details) -> None:
    if details:
        detail_str = " ".join(f"{key}={value}" for key, value in details.items())
        logger.info("Pipeline stage=%s %s", stage, detail_str)
    else:
        logger.info("Pipeline stage=%s", stage)


def build_components():
    config = load_config()
    schema = SchemaLoader()
    llm = LLMClient(config=config)
    normalizer = Normalizer(schema)
    ner = NamedEntityRecognition(schema, normalizer, llm, config)
    pair_generator = PairGenerator(schema)
    re = RelationExtraction(llm, config)
    postprocessor = PostProcessor()
    return ner, pair_generator, re, postprocessor


def main():
    config = load_config()
    configure_logging(config["logging"]["level"])

    log_stage("build_components")
    ner, pair_generator, re, postprocessor = build_components()
    log_stage("build_components_complete")
    input_path = Path(config["paths"]["input"])
    log_path = Path(config["paths"]["log"])
    raw_results = []
    sentence_count = 0
    entity_total = 0
    pair_total = 0

    logger.info(
        "Starting pipeline input=%s output=%s log=%s",
        input_path,
        config["paths"]["output"],
        log_path,
    )

    sentences = list(load_sentences(input_path))
    log_stage("entity_queue", sentences=len(sentences))
    ner.add_sentences(sentences)
    log_stage("entity_execute", sentences=ner.total_sentences)
    entity_mapping = ner.run()
    log_stage("entity_results", sentences=len(entity_mapping))

    for sentence in sentences:
        sentence_count += 1
        entities = entity_mapping.get((sentence.pmid, sentence.sentence_id), [])
        if not entities:
            logger.debug(
                "No entities for pmid=%s sentence_id=%s", sentence.pmid, sentence.sentence_id
            )
            continue
        entity_total += len(entities)
        log_stage(
            "pair_generation",
            pmid=sentence.pmid,
            sentence_id=sentence.sentence_id,
            entity_count=len(entities),
        )
        pairs = pair_generator.generate(sentence, entities)
        pair_total += len(pairs)
        if not pairs:
            continue
        log_stage(
            "relation_extraction",
            pmid=sentence.pmid,
            sentence_id=sentence.sentence_id,
            pair_count=len(pairs),
        )
        re.add_pairs(pairs)
        if sentence_count and sentence_count % 50 == 0:
            logger.info(
                "Processed %d sentences (%d entities, %d pairs so far)",
                sentence_count,
                entity_total,
                pair_total,
            )

    log_stage("relation_execute", total_pairs=re.total_pairs)
    for classification in re.run():
        log_result(classification, log_path)
        raw_results.append(classification)

    postprocessor.threshold = config["relation_extraction"]["threshold"]
    log_stage("postprocess_filter", total=len(raw_results))
    filtered = postprocessor.filter(raw_results)
    log_stage("postprocess_aggregate", filtered=len(filtered))
    aggregated = postprocessor.aggregate(filtered)
    log_stage("write_output", aggregated=len(aggregated), output=config["paths"]["output"])
    write_jsonl(Path(config["paths"]["output"]), aggregated)
    logger.info(
        "Finished: sentences=%d edges=%d filtered=%d aggregated=%d",
        sentence_count,
        len(raw_results),
        len(filtered),
        len(aggregated),
    )


if __name__ == "__main__":
    main()

