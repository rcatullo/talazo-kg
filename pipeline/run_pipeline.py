import argparse
import logging
import sys
from pathlib import Path
from typing import List

if __package__ is None or __package__ == "":
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))

from pipeline.model.entity_extractor import EntityExtractor
from pipeline.model.llm_client import LLMClient
from pipeline.model.pairing import PairGenerator
from pipeline.model.relation_classifier import RelationClassifier
from pipeline.schema.loader import SchemaLoader
from pipeline.schema.normalizer import Normalizer
from pipeline.utils.postprocess import PostProcessor, log_result
from pipeline.utils.sentence_splitter import Sentence, load_sentences
from pipeline.utils.settings import load_settings, write_jsonl

logger = logging.getLogger("pipeline.run")


def build_components():
    settings = load_settings()
    schema = SchemaLoader()
    llm = LLMClient(settings=settings)
    normalizer = Normalizer(schema)
    extractor = EntityExtractor(schema, normalizer, llm)
    pair_generator = PairGenerator(schema)
    classifier = RelationClassifier(schema, llm, settings)
    postprocessor = PostProcessor()
    return extractor, pair_generator, classifier, postprocessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/pubmed_talazoparib.jsonl")
    parser.add_argument("--output", default="data/relations.jsonl")
    parser.add_argument("--log", default="data/relation_log.jsonl")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--entity-batch", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    extractor, pair_generator, classifier, postprocessor = build_components()
    input_path = Path(args.input)
    log_path = Path(args.log)
    raw_results = []
    sentence_count = 0
    entity_total = 0
    pair_total = 0

    logger.info("Starting pipeline input=%s output=%s log=%s", input_path, args.output, log_path)

    batch: List[Sentence] = []
    batch_size = max(1, args.entity_batch)

    def process_batch(pending: List[Sentence]):
        nonlocal sentence_count, entity_total, pair_total, raw_results
        if not pending:
            return
        mapping = extractor.extract_batch(pending)
        for sentence in pending:
            sentence_count += 1
            entities = mapping.get((sentence.pmid, sentence.sentence_id), [])
            if not entities:
                logger.debug(
                    "No entities for pmid=%s sentence_id=%s", sentence.pmid, sentence.sentence_id
                )
                continue
            entity_total += len(entities)
            pairs = pair_generator.generate(sentence, entities)
            pair_total += len(pairs)
            for pair in pairs:
                classification = classifier.classify(pair)
                if not classification:
                    continue
                log_result(classification, log_path)
                raw_results.append(classification)

    for sentence in load_sentences(input_path):
        batch.append(sentence)
        if len(batch) >= batch_size:
            process_batch(batch)
            batch = []
        if sentence_count and sentence_count % 50 == 0:
            logger.info(
                "Processed %d sentences (%d entities, %d pairs so far)",
                sentence_count,
                entity_total,
                pair_total,
            )

    process_batch(batch)

    postprocessor.threshold = args.threshold
    filtered = postprocessor.filter(raw_results)
    aggregated = postprocessor.aggregate(filtered)
    write_jsonl(Path(args.output), aggregated)
    logger.info(
        "Finished: sentences=%d edges=%d filtered=%d aggregated=%d",
        sentence_count,
        len(raw_results),
        len(filtered),
        len(aggregated),
    )


if __name__ == "__main__":
    main()

