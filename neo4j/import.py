import argparse
import json
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase

URI = "neo4j+s://4e9b2d24.databases.neo4j.io"
USER = "neo4j"
PASSWORD = "4MyZedIjW_ZdPpR7xzXx5ocCENECin_jZh6WThHHktU"
DATABASE = "neo4j"

def parse_sentences(sentences: Optional[List[Dict[str, Any]]]) -> List[str]:
    """Extract plain sentence strings from the 'sentences' field."""
    if not sentences:
        return []
    return [
        s.get("sentence")
        for s in sentences
        if isinstance(s, dict) and s.get("sentence")
    ]


def import_triple(tx, record: Dict[str, Any]) -> None:
    """Import a single JSONL record as nodes + relationship into Neo4j."""

    subj = record.get("subject") or {}
    obj = record.get("object") or {}
    predicate = record.get("predicate")

    if not subj or not obj or not predicate:
        return

    rel_type = predicate.upper()
    confidence = record.get("confidence")

    s_class = subj.get("class")
    o_class = obj.get("class")

    s_text = subj.get("text")
    o_text = obj.get("text")

    s_id = subj.get("id") or f"{s_class}:{s_text}"
    o_id = obj.get("id") or f"{o_class}:{o_text}"

    s_name = s_text
    o_name = o_text

    pmids = record.get("pmids") or []
    pmid = pmids[0] if pmids else None

    sentences = parse_sentences(record.get("sentences"))
    timestamp = record.get("timestamp")
    model_name = record.get("model_name")
    model_version = record.get("model_version")

    query = f"""
    MERGE (s:{s_class} {{id: $s_id}})
      ON CREATE SET s.name = $s_name
    MERGE (o:{o_class} {{id: $o_id}})
      ON CREATE SET o.name = $o_name
    MERGE (s)-[r:{rel_type}]->(o)
    SET
      r.pmid          = coalesce($pmid, r.pmid),
      r.pmids         = coalesce($pmids, r.pmids),
      r.confidence    = coalesce($confidence, r.confidence),
      r.extractor     = coalesce(r.extractor, 'LLM'),
      r.created_at    = coalesce($timestamp, r.created_at),
      r.model_name    = coalesce($model_name, r.model_name),
      r.model_version = coalesce($model_version, r.model_version),
      r.sentences     = coalesce($sentences, r.sentences)
    """

    tx.run(
        query,
        s_id=s_id,
        s_name=s_name,
        o_id=o_id,
        o_name=o_name,
        pmid=pmid,
        pmids=pmids,
        confidence=confidence,
        timestamp=timestamp,
        model_name=model_name,
        model_version=model_version,
        sentences=sentences,
    )


def import_file(path: str) -> None:
    """Stream a JSONL file and import each triple into Neo4j."""
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    total = 0
    imported = 0

    with driver.session(database=DATABASE) as session:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Line {line_num}: JSON error, skipping: {e}")
                    continue

                try:
                    session.execute_write(import_triple, record)
                    imported += 1
                except Exception as e:
                    print(f"Line {line_num}: Neo4j error, skipping: {e}")

    driver.close()
    print(f"Done. Processed {total} lines, imported (attempted) {imported}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import JSONL triples into Neo4j.")
    parser.add_argument("relations_path", help="Path to the relations.jsonl file with triples")
    args = parser.parse_args()

    import_file(args.relations_path)
