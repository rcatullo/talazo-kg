# Biomedical Knowledge Graph Pipeline

A lightweight LLM-assisted workflow for extracting oncology resistance relations and writing them into a Biolink-compliant knowledge graph.

## Requirements

- Python 3.11+
- `pip install -r requirements.txt`
- `OPENAI_API_KEY` set to a model with JSON-mode chat completions

Configurable parameters lie in `config.yaml`

## Gather Data: Fetch Pubmed Articles

The file `fetch_pubmed.py` collects recent PubMed abstracts for "talazoparib resistance" and writes JSONL into `data/`.

```bash
fetch_pubmed.py --years 10 --output data/pubmed_talazoparib.jsonl
```

## Pipeline

Run the pipeline script by calling

```bash
python run_pipeline.py \
  --input data/pubmed_talazoparib.jsonl \
  --output data/relations.jsonl
```

The script does the following in order.

### Named-Entity Recognition (NER)
Load config + schema metadata, split abstracts into sentences, and queue every sentence for **named-entity recognition** through the shared OpenAI request worker. There are two files generated during this phase, both in `named_entity_recognition/tmp/`:
1. `requests.json` - the requests being posted to the OpenAI API in parallel.
2. `results.json` - the results of the requests, returned not necessarily in the same order.

### Relation Extraction (RE)

Entities are normalized with `schema/idpolicy.yaml`, candidate pairs are filtered by the domains/ranges defined in `schema/model.yaml`, and the LLM is prompted to perform **relation extraction** using concise predicate descriptions derived from `schema/annotation_guideline.yaml`. 

Requests and results jsons of relation extraction similarly are logged to `relation_extraction/tmp/`.

Each evaluated pair is logged to `logs/relation_log.jsonl`. Low-confidence edges are dropped, duplicates (same subject–predicate–object) are merged, and results are written to `data/relations.jsonl` with pmids, confidence, and model metadata.

## TL;DR Typical Run

```bash
export OPENAI_API_KEY=sk-...
python fetch_pubmed.py --years 10 --output data/pubmed_talazoparib.jsonl
python run_pipeline.py \
  --input data/pubmed_talazoparib.jsonl \
  --output data/relations.jsonl
```