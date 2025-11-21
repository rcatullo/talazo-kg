import argparse
import datetime as dt
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


def chunked(seq: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def esearch_ids(query: str, mindate: str, maxdate: str, batch_size: int, max_articles: Optional[int] = None) -> List[str]:
    ids: List[str] = []
    retstart = 0
    count = None
    fetched = 0
    while True:
        # If max_articles is set, do not request beyond that limit
        curr_batch_size = batch_size
        if max_articles is not None:
            remaining = max_articles - fetched
            if remaining <= 0:
                break
            curr_batch_size = min(curr_batch_size, remaining)
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "xml",
            "retmax": curr_batch_size,
            "retstart": retstart,
            "datetype": "pdat",
            "mindate": mindate,
            "maxdate": maxdate,
        }
        resp = requests.get(f"{EUTILS_BASE}esearch.fcgi", params=params, timeout=30)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        if count is None:
            count_text = root.findtext("Count", default="0")
            count = int(count_text)
        batch_ids = [elem.text for elem in root.findall(".//IdList/Id") if elem.text]
        if not batch_ids:
            break
        ids.extend(batch_ids)
        retstart += len(batch_ids)
        fetched += len(batch_ids)
        if retstart >= count:
            break
        if max_articles is not None and fetched >= max_articles:
            break
        time.sleep(0.34)
    if max_articles is not None:
        ids = ids[:max_articles]
    return ids


def parse_article(article: ET.Element) -> Dict[str, object]:
    pmid = article.findtext(".//PMID")
    title = article.findtext(".//Article/ArticleTitle") or ""
    abstract_nodes = article.findall(".//Abstract/AbstractText")
    abstract_parts = []
    for node in abstract_nodes:
        text = node.text or ""
        label = node.get("Label")
        abstract_parts.append(f"{label}: {text}" if label else text)
    abstract = "\n".join(part.strip() for part in abstract_parts if part.strip())
    pub_date = article.find(".//Article/Journal/JournalIssue/PubDate")
    year = pub_date.findtext("Year") if pub_date is not None else None
    if not year:
        medline_date = pub_date.findtext("MedlineDate") if pub_date is not None else ""
        if medline_date:
            year = medline_date.split(" ")[0]
    journal = article.findtext(".//Article/Journal/Title")
    mesh_terms = [
        node.text for node in article.findall(".//MeshHeadingList/MeshHeading/DescriptorName") if node.text
    ]
    authors = []
    for author in article.findall(".//AuthorList/Author"):
        last = author.findtext("LastName")
        fore = author.findtext("ForeName")
        collective = author.findtext("CollectiveName")
        if collective:
            authors.append(collective)
        elif last or fore:
            authors.append(", ".join(filter(None, [last, fore])))
    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "meta": {"year": year, "journal": journal, "mesh_terms": mesh_terms, "authors": authors},
    }


def efetch_records(ids: List[str], batch_size: int, max_records: Optional[int] = None) -> Iterable[dict]:
    yielded = 0
    for batch in chunked(ids, batch_size):
        if max_records is not None:
            remaining = max_records - yielded
            if remaining <= 0:
                break
            batch = batch[:remaining]
        params = {
            "db": "pubmed",
            "retmode": "xml",
            "rettype": "abstract",
            "id": ",".join(batch),
        }
        resp = requests.get(f"{EUTILS_BASE}efetch.fcgi", params=params, timeout=60)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        for article in root.findall(".//PubmedArticle"):
            if max_records is not None and yielded >= max_records:
                break
            yield parse_article(article)
            yielded += 1
        if max_records is not None and yielded >= max_records:
            break
        time.sleep(0.34)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="talazoparib resistance")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--esearch-batch", type=int, default=100)
    parser.add_argument("--efetch-batch", type=int, default=20)
    parser.add_argument("--output", default="data/pubmed_talazoparib.jsonl")
    parser.add_argument("--max-articles", type=int, default=None, help="Maximum number of PubMed articles to fetch")
    args = parser.parse_args()

    today = dt.date.today()
    start_date = today - dt.timedelta(days=365 * args.years)
    mindate = start_date.strftime("%Y/%m/%d")
    maxdate = today.strftime("%Y/%m/%d")

    ids = esearch_ids(args.query, mindate, maxdate, args.esearch_batch, args.max_articles)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_pmids = set()
    num_written = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for record in efetch_records(ids, args.efetch_batch, args.max_articles):
            pmid = record.get("pmid")
            if not pmid or pmid in seen_pmids:
                continue
            seen_pmids.add(pmid)
            fh.write(json.dumps(record) + "\n")
            num_written += 1
            if args.max_articles is not None and num_written >= args.max_articles:
                break


if __name__ == "__main__":
    main()