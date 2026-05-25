"""
Build ChromaDB index from preprocessed Yelp businesses.

Embeds each business description with NaijaBERT and inserts
into ChromaDB for ANN retrieval during Task B inference.

Usage:
    python data/build_chroma_index.py \
        --input    data/yelp_businesses.jsonl \
        --chroma-host localhost \
        --chroma-port 8001 \
        --batch-size 256
"""
import json, argparse
import numpy as np
from pathlib import Path

def build_description(record: dict) -> str:
    """Build text description of business for embedding."""
    parts = [record.get("item_title","")]
    if record.get("categories"):
        parts.append(record["categories"])
    if record.get("city"):
        parts.append(f"Located in {record['city']}")
    attrs = record.get("attributes", {})
    for k, v in list(attrs.items())[:5]:
        parts.append(f"{k}: {v}")
    return ". ".join(parts)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       default="data/yelp_businesses.jsonl")
    parser.add_argument("--chroma-host", default="localhost")
    parser.add_argument("--chroma-port", type=int, default=8001)
    parser.add_argument("--batch-size",  type=int, default=256)
    args = parser.parse_args()

    from src.nlp.naija_bert import naija_bert
    from src.agent.retriever import retriever

    print("Loading NaijaBERT...")
    naija_bert.warmup()

    print(f"Connecting to ChromaDB at {args.chroma_host}:{args.chroma_port}...")
    retriever.connect(args.chroma_host, args.chroma_port)

    records = []
    with open(args.input) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"Indexing {len(records)} businesses in batches of {args.batch_size}...")
    total = 0
    for i in range(0, len(records), args.batch_size):
        batch = records[i:i+args.batch_size]
        descriptions = [build_description(r) for r in batch]
        embeddings = naija_bert.embed(descriptions)  # (N, 768)

        items = [{
            "item_id":   r["item_id"],
            "embedding": embeddings[j],
            "metadata":  {k: str(v) for k, v in r.items() if k != "item_id"},
        } for j, r in enumerate(batch)]

        retriever.upsert_items(items)
        total += len(batch)
        print(f"  {total}/{len(records)} indexed...")

    print(f"Done. Total indexed: {retriever.get_collection_size()}")

if __name__ == "__main__":
    main()
