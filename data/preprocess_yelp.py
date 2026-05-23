"""
Preprocess Yelp Open Dataset for MotiveAi Task B retrieval.

Flattens nested Yelp business JSON and extracts attributes
relevant to multi-constraint reasoning.

Usage:
    python data/preprocess_yelp.py \
        --business-file data/raw/yelp_academic_dataset_business.json \
        --output        data/yelp_businesses.jsonl \
        --max-records   50000
"""
import json, argparse
from pathlib import Path

TARGET_CATEGORIES = {
    "Restaurants", "Food", "Bars", "Cafes", "Nightlife",
    "Fast Food", "Burgers", "Pizza", "Nigerian", "African",
    "Continental", "American", "Italian", "Chinese", "Indian",
}

def parse_attributes(attrs: dict | None) -> dict:
    if not attrs:
        return {}
    flat = {}
    for k, v in attrs.items():
        if isinstance(v, str):
            flat[k.lower()] = v.strip("'\"")
        elif isinstance(v, dict):
            for sk, sv in v.items():
                flat[f"{k.lower()}_{sk.lower()}"] = str(sv).strip("'\"")
        else:
            flat[k.lower()] = str(v)
    return flat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--business-file", required=True)
    parser.add_argument("--output",        default="data/yelp_businesses.jsonl")
    parser.add_argument("--max-records",   type=int, default=50000)
    args = parser.parse_args()

    written = 0
    with open(args.business_file) as fin, open(args.output, "w") as fout:
        for line in fin:
            if written >= args.max_records:
                break
            try:
                b = json.loads(line)
            except json.JSONDecodeError:
                continue

            cats = set((b.get("categories") or "").split(", "))
            if not cats.intersection(TARGET_CATEGORIES):
                continue
            if b.get("stars", 0) < 2.0:
                continue

            record = {
                "item_id":       b["business_id"],
                "item_title":    b["name"],
                "item_category": "Restaurants",
                "stars":         b.get("stars"),
                "review_count":  b.get("review_count"),
                "city":          b.get("city"),
                "state":         b.get("state"),
                "categories":    b.get("categories",""),
                "attributes":    parse_attributes(b.get("attributes")),
            }
            fout.write(json.dumps(record) + "\n")
            written += 1

    print(f"Exported {written} Yelp businesses → {args.output}")

if __name__ == "__main__":
    main()
