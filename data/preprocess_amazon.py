"""
Preprocess Amazon Reviews 2023 for MotiveAi.

Filters users who appear in BOTH Electronics AND Movies & TV domains
(minimum 3 reviews each) — required for Lasso cross-domain simulation.

Usage:
    python data/preprocess_amazon.py \
        --electronics-file data/raw/amazon_electronics.jsonl \
        --movies-file      data/raw/amazon_movies.jsonl \
        --output-dir       data/
"""
import json, argparse
from pathlib import Path
from collections import defaultdict

MIN_REVIEWS_PER_DOMAIN = 3
MIN_TOTAL_REVIEWS = 8

def load_reviews(path: str) -> list[dict]:
    reviews = []
    with open(path) as f:
        for line in f:
            try:
                reviews.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return reviews

def filter_cross_domain_users(elec: list[dict], movies: list[dict]):
    elec_users  = defaultdict(list)
    movie_users = defaultdict(list)

    for r in elec:
        elec_users[r["user_id"]].append(r)
    for r in movies:
        movie_users[r["user_id"]].append(r)

    qualifying = []
    for uid in elec_users:
        if uid not in movie_users:
            continue
        e_reviews = elec_users[uid]
        m_reviews = movie_users[uid]
        if (len(e_reviews) >= MIN_REVIEWS_PER_DOMAIN
                and len(m_reviews) >= MIN_REVIEWS_PER_DOMAIN
                and len(e_reviews) + len(m_reviews) >= MIN_TOTAL_REVIEWS):
            qualifying.append({
                "user_id": uid,
                "electronics": e_reviews,
                "movies": m_reviews,
            })

    print(f"Qualifying cross-domain users: {len(qualifying)}")
    return qualifying

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--electronics-file", required=True)
    parser.add_argument("--movies-file",      required=True)
    parser.add_argument("--output-dir",       default="data/")
    args = parser.parse_args()

    print("Loading Electronics reviews...")
    elec = load_reviews(args.electronics_file)
    print(f"  Loaded {len(elec)} reviews")

    print("Loading Movies & TV reviews...")
    movies = load_reviews(args.movies_file)
    print(f"  Loaded {len(movies)} reviews")

    users = filter_cross_domain_users(elec, movies)

    out = Path(args.output_dir) / "amazon_users.jsonl"
    with open(out, "w") as f:
        for u in users:
            f.write(json.dumps(u) + "\n")
    print(f"Saved {len(users)} cross-domain users → {out}")

if __name__ == "__main__":
    main()
