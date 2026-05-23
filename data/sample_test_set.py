"""
Sample held-out test sets for Task A and Task B evaluation.

Splits preprocessed Amazon user data into train/test,
ensuring cold-start users are represented in Task B test set.

Usage:
    python data/sample_test_set.py \
        --amazon-file data/amazon_users.jsonl \
        --output-dir  data/test/ \
        --test-size   200
"""
import json, argparse, random
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amazon-file", default="data/amazon_users.jsonl")
    parser.add_argument("--output-dir",  default="data/test/")
    parser.add_argument("--test-size",   type=int, default=200)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    users = []
    with open(args.amazon_file) as f:
        for line in f:
            try:
                users.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    random.shuffle(users)
    test_users = users[:args.test_size]

    # Task A test: for each user, hold out their last review as ground truth
    task_a = []
    for u in test_users:
        history = sorted(u["electronics"], key=lambda r: r.get("timestamp",""))
        if len(history) < 2:
            continue
        train_history = history[:-1]
        ground_truth  = history[-1]
        task_a.append({
            "user_id":       u["user_id"],
            "user_history":  train_history,
            "target_item": {
                "item_id":       ground_truth["item_id"],
                "item_title":    ground_truth.get("item_title",""),
                "item_category": "Electronics",
                "item_metadata": {},
            },
            "ground_truth_rating": ground_truth["rating"],
            "ground_truth_review": ground_truth["review_text"],
        })

    # Task B test: use movies history as source, restaurants as cold-start target
    task_b = []
    for u in test_users:
        task_b.append({
            "user_id": u["user_id"],
            "user_persona": {
                "source_domain":  "Electronics",
                "source_history": u["electronics"],
                "target_domain":  "Restaurants",
                "target_history": [],  # cold-start
                "conversational_context": [],
            },
            "top_k": 10,
        })

    a_out = Path(args.output_dir) / "task_a_test.jsonl"
    b_out = Path(args.output_dir) / "task_b_test.jsonl"

    with open(a_out, "w") as f:
        for r in task_a:
            f.write(json.dumps(r) + "\n")
    with open(b_out, "w") as f:
        for r in task_b:
            f.write(json.dumps(r) + "\n")

    print(f"Task A test set: {len(task_a)} cases → {a_out}")
    print(f"Task B test set: {len(task_b)} cases → {b_out}")

if __name__ == "__main__":
    main()
