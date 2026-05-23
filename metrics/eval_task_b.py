"""
Task B Evaluation Script.

Runs the /recommend endpoint against the test set and computes
NDCG@10, Hit Rate@10, and cold-start coverage.

Usage:
    python metrics/eval_task_b.py \
        --test-file    data/test/task_b_test.jsonl \
        --ground-truth data/test/task_b_ground_truth.jsonl \
        --api-url      http://localhost:8000 \
        --output       metrics/task_b_metrics.json
"""
import json, argparse, math
import httpx

def dcg_at_k(relevances: list[int], k: int) -> float:
    return sum(
        rel / math.log2(i + 2)
        for i, rel in enumerate(relevances[:k])
    )

def ndcg_at_k(pred_ids: list[str], relevant_ids: set, k: int = 10) -> float:
    relevances = [1 if item_id in relevant_ids else 0 for item_id in pred_ids[:k]]
    ideal = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(relevances, k) / idcg

def hit_rate_at_k(pred_ids: list[str], relevant_ids: set, k: int = 10) -> int:
    return int(any(item_id in relevant_ids for item_id in pred_ids[:k]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file",    default="data/test/task_b_test.jsonl")
    parser.add_argument("--ground-truth", default="data/test/task_b_ground_truth.jsonl")
    parser.add_argument("--api-url",      default="http://localhost:8000")
    parser.add_argument("--output",       default="metrics/task_b_metrics.json")
    parser.add_argument("--max-cases",    type=int, default=200)
    parser.add_argument("--k",            type=int, default=10)
    args = parser.parse_args()

    cases = []
    with open(args.test_file) as f:
        for line in f:
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    cases = cases[:args.max_cases]

    # Load ground truth relevant items per user
    ground_truth = {}
    try:
        with open(args.ground_truth) as f:
            for line in f:
                gt = json.loads(line)
                ground_truth[gt["user_id"]] = set(gt["relevant_item_ids"])
    except FileNotFoundError:
        print(f"  WARNING: Ground truth file not found ({args.ground_truth})")
        print("  NDCG and Hit Rate will be computed against stub ground truth.")
        ground_truth = {c["user_id"]: set() for c in cases}

    print(f"Evaluating {len(cases)} Task B cases (k={args.k})...")

    ndcg_scores, hit_rates, cold_start_covered = [], [], 0
    cold_start_total, errors = 0, 0

    with httpx.Client(base_url=args.api_url, timeout=30.0) as client:
        for i, case in enumerate(cases):
            is_cold = len(case["user_persona"].get("target_history", [])) == 0
            if is_cold:
                cold_start_total += 1

            try:
                resp = client.post("/recommend", json=case)
                resp.raise_for_status()
                result = resp.json()

                pred_ids = [r["item_id"] for r in result["recommendations"]]
                relevant = ground_truth.get(case["user_id"], set())

                ndcg_scores.append(ndcg_at_k(pred_ids, relevant, args.k))
                hit_rates.append(hit_rate_at_k(pred_ids, relevant, args.k))

                if is_cold and len(pred_ids) > 0:
                    cold_start_covered += 1

            except Exception as e:
                print(f"  Error on case {i}: {e}")
                errors += 1

            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(cases)} processed...")

    if not ndcg_scores:
        print("No successful predictions.")
        return

    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)
    avg_hit  = sum(hit_rates)   / len(hit_rates)
    cold_coverage = cold_start_covered / cold_start_total if cold_start_total else 1.0

    results = {
        "n_cases":                len(ndcg_scores),
        "n_errors":               errors,
        "ndcg_at_10":             round(avg_ndcg, 4),
        "hit_rate_at_10":         round(avg_hit,  4),
        "cold_start_total":       cold_start_total,
        "cold_start_covered":     cold_start_covered,
        "cold_start_coverage":    round(cold_coverage, 4),
        "target_ndcg":            0.45,
        "target_hit_rate":        0.58,
        "target_cold_coverage":   1.0,
        "ndcg_pass":              avg_ndcg >= 0.45,
        "hit_rate_pass":          avg_hit  >= 0.58,
        "cold_coverage_pass":     cold_coverage >= 1.0,
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n── Task B Results ──")
    print(f"  NDCG@10:           {results['ndcg_at_10']} (target ≥ 0.45) {'✓' if results['ndcg_pass'] else '✗'}")
    print(f"  Hit Rate@10:       {results['hit_rate_at_10']} (target ≥ 0.58) {'✓' if results['hit_rate_pass'] else '✗'}")
    print(f"  Cold-Start Cov:    {results['cold_start_coverage']:.0%} (target 100%) {'✓' if results['cold_coverage_pass'] else '✗'}")
    print(f"\nResults saved → {args.output}")

if __name__ == "__main__":
    main()
