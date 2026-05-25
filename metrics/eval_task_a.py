"""
Task A Evaluation Script.

Runs the /simulate-review endpoint against the held-out test set
and computes ROUGE-L, BERTScore F1, and RMSE.

Usage:
    python metrics/eval_task_a.py \
        --test-file data/test/task_a_test.jsonl \
        --api-url   http://localhost:8000 \
        --output    metrics/task_a_metrics.json
"""
import json, argparse, math
import httpx
from collections import defaultdict

def compute_rmse(predictions: list[float], ground_truth: list[float]) -> float:
    n = len(predictions)
    return math.sqrt(sum((p - g) ** 2 for p, g in zip(predictions, ground_truth)) / n)

def compute_rouge_l(pred: str, ref: str) -> float:
    """Simple ROUGE-L implementation using LCS."""
    pred_tokens = pred.lower().split()
    ref_tokens  = ref.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    m, n = len(ref_tokens), len(pred_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if ref_tokens[i-1] == pred_tokens[j-1] \
                       else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    precision = lcs / n if n else 0
    recall    = lcs / m if m else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", default="data/test/task_a_test.jsonl")
    parser.add_argument("--api-url",   default="http://localhost:8000")
    parser.add_argument("--output",    default="metrics/task_a_metrics.json")
    parser.add_argument("--max-cases", type=int, default=200)
    args = parser.parse_args()

    cases = []
    with open(args.test_file) as f:
        for line in f:
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    cases = cases[:args.max_cases]
    print(f"Evaluating {len(cases)} Task A cases...")

    pred_ratings, true_ratings, rouge_scores = [], [], []
    errors = 0

    with httpx.Client(base_url=args.api_url, timeout=30.0) as client:
        for i, case in enumerate(cases):
            payload = {
                "user_id":      case["user_id"],
                "user_history": case["user_history"],
                "target_item":  case["target_item"],
            }
            try:
                resp = client.post("/simulate-review", json=payload)
                resp.raise_for_status()
                result = resp.json()

                pred_ratings.append(result["predicted_rating"])
                true_ratings.append(float(case["ground_truth_rating"]))
                rouge_scores.append(compute_rouge_l(
                    result["generated_review"],
                    case["ground_truth_review"]
                ))

            except Exception as e:
                print(f"  Error on case {i}: {e}")
                errors += 1

            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(cases)} processed...")

    if not pred_ratings:
        print("No successful predictions. Exiting.")
        return

    rmse = compute_rmse(pred_ratings, true_ratings)
    avg_rouge = sum(rouge_scores) / len(rouge_scores)

    # BERTScore requires bert_score library
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(
            [c["generated_review"] for c in cases[:len(rouge_scores)]],
            [c["ground_truth_review"] for c in cases[:len(rouge_scores)]],
            lang="en", verbose=False
        )
        avg_bert_f1 = F1.mean().item()
    except Exception:
        avg_bert_f1 = None
        print("  BERTScore unavailable — skipping")

    results = {
        "n_cases":          len(pred_ratings),
        "n_errors":         errors,
        "rmse":             round(rmse, 4),
        "rouge_l":          round(avg_rouge, 4),
        "bert_score_f1":    round(avg_bert_f1, 4) if avg_bert_f1 else None,
        "target_rmse":      0.80,
        "target_rouge_l":   0.35,
        "target_bert_f1":   0.85,
        "rmse_pass":        rmse <= 0.80,
        "rouge_l_pass":     avg_rouge >= 0.35,
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n── Task A Results ──")
    print(f"  RMSE:          {results['rmse']} (target ≤ 0.80)  {'✓' if results['rmse_pass'] else '✗'}")
    print(f"  ROUGE-L:       {results['rouge_l']} (target ≥ 0.35) {'✓' if results['rouge_l_pass'] else '✗'}")
    if avg_bert_f1:
        print(f"  BERTScore F1:  {results['bert_score_f1']} (target ≥ 0.85)")
    print(f"\nResults saved → {args.output}")

if __name__ == "__main__":
    main()
