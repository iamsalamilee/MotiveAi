"""
Build and validate the Nigerian Pidgin VADER lexicon.

Merges manually curated overrides with base VADER lexicon.
Validates that critical overrides produce correct sentiment directions.

Usage:
    python data/build_pidgin_lexicon.py --validate
"""
import json, argparse
from pathlib import Path

CURATED_OVERRIDES = {
    # Positive re-mappings
    "ginger": 0.7,   "hammer": 0.8,   "sabi": 0.5,    "fire": 0.7,
    "sharp":  0.6,   "kpali": 0.4,    "jejely": 0.3,  "gra-gra": -0.2,
    "wire":   0.3,   "level": 0.5,    "correct": 0.6, "tush": 0.4,
    # Negative re-mappings
    "wash":  -0.6,   "mumu": -0.5,    "vex": -0.7,    "wahala": -0.3,
    "scatter":-0.4,  "yab": -0.5,     "show": 0.1,    "suffer": -0.7,
    # Near-neutral (context-dependent)
    "yarn":  -0.1,   "yarn": -0.1,    "chop": 0.1,    "dey": 0.0,
    "comot":  0.0,   "enter": 0.0,    "follow": 0.0,  "run": -0.1,
}

VALIDATION_CASES = [
    ("E dey ginger me", "positive"),
    ("Dem wash am bad", "negative"),
    ("E don hammer today", "positive"),
    ("Too much wahala", "negative"),
    ("The guy sabi well", "positive"),
]

def validate(lexicon_path: str):
    from src.nlp.pidgin_vader import PidginVader
    vader = PidginVader(lexicon_path)
    passed = 0
    for text, expected in VALIDATION_CASES:
        score = vader.score_sentiment(text)["compound"]
        direction = "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"
        ok = direction == expected
        status = "✓" if ok else "✗"
        print(f"  {status}  '{text}' → {direction} ({score:.3f}) [expected {expected}]")
        if ok:
            passed += 1
    print(f"\n{passed}/{len(VALIDATION_CASES)} validation cases passed.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",   default="data/lexicons/pidgin_vader.json")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(CURATED_OVERRIDES, f, indent=2)
    print(f"Lexicon saved: {len(CURATED_OVERRIDES)} overrides → {args.output}")

    if args.validate:
        print("\nRunning validation...")
        validate(args.output)

if __name__ == "__main__":
    main()
