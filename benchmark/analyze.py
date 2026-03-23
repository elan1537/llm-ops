"""Benchmark result analysis CLI.

Usage: python -m benchmark.analyze <results.json> [--dataset NAME] [--model NAME]
"""

import argparse
import json
import re
import sys

from benchmark.evaluators.common import strip_thinking


def _classify_prediction(pred: str) -> str:
    """Classify prediction into pattern categories."""
    pred = strip_thinking(pred).strip()
    if not pred:
        return "empty"
    if pred.startswith("ERROR:"):
        return "error"
    if re.match(r"^[A-D]\.?$", pred.strip(), re.IGNORECASE):
        return "clean_letter"
    if re.match(r"^[-+]?\d[\d,]*\.?\d*$", pred.strip()):
        return "clean_number"
    if re.search(r"(?:Answer|답)\s*:", pred, re.IGNORECASE):
        return "has_answer_tag"
    if len(pred) < 50:
        return "short_verbose"
    return "long_verbose"


def analyze_dataset(ds_name: str, model_results: dict, model_name: str):
    """Analyze results for one dataset-model pair."""
    for temp_key, result in model_results.items():
        temp = temp_key.replace("temperature_", "t=")
        details = result.get("details", [])
        if not details:
            continue

        score_keys = ["score", "f1", "anls", "cer", "mean_score"]
        score_val = None
        for sk in score_keys:
            if sk in result:
                score_val = result[sk]
                break

        total = result.get("total", len(details))
        correct_count = sum(1 for d in details if d.get("correct", False))
        error_count = sum(1 for d in details if "error" in d)

        print(f"\n=== {ds_name.upper()} ({model_name}, {temp}) ===")
        if score_val is not None:
            if isinstance(score_val, float) and score_val <= 1.0:
                print(f"Score: {score_val:.1%} ({correct_count}/{total})")
            else:
                print(f"Score: {score_val}")

        # Wrong answers
        wrong = [d for d in details if not d.get("correct", True) and "error" not in d]
        if wrong:
            print(f"\nWrong answers ({len(wrong)}):")
            for i, d in enumerate(wrong[:10]):
                pred_short = str(d.get("prediction", ""))[:80].replace("\n", " ")
                ref = str(d.get("reference", ""))[:40]
                extracted = d.get("extracted", "?")
                print(f"  #{i+1}  Ref: {ref:<35} Extracted: {extracted}")
                print(f"       Pred: {pred_short}")

        # Errors
        errors = [d for d in details if "error" in d]
        if errors:
            print(f"\nErrors ({len(errors)}):")
            for d in errors[:5]:
                print(f"  {str(d['error'])[:100]}")

        # Pattern statistics
        preds = [d.get("prediction", "") for d in details]
        patterns = {}
        for p in preds:
            cat = _classify_prediction(p)
            patterns[cat] = patterns.get(cat, 0) + 1

        print(f"\nPattern stats:")
        for cat, count in sorted(patterns.items(), key=lambda x: -x[1]):
            print(f"  {cat:<20} {count}/{total} ({count/total:.0%})")

        # Potential eval issues: reference found in prediction but marked wrong
        potential_issues = []
        for d in details:
            if not d.get("correct", True):
                ref = str(d.get("reference", "")).lower()
                pred = strip_thinking(str(d.get("prediction", ""))).lower()
                if ref and ref in pred:
                    potential_issues.append(d)

        if potential_issues:
            print(f"\nPotential eval issues ({len(potential_issues)} — ref found in pred but scored wrong):")
            for d in potential_issues[:5]:
                ref = str(d.get("reference", ""))[:30]
                extracted = d.get("extracted", "?")
                print(f"  Ref: {ref}  Extracted: {extracted}")


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("results_file", help="Path to results JSON file")
    parser.add_argument("--dataset", help="Analyze only this dataset")
    parser.add_argument("--model", help="Analyze only this model")
    args = parser.parse_args()

    with open(args.results_file) as f:
        data = json.load(f)

    results = data.get("results", {})

    for ds_name, ds_results in results.items():
        if args.dataset and ds_name != args.dataset:
            continue
        for model_name, model_results in ds_results.items():
            if args.model and model_name != args.model:
                continue
            analyze_dataset(ds_name, model_results, model_name)

    print()


if __name__ == "__main__":
    main()
