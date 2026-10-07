"""Evaluation Script for AIShield V1 Prototype.

Evaluates the complete end-to-end AIShield pipeline (Detector → Adapter → Risk → Policy → Firewall)
on evaluation datasets and compares security enforcement vs a baseline without a firewall.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.pipeline import AIShieldPipeline

DEFAULT_DATASET = PROJECT_ROOT / "src" / "detector" / "evaluation_examples.csv"
FALLBACK_DATASET = PROJECT_ROOT / "src" / "detector" / "data.csv"


def load_dataset(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found at {csv_path}")

    rows = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def evaluate(csv_path: Path) -> Dict[str, Any]:
    dataset = load_dataset(csv_path)
    pipeline = AIShieldPipeline()

    total = len(dataset)
    if total == 0:
        raise ValueError("Dataset is empty.")

    # Counters
    tp = 0  # True Positives (actual threat, flagged threat)
    fp = 0  # False Positives (actual safe, flagged threat)
    tn = 0  # True Negatives (actual safe, flagged safe)
    fn = 0  # False Negatives (actual threat, flagged safe)

    allow_count = 0
    review_count = 0
    block_count = 0

    threat_risk_scores: List[float] = []
    safe_risk_scores: List[float] = []
    latencies_ms: List[float] = []

    false_positives_list: List[Dict[str, Any]] = []
    false_negatives_list: List[Dict[str, Any]] = []

    for item in dataset:
        prompt = item["text"]
        actual_label = item["label"]
        actual_is_threat = actual_label in ("prompt_injection", "jailbreak")

        start_t = time.perf_counter()
        result = pipeline.analyse(prompt)
        elapsed_ms = (time.perf_counter() - start_t) * 1000
        latencies_ms.append(elapsed_ms)

        pred_is_threat = result.is_threat

        # Risk scoring stats
        if actual_is_threat:
            threat_risk_scores.append(result.risk_score)
        else:
            safe_risk_scores.append(result.risk_score)

        # Policy counters
        if result.action == "ALLOW":
            allow_count += 1
        elif result.action == "REVIEW":
            review_count += 1
        elif result.action == "BLOCK":
            block_count += 1

        # Confusion Matrix
        if actual_is_threat and pred_is_threat:
            tp += 1
        elif not actual_is_threat and pred_is_threat:
            fp += 1
            false_positives_list.append({"text": prompt, "label": actual_label, "result": result})
        elif not actual_is_threat and not pred_is_threat:
            tn += 1
        elif actual_is_threat and not pred_is_threat:
            fn += 1
            false_negatives_list.append({"text": prompt, "label": actual_label, "result": result})

    # Performance metrics
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    avg_threat_risk = sum(threat_risk_scores) / len(threat_risk_scores) if threat_risk_scores else 0.0
    avg_safe_risk = sum(safe_risk_scores) / len(safe_risk_scores) if safe_risk_scores else 0.0

    return {
        "total": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "allow_count": allow_count,
        "review_count": review_count,
        "block_count": block_count,
        "block_rate": (review_count + block_count) / total if total > 0 else 0.0,
        "avg_latency_ms": avg_latency,
        "avg_threat_risk": avg_threat_risk,
        "avg_safe_risk": avg_safe_risk,
        "false_positives": false_positives_list,
        "false_negatives": false_negatives_list,
    }


def print_report(metrics: Dict[str, Any], dataset_name: str) -> None:
    print("=" * 64)
    print("           AIShield V1 Firewall Security Evaluation")
    print(f"Dataset: {dataset_name} ({metrics['total']} prompts)")
    print("=" * 64)

    print("\n[1] Threat Detection Metrics")
    print(f"  Accuracy       : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision      : {metrics['precision']:.4f}")
    print(f"  Recall         : {metrics['recall']:.4f}")
    print(f"  F1 Score       : {metrics['f1']:.4f}")

    print("\n[2] Confusion Matrix (Threat vs Safe)")
    print(f"  True Positives  (Threat -> Flagged Threat) : {metrics['tp']}")
    print(f"  False Positives (Safe   -> Flagged Threat) : {metrics['fp']}")
    print(f"  True Negatives  (Safe   -> Flagged Safe)   : {metrics['tn']}")
    print(f"  False Negatives (Threat -> Flagged Safe)   : {metrics['fn']}")

    print("\n[3] Risk & Policy Enforcement")
    print(f"  ALLOW  (Pass to Downstream) : {metrics['allow_count']:3d} ({metrics['allow_count']/metrics['total']*100:.1f}%)")
    print(f"  REVIEW (Blocked, Needs Rev): {metrics['review_count']:3d} ({metrics['review_count']/metrics['total']*100:.1f}%)")
    print(f"  BLOCK  (Blocked Immediately): {metrics['block_count']:3d} ({metrics['block_count']/metrics['total']*100:.1f}%)")
    print(f"  Total Non-ALLOW Block Rate  : {metrics['block_rate']*100:.1f}%")

    print("\n[4] Risk Score Distributions & Performance")
    print(f"  Avg Risk Score (Threat Prompts): {metrics['avg_threat_risk']:.4f}")
    print(f"  Avg Risk Score (Safe Prompts)  : {metrics['avg_safe_risk']:.4f}")
    print(f"  Avg Latency per Prompt         : {metrics['avg_latency_ms']:.2f} ms")

    print("\n[5] Architecture Impact Comparison")
    print("+--------------------------------+-----------------+------------------+")
    print("| Metric                         | WITHOUT AIShield| WITH AIShield V1 |")
    print("+--------------------------------+-----------------+------------------+")
    print(f"| Prompts Reaching LLM           | 100.0% ({metrics['total']}/{metrics['total']})    | {metrics['allow_count']/metrics['total']*100:5.1f}% ({metrics['allow_count']}/{metrics['total']})     |")
    print(f"| Threat Prevention Rate         |   0.0% (0/{metrics['tp']+metrics['fn']})      | {metrics['tp']/(metrics['tp']+metrics['fn'])*100 if (metrics['tp']+metrics['fn'])>0 else 0:5.1f}% ({metrics['tp']}/{metrics['tp']+metrics['fn']})     |")
    print("+--------------------------------+-----------------+------------------+")
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AIShield V1 Firewall")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET),
        help="Path to evaluation CSV dataset",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_file() and DEFAULT_DATASET.is_file():
        dataset_path = DEFAULT_DATASET

    metrics = evaluate(dataset_path)
    print_report(metrics, dataset_path.name)


if __name__ == "__main__":
    main()
