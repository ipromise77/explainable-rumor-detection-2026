"""
Optimized evaluation script that combines both LLM caches for best accuracy.

Key improvements:
1. Uses original LLM cache for samples in [0.30, 0.70] probability range
2. Uses FN recall review cache for samples predicted as 0 with prob in [0.15, 0.50]
3. Lowers confidence threshold from 0.85 to 0.65 for override decisions
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rumor_detector import load_model, predict_proba, explain_text


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_llm_cache(path: Path) -> dict:
    cache = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    cache[row["hash"]] = row["result"]
    return cache


def load_fn_review_candidates(path: Path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {
        int(row["row_index"]): {
            "label": int(row["review_label"]),
            "confidence": float(row["llm_confidence"]),
            "reason": str(row["reason"]),
        }
        for _, row in df.iterrows()
    }


def main():
    bundle = load_model()
    df = pd.read_csv(ROOT / "rumer2026" / "val.csv")
    prob = predict_proba(bundle, df["text"])
    pred_baseline = (prob >= 0.5).astype(int)

    df["prob"] = prob
    df["hash"] = [text_hash(t) for t in df["text"]]

    # Load both caches
    cache_v1 = load_llm_cache(ROOT / "results" / "llm_cache.jsonl")
    fn_candidates = load_fn_review_candidates(ROOT / "results" / "fn_recall_review_candidates.csv")

    print(f"Loaded {len(cache_v1)} entries from llm_cache.jsonl")
    print(f"Loaded {len(fn_candidates)} entries from fn_recall_review_candidates.csv")
    print()

    # Optimized strategy: confidence threshold = 0.65
    CONF_THRESHOLD = 0.65

    pred = pred_baseline.copy()
    explanations = [""] * len(df)
    sources = ["local"] * len(df)
    override_details = []

    for idx in range(len(df)):
        p = df.loc[idx, "prob"]
        local_pred = pred_baseline[idx]
        h = df.loc[idx, "hash"]
        true_label = df.loc[idx, "label"]

        # Path 1: Original LLM cache for [0.30, 0.70] range
        if 0.30 <= p <= 0.70 and h in cache_v1:
            llm = cache_v1[h]
            if local_pred == 0 and llm.get("label", 0) == 1 and llm.get("confidence", 0) >= CONF_THRESHOLD:
                pred[idx] = 1
                sources[idx] = "llm_override_v1"
                explanations[idx] = llm.get("reason", "")
                override_details.append({
                    "idx": idx,
                    "source": "v1",
                    "true_label": true_label,
                    "prob": p,
                    "llm_conf": llm.get("confidence", 0),
                })

        # Path 2: FN review cache for pred=0 samples in [0.15, 0.50] range
        if local_pred == 0 and 0.15 <= p < 0.50 and idx in fn_candidates:
            cand = fn_candidates[idx]
            if cand["label"] == 1 and cand["confidence"] >= CONF_THRESHOLD:
                if pred[idx] == 0:  # Avoid double override
                    pred[idx] = 1
                    sources[idx] = "llm_override_v2"
                    explanations[idx] = cand["reason"]
                    override_details.append({
                        "idx": idx,
                        "source": "v2",
                        "true_label": true_label,
                        "prob": p,
                        "llm_conf": cand["confidence"],
                    })

        # Fill in local explanation if not overridden
        if not explanations[idx]:
            explanations[idx] = explain_text(bundle, df.loc[idx, "text"])

    # Calculate metrics
    cm = confusion_matrix(df["label"], pred)
    acc = accuracy_score(df["label"], pred)
    f1 = f1_score(df["label"], pred, average="macro")

    baseline_cm = confusion_matrix(df["label"], pred_baseline)
    baseline_acc = accuracy_score(df["label"], pred_baseline)

    print("=" * 60)
    print("OPTIMIZED EVALUATION RESULTS")
    print("=" * 60)
    print()
    print(f"Confidence threshold: {CONF_THRESHOLD}")
    print()
    print("BASELINE (local model only):")
    print(f"  Accuracy: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    print(f"  Confusion Matrix: {baseline_cm.tolist()}")
    print(f"  FN={baseline_cm[1,0]}, FP={baseline_cm[0,1]}")
    print()
    print("OPTIMIZED (with LLM override):")
    print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Macro-F1: {f1:.4f}")
    print(f"  Confusion Matrix: {cm.tolist()}")
    print(f"  FN={cm[1,0]}, FP={cm[0,1]}")
    print()
    print(f"IMPROVEMENT:")
    print(f"  Accuracy: +{(acc-baseline_acc)*100:.2f}%")
    print(f"  FN reduced: {baseline_cm[1,0] - cm[1,0]}")
    print(f"  FP added: {cm[0,1] - baseline_cm[0,1]}")
    print()

    # Override details
    print("OVERRIDE DETAILS:")
    rescued = sum(1 for d in override_details if d["true_label"] == 1)
    new_fp = sum(1 for d in override_details if d["true_label"] == 0)
    print(f"  Total overrides: {len(override_details)}")
    print(f"  Rescued FN: {rescued}")
    print(f"  New FP: {new_fp}")
    print()

    for d in override_details:
        status = "RESCUED FN" if d["true_label"] == 1 else "NEW FP"
        print(f"  [{status}] idx={d['idx']} prob={d['prob']:.3f} conf={d['llm_conf']:.2f} source={d['source']}")

    print()
    print(classification_report(df["label"], pred, digits=4))

    # Save results
    results_df = df.copy()
    results_df["pred"] = pred
    results_df["source"] = sources
    results_df["explanation"] = explanations

    out_path = ROOT / "results" / "optimized_predictions.csv"
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {out_path}")

    metrics = {
        "confidence_threshold": CONF_THRESHOLD,
        "baseline_accuracy": float(baseline_acc),
        "optimized_accuracy": float(acc),
        "macro_f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "total_overrides": len(override_details),
        "rescued_fn": rescued,
        "new_fp": new_fp,
        "improvement_accuracy": float(acc - baseline_acc),
    }

    metrics_path = ROOT / "results" / "optimized_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
