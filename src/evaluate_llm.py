from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .llm_enhanced import LLMEnhancedRumorDetector
from .rumor_detector import evidence_for_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LLM-enhanced rumor detection.")
    parser.add_argument("--csv", default="rumer2026/val.csv")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N rows.")
    parser.add_argument("--force-llm", action="store_true", help="Call LLM for every evaluated row.")
    parser.add_argument(
        "--only-uncertain",
        action="store_true",
        help="Evaluate only rows whose local probability falls in [low, high].",
    )
    parser.add_argument("--low", type=float, default=0.30)
    parser.add_argument("--high", type=float, default=0.70)
    parser.add_argument("--allow-override", action="store_true")
    parser.add_argument("--override-confidence", type=float, default=0.85)
    parser.add_argument("--out", default="results/llm_val_predictions.csv")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    detector = LLMEnhancedRumorDetector(
        low=args.low,
        high=args.high,
        allow_override=args.allow_override,
        override_confidence=args.override_confidence,
    )
    if args.only_uncertain:
        local_records = []
        for _, row in df.iterrows():
            local = evidence_for_text(detector.bundle, str(row["text"]), top_k=1)
            if args.low <= float(local["prob_rumor"]) <= args.high:
                local_records.append(row)
        df = pd.DataFrame(local_records).reset_index(drop=True)
    if args.limit is not None:
        df = df.head(args.limit).copy()
    records = []
    for i, row in df.iterrows():
        result = detector.predict(str(row["text"]), force_llm=args.force_llm)
        records.append(
            {
                "id": row["id"],
                "text": row["text"],
                "label": int(row["label"]),
                "pred": int(result["label"]),
                "source": result["source"],
                "prob_rumor": result["prob_rumor"],
                "explanation": result["explanation"],
                "local_label": result.get("local_label", result["label"]),
                "llm_label": result.get("llm_label", ""),
                "llm_confidence": result.get("llm_confidence", ""),
            }
        )
        if not args.quiet:
            print(
                f"[{len(records)}/{len(df)}] gold={int(row['label'])} "
                f"pred={int(result['label'])} source={result['source']} "
                f"prob={float(result['prob_rumor']):.4f}"
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame(records)
    pred_df.to_csv(out, index=False, encoding="utf-8-sig")

    y_true = pred_df["label"]
    y_pred = pred_df["pred"]
    metrics = {
        "size": int(len(pred_df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "llm_calls": int(pred_df["source"].astype(str).str.startswith("llm").sum()),
        "llm_overrides": int((pred_df["source"] == "llm_override").sum()),
        "local_only": int((pred_df["source"] == "local").sum()),
    }
    metrics_path = out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(classification_report(y_true, y_pred, digits=4))
    print(f"Wrote {out}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
