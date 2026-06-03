from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .rumor_detector import explain_text, predict_label, predict_proba, save_model, train_ensemble


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the explainable rumor detector.")
    parser.add_argument("--data-dir", default="rumer2026", help="Directory containing train.csv and val.csv.")
    parser.add_argument(
        "--train-file",
        default=None,
        help="Training CSV filename. Defaults to train_clean.csv if it exists, otherwise train.csv.",
    )
    parser.add_argument("--model-out", default="models/rumor_ensemble.joblib")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--with-explanations",
        action="store_true",
        help="Also write explanations for each validation example.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_file = args.train_file
    if train_file is None:
        train_file = "train_clean.csv" if (data_dir / "train_clean.csv").exists() else "train.csv"

    train_df = pd.read_csv(data_dir / train_file)
    val_df = pd.read_csv(data_dir / "val.csv")

    bundle = train_ensemble(train_df["text"], train_df["label"])
    val_prob = predict_proba(bundle, val_df["text"])
    val_pred = predict_label(bundle, val_df["text"])

    metrics = {
        "train_size": int(len(train_df)),
        "train_file": train_file,
        "val_size": int(len(val_df)),
        "threshold": float(bundle["threshold"]),
        "accuracy": float(accuracy_score(val_df["label"], val_pred)),
        "macro_f1": float(f1_score(val_df["label"], val_pred, average="macro")),
        "rumor_f1": float(f1_score(val_df["label"], val_pred)),
        "confusion_matrix": confusion_matrix(val_df["label"], val_pred).tolist(),
        "classification_report": classification_report(
            val_df["label"], val_pred, digits=4, output_dict=True
        ),
        "model_names": [item["name"] for item in bundle["models"]],
    }
    bundle["validation_metrics"] = metrics
    save_model(bundle, args.model_out)

    predictions = val_df.copy()
    predictions["prob_rumor"] = val_prob
    predictions["pred"] = val_pred
    if args.with_explanations:
        predictions["explanation"] = [explain_text(bundle, text) for text in predictions["text"]]
    predictions.to_csv(results_dir / "val_predictions.csv", index=False, encoding="utf-8-sig")

    with (results_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved model to {args.model_out}")
    print(f"Saved metrics to {results_dir / 'metrics.json'}")
    print(f"Val accuracy: {metrics['accuracy']:.4f}")
    print(f"Val macro-F1: {metrics['macro_f1']:.4f}")
    print(classification_report(val_df["label"], val_pred, digits=4))


if __name__ == "__main__":
    main()
