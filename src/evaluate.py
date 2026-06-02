from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .rumor_detector import explain_text, load_model, predict_label, predict_proba


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved rumor detector.")
    parser.add_argument("--csv", default="rumer2026/val.csv", help="CSV file with text and label columns.")
    parser.add_argument("--model", default="models/rumor_ensemble.joblib")
    parser.add_argument("--show-examples", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_model(args.model)
    df = pd.read_csv(Path(args.csv))
    prob = predict_proba(bundle, df["text"])
    pred = predict_label(bundle, df["text"])

    print(f"Accuracy: {accuracy_score(df['label'], pred):.4f}")
    print(f"Macro-F1: {f1_score(df['label'], pred, average='macro'):.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(df["label"], pred))
    print(classification_report(df["label"], pred, digits=4))

    for i in range(min(args.show_examples, len(df))):
        print("-" * 80)
        print("Text:", df.loc[i, "text"])
        print("Gold:", int(df.loc[i, "label"]), "Pred:", int(pred[i]), f"Prob_rumor: {prob[i]:.4f}")
        print(explain_text(bundle, df.loc[i, "text"]))


if __name__ == "__main__":
    main()

