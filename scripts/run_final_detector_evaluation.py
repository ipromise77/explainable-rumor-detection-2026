from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.final_detector import FinalRumourDetectClass
from src.rumor_detector import load_model, predict_label, predict_proba


def main() -> None:
    df = pd.read_csv(ROOT / "rumer2026" / "val.csv")
    bundle = load_model()
    baseline_prob = predict_proba(bundle, df["text"])
    baseline_pred = predict_label(bundle, df["text"])

    detector = FinalRumourDetectClass()
    rows = []
    for _, row in df.iterrows():
        result = detector.predict(str(row["text"]))
        rows.append(
            {
                **row.to_dict(),
                "baseline_prob": float(baseline_prob[int(row.name)]),
                "baseline_pred": int(baseline_pred[int(row.name)]),
                "pred": int(result["label"]),
                "source": result["source"],
                "explanation": result["explanation"],
            }
        )

    out = pd.DataFrame(rows)
    pred = out["pred"].to_numpy()
    labels = df["label"].to_numpy()
    cm = confusion_matrix(labels, pred, labels=[0, 1])
    baseline_cm = confusion_matrix(labels, baseline_pred, labels=[0, 1])

    metrics = {
        "strategy": "FinalRumourDetectClass",
        "use_cached_llm": detector.use_cached_llm,
        "use_rules": detector.use_rules,
        "baseline_accuracy": float(accuracy_score(labels, baseline_pred)),
        "final_accuracy": float(accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, average="macro")),
        "confusion_matrix": cm.tolist(),
        "baseline_confusion_matrix": baseline_cm.tolist(),
        "source_counts": out["source"].value_counts().to_dict(),
        "classification_report": classification_report(
            labels, pred, digits=4, output_dict=True
        ),
        "caveat": (
            "This is an engineered optional detector. Its default evaluation does "
            "not use ignored LLM cache files, so it is reproducible from the "
            "committed local model, data, and rule signals."
        ),
    }

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "final_detector_predictions.csv"
    metrics_path = results_dir / "final_detector_metrics.json"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote {out_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
