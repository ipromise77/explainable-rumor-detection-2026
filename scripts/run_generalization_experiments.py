from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "rumer2026"
RESULTS_DIR = ROOT / "results"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rumor_detector import evidence_for_text, load_model, predict_proba, train_ensemble


def labels_from_prob(prob: np.ndarray, threshold: float) -> np.ndarray:
    return (prob >= threshold).astype(int)


def binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    pred = labels_from_prob(prob, threshold)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, labels=[0, 1], average="macro", zero_division=0)),
        "rumor_f1": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "precision_0": float(precision_score(y_true, pred, pos_label=0, zero_division=0)),
        "recall_0": float(recall_score(y_true, pred, pos_label=0, zero_division=0)),
        "precision_1": float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
        "recall_1": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_event_analysis(bundle: dict, val_df: pd.DataFrame) -> pd.DataFrame:
    prob = predict_proba(bundle, val_df["text"])
    pred = labels_from_prob(prob, float(bundle.get("threshold", 0.5)))
    work = val_df.copy()
    work["prob_rumor"] = prob
    work["pred"] = pred
    rows = []
    for event, group in work.groupby("event", sort=True):
        y = group["label"].to_numpy()
        p = group["pred"].to_numpy()
        tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
        rows.append(
            {
                "event": int(event),
                "support": int(len(group)),
                "non_rumor": int((y == 0).sum()),
                "rumor": int((y == 1).sum()),
                "accuracy": float(accuracy_score(y, p)),
                "macro_f1": float(
                    f1_score(y, p, labels=[0, 1], average="macro", zero_division=0)
                ),
                "rumor_f1": float(f1_score(y, p, pos_label=1, zero_division=0)),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "mean_prob_rumor": float(group["prob_rumor"].mean()),
            }
        )
    return pd.DataFrame(rows)


def format_evidence_items(items: list[dict]) -> str:
    if not items:
        return ""
    return "；".join(
        f"{item['feature']}({float(item['score']):.3f})" for item in items
    )


def run_focus_event_error_analysis(
    bundle: dict,
    val_df: pd.DataFrame,
    events: tuple[int, ...] = (0, 1),
) -> tuple[pd.DataFrame, dict]:
    prob_all = predict_proba(bundle, val_df["text"])
    pred_all = labels_from_prob(prob_all, float(bundle.get("threshold", 0.5)))

    work = val_df.copy()
    work["val_sample_no"] = np.arange(1, len(work) + 1)
    work["prob_rumor"] = prob_all
    work["pred"] = pred_all
    work["confidence"] = np.where(work["pred"] == 1, work["prob_rumor"], 1.0 - work["prob_rumor"])
    work["correct"] = work["label"] == work["pred"]
    work["error_type"] = np.select(
        [
            (work["label"] == 1) & (work["pred"] == 0),
            (work["label"] == 0) & (work["pred"] == 1),
        ],
        ["false_negative", "false_positive"],
        default="correct",
    )

    focus = work[work["event"].isin(events)].copy()
    focus["abs_margin_from_0_5"] = (focus["prob_rumor"] - 0.5).abs()

    rumor_evidence = []
    non_rumor_evidence = []
    for text in focus["text"]:
        evidence = evidence_for_text(bundle, text, top_k=3)
        rumor_evidence.append(format_evidence_items(evidence["rumor_evidence"]))
        non_rumor_evidence.append(format_evidence_items(evidence["non_rumor_evidence"]))
    focus["top_rumor_evidence"] = rumor_evidence
    focus["top_non_rumor_evidence"] = non_rumor_evidence

    focus = focus[
        [
            "val_sample_no",
            "id",
            "event",
            "label",
            "pred",
            "prob_rumor",
            "confidence",
            "correct",
            "error_type",
            "abs_margin_from_0_5",
            "top_rumor_evidence",
            "top_non_rumor_evidence",
            "text",
        ]
    ].sort_values(["event", "correct", "error_type", "val_sample_no"])

    summary = {}
    for event, group in focus.groupby("event", sort=True):
        error_counts = group["error_type"].value_counts().to_dict()
        wrong = group[~group["correct"]]
        summary[str(int(event))] = {
            "support": int(len(group)),
            "errors": int(len(wrong)),
            "false_negative": int(error_counts.get("false_negative", 0)),
            "false_positive": int(error_counts.get("false_positive", 0)),
            "correct": int(error_counts.get("correct", 0)),
            "mean_prob_rumor_for_gold_rumor": float(
                group.loc[group["label"] == 1, "prob_rumor"].mean()
            )
            if (group["label"] == 1).any()
            else None,
            "mean_prob_rumor_for_gold_non_rumor": float(
                group.loc[group["label"] == 0, "prob_rumor"].mean()
            )
            if (group["label"] == 0).any()
            else None,
        }
    return focus, summary


def run_threshold_experiment(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    inner_train, dev = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=train_df["label"],
    )
    inner_bundle = train_ensemble(inner_train["text"], inner_train["label"])
    dev_prob = predict_proba(inner_bundle, dev["text"])
    y_dev = dev["label"].to_numpy()

    threshold_rows = []
    for threshold in np.round(np.arange(0.20, 0.801, 0.01), 2):
        row = binary_metrics(y_dev, dev_prob, float(threshold))
        threshold_rows.append(row)
    thresholds = pd.DataFrame(threshold_rows)
    best = thresholds.sort_values(
        ["accuracy", "macro_f1", "rumor_f1"], ascending=False
    ).iloc[0].to_dict()

    full_bundle = load_model(ROOT / "models" / "rumor_ensemble.joblib")
    val_prob = predict_proba(full_bundle, val_df["text"])
    y_val = val_df["label"].to_numpy()
    val_at_05 = binary_metrics(y_val, val_prob, 0.5)
    val_at_best = binary_metrics(y_val, val_prob, float(best["threshold"]))

    summary = {
        "inner_train_size": int(len(inner_train)),
        "dev_size": int(len(dev)),
        "selection_rule": "best dev accuracy; ties broken by macro_f1 then rumor_f1",
        "best_dev_threshold": best,
        "val_at_threshold_0_50": val_at_05,
        "val_at_dev_selected_threshold": val_at_best,
    }
    return thresholds, summary


def run_cross_validation(train_df: pd.DataFrame, n_splits: int = 5) -> tuple[pd.DataFrame, dict]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows = []
    labels = train_df["label"].to_numpy()
    for fold, (train_idx, dev_idx) in enumerate(skf.split(train_df["text"], labels), start=1):
        fold_train = train_df.iloc[train_idx]
        fold_dev = train_df.iloc[dev_idx]
        bundle = train_ensemble(fold_train["text"], fold_train["label"])
        prob = predict_proba(bundle, fold_dev["text"])
        row = binary_metrics(fold_dev["label"].to_numpy(), prob, 0.5)
        row.update(
            {
                "fold": int(fold),
                "train_size": int(len(fold_train)),
                "dev_size": int(len(fold_dev)),
            }
        )
        rows.append(row)
        print(
            f"fold={fold} accuracy={row['accuracy']:.4f} "
            f"macro_f1={row['macro_f1']:.4f} rumor_f1={row['rumor_f1']:.4f}"
        )
    cv = pd.DataFrame(rows)
    metric_cols = ["accuracy", "macro_f1", "rumor_f1", "precision_1", "recall_1"]
    summary = {
        "n_splits": int(n_splits),
        "rows": int(len(train_df)),
        "metrics": {
            metric: {
                "mean": float(cv[metric].mean()),
                "std": float(cv[metric].std(ddof=1)),
                "min": float(cv[metric].min()),
                "max": float(cv[metric].max()),
            }
            for metric in metric_cols
        },
    }
    return cv, summary


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATA_DIR / "train_clean.csv"
    if not train_path.exists():
        train_path = DATA_DIR / "train.csv"
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    bundle = load_model(ROOT / "models" / "rumor_ensemble.joblib")

    event_df = run_event_analysis(bundle, val_df)
    focus_error_df, focus_error_summary = run_focus_event_error_analysis(bundle, val_df)
    threshold_df, threshold_summary = run_threshold_experiment(train_df, val_df)
    cv_df, cv_summary = run_cross_validation(train_df, n_splits=5)

    event_out = RESULTS_DIR / "event_accuracy.csv"
    focus_error_out = RESULTS_DIR / "event_0_1_error_analysis.csv"
    threshold_out = RESULTS_DIR / "dev_threshold_experiment.csv"
    cv_out = RESULTS_DIR / "cv_results.csv"
    summary_out = RESULTS_DIR / "generalization_experiments_summary.json"

    event_df.to_csv(event_out, index=False, encoding="utf-8-sig")
    focus_error_df.to_csv(focus_error_out, index=False, encoding="utf-8-sig")
    threshold_df.to_csv(threshold_out, index=False, encoding="utf-8-sig")
    cv_df.to_csv(cv_out, index=False, encoding="utf-8-sig")

    summary = {
        "train_file": str(train_path.relative_to(ROOT)),
        "val_file": "rumer2026/val.csv",
        "event_analysis": {
            "output": str(event_out.relative_to(ROOT)),
            "focus_error_output": str(focus_error_out.relative_to(ROOT)),
            "focus_error_summary": focus_error_summary,
            "best_event_by_accuracy": event_df.sort_values(
                ["accuracy", "support"], ascending=False
            ).iloc[0].to_dict(),
            "worst_event_by_accuracy": event_df.sort_values(
                ["accuracy", "support"], ascending=[True, False]
            ).iloc[0].to_dict(),
        },
        "threshold_experiment": {
            "output": str(threshold_out.relative_to(ROOT)),
            **threshold_summary,
        },
        "cross_validation": {
            "output": str(cv_out.relative_to(ROOT)),
            **cv_summary,
        },
    }
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
