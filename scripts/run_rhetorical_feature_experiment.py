from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "rumer2026"
RESULTS_DIR = ROOT / "results"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rumor_detector import MODEL_CONFIGS, preprocess_many, predict_proba, train_ensemble


ANON_WORDS = ("anonymous", "unnamed", "unidentified", "undisclosed")
LEAK_WORDS = ("obtained", "acquired", "leaked", "uncovered", "secured")
SMEAR_WORDS = ("smear", "slander", "defame", "discredit")
HIDE_WORDS = ("hiding", "concealing", "covering up", "cover up", "withholding", "suppressing")
DEVOLVE_WORDS = ("devolved", "deteriorated", "descended", "degenerated")

RHETORICAL_FEATURE_NAMES = [
    "anonymous_leak_signal",
    "smear_campaign_signal",
    "concealment_signal",
    "extreme_worst_signal",
    "rhetorical_signal_count",
]

COMPONENT_FEATURE_NAMES = [
    "anonymous_word_any",
    "leak_word_any",
    "anonymous_leak_signal",
    "smear_word_any",
    "campaign_word",
    "smear_campaign_signal",
    "concealment_signal",
    "devolve_word_any",
    "worst_word",
    "extreme_worst_signal",
    "component_signal_count",
]


class RhetoricalPatternTransformer(BaseEstimator, TransformerMixin):
    """Binary/counted features for pre-declared rumor-rhetoric patterns."""

    def __init__(self, scale: float = 1.0, feature_set: str = "strict"):
        self.scale = scale
        self.feature_set = feature_set

    def fit(self, x: Iterable[str], y: Iterable[int] | None = None) -> "RhetoricalPatternTransformer":
        return self

    def transform(self, x: Iterable[str]) -> sparse.csr_matrix:
        rows = []
        for text in x:
            rows.append(extract_rhetorical_features(str(text), feature_set=self.feature_set))
        return sparse.csr_matrix(np.asarray(rows, dtype=float) * float(self.scale))

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        return np.asarray(feature_names(self.feature_set), dtype=object)


def has_word(text: str, word: str) -> bool:
    return re.search(rf"(?<![a-z0-9_]){re.escape(word)}(?![a-z0-9_])", text) is not None


def has_any(text: str, words: tuple[str, ...]) -> bool:
    return any(has_word(text, word) for word in words)


def feature_names(feature_set: str) -> list[str]:
    if feature_set == "strict":
        return RHETORICAL_FEATURE_NAMES
    if feature_set == "component":
        return COMPONENT_FEATURE_NAMES
    raise ValueError(f"Unknown feature_set: {feature_set}")


def extract_rhetorical_features(text: str, feature_set: str = "strict") -> list[float]:
    t = str(text).lower()
    anon_any = has_any(t, ANON_WORDS)
    leak_any = has_any(t, LEAK_WORDS)
    smear_any = has_any(t, SMEAR_WORDS)
    campaign = has_word(t, "campaign")
    concealment = has_any(t, HIDE_WORDS)
    devolve_any = has_any(t, DEVOLVE_WORDS)
    worst = has_word(t, "worst")

    if feature_set == "component":
        anonymous_leak = anon_any and leak_any
        smear_campaign = smear_any and campaign
        extreme_worst = devolve_any and worst
        flags = [
            anon_any,
            leak_any,
            anonymous_leak,
            smear_any,
            campaign,
            smear_campaign,
            concealment,
            devolve_any,
            worst,
            extreme_worst,
        ]
        return [float(flag) for flag in flags] + [float(sum(flags))]

    anonymous_leak = has_any(t, ANON_WORDS) and has_any(t, LEAK_WORDS)
    smear_campaign = has_any(t, SMEAR_WORDS) and has_word(t, "campaign")
    extreme_worst = has_any(t, DEVOLVE_WORDS) and has_word(t, "worst")
    flags = [anonymous_leak, smear_campaign, concealment, extreme_worst]
    return [float(flag) for flag in flags] + [float(sum(flags))]


def build_rhetorical_pipeline(config: dict, scale: float, feature_set: str) -> Pipeline:
    word_vectorizer = TfidfVectorizer(
        stop_words=config["stop_words"],
        ngram_range=tuple(config["word_ngram"]),
        min_df=1,
        sublinear_tf=True,
        norm="l2",
        max_features=40000,
        token_pattern=r"(?u)\b\w\w+\b|[#@]\w+",
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=tuple(config["char_ngram"]),
        min_df=1,
        sublinear_tf=True,
        norm="l2",
        max_features=40000,
    )
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        ("word", word_vectorizer),
                        ("char", char_vectorizer),
                        ("rhetoric", RhetoricalPatternTransformer(scale=scale, feature_set=feature_set)),
                    ]
                ),
            ),
            ("classifier", LogisticRegression(max_iter=5000, C=config["c"], solver="liblinear")),
        ]
    )


def train_rhetorical_ensemble(
    texts: Iterable[str],
    labels: Iterable[int],
    scale: float,
    feature_set: str,
) -> dict:
    clean_texts = preprocess_many(texts)
    labels = np.asarray(list(labels), dtype=int)
    models = []
    for config in MODEL_CONFIGS:
        pipeline = build_rhetorical_pipeline(config, scale=scale, feature_set=feature_set)
        pipeline.fit(clean_texts, labels)
        models.append(
            {
                "name": f"{config['name']}_{feature_set}_rhetoric_s{scale:g}",
                "config": config,
                "pipeline": pipeline,
            }
        )
    return {
        "version": "2026.rhetorical-experiment",
        "threshold": 0.5,
        "label_meaning": {"0": "non-rumor", "1": "rumor"},
        "rhetorical_feature_scale": float(scale),
        "rhetorical_feature_set": feature_set,
        "models": models,
    }


def labels_from_prob(prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (prob >= threshold).astype(int)


def metrics(y_true: np.ndarray, pred: np.ndarray, strategy: str) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "strategy": strategy,
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "rumor_f1": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "precision_1": float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
        "recall_1": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def active_feature_summary(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    rows = []
    labels = df["label"].to_numpy()
    names = feature_names(feature_set)
    for name, values in zip(
        names,
        np.asarray([extract_rhetorical_features(text, feature_set=feature_set) for text in df["text"]]).T,
    ):
        active = values > 0
        rows.append(
            {
                "feature": name,
                "feature_set": feature_set,
                "active_rows": int(active.sum()),
                "active_rumor": int(((labels == 1) & active).sum()),
                "active_non_rumor": int(((labels == 0) & active).sum()),
                "rumor_rate_when_active": float(labels[active].mean()) if active.any() else None,
            }
        )
    return pd.DataFrame(rows)


def run_cv(
    train_df: pd.DataFrame,
    scales: list[float],
    feature_sets: list[str],
    n_splits: int = 5,
) -> pd.DataFrame:
    rows = []
    labels = train_df["label"].to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (train_idx, dev_idx) in enumerate(skf.split(train_df["text"], labels), start=1):
        fold_train = train_df.iloc[train_idx]
        fold_dev = train_df.iloc[dev_idx]
        y_dev = fold_dev["label"].to_numpy()

        baseline = train_ensemble(fold_train["text"], fold_train["label"])
        baseline_pred = labels_from_prob(predict_proba(baseline, fold_dev["text"]))
        base_row = metrics(y_dev, baseline_pred, "baseline")
        base_row.update({"fold": int(fold), "scale": 0.0, "feature_set": "baseline"})
        rows.append(base_row)

        for feature_set in feature_sets:
            for scale in scales:
                bundle = train_rhetorical_ensemble(
                    fold_train["text"],
                    fold_train["label"],
                    scale=scale,
                    feature_set=feature_set,
                )
                pred = labels_from_prob(predict_proba(bundle, fold_dev["text"]))
                row = metrics(y_dev, pred, f"{feature_set}_rhetorical_scale_{scale:g}")
                row.update({"fold": int(fold), "scale": float(scale), "feature_set": feature_set})
                rows.append(row)
        print(f"finished fold={fold}")
    return pd.DataFrame(rows)


def aggregate_cv(cv_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["accuracy", "macro_f1", "rumor_f1", "precision_1", "recall_1", "fp", "fn"]
    rows = []
    for strategy, group in cv_df.groupby("strategy", sort=False):
        row = {
            "strategy": strategy,
            "feature_set": str(group["feature_set"].iloc[0]),
            "scale": float(group["scale"].iloc[0]),
            "folds": int(len(group)),
        }
        for metric in metric_cols:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["accuracy_mean", "macro_f1_mean"], ascending=False)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATA_DIR / "train_clean.csv"
    if not train_path.exists():
        train_path = DATA_DIR / "train.csv"
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    scales = [0.5, 1.0, 2.0, 4.0]
    feature_sets = ["strict", "component"]

    train_feature_stats = pd.concat(
        [active_feature_summary(train_df, feature_set=feature_set) for feature_set in feature_sets],
        ignore_index=True,
    )
    val_feature_stats = pd.concat(
        [active_feature_summary(val_df, feature_set=feature_set) for feature_set in feature_sets],
        ignore_index=True,
    )
    cv_df = run_cv(train_df, scales=scales, feature_sets=feature_sets, n_splits=5)
    cv_summary = aggregate_cv(cv_df)

    baseline_cv = cv_summary[cv_summary["strategy"] == "baseline"].iloc[0]
    candidates = cv_summary[cv_summary["strategy"] != "baseline"].copy()
    candidates["delta_accuracy_vs_baseline"] = candidates["accuracy_mean"] - float(baseline_cv["accuracy_mean"])
    candidates["delta_macro_f1_vs_baseline"] = candidates["macro_f1_mean"] - float(baseline_cv["macro_f1_mean"])
    best = candidates.sort_values(["accuracy_mean", "macro_f1_mean", "rumor_f1_mean"], ascending=False).iloc[0]
    selected_scale = float(best["scale"])

    baseline_bundle = train_ensemble(train_df["text"], train_df["label"])
    baseline_val_pred = labels_from_prob(predict_proba(baseline_bundle, val_df["text"]))
    baseline_val = metrics(val_df["label"].to_numpy(), baseline_val_pred, "baseline_val")

    selected_feature_set = str(best["feature_set"])
    rhetorical_bundle = train_rhetorical_ensemble(
        train_df["text"],
        train_df["label"],
        scale=selected_scale,
        feature_set=selected_feature_set,
    )
    rhetorical_val_prob = predict_proba(rhetorical_bundle, val_df["text"])
    rhetorical_val_pred = labels_from_prob(rhetorical_val_prob)
    rhetorical_val = metrics(val_df["label"].to_numpy(), rhetorical_val_pred, f"rhetorical_val_scale_{selected_scale:g}")
    rhetorical_val.update(
        {
            "selected_by": "best 5-fold train_clean accuracy; ties by macro_f1 and rumor_f1",
            "selected_scale": selected_scale,
            "selected_feature_set": selected_feature_set,
            "delta_accuracy_vs_baseline": rhetorical_val["accuracy"] - baseline_val["accuracy"],
            "delta_macro_f1_vs_baseline": rhetorical_val["macro_f1"] - baseline_val["macro_f1"],
            "changed_predictions": int((baseline_val_pred != rhetorical_val_pred).sum()),
        }
    )

    val_predictions = val_df.copy()
    val_predictions["baseline_pred"] = baseline_val_pred
    val_predictions["rhetorical_pred"] = rhetorical_val_pred
    val_predictions["rhetorical_prob_rumor"] = rhetorical_val_prob
    val_predictions["prediction_changed"] = baseline_val_pred != rhetorical_val_pred

    outputs = {
        "train_feature_stats": RESULTS_DIR / "rhetorical_feature_train_stats.csv",
        "val_feature_stats": RESULTS_DIR / "rhetorical_feature_val_stats.csv",
        "cv": RESULTS_DIR / "rhetorical_feature_cv.csv",
        "cv_summary": RESULTS_DIR / "rhetorical_feature_cv_summary.csv",
        "val_predictions": RESULTS_DIR / "rhetorical_feature_val_predictions.csv",
        "summary": RESULTS_DIR / "rhetorical_feature_experiment_summary.json",
    }
    train_feature_stats.to_csv(outputs["train_feature_stats"], index=False, encoding="utf-8-sig")
    val_feature_stats.to_csv(outputs["val_feature_stats"], index=False, encoding="utf-8-sig")
    cv_df.to_csv(outputs["cv"], index=False, encoding="utf-8-sig")
    cv_summary.to_csv(outputs["cv_summary"], index=False, encoding="utf-8-sig")
    val_predictions.to_csv(outputs["val_predictions"], index=False, encoding="utf-8-sig")

    summary = {
        "train_file": str(train_path.relative_to(ROOT)),
        "val_file": "rumer2026/val.csv",
        "rhetorical_patterns": {
            "anonymous_leak_signal": {"anonymous_words": ANON_WORDS, "leak_words": LEAK_WORDS},
            "smear_campaign_signal": {"smear_words": SMEAR_WORDS, "required_word": "campaign"},
            "concealment_signal": {"hide_words": HIDE_WORDS},
            "extreme_worst_signal": {"devolve_words": DEVOLVE_WORDS, "required_word": "worst"},
        },
        "selection_rule": "select scale using train_clean 5-fold only; val is used once for final comparison",
        "baseline_cv": baseline_cv.to_dict(),
        "best_rhetorical_cv": best.to_dict(),
        "baseline_val": baseline_val,
        "rhetorical_val_selected_by_cv": rhetorical_val,
        "outputs": {key: str(path.relative_to(ROOT)) for key, path in outputs.items()},
    }
    outputs["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
