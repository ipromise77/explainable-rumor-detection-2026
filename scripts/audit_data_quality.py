from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "rumer2026"
RESULTS_DIR = ROOT / "results"


def preprocess_text(text: str) -> str:
    """Same normalization rule used by src/rumor_detector.py."""
    text = html.unescape(str(text)).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URLTOKEN ", text)
    text = re.sub(r"@\w+", " USERTOKEN ", text)
    text = re.sub(r"#(\w+)", r" HASHTAG_\1 \1 ", text)
    return re.sub(r"\s+", " ", text).strip()


def find_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work.insert(0, "row_index", range(len(work)))
    work.insert(1, "csv_line_no", work["row_index"] + 2)
    work["normalized_text"] = work["text"].map(preprocess_text)

    conflict_keys = []
    for key, group in work.groupby("normalized_text", sort=False):
        if group["label"].nunique() > 1:
            conflict_keys.append(key)

    conflict_rows = work[work["normalized_text"].isin(conflict_keys)].copy()
    key_to_group = {key: i + 1 for i, key in enumerate(conflict_keys)}
    conflict_rows.insert(
        0, "conflict_group", conflict_rows["normalized_text"].map(key_to_group)
    )
    conflict_rows["label_meaning"] = conflict_rows["label"].map({0: "非谣言", 1: "谣言"})
    conflict_rows["same_raw_text_conflict"] = (
        conflict_rows.groupby("text")["label"].transform("nunique") > 1
    )
    reason_by_key = {}
    for key, group in conflict_rows.groupby("normalized_text", sort=False):
        if group["text"].nunique() == 1:
            reason = "原始文本完全相同，但同一文本被标注为0和1。"
        else:
            reason = "原始文本主要差异来自短链接；URL归一化后模型输入相同，但标签同时包含0和1。"
        reason_by_key[key] = reason
    conflict_rows["issue_reason"] = conflict_rows["normalized_text"].map(reason_by_key)
    conflict_rows = conflict_rows.sort_values(["conflict_group", "row_index"]).copy()
    conflict_rows.insert(0, "issue_no", range(1, len(conflict_rows) + 1))

    columns = [
        "issue_no",
        "conflict_group",
        "row_index",
        "csv_line_no",
        "id",
        "label",
        "label_meaning",
        "event",
        "same_raw_text_conflict",
        "issue_reason",
        "text",
        "normalized_text",
    ]
    return conflict_rows[columns]


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")

    train_conflicts = find_conflicts(train)
    val_conflicts = find_conflicts(val)

    train_out = RESULTS_DIR / "train_preprocess_label_conflicts.csv"
    val_out = RESULTS_DIR / "val_preprocess_label_conflicts.csv"
    summary_out = RESULTS_DIR / "data_quality_summary.json"
    clean_train_out = DATA_DIR / "train_clean.csv"

    train_conflicts.to_csv(train_out, index=False, encoding="utf-8-sig")
    val_conflicts.to_csv(val_out, index=False, encoding="utf-8-sig")
    if len(train_conflicts):
        train_clean = train.drop(index=train_conflicts["row_index"].astype(int)).reset_index(
            drop=True
        )
    else:
        train_clean = train.copy()
    train_clean.to_csv(clean_train_out, index=False, encoding="utf-8-sig")

    summary = {
        "normalization_rule": "html_unescape + lowercase + URLTOKEN + USERTOKEN + HASHTAG expansion + whitespace collapse",
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "train_conflict_groups": int(train_conflicts["conflict_group"].nunique())
        if len(train_conflicts)
        else 0,
        "train_conflict_rows": int(len(train_conflicts)),
        "train_clean_rows": int(len(train_clean)),
        "removed_train_rows": int(len(train) - len(train_clean)),
        "val_conflict_groups": int(val_conflicts["conflict_group"].nunique())
        if len(val_conflicts)
        else 0,
        "val_conflict_rows": int(len(val_conflicts)),
        "recommended_action": "Remove training rows whose normalized text maps to conflicting labels; keep val.csv unchanged for fair evaluation.",
        "outputs": {
            "train_conflict_table": str(train_out.relative_to(ROOT)),
            "val_conflict_table": str(val_out.relative_to(ROOT)),
            "clean_train": str(clean_train_out.relative_to(ROOT)),
        },
    }
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
