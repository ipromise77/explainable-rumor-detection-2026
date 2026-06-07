from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "rumer2026"
RESULTS_DIR = ROOT / "results"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rumor_detector import evidence_for_text, load_model, predict_proba, preprocess_many


PROMPT_VERSION = "fn_recall_review_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run low-confidence false-negative recall review with the school LLM."
    )
    parser.add_argument("--csv", default="rumer2026/val.csv")
    parser.add_argument("--model", default="models/rumor_ensemble.joblib")
    parser.add_argument("--data-dir", default="rumer2026")
    parser.add_argument("--min-low", type=float, default=0.20)
    parser.add_argument("--high", type=float, default=0.50)
    parser.add_argument("--default-low", type=float, default=0.20)
    parser.add_argument("--default-confidence", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--cache", default="results/fn_review_llm_cache.jsonl")
    parser.add_argument("--out-prefix", default="results/fn_recall_review")
    parser.add_argument(
        "--reuse-cache-only",
        action="store_true",
        help="Do not call the API; fail if any candidate is missing from cache.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Review only the first N candidates. Intended for debugging.",
    )
    return parser.parse_args()


def text_hash(text: str) -> str:
    payload = f"{PROMPT_VERSION}\n{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def read_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    cache: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cache[row["hash"]] = row["result"]
    return cache


def append_cache(path: Path, text: str, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"hash": text_hash(text), "prompt_version": PROMPT_VERSION, "result": result},
                ensure_ascii=False,
            )
            + "\n"
        )


def format_items(items: list[dict]) -> list[dict]:
    return [
        {"feature": item["feature"], "score": round(float(item["score"]), 4)}
        for item in items
    ]


def build_retriever(bundle: dict, data_dir: Path) -> tuple[pd.DataFrame, Any, Any]:
    train_path = data_dir / "train_clean.csv"
    if not train_path.exists():
        train_path = data_dir / "train.csv"
    train_df = pd.read_csv(train_path)
    features = bundle["models"][0]["pipeline"].named_steps["features"]
    train_matrix = features.transform(preprocess_many(train_df["text"]))
    return train_df, features, train_matrix


def retrieve_examples(
    text: str,
    train_df: pd.DataFrame,
    features: Any,
    train_matrix: Any,
    top_k: int,
) -> list[dict]:
    query = features.transform(preprocess_many([text]))
    scores = (train_matrix @ query.T).toarray().ravel()
    order = np.argsort(scores)[::-1][:top_k]
    examples = []
    for idx in order:
        row = train_df.iloc[int(idx)]
        examples.append(
            {
                "label": int(row["label"]),
                "similarity": round(float(scores[idx]), 4),
                "text": str(row["text"])[:280],
            }
        )
    return examples


def prompt_messages(text: str, local: dict, examples: list[dict]) -> list[dict]:
    system = (
        "你是一个课程项目中的谣言检测复核器。标签定义：0=非谣言，1=谣言。"
        "当前任务只复核本地模型判为0但可能漏报谣言的样本。"
        "只有当文本线索和相似训练样本都强烈支持谣言时，才建议改为1；"
        "如果证据不足或只是一般新闻陈述，必须保持0。"
        "不要使用或猜测验证集真实标签，不要编造外部事实。只输出JSON。"
    )
    user_payload = {
        "task": "复核该推文是否应从本地模型的0改为1。",
        "output_schema": {
            "review_label": "integer, only 0 or 1",
            "confidence": "number between 0 and 1",
            "reason": "one or two concise Chinese sentences",
        },
        "decision_rules": [
            "review_label=1表示建议覆盖本地模型并判为谣言。",
            "review_label=0表示保持本地模型的非谣言判断。",
            "宁可少覆盖，也不要把证据不足的非谣言改成谣言。",
            "reason必须引用输入文本、本地证据或相似训练样本中的具体线索。",
        ],
        "tweet": text,
        "local_model": local,
        "similar_labeled_train_examples": examples,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def call_llm(client: OpenAI, model: str, text: str, local: dict, examples: list[dict]) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=prompt_messages(text, local, examples),
        temperature=0,
    )
    parsed = extract_json(response.choices[0].message.content or "")
    label = int(parsed.get("review_label", 0))
    if label not in (0, 1):
        label = 0
    confidence = float(parsed.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))
    return {
        "review_label": label,
        "confidence": confidence,
        "reason": str(parsed.get("reason", "")).strip(),
    }


def metric_row(y_true: np.ndarray, pred: np.ndarray, name: str) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "strategy": name,
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


def apply_review_policy(
    base: pd.DataFrame,
    review: pd.DataFrame,
    low: float,
    high: float,
    confidence_threshold: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    pred = base["local_pred"].to_numpy().copy()
    selected = review[
        (review["prob_rumor"] >= low)
        & (review["prob_rumor"] < high)
        & (review["review_label"] == 1)
        & (review["llm_confidence"] >= confidence_threshold)
    ].copy()
    if not selected.empty:
        pred[selected["row_index"].astype(int).to_numpy()] = 1
    return pred, selected


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("SJTU_API_KEY", "").strip()
    base_url = os.getenv("SJTU_BASE_URL", "").strip()
    model_name = os.getenv("SJTU_MODEL", "deepseek-reasoner").strip()
    if args.reuse_cache_only:
        client = None
    else:
        if not api_key or "请在这里" in api_key:
            raise RuntimeError("SJTU_API_KEY is missing. Please fill it in .env first.")
        if not base_url or "请在这里" in base_url:
            raise RuntimeError("SJTU_BASE_URL is missing. Please fill it in .env first.")
        client = OpenAI(api_key=api_key, base_url=base_url)

    bundle = load_model(ROOT / args.model)
    val_df = pd.read_csv(ROOT / args.csv).reset_index(drop=True)
    train_df, features, train_matrix = build_retriever(bundle, ROOT / args.data_dir)

    prob = predict_proba(bundle, val_df["text"])
    local_pred = (prob >= float(bundle.get("threshold", 0.5))).astype(int)
    base = val_df.copy()
    base["row_index"] = np.arange(len(base))
    base["prob_rumor"] = prob
    base["local_pred"] = local_pred
    base["local_correct"] = base["label"] == base["local_pred"]

    candidates = base[
        (base["local_pred"] == 0)
        & (base["prob_rumor"] >= args.min_low)
        & (base["prob_rumor"] < args.high)
    ].copy()
    candidates = candidates.sort_values("prob_rumor", ascending=False)
    if args.limit is not None:
        candidates = candidates.head(args.limit).copy()

    cache_path = ROOT / args.cache
    cache = read_cache(cache_path)
    review_records = []
    total = len(candidates)
    for done, (_, row) in enumerate(candidates.iterrows(), start=1):
        text = str(row["text"])
        local_evidence = evidence_for_text(bundle, text, top_k=5)
        local_payload = {
            "label": int(row["local_pred"]),
            "prob_rumor": round(float(row["prob_rumor"]), 4),
            "confidence": round(1.0 - float(row["prob_rumor"]), 4),
            "rumor_evidence": format_items(local_evidence["rumor_evidence"]),
            "non_rumor_evidence": format_items(local_evidence["non_rumor_evidence"]),
        }
        examples = retrieve_examples(text, train_df, features, train_matrix, args.top_k)
        key = text_hash(text)
        if key in cache:
            llm_result = cache[key]
            source = "cache"
        else:
            if args.reuse_cache_only or client is None:
                raise RuntimeError(f"Missing cache for candidate id={row['id']}")
            llm_result = call_llm(client, model_name, text, local_payload, examples)
            append_cache(cache_path, text, llm_result)
            cache[key] = llm_result
            source = "api"
        review_records.append(
            {
                "row_index": int(row["row_index"]),
                "id": row["id"],
                "event": int(row["event"]) if "event" in row else "",
                "label": int(row["label"]),
                "local_pred": int(row["local_pred"]),
                "prob_rumor": float(row["prob_rumor"]),
                "review_label": int(llm_result["review_label"]),
                "llm_confidence": float(llm_result["confidence"]),
                "review_source": source,
                "would_rescue_false_negative": bool(
                    int(row["label"]) == 1 and int(llm_result["review_label"]) == 1
                ),
                "would_create_false_positive": bool(
                    int(row["label"]) == 0 and int(llm_result["review_label"]) == 1
                ),
                "reason": llm_result["reason"],
                "similar_examples": json.dumps(examples, ensure_ascii=False),
                "text": text,
            }
        )
        print(
            f"[{done}/{total}] id={row['id']} prob={float(row['prob_rumor']):.4f} "
            f"llm={int(llm_result['review_label'])} conf={float(llm_result['confidence']):.2f} "
            f"source={source}"
        )

    review_df = pd.DataFrame(review_records)
    candidate_out = ROOT / f"{args.out_prefix}_candidates.csv"
    review_df.to_csv(candidate_out, index=False, encoding="utf-8-sig")

    y_true = base["label"].to_numpy()
    baseline = metric_row(y_true, base["local_pred"].to_numpy(), "local_baseline")
    sweep_rows = []
    lows = [0.20, 0.25, 0.30, 0.35, 0.40]
    confidence_thresholds = [0.80, 0.85, 0.90, 0.95]
    for low in lows:
        if low < args.min_low:
            continue
        for conf in confidence_thresholds:
            pred, selected = apply_review_policy(base, review_df, low, args.high, conf)
            row = metric_row(y_true, pred, f"fn_review_low_{low:.2f}_conf_{conf:.2f}")
            row.update(
                {
                    "candidate_low": float(low),
                    "candidate_high": float(args.high),
                    "override_confidence": float(conf),
                    "llm_calls": int(
                        (
                            (review_df["prob_rumor"] >= low)
                            & (review_df["prob_rumor"] < args.high)
                        ).sum()
                    ),
                    "overrides": int(len(selected)),
                    "rescued_false_negatives": int((selected["label"] == 1).sum()),
                    "new_false_positives": int((selected["label"] == 0).sum()),
                }
            )
            row["delta_accuracy"] = row["accuracy"] - baseline["accuracy"]
            row["delta_recall_1"] = row["recall_1"] - baseline["recall_1"]
            row["delta_fp"] = row["fp"] - baseline["fp"]
            row["delta_fn"] = row["fn"] - baseline["fn"]
            sweep_rows.append(row)
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_out = ROOT / f"{args.out_prefix}_sweep.csv"
    sweep_df.to_csv(sweep_out, index=False, encoding="utf-8-sig")

    default_pred, default_selected = apply_review_policy(
        base, review_df, args.default_low, args.high, args.default_confidence
    )
    default_metrics = metric_row(y_true, default_pred, "predeclared_default")
    default_metrics.update(
        {
            "candidate_low": float(args.default_low),
            "candidate_high": float(args.high),
            "override_confidence": float(args.default_confidence),
            "llm_calls": int(
                (
                    (review_df["prob_rumor"] >= args.default_low)
                    & (review_df["prob_rumor"] < args.high)
                ).sum()
            ),
            "overrides": int(len(default_selected)),
            "rescued_false_negatives": int((default_selected["label"] == 1).sum()),
            "new_false_positives": int((default_selected["label"] == 0).sum()),
            "delta_accuracy": default_metrics["accuracy"] - baseline["accuracy"],
            "delta_recall_1": default_metrics["recall_1"] - baseline["recall_1"],
            "delta_fp": default_metrics["fp"] - baseline["fp"],
            "delta_fn": default_metrics["fn"] - baseline["fn"],
        }
    )

    best = sweep_df.sort_values(
        ["accuracy", "macro_f1", "rumor_f1"], ascending=False
    ).iloc[0].to_dict()
    summary = {
        "prompt_version": PROMPT_VERSION,
        "model": model_name,
        "val_file": args.csv,
        "candidate_rule": {
            "local_pred": 0,
            "min_low": float(args.min_low),
            "high": float(args.high),
        },
        "candidate_count": int(len(candidates)),
        "candidate_gold_rumor": int((candidates["label"] == 1).sum()),
        "candidate_gold_non_rumor": int((candidates["label"] == 0).sum()),
        "api_calls": int((review_df["review_source"] == "api").sum()) if len(review_df) else 0,
        "cache_hits": int((review_df["review_source"] == "cache").sum()) if len(review_df) else 0,
        "baseline": baseline,
        "predeclared_default": default_metrics,
        "best_exploratory_on_val": best,
        "outputs": {
            "candidates": str(candidate_out.relative_to(ROOT)),
            "sweep": str(sweep_out.relative_to(ROOT)),
        },
        "note": (
            "best_exploratory_on_val is diagnostic only because it is selected using val labels; "
            "predeclared_default is the conservative policy configured before the sweep."
        ),
    }
    summary_out = ROOT / f"{args.out_prefix}_summary.json"
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {candidate_out}")
    print(f"Wrote {sweep_out}")
    print(f"Wrote {summary_out}")


if __name__ == "__main__":
    main()
