from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from .rumor_detector import (
    DEFAULT_MODEL_PATH,
    evidence_for_text,
    explain_text,
    load_model,
    preprocess_many,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "rumer2026"


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_json(text: str) -> dict[str, Any]:
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


class LLMEnhancedRumorDetector:
    """Use the local model first, then call the school LLM for uncertain cases."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        data_dir: str | Path = DEFAULT_DATA_DIR,
        low: float = 0.30,
        high: float = 0.70,
        top_k: int = 4,
        allow_override: bool = False,
        override_confidence: float = 0.85,
        cache_path: str | Path = PROJECT_ROOT / "results" / "llm_cache.jsonl",
    ):
        load_dotenv(PROJECT_ROOT / ".env")
        self.bundle = load_model(model_path)
        self.low = low
        self.high = high
        self.top_k = top_k
        self.allow_override = allow_override
        self.override_confidence = override_confidence
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = self._read_cache()

        api_key = os.getenv("SJTU_API_KEY", "").strip()
        base_url = os.getenv("SJTU_BASE_URL", "").strip()
        self.model = os.getenv("SJTU_MODEL", "deepseek-reasoner").strip()
        if not api_key or "请在这里" in api_key:
            raise RuntimeError("SJTU_API_KEY is missing. Please fill it in .env first.")
        if not base_url or "请在这里" in base_url:
            raise RuntimeError("SJTU_BASE_URL is missing. Please fill it in .env first.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        self.train_df = pd.read_csv(Path(data_dir) / "train_clean.csv")
        first_pipeline = self.bundle["models"][0]["pipeline"]
        self.retriever_features = first_pipeline.named_steps["features"]
        self.train_matrix = self.retriever_features.transform(preprocess_many(self.train_df["text"]))

    def _read_cache(self) -> dict[str, dict]:
        if not self.cache_path.exists():
            return {}
        cache: dict[str, dict] = {}
        with self.cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                cache[row["hash"]] = row["result"]
        return cache

    def _write_cache(self, text: str, result: dict) -> None:
        key = _text_hash(text)
        self.cache[key] = result
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"hash": key, "result": result}, ensure_ascii=False) + "\n")

    def should_call_llm(self, prob_rumor: float, force: bool = False) -> bool:
        if force:
            return True
        return self.low <= prob_rumor <= self.high

    def retrieve_examples(self, text: str) -> list[dict]:
        query = self.retriever_features.transform(preprocess_many([text]))
        scores = (self.train_matrix @ query.T).toarray().ravel()
        order = np.argsort(scores)[::-1][: self.top_k]
        examples = []
        for idx in order:
            row = self.train_df.iloc[int(idx)]
            examples.append(
                {
                    "label": int(row["label"]),
                    "text": str(row["text"])[:260],
                    "similarity": round(float(scores[idx]), 4),
                }
            )
        return examples

    def _prompt(self, text: str, local: dict, examples: list[dict]) -> list[dict]:
        system = (
            "你是一个用于课程项目复现的英文推文谣言检测器。标签定义：0=非谣言，1=谣言。"
            "你必须结合输入文本、本地模型证据和相似训练样本判断，不要编造外部事实。"
            "请只输出JSON，不要输出Markdown。"
        )
        user_payload = {
            "task": "判断输入推文是否为谣言，并给出简洁中文判断依据。",
            "output_schema": {
                "label": "integer, only 0 or 1",
                "confidence": "number between 0 and 1",
                "reason": "1-2 sentences in Chinese",
            },
            "rules": [
                "如果本地模型概率和相似训练样本一致，优先保持本地模型判断。",
                "如果文本证据与相似样本明显冲突，可以修正本地模型判断。",
                "解释应指出具体文本线索或与相似样本的关系。",
                "不得提到验证集真实标签，也不得输出多余字段。",
            ],
            "tweet": text,
            "local_model": local,
            "similar_labeled_examples": examples,
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

    def call_llm(self, text: str, local: dict, examples: list[dict]) -> dict:
        key = _text_hash(text)
        if key in self.cache:
            return self.cache[key]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._prompt(text, local, examples),
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        parsed = _extract_json(content)
        label = int(parsed.get("label", local["label"]))
        if label not in (0, 1):
            label = int(local["label"])
        confidence = float(parsed.get("confidence", local["confidence"]))
        confidence = max(0.0, min(1.0, confidence))
        result = {
            "label": label,
            "confidence": confidence,
            "reason": str(parsed.get("reason", "")).strip() or explain_text(self.bundle, text),
        }
        self._write_cache(text, result)
        return result

    def predict(self, text: str, force_llm: bool = False) -> dict:
        local = evidence_for_text(self.bundle, text, top_k=5)
        local_summary = {
            "label": int(local["label"]),
            "prob_rumor": round(float(local["prob_rumor"]), 4),
            "confidence": round(float(local["confidence"]), 4),
            "rumor_evidence": local["rumor_evidence"],
            "non_rumor_evidence": local["non_rumor_evidence"],
        }
        if not self.should_call_llm(float(local["prob_rumor"]), force_llm):
            return {
                "label": int(local["label"]),
                "prob_rumor": float(local["prob_rumor"]),
                "source": "local",
                "explanation": explain_text(self.bundle, text),
            }

        examples = self.retrieve_examples(text)
        llm_result = self.call_llm(text, local_summary, examples)
        llm_label = int(llm_result["label"])
        local_label = int(local["label"])
        use_override = (
            self.allow_override
            and llm_label != local_label
            and float(llm_result["confidence"]) >= self.override_confidence
        )
        return {
            "label": llm_label if use_override else local_label,
            "prob_rumor": float(local["prob_rumor"]),
            "source": "llm_override" if use_override else "llm_explain",
            "explanation": llm_result["reason"],
            "llm_confidence": float(llm_result["confidence"]),
            "local_label": local_label,
            "llm_label": llm_label,
            "similar_examples": examples,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with optional LLM enhancement.")
    parser.add_argument("text")
    parser.add_argument("--force-llm", action="store_true")
    parser.add_argument("--allow-override", action="store_true")
    parser.add_argument("--override-confidence", type=float, default=0.85)
    args = parser.parse_args()
    detector = LLMEnhancedRumorDetector(
        allow_override=args.allow_override,
        override_confidence=args.override_confidence,
    )
    print(json.dumps(detector.predict(args.text, force_llm=args.force_llm), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
