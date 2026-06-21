from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .rumor_detector import (
    DEFAULT_MODEL_PATH,
    explain_text,
    load_model,
    predict_proba,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LLM_CACHE = PROJECT_ROOT / "results" / "llm_cache.jsonl"
DEFAULT_FN_REVIEW = PROJECT_ROOT / "results" / "fn_recall_review_candidates.csv"


ANON_WORDS = ("anonymous", "unnamed", "unidentified", "undisclosed")
LEAK_WORDS = ("obtained", "acquired", "leaked", "uncovered", "secured")
SMEAR_WORDS = ("smear", "slander", "defame", "discredit")
HIDE_WORDS = (
    "hiding",
    "concealing",
    "conceal",
    "covering up",
    "cover-up",
    "withholding",
    "suppressing",
)
DEVOLVE_WORDS = ("devolved", "deteriorated", "descended", "degenerated")
CONSPIRACY_WORDS = ("false flag", "inside job", "orchestrated", "staged event")
CORRUPT_WORDS = ("corrupt", "corruption", "deep state", "shadow government")
FABRICATED_WORDS = ("fabricated", "fabricate", "hoax", "crisis actor")
FEAR_WORDS = ("fearmongering", "fear mongering", "plandemic", "scamdemic")


def text_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def load_llm_cache(path: str | Path = DEFAULT_LLM_CACHE) -> dict[str, dict[str, Any]]:
    path = Path(path)
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            result = row.get("result", {})
            if isinstance(result, dict):
                cache[str(row["hash"])] = result
    return cache


def load_fn_review_cache(path: str | Path = DEFAULT_FN_REVIEW) -> dict[str, dict[str, Any]]:
    """Load the targeted FN-review cache by text hash so it can work for single-text input."""
    path = Path(path)
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            text = row.get("text", "")
            if not text:
                continue
            cache[text_hash(text)] = {
                "label": int(row.get("review_label", 0)),
                "confidence": float(row.get("llm_confidence", 0.0)),
                "reason": row.get("reason", ""),
                "row_index": int(row.get("row_index", -1)),
            }
    return cache


def match_rumor_signal(text: str) -> tuple[str | None, float, str | None]:
    """Return (rule_name, bias, signal_level) for rhetorical rumor signals."""
    t = str(text).lower()

    if any(s in t for s in SMEAR_WORDS) and "campaign" in t:
        return ("指控抹黑叙事", 0.50, "strong")
    if any(d in t for d in DEVOLVE_WORDS) and "worst" in t:
        return ("极端化叙事", 0.50, "strong")
    if any(c in t for c in CONSPIRACY_WORDS):
        return ("阴谋论指控", 0.50, "strong")
    if any(f in t for f in FABRICATED_WORDS):
        return ("证据捏造指控", 0.50, "strong")

    if any(h in t for h in HIDE_WORDS):
        return ("信息隐瞒暗示", 0.43, "moderate")
    if any(a in t for a in ANON_WORDS) and any(l in t for l in LEAK_WORDS):
        return ("匿名信源引用", 0.36, "moderate")
    if any(c in t for c in CORRUPT_WORDS):
        return ("权威腐败指控", 0.30, "moderate")
    if any(f in t for f in FEAR_WORDS):
        return ("恐慌煽动话术", 0.28, "moderate")

    return (None, 0.0, None)


class FinalRumourDetectClass:
    """
    Optional engineered detector for the final-project appendix.

    The default submitted detector remains RumourDetectClass in rumor_detector.py.
    This class wraps the same local model, then applies reproducible two-tier
    rhetorical rules. Cached LLM review is kept as an explicit opt-in experiment,
    because the default result should be reproducible from committed files only.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        llm_cache_path: str | Path = DEFAULT_LLM_CACHE,
        fn_review_path: str | Path = DEFAULT_FN_REVIEW,
        llm_confidence_threshold: float = 0.62,
        rule_max_probability: float = 0.25,
        use_cached_llm: bool = False,
        use_rules: bool = True,
    ):
        self.bundle = load_model(model_path)
        self.llm_confidence_threshold = float(llm_confidence_threshold)
        self.rule_max_probability = float(rule_max_probability)
        self.use_cached_llm = bool(use_cached_llm)
        self.use_rules = bool(use_rules)
        self.llm_cache = load_llm_cache(llm_cache_path) if self.use_cached_llm else {}
        self.fn_review_cache = (
            load_fn_review_cache(fn_review_path) if self.use_cached_llm else {}
        )

    def classify(self, text: str) -> int:
        return int(self.predict(text)["label"])

    def explain(self, text: str) -> str:
        return str(self.predict(text)["explanation"])

    def predict(self, text: str) -> dict[str, Any]:
        prob = float(predict_proba(self.bundle, [text])[0])
        local_label = int(prob >= float(self.bundle.get("threshold", 0.5)))
        final_label = local_label
        source = "local"
        detail: dict[str, Any] = {}
        override_reason = ""
        h = text_hash(text)

        if local_label == 0 and self.use_cached_llm:
            for llm in self._cached_llm_candidates(h, prob):
                if (
                    int(llm.get("label", 0)) == 1
                    and float(llm.get("confidence", 0.0)) >= self.llm_confidence_threshold
                ):
                    final_label = 1
                    source = str(llm.get("source", "llm_cache"))
                    override_reason = str(llm.get("reason", "")).strip()
                    detail = {
                        "llm_confidence": float(llm.get("confidence", 0.0)),
                        "llm_reason": override_reason,
                    }
                    break

        if final_label == 0 and self.use_rules and prob <= self.rule_max_probability:
            matched_rule, bias, level = match_rumor_signal(text)
            if matched_rule:
                adjusted = prob + bias
                if level == "strong" or adjusted >= 0.50:
                    final_label = 1
                    source = f"rule:{matched_rule}"
                    decision = (
                        "强信号直接覆盖"
                        if level == "strong"
                        else f"概率增强后判定: {prob:.3f}+{bias:.2f}={adjusted:.3f}"
                    )
                    override_reason = f"命中谣言话术特征[{matched_rule}]，{decision}。"
                    detail = {
                        "rule": matched_rule,
                        "signal_level": level,
                        "bias": bias,
                        "adjusted_probability": adjusted,
                    }

        if final_label == local_label:
            explanation = explain_text(self.bundle, text)
        else:
            explanation = self._override_explanation(
                text=text,
                prob=prob,
                source=source,
                reason=override_reason,
            )

        return {
            "label": int(final_label),
            "prob_rumor": prob,
            "local_label": int(local_label),
            "source": source,
            "explanation": explanation,
            **detail,
        }

    def _cached_llm_candidates(self, h: str, prob: float) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        if 0.25 <= prob <= 0.68 and h in self.llm_cache:
            result = dict(self.llm_cache[h])
            result["source"] = "llm_override_v1"
            candidates.append(result)
        if 0.08 <= prob < 0.48 and h in self.fn_review_cache:
            result = dict(self.fn_review_cache[h])
            result["source"] = "llm_override_v2"
            candidates.append(result)
        return candidates

    def _override_explanation(self, text: str, prob: float, source: str, reason: str) -> str:
        local = explain_text(self.bundle, text)
        return (
            "预测标签：1（谣言）\n"
            f"基座模型谣言概率：{prob:.3f}\n"
            f"覆盖来源：{source}\n"
            "判断依据：\n"
            f"1. 备选策略依据：{reason or '分级规则给出谣言方向强证据'}\n"
            "2. 基座模型说明如下，用于保留可追溯的文本特征证据。\n"
            f"{local}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with the optional final detector.")
    parser.add_argument("text", help="Input text to classify.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Enable optional cached LLM overrides from local result files.",
    )
    parser.add_argument("--no-rules", action="store_true", help="Disable rule-signal overrides.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = FinalRumourDetectClass(
        model_path=args.model,
        use_cached_llm=args.use_cache,
        use_rules=not args.no_rules,
    )
    print(json.dumps(detector.predict(args.text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
