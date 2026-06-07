from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "rumor_ensemble.joblib"


MODEL_CONFIGS = [
    {
        "name": "word12_char36_c4",
        "word_ngram": (1, 2),
        "char_ngram": (3, 6),
        "c": 4.0,
        "stop_words": None,
    },
    {
        "name": "word12_char25_c8",
        "word_ngram": (1, 2),
        "char_ngram": (2, 5),
        "c": 8.0,
        "stop_words": None,
    },
    {
        "name": "word13_char35_c4",
        "word_ngram": (1, 3),
        "char_ngram": (3, 5),
        "c": 4.0,
        "stop_words": None,
    },
]


def preprocess_text(text: str) -> str:
    """Normalize tweet-like text while preserving useful platform cues."""
    text = html.unescape(str(text)).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URLTOKEN ", text)
    text = re.sub(r"@\w+", " USERTOKEN ", text)
    text = re.sub(r"#(\w+)", r" HASHTAG_\1 \1 ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_many(texts: str | Iterable[str]) -> list[str]:
    if isinstance(texts, str):
        return [preprocess_text(texts)]
    return [preprocess_text(text) for text in texts]


def build_pipeline(config: dict) -> Pipeline:
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
            ("features", FeatureUnion([("word", word_vectorizer), ("char", char_vectorizer)])),
            (
                "classifier",
                LogisticRegression(max_iter=5000, C=config["c"], solver="liblinear"),
            ),
        ]
    )


def train_ensemble(texts: Iterable[str], labels: Iterable[int]) -> dict:
    clean_texts = preprocess_many(texts)
    labels = np.asarray(list(labels), dtype=int)
    models = []
    for config in MODEL_CONFIGS:
        pipeline = build_pipeline(config)
        pipeline.fit(clean_texts, labels)
        models.append({"name": config["name"], "config": config, "pipeline": pipeline})
    return {
        "version": "2026.1",
        "threshold": 0.5,
        "label_meaning": {"0": "non-rumor", "1": "rumor"},
        "models": models,
    }


def save_model(bundle: dict, path: str | Path = DEFAULT_MODEL_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_model(path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. Run `python -m src.train` first."
        )
    return joblib.load(path)


def predict_proba(bundle: dict, texts: str | Iterable[str]) -> np.ndarray:
    clean_texts = preprocess_many(texts)
    probs = []
    for item in bundle["models"]:
        probs.append(item["pipeline"].predict_proba(clean_texts)[:, 1])
    return np.mean(np.vstack(probs), axis=0)


def predict_label(bundle: dict, texts: str | Iterable[str]) -> np.ndarray:
    threshold = float(bundle.get("threshold", 0.5))
    return (predict_proba(bundle, texts) >= threshold).astype(int)


def _feature_names(feature_union: FeatureUnion) -> np.ndarray:
    names = []
    for name, transformer in feature_union.transformer_list:
        for feature in transformer.get_feature_names_out():
            names.append(f"{name}__{feature}")
    return np.asarray(names, dtype=object)


def _display_feature(raw_name: str) -> str:
    kind, value = raw_name.split("__", 1)
    value = value.replace("HASHTAG_", "#").replace("urltoken", "URL").replace(
        "usertoken", "USER"
    )
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        return ""
    if kind == "word":
        return f"词项'{value}'"
    return f"字符片段'{value}'"


def evidence_for_text(bundle: dict, text: str, top_k: int = 5) -> dict:
    clean = preprocess_text(text)
    rumor_score: dict[str, float] = {}
    non_rumor_score: dict[str, float] = {}

    for item in bundle["models"]:
        pipeline = item["pipeline"]
        features = pipeline.named_steps["features"]
        classifier = pipeline.named_steps["classifier"]
        names = _feature_names(features)
        row = features.transform([clean])
        coef = classifier.coef_[0]
        contribution = row.multiply(coef).tocoo()

        for idx, value in zip(contribution.col, contribution.data):
            if abs(value) < 1e-6:
                continue
            shown = _display_feature(str(names[idx]))
            if not shown:
                continue
            if value >= 0:
                rumor_score[shown] = rumor_score.get(shown, 0.0) + float(value)
            else:
                non_rumor_score[shown] = non_rumor_score.get(shown, 0.0) + float(-value)

    prob = float(predict_proba(bundle, [text])[0])
    label = int(prob >= float(bundle.get("threshold", 0.5)))

    def top_items(scores: dict[str, float]) -> list[dict]:
        return [
            {"feature": name, "score": round(score, 4)}
            for name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        ]

    return {
        "label": label,
        "prob_rumor": prob,
        "confidence": prob if label == 1 else 1.0 - prob,
        "rumor_evidence": top_items(rumor_score),
        "non_rumor_evidence": top_items(non_rumor_score),
    }


def explain_text(bundle: dict, text: str, top_k: int = 4) -> str:
    evidence = evidence_for_text(bundle, text, top_k=top_k)
    label = int(evidence["label"])
    prob = float(evidence["prob_rumor"])
    confidence = float(evidence["confidence"])

    def top_items(items: list[dict]) -> str:
        if not items:
            return "未发现明显关键词证据"
        return "、".join(
            f"{item['feature']}(贡献{float(item['score']):.3f})" for item in items
        )

    pro_rumor = top_items(evidence["rumor_evidence"])
    pro_non_rumor = top_items(evidence["non_rumor_evidence"])
    label_name = "谣言" if label == 1 else "非谣言"
    if label == 1:
        return (
            f"预测标签：{label}（{label_name}）\n"
            f"谣言概率：{prob:.3f}\n"
            f"分类置信度：{confidence:.3f}\n"
            "判断依据：\n"
            f"1. 支持谣言的主要证据：{pro_rumor}。\n"
            f"2. 支持非谣言的反向证据：{pro_non_rumor}。\n"
            "3. 综合判断：上述证据来自当前文本的词级和字符级 TF-IDF 特征贡献，"
            "贡献越大说明该特征越推动模型做出对应方向的判断。本样本中谣言方向证据"
            "整体更强，因此判为谣言；解释不引入外部事实，只反映模型在该样本上的局部判定依据。"
        )
    return (
        f"预测标签：{label}（{label_name}）\n"
        f"谣言概率：{prob:.3f}\n"
        f"分类置信度：{confidence:.3f}\n"
        "判断依据：\n"
        f"1. 支持非谣言的主要证据：{pro_non_rumor}。\n"
        f"2. 支持谣言的反向证据：{pro_rumor}。\n"
        "3. 综合判断：上述证据来自当前文本的词级和字符级 TF-IDF 特征贡献，"
        "贡献越大说明该特征越推动模型做出对应方向的判断。本样本中非谣言方向证据"
        "整体更强，因此判为非谣言；解释不引入外部事实，只反映模型在该样本上的局部判定依据。"
    )


class RumourDetectClass:
    """Course-required callable detector: input text, output label and rationale."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        self.bundle = load_model(model_path)

    def classify(self, text: str) -> int:
        return int(predict_label(self.bundle, [text])[0])

    def explain(self, text: str) -> str:
        return explain_text(self.bundle, text)

    def predict(self, text: str) -> dict:
        evidence = evidence_for_text(self.bundle, text)
        return {
            "label": int(evidence["label"]),
            "prob_rumor": float(evidence["prob_rumor"]),
            "explanation": explain_text(self.bundle, text),
        }
