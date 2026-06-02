from __future__ import annotations

import argparse
import json

from .rumor_detector import RumourDetectClass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one tweet-like text.")
    parser.add_argument("text", help="Input text to classify.")
    parser.add_argument("--model", default="models/rumor_ensemble.joblib")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = RumourDetectClass(args.model)
    print(json.dumps(detector.predict(args.text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

