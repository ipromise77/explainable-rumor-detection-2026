from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "demo"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.final_detector import FinalRumourDetectClass
from src.rumor_detector import RumourDetectClass


MAIN_DETECTOR = RumourDetectClass()
FINAL_DETECTOR = FinalRumourDetectClass()


def predict_payload(text: str, detector_name: str) -> dict:
    detector_name = detector_name if detector_name in {"main", "final"} else "main"
    detector = FINAL_DETECTOR if detector_name == "final" else MAIN_DETECTOR
    result = detector.predict(text)
    label = int(result["label"])
    return {
        "detector": detector_name,
        "label": label,
        "label_name": "谣言" if label == 1 else "非谣言",
        "prob_rumor": float(result.get("prob_rumor", 0.0)),
        "source": result.get("source", "local"),
        "local_label": result.get("local_label"),
        "explanation": result.get("explanation", ""),
    }


class DemoHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        return

    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        content_types = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
        }
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_types.get(path.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        rel = parsed.path.lstrip("/") or "index.html"
        path = (DEMO_DIR / rel).resolve()
        if not str(path).startswith(str(DEMO_DIR.resolve())):
            self.send_error(403)
            return
        self._send_file(path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/predict":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            text = str(payload.get("text", "")).strip()
            detector = str(payload.get("detector", "main")).strip()
            if not text:
                self._send_json({"error": "请输入待检测文本。"}, status=400)
                return
            self._send_json(predict_payload(text, detector))
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)


def main() -> None:
    host = "127.0.0.1"
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = ThreadingHTTPServer((host, port), DemoHandler)
    print(f"Demo page: http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
