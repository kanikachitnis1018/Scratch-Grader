import json
import os
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

try:
    from .generate_prompt_from_test_json import write_prompt_for_url
    from .ollama_grader import query_ollama
except ImportError:  # pragma: no cover - supports running the file directly
    from generate_prompt_from_test_json import write_prompt_for_url
    from ollama_grader import query_ollama


def _extract_grades_from_response(text: str):
    if not text:
        return None

    for pattern in [
        re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE),
        re.compile(r"\{[\s\S]*\}", re.IGNORECASE),
    ]:
        for match in pattern.finditer(text):
            candidate = match.group(1).strip() if match.lastindex else match.group(0).strip()
            if not candidate.startswith("{"):
                continue
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

    return {"raw_response": text.strip()}


def build_response_payload(url: str, output_path: Path | str, grades: dict | None) -> dict:
    return {
        "status": "ok",
        "url": url,
        "prompt_path": str(output_path),
        "grades": grades,
    }


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")

        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "invalid JSON"}).encode("utf-8"))
            return

        url = payload.get("url")
        prompt_style = payload.get("prompt_style", "few_shot")
        if not url:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "missing url"}).encode("utf-8"))
            return

        output_path = Path(__file__).resolve().parent.parent / "few_shot_prompt.txt"
        prompt = write_prompt_for_url(url, output_path=output_path, prompt_style=prompt_style)
        ollama_response = query_ollama(prompt)

        grades = None
        if ollama_response:
            grades = _extract_grades_from_response(ollama_response)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(build_response_payload(url, output_path, grades), indent=2).encode("utf-8"))

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Scratch URL receiver is running")


def main():
    port = int(os.getenv("PORT", "8000"))
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"Listening on http://0.0.0.0:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
