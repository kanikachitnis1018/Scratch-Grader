import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import requests

try:
    from .grader import extract_project_features
    from .few_shot_prompt import build_few_shot_prompt, build_prompt_chain_of_thought
    from .scratch_loader import fetch_project_json
except ImportError:  # pragma: no cover - supports running the file directly
    from grader import extract_project_features
    from few_shot_prompt import build_few_shot_prompt, build_prompt_chain_of_thought
    from scratch_loader import fetch_project_json


def extract_project_id_from_url(url: str) -> int:
    match = re.search(r"/projects/(\d+)", url)
    if not match:
        raise ValueError(f"Could not extract project ID from URL: {url}")
    return int(match.group(1))


def get_project_features_from_url(url: str) -> Dict[str, Any]:
    project_id = extract_project_id_from_url(url)
    project_json = fetch_project_json(project_id)
    if not project_json:
        raise ValueError(f"Could not fetch Scratch project for {url}")
    return extract_project_features(project_json)


def build_prompt_for_url(url: str, records: List[Dict[str, Any]], num_examples: int = 3, prompt_style: str = "few_shot") -> str:
    features = get_project_features_from_url(url)
    prompt_records = records[:max(1, min(num_examples, len(records) - 1))] + [{"id": None, "features": features, "grades": {}}]
    builder = build_prompt_chain_of_thought if prompt_style == "chain_of_thought" else build_few_shot_prompt
    return builder(prompt_records, num_examples=num_examples)


def query_ollama(prompt: str, model: str = "llama3:latest") -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
    response.raise_for_status()
    payload_json = response.json()
    return payload_json.get("response", "")


def parse_grades_from_response(text: str) -> Dict[str, int]:
    if not text:
        return {}

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
                    return {key: int(value) for key, value in parsed.items() if isinstance(value, (int, float))}
            except json.JSONDecodeError:
                continue

    return {}


def grade_prompt_with_ollama(prompt: str, model: str = "llama3:latest") -> Dict[str, int]:
    response_text = query_ollama(prompt, model=model)
    return parse_grades_from_response(response_text)


def save_predictions(predictions: Dict[str, Any], output_path: str) -> None:
    output_path = Path(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(predictions, handle, indent=2)


def main():
    base_dir = Path(__file__).resolve().parent.parent
    dataset_path = base_dir / "enriched_dataset.json"
    output_path = base_dir / "ollama_predictions.json"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    url = os.getenv("SCRATCH_TEST_URL", "https://scratch.mit.edu/projects/1270822989")
    prompt = build_prompt_for_url(url, records, num_examples=3)
    print(prompt)
    print("\nCalling Ollama...\n")
    response = query_ollama(prompt)
    print(response)
    save_predictions({"url": url, "prompt": prompt, "response": response}, str(output_path))


if __name__ == "__main__":
    main()
