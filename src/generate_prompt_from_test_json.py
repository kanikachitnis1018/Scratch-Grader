import json
import os
from pathlib import Path

try:
    from .few_shot_prompt import build_few_shot_prompt, build_prompt_chain_of_thought
    from .ollama_grader import get_project_features_from_url
except ImportError:  # pragma: no cover - supports running the file directly
    from few_shot_prompt import build_few_shot_prompt, build_prompt_chain_of_thought
    from ollama_grader import get_project_features_from_url


def load_dataset_records(base_dir: Path):
    dataset_path = base_dir / "enriched_dataset.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_prompt_for_url(url: str, output_path: Path | None = None, prompt_style: str = "few_shot") -> str:
    base_dir = Path(__file__).resolve().parent.parent
    output_path = output_path or (base_dir / "few_shot_prompt.txt")

    records = load_dataset_records(base_dir)
    features = get_project_features_from_url(url)

    test_project = {"id": None, "features": features, "grades": {}}
    prompt_records = records[:3] + [test_project]
    builder = build_prompt_chain_of_thought if prompt_style == "chain_of_thought" else build_few_shot_prompt
    prompt = builder(prompt_records, num_examples=3)

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(prompt)

    return prompt


def main():
    url = os.getenv("SCRATCH_TEST_URL", "https://scratch.mit.edu/projects/1270822989")
    prompt = write_prompt_for_url(url)
    output_path = Path(__file__).resolve().parent.parent / "few_shot_prompt.txt"
    print(f"Wrote prompt to {output_path}")
    print(prompt)


if __name__ == "__main__":
    main()
