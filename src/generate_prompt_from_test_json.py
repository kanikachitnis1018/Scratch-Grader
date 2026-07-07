import json
from pathlib import Path

from few_shot_prompt import build_few_shot_prompt


def main():
    base_dir = Path(__file__).resolve().parent.parent
    dataset_path = base_dir / "enriched_dataset.json"
    test_path = base_dir / "test_project_features.json"
    output_path = base_dir / "few_shot_prompt_test.txt"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test project feature file not found at {test_path}")

    with dataset_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    with test_path.open("r", encoding="utf-8") as handle:
        test_project = json.load(handle)

    prompt_records = records[:3] + [test_project]
    prompt = build_few_shot_prompt(prompt_records, num_examples=3)

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(prompt)

    print(f"Wrote prompt to {output_path}")
    print(prompt)


if __name__ == "__main__":
    main()
