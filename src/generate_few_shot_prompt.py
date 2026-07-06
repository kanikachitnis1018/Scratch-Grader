import json
import os
from pathlib import Path

from few_shot_prompt import save_prompt_to_file


def main():
    dataset_path = Path(__file__).resolve().parent.parent / "enriched_dataset.json"
    output_path = Path(__file__).resolve().parent.parent / "few_shot_prompt.txt"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    prompt = save_prompt_to_file(records, str(output_path), num_examples=3)
    print(f"Saved few-shot prompt to {output_path}")
    print(prompt)


if __name__ == "__main__":
    main()
