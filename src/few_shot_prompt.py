import json
from typing import List, Dict, Any


def _format_features(features: Dict[str, Any]) -> str:
    parts = []
    for key, value in features.items():
        if isinstance(value, bool):
            rendered = "True" if value else "False"
        elif isinstance(value, str):
            rendered = value
        else:
            rendered = str(value)
        parts.append(f"{key}={rendered}")
    return ", ".join(parts)


def _format_grades(grades: Dict[str, Any]) -> str:
    parts = [f"{key}={value}" for key, value in grades.items()]
    return ", ".join(parts)


def build_few_shot_prompt(records: List[Dict[str, Any]], num_examples: int = 3) -> str:
    if not records:
        raise ValueError("records must not be empty")

    examples = records[:max(1, min(num_examples, len(records) - 1))]
    target = records[min(len(records) - 1, num_examples)] if len(records) > num_examples else records[-1]

    prompt_lines = []
    for index, record in enumerate(examples, start=1):
        prompt_lines.append(f"Example {index}:")
        prompt_lines.append(f"Features: {_format_features(record['features'])}")
        prompt_lines.append(f"Grades: {_format_grades(record['grades'])}")
        prompt_lines.append("")

    prompt_lines.append("Now grade this new project:")
    prompt_lines.append(f"Features: {_format_features(target['features'])}")
    prompt_lines.append("")
    prompt_lines.append("Return only the grades as a JSON object with the same rubric keys.")
    return "\n".join(prompt_lines)


def save_prompt_to_file(records: List[Dict[str, Any]], output_path: str, num_examples: int = 3) -> str:
    prompt = build_few_shot_prompt(records, num_examples=num_examples)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(prompt)
    return prompt
