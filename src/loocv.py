import json
from typing import Any, Callable, Dict, List

import pandas as pd

try:
    from .few_shot_prompt import build_few_shot_prompt, build_prompt_chain_of_thought
except ImportError:  # pragma: no cover - supports running the file directly
    from few_shot_prompt import build_few_shot_prompt, build_prompt_chain_of_thought


DEFAULT_RUBRIC_DIMENSIONS = [
    "problem_decomposition",
    "sequencing",
    "loops",
    "conditionals",
    "variables",
    "event_handling",
    "debugging",
    "procedures",
    "coordinates",
    "cloning",
    "collision",
    "animation",
    "sound",
    "ui_feedback",
    "lists",
    "math",
    "messaging",
    "algorithms",
    "nesting",
    "integration",
]


def _cohens_kappa(actual: List[int], predicted: List[int]) -> float:
    if not actual:
        return 0.0

    actual_arr = pd.Series(actual)
    predicted_arr = pd.Series(predicted)
    observed = (actual_arr == predicted_arr).mean()

    categories = sorted(set(actual) | set(predicted))
    if len(categories) <= 1:
        return 1.0 if observed == 1.0 else 0.0

    expected = 0.0
    for category in categories:
        actual_prob = (actual_arr == category).mean()
        predicted_prob = (predicted_arr == category).mean()
        expected += actual_prob * predicted_prob

    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def _mean_absolute_error(actual: Dict[str, int], predicted: Dict[str, int]) -> Dict[str, float]:
    return {
        dimension: abs(int(actual.get(dimension, 0)) - int(predicted.get(dimension, 0)))
        for dimension in DEFAULT_RUBRIC_DIMENSIONS
    }


def _exact_match_percent(actual: Dict[str, int], predicted: Dict[str, int]) -> float:
    matches = sum(
        int(actual.get(dimension, 0)) == int(predicted.get(dimension, 0))
        for dimension in DEFAULT_RUBRIC_DIMENSIONS
    )
    return (matches / len(DEFAULT_RUBRIC_DIMENSIONS)) * 100.0


def run_loocv(
    records: List[Dict[str, Any]],
    model_fn: Callable[[str], Dict[str, int]],
    num_examples: int = 3,
    prompt_builder: Callable = None,
) -> Dict[str, Any]:
    if prompt_builder is None:
        prompt_builder = build_few_shot_prompt
    if not records:
        raise ValueError("records must not be empty")

    sample_results = []
    for idx, test_record in enumerate(records):
        print(f"LOOCV progress: {idx + 1}/{len(records)}", flush=True)
        train_records = records[:idx] + records[idx + 1 :]
        example_records = train_records[:min(num_examples, len(train_records))]
        prompt = prompt_builder(example_records + [test_record], num_examples=min(num_examples, len(train_records)))
        predicted = model_fn(prompt)
        actual = test_record.get("grades", {})

        sample_results.append({
            "test_id": test_record.get("id"),
            "predicted": predicted,
            "actual": actual,
            "mae": _mean_absolute_error(actual, predicted),
            "exact_match_percent": _exact_match_percent(actual, predicted),
        })

    mae_by_dimension = {
        dimension: sum(result["mae"].get(dimension, 0.0) for result in sample_results) / len(sample_results)
        for dimension in DEFAULT_RUBRIC_DIMENSIONS
    }

    all_actual = []
    all_predicted = []
    for result in sample_results:
        actual_values = [int(result["actual"].get(dimension, 0)) for dimension in DEFAULT_RUBRIC_DIMENSIONS]
        predicted_values = [int(result["predicted"].get(dimension, 0)) for dimension in DEFAULT_RUBRIC_DIMENSIONS]
        all_actual.extend(actual_values)
        all_predicted.extend(predicted_values)

    summary = {
        "num_samples": len(records),
        "num_dimensions": len(DEFAULT_RUBRIC_DIMENSIONS),
        "mae_by_dimension": mae_by_dimension,
        "exact_match_percent": sum(result["exact_match_percent"] for result in sample_results) / len(sample_results),
        "cohens_kappa": _cohens_kappa(all_actual, all_predicted),
    }

    return {
        "summary": summary,
        "sample_results": sample_results,
    }


def run_loocv_to_dataframe(
    records: List[Dict[str, Any]],
    model_fn: Callable[[str], Dict[str, int]],
    num_examples: int = 3,
    prompt_builder: Callable = None,
) -> pd.DataFrame:
    result = run_loocv(records, model_fn=model_fn, num_examples=num_examples, prompt_builder=prompt_builder)
    rows = []
    for item in result["sample_results"]:
        row = {"test_id": item["test_id"], "exact_match_percent": item["exact_match_percent"]}
        row.update(item["mae"])
        rows.append(row)
    return pd.DataFrame(rows)
