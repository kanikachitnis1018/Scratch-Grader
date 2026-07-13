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


def _precision_per_class(actual: List[int], predicted: List[int]) -> Dict[int, float]:
    """Compute precision for each class (0-5)."""
    precision_by_class = {}
    for score_class in range(0, 6):
        tp = sum(1 for a, p in zip(actual, predicted) if p == score_class and a == p)
        fp = sum(1 for a, p in zip(actual, predicted) if p == score_class and a != p)
        precision_by_class[score_class] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return precision_by_class


def _recall_per_class(actual: List[int], predicted: List[int]) -> Dict[int, float]:
    """Compute recall for each class (0-5)."""
    recall_by_class = {}
    for score_class in range(0, 6):
        tp = sum(1 for a, p in zip(actual, predicted) if a == score_class and a == p)
        fn = sum(1 for a, p in zip(actual, predicted) if a == score_class and a != p)
        recall_by_class[score_class] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return recall_by_class


def _f1_score_per_class(precision_by_class: Dict[int, float], recall_by_class: Dict[int, float]) -> Dict[int, float]:
    """Compute F1-score for each class (0-5)."""
    f1_by_class = {}
    for score_class in range(0, 6):
        p = precision_by_class.get(score_class, 0.0)
        r = recall_by_class.get(score_class, 0.0)
        if (p + r) > 0:
            f1_by_class[score_class] = 2 * (p * r) / (p + r)
        else:
            f1_by_class[score_class] = 0.0
    return f1_by_class


def _validate_and_clamp_predictions(predicted: Dict[str, int]) -> Dict[str, int]:
    """Ensure all predicted values are valid integers in range 0-5.
    
    If a value is missing, out of range, or invalid, it's clamped to 0-5.
    """
    valid_predicted = {}
    for dimension in DEFAULT_RUBRIC_DIMENSIONS:
        value = predicted.get(dimension, 0)  # Default to 0 if missing
        try:
            int_value = int(value)
            # Clamp to valid range 0-5
            valid_predicted[dimension] = max(0, min(5, int_value))
        except (ValueError, TypeError):
            valid_predicted[dimension] = 0  # Default to 0 if conversion fails
    return valid_predicted


def run_loocv(
    records: List[Dict[str, Any]],
    model_fn: Callable[[str], Dict[str, int]],
    num_examples: int = 3,
    prompt_builder: Callable = None,
    debug: bool = False,
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
        predicted_raw = model_fn(prompt)
        predicted = _validate_and_clamp_predictions(predicted_raw)
        actual = test_record.get("grades", {})

        if debug and idx == 0:
            print(f"\n[DEBUG] Sample 1 actual grades: {actual}", flush=True)
            print(f"[DEBUG] Sample 1 predicted (raw): {predicted_raw}", flush=True)
            print(f"[DEBUG] Sample 1 predicted (clamped): {predicted}", flush=True)
            actual_values = [int(actual.get(dimension, 0)) for dimension in DEFAULT_RUBRIC_DIMENSIONS]
            pred_values = [int(predicted.get(dimension, 0)) for dimension in DEFAULT_RUBRIC_DIMENSIONS]
            print(f"[DEBUG] Actual value range: {min(actual_values)}-{max(actual_values)}", flush=True)
            print(f"[DEBUG] Predicted value range: {min(pred_values)}-{max(pred_values)}", flush=True)

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

    precision_by_class = _precision_per_class(all_actual, all_predicted)
    recall_by_class = _recall_per_class(all_actual, all_predicted)
    f1_by_class = _f1_score_per_class(precision_by_class, recall_by_class)

    macro_precision = sum(precision_by_class.values()) / len(precision_by_class) if precision_by_class else 0.0
    macro_recall = sum(recall_by_class.values()) / len(recall_by_class) if recall_by_class else 0.0
    macro_f1 = sum(f1_by_class.values()) / len(f1_by_class) if f1_by_class else 0.0

    summary = {
        "num_samples": len(records),
        "num_dimensions": len(DEFAULT_RUBRIC_DIMENSIONS),
        "mae_by_dimension": mae_by_dimension,
        "accuracy": sum(result["exact_match_percent"] for result in sample_results) / len(sample_results),
        "cohens_kappa": _cohens_kappa(all_actual, all_predicted),
        "precision_by_class": precision_by_class,
        "recall_by_class": recall_by_class,
        "f1_by_class": f1_by_class,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
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
