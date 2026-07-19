import json
from pathlib import Path
from typing import Any, Dict, List

try:
    from .loocv import (
        DEFAULT_RUBRIC_DIMENSIONS,
        _cohens_kappa,
        _exact_match_percent,
        _f1_score_per_class,
        _mean_absolute_error,
        _precision_per_class,
        _recall_per_class,
        _validate_and_clamp_predictions,
    )
except ImportError:  # pragma: no cover - supports running the file directly
    from loocv import (
        DEFAULT_RUBRIC_DIMENSIONS,
        _cohens_kappa,
        _exact_match_percent,
        _f1_score_per_class,
        _mean_absolute_error,
        _precision_per_class,
        _recall_per_class,
        _validate_and_clamp_predictions,
    )


def load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_stage_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        raise ValueError("records must not be empty")

    sample_results = []
    for record in records:
        actual = record.get("grades", {})
        predicted_raw = record.get("model_scores", {})
        predicted = _validate_and_clamp_predictions(predicted_raw)

        sample_results.append({
            "test_id": record.get("id"),
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


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    dataset_path = str(base_dir / "enriched_dataset.json")

    records = load_records(dataset_path)
    result = evaluate_stage_records(records)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()