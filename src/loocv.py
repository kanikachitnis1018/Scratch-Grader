import json
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd

try:
    from .few_shot_prompt import (
        build_few_shot_prompt, 
        build_prompt_chain_of_thought, 
        build_prompt_hybrid,
        build_prompt_category_batch, # <--- ADD HERE
    )
except ImportError:
    from few_shot_prompt import (
        build_few_shot_prompt, 
        build_prompt_chain_of_thought, 
        build_prompt_hybrid,
        build_prompt_category_batch, # <--- ADD HERE
    )


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


def _apply_prediction_calibration(
    predicted: Dict[str, int],
    test_record: Dict[str, Any],
    calibration_mode: str,
) -> Dict[str, int]:
    """Apply optional feature-based post-calibration.

    Calibration is intentionally conservative and only uses test features,
    avoiding any access to target grades.
    """
    if calibration_mode in {"", "off", "none"}:
        return predicted

    calibrated = dict(predicted)
    features = test_record.get("features", {}) if isinstance(test_record, dict) else {}

    block_count = int(features.get("block_count", 0) or 0)
    sprite_count = int(features.get("sprite_count", 0) or 0)
    uses_event_handling = bool(features.get("uses_event_handling"))
    uses_messaging = bool(features.get("uses_messaging"))
    uses_nesting = bool(features.get("uses_nesting"))
    uses_lists = bool(features.get("uses_lists"))
    uses_sound = bool(features.get("uses_sound"))
    uses_collision = bool(features.get("uses_collision"))
    uses_animation = bool(features.get("uses_animation"))
    uses_math = bool(features.get("uses_math"))
    uses_algorithms = bool(features.get("uses_algorithms"))
    uses_variables = bool(features.get("uses_variables"))

    # Helper function: Only apply a ceiling cap if the raw prediction isn't showing strong mastery (>= 4)
    def _cap_if_low_raw(dimension: str, cap: int) -> None:
        raw_val = calibrated.get(dimension, 0)
        # Protect strong 4s and 5s predicted by the model from being pulled down to 1-3
        if raw_val < 4:
            calibrated[dimension] = min(raw_val, cap)

    # Prevent high scores when core evidence flags are absent (relaxed for high predictions)
    if not uses_event_handling:
        _cap_if_low_raw("event_handling", 2)
    if not uses_messaging:
        _cap_if_low_raw("messaging", 2)
    if not uses_nesting:
        _cap_if_low_raw("nesting", 2)
    if not uses_lists:
        _cap_if_low_raw("lists", 2)
    if not uses_sound:
        _cap_if_low_raw("sound", 2)
    if not uses_collision:
        _cap_if_low_raw("collision", 2)

    # Small projects tend to be over-scored on abstract logic dimensions.
    # Preserve 4s and 5s if the LLM confidently identifies advanced logic in concise code.
    if block_count < 80:
        _cap_if_low_raw("algorithms", 3)
        _cap_if_low_raw("conditionals", 3)
        _cap_if_low_raw("loops", 3)

    # v2 applies extra evidence gates for dimensions that are frequently
    # over-scored in sparse projects.
    if calibration_mode == "v2":
        if not uses_animation:
            _cap_if_low_raw("animation", 2)
        if not uses_math:
            _cap_if_low_raw("math", 2)
        if not uses_algorithms:
            _cap_if_low_raw("algorithms", 2)
        if not uses_variables:
            _cap_if_low_raw("variables", 2)

        if sprite_count <= 1:
            _cap_if_low_raw("problem_decomposition", 3)
            _cap_if_low_raw("integration", 2)
            _cap_if_low_raw("messaging", 1)

        if block_count < 45:
            _cap_if_low_raw("sequencing", 3)
            _cap_if_low_raw("coordinates", 3)
            _cap_if_low_raw("procedures", 2)
            _cap_if_low_raw("nesting", 1)

        if block_count < 25:
            _cap_if_low_raw("loops", 2)
            _cap_if_low_raw("conditionals", 2)
            _cap_if_low_raw("algorithms", 2)

    return _validate_and_clamp_predictions(calibrated)


def _blend_with_retrieval_prior(
    predicted: Dict[str, int],
    example_records: List[Dict[str, Any]],
    test_record: Dict[str, Any],
    blend_weight: float,
) -> Dict[str, int]:
    """Blend model output with a similarity-weighted prior from retrieved examples.

    This reduces noisy per-dimension jumps by anchoring predictions to grades from
    the selected few-shot examples that are most similar to the target features.
    """
    if blend_weight <= 0 or not example_records:
        return _validate_and_clamp_predictions(predicted)

    weight = max(0.0, min(0.5, float(blend_weight)))
    similarities = [_feature_similarity_score(example, test_record) for example in example_records]
    if not similarities:
        return _validate_and_clamp_predictions(predicted)

    min_similarity = min(similarities)
    shifted_weights = [(similarity - min_similarity) + 1e-6 for similarity in similarities]
    weight_sum = sum(shifted_weights)
    if weight_sum <= 0:
        return _validate_and_clamp_predictions(predicted)

    priors: Dict[str, float] = {dimension: 0.0 for dimension in DEFAULT_RUBRIC_DIMENSIONS}
    for example, similarity_weight in zip(example_records, shifted_weights):
        grades = example.get("grades", {})
        for dimension in DEFAULT_RUBRIC_DIMENSIONS:
            priors[dimension] += similarity_weight * float(grades.get(dimension, 0) or 0)
    for dimension in DEFAULT_RUBRIC_DIMENSIONS:
        priors[dimension] /= weight_sum

    blended: Dict[str, int] = {}
    for dimension in DEFAULT_RUBRIC_DIMENSIONS:
        pred_value = float(predicted.get(dimension, 0) or 0)
        prior_value = priors.get(dimension, pred_value)
        blended[dimension] = int(round(((1.0 - weight) * pred_value) + (weight * prior_value)))

    return _validate_and_clamp_predictions(blended)


def _feature_similarity_score(candidate: Dict[str, Any], target: Dict[str, Any]) -> float:
    candidate_features = candidate.get("features", {})
    target_features = target.get("features", {})

    score = 0.0
    numeric_keys = ["sprite_count", "block_count", "variable_count", "list_count"]
    boolean_keys = [
        "uses_loops",
        "uses_conditionals",
        "uses_variables",
        "uses_event_handling",
        "uses_cloning",
        "uses_collision",
        "uses_animation",
        "uses_sound",
        "uses_ui_feedback",
        "uses_lists",
        "uses_math",
        "uses_messaging",
        "uses_algorithms",
        "uses_nesting",
        "uses_integration",
    ]

    for key in boolean_keys:
        if candidate_features.get(key) == target_features.get(key):
            score += 2.0

    for key in numeric_keys:
        candidate_value = float(candidate_features.get(key, 0) or 0)
        target_value = float(target_features.get(key, 0) or 0)
        max_value = max(abs(candidate_value), abs(target_value), 1.0)
        score += 1.0 - (abs(candidate_value - target_value) / max_value)

    # Dimension-aware profiles help hybrid retrieval favor projects that are
    # similar in the same kinds of skills, not just overall project size.
    profile_groups = [
        {
            "bool": ["uses_loops", "uses_conditionals", "uses_algorithms", "uses_nesting", "uses_math"],
            "num": ["block_count"],
            "bool_weight": 1.8,
            "num_weight": 1.0,
        },
        {
            "bool": ["uses_event_handling", "uses_messaging", "uses_ui_feedback"],
            "num": ["sprite_count"],
            "bool_weight": 2.0,
            "num_weight": 1.1,
        },
        {
            "bool": ["uses_variables", "uses_lists", "uses_math"],
            "num": ["variable_count", "list_count"],
            "bool_weight": 1.7,
            "num_weight": 1.0,
        },
    ]

    for group in profile_groups:
        for key in group["bool"]:
            if candidate_features.get(key) == target_features.get(key):
                score += group["bool_weight"]
        for key in group["num"]:
            candidate_value = float(candidate_features.get(key, 0) or 0)
            target_value = float(target_features.get(key, 0) or 0)
            max_value = max(abs(candidate_value), abs(target_value), 1.0)
            score += group["num_weight"] * (1.0 - (abs(candidate_value - target_value) / max_value))

    return score


def _adaptive_mmr_weights(similarities: List[float]) -> Tuple[float, float]:
    """Set similarity/diversity weights from candidate pool shape.

    - Near-duplicate pools (low variance) get more diversity weight.
    - Noisy/sparse pools (high variance) get more similarity weight.
    """
    if not similarities:
        return 0.75, 0.25

    sorted_scores = sorted(similarities, reverse=True)
    top_scores = sorted_scores[: min(8, len(sorted_scores))]
    mean_score = sum(top_scores) / len(top_scores)

    variance = sum((value - mean_score) ** 2 for value in top_scores) / len(top_scores)
    std_dev = variance ** 0.5
    coefficient_of_variation = std_dev / mean_score if mean_score > 0 else 0.0

    # 0.12 -> near-duplicates (favor diversity), 0.35 -> noisy/sparse (favor similarity)
    lower, upper = 0.12, 0.35
    if coefficient_of_variation <= lower:
        similarity_weight = 0.62
    elif coefficient_of_variation >= upper:
        similarity_weight = 0.84
    else:
        ratio = (coefficient_of_variation - lower) / (upper - lower)
        similarity_weight = 0.62 + (0.22 * ratio)

    if mean_score < 12.0:
        similarity_weight = min(0.90, similarity_weight + 0.04)

    diversity_weight = 1.0 - similarity_weight
    return similarity_weight, diversity_weight


def _complexity_score(features: Dict[str, Any]) -> float:
    """Compute a coarse project complexity score from extracted features."""
    sprite_count = float(features.get("sprite_count", 0) or 0)
    block_count = float(features.get("block_count", 0) or 0)
    variable_count = float(features.get("variable_count", 0) or 0)
    list_count = float(features.get("list_count", 0) or 0)

    boolean_signals = [
        "uses_loops",
        "uses_conditionals",
        "uses_variables",
        "uses_event_handling",
        "uses_cloning",
        "uses_collision",
        "uses_animation",
        "uses_sound",
        "uses_ui_feedback",
        "uses_lists",
        "uses_math",
        "uses_messaging",
        "uses_algorithms",
        "uses_nesting",
        "uses_integration",
    ]
    enabled_count = sum(1 for key in boolean_signals if bool(features.get(key)))

    return (
        (sprite_count * 1.2)
        + (block_count / 25.0)
        + (variable_count * 1.5)
        + (list_count * 2.0)
        + (enabled_count * 2.2)
    )


def _complexity_band(features: Dict[str, Any]) -> int:
    """Bucket projects into low/medium/high complexity bands."""
    score = _complexity_score(features)
    if score < 28.0:
        return 0
    if score < 52.0:
        return 1
    return 2


def _filter_candidates_by_complexity(
    train_records: List[Dict[str, Any]],
    test_record: Dict[str, Any],
    min_required: int,
) -> List[Dict[str, Any]]:
    """Prefer nearby complexity bands and widen gradually when needed."""
    test_features = test_record.get("features", {})
    test_band = _complexity_band(test_features)

    def _band_distance(record: Dict[str, Any]) -> int:
        return abs(_complexity_band(record.get("features", {})) - test_band)

    exact_band = [record for record in train_records if _band_distance(record) == 0]
    if len(exact_band) >= min_required:
        return exact_band

    near_band = [record for record in train_records if _band_distance(record) <= 1]
    if len(near_band) >= min_required:
        return near_band

    return train_records


def _grade_distance_score(candidate: Dict[str, Any], selected: List[Dict[str, Any]]) -> float:
    """Reward examples whose grade vectors differ from already selected ones."""
    if not selected:
        return 0.0

    candidate_grades = candidate.get("grades", {})
    if not isinstance(candidate_grades, dict) or not candidate_grades:
        return 0.0

    distances = []
    for selected_record in selected:
        selected_grades = selected_record.get("grades", {})
        distance = 0.0
        for dimension in DEFAULT_RUBRIC_DIMENSIONS:
            candidate_value = int(candidate_grades.get(dimension, 0))
            selected_value = int(selected_grades.get(dimension, 0))
            distance += abs(candidate_value - selected_value)
        distances.append(distance / len(DEFAULT_RUBRIC_DIMENSIONS))

    return min(distances) if distances else 0.0


def _select_example_records(
    train_records: List[Dict[str, Any]],
    test_record: Dict[str, Any],
    num_examples: int,
    prompt_builder: Callable,
) -> List[Dict[str, Any]]:
    example_count = min(num_examples, len(train_records))
    if example_count <= 0:
        return []

    # Enable MMR greedy example selection for both hybrid and category batching
    if prompt_builder in (build_prompt_hybrid, build_prompt_category_batch):
        # Greedy MMR-style selection: keep examples similar to target while
        # increasing grade diversity across selected examples.
        minimum_pool = min(len(train_records), max(example_count * 2, example_count + 2))
        complexity_filtered = _filter_candidates_by_complexity(train_records, test_record, minimum_pool)
        available = list(complexity_filtered)
        selected = []
        candidate_similarities = [_feature_similarity_score(candidate, test_record) for candidate in available]
        similarity_weight, diversity_weight = _adaptive_mmr_weights(candidate_similarities)

        while len(selected) < example_count and available:
            best_index = 0
            best_score = float("-inf")

            for index, candidate in enumerate(available):
                similarity = _feature_similarity_score(candidate, test_record)
                diversity = _grade_distance_score(candidate, selected)
                combined = (similarity_weight * similarity) + (diversity_weight * diversity)
                if combined > best_score:
                    best_score = combined
                    best_index = index

            selected.append(available.pop(best_index))

        return selected

    return train_records[:example_count]


def run_loocv(
    records: List[Dict[str, Any]],
    model_fn: Callable[[Any], Dict[str, int]],  
    num_examples: int = 3,
    prompt_builder: Callable = None,
    debug: bool = False,
    calibration_mode: str = "off",
    retrieval_blend_weight: float = 0.0,
) -> Dict[str, Any]:
    if prompt_builder is None:
        prompt_builder = build_few_shot_prompt
    if not records:
        raise ValueError("records must not be empty")

    sample_results = []
    for idx, test_record in enumerate(records):
        print(f"LOOCV progress: {idx + 1}/{len(records)}", flush=True)
        train_records = records[:idx] + records[idx + 1 :]
        example_records = _select_example_records(train_records, test_record, num_examples, prompt_builder)

        prompt_records = example_records + [test_record]
        if prompt_builder is build_prompt_category_batch:
            predicted_raw = model_fn(prompt_records)
        else:
            prompt = prompt_builder(prompt_records, num_examples=min(num_examples, len(train_records)))
            predicted_raw = model_fn(prompt)

        # predicted_raw = model_fn(prompt)  # <--- REMOVE THIS LINE, already handled above
        predicted = _validate_and_clamp_predictions(predicted_raw)
        predicted = _blend_with_retrieval_prior(predicted, example_records, test_record, retrieval_blend_weight)
        predicted = _apply_prediction_calibration(predicted, test_record, calibration_mode)
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
    calibration_mode: str = "off",
    retrieval_blend_weight: float = 0.0,
) -> pd.DataFrame:
    result = run_loocv(
        records,
        model_fn=model_fn,
        num_examples=num_examples,
        prompt_builder=prompt_builder,
        calibration_mode=calibration_mode,
        retrieval_blend_weight=retrieval_blend_weight,
    )
    rows = []
    for item in result["sample_results"]:
        row = {"test_id": item["test_id"], "exact_match_percent": item["exact_match_percent"]}
        row.update(item["mae"])
        rows.append(row)
    return pd.DataFrame(rows)
