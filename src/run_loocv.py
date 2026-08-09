# Add a small sys.path bootstrap so `from calibration...` works when running the script directly.
import sys
import pathlib

# Ensure the 'src' directory is on sys.path when running this file directly
_SRC_DIR = pathlib.Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
	sys.path.insert(0, str(_SRC_DIR))

import json
import os
import statistics
from pathlib import Path
from typing import Any, Dict

# Import calibration helpers using the local package path (calibration.*).
# Fall back gracefully if the module is not available.
try:
	from calibration.confusion_map import apply_confusion_remap, load_confusion_map, build_confusion_map, save_confusion_map
except Exception:
	apply_confusion_remap = None
	load_confusion_map = lambda p: {}
	build_confusion_map = None
	save_confusion_map = None

try:
    from .loocv import run_loocv
    from .ollama_grader import grade_prompt_with_provider
    from .few_shot_prompt import build_few_shot_prompt, build_prompt_chain_of_thought, build_prompt_rubric_reference, build_prompt_hybrid, build_prompt_question_based
    from .balanced_selection import select_balanced_examples
except ImportError:  # pragma: no cover - supports running the file directly
    from loocv import run_loocv
    from ollama_grader import grade_prompt_with_provider
    from few_shot_prompt import build_few_shot_prompt, build_prompt_chain_of_thought, build_prompt_rubric_reference, build_prompt_hybrid, build_prompt_question_based
    from balanced_selection import select_balanced_examples

try:
    from .ollama_grader import (
        grade_prompt_with_provider,
        grade_prompt_category_batched,
    )
    from .few_shot_prompt import (
        build_few_shot_prompt,
        build_prompt_chain_of_thought,
        build_prompt_rubric_reference,
        build_prompt_hybrid,
        build_prompt_question_based,
        build_prompt_category_batch,
    )
except ImportError:
    from ollama_grader import (
        grade_prompt_with_provider,
        grade_prompt_category_batched,
    )
    from few_shot_prompt import (
        build_few_shot_prompt,
        build_prompt_chain_of_thought,
        build_prompt_rubric_reference,
        build_prompt_hybrid,
        build_prompt_question_based,
        build_prompt_category_batch,
    )


def load_records(path: str) -> list[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _run_single_pass(item, provider_args, pass_index):
    # set deterministic seed per pass
    base_seed = int(os.getenv("OLLAMA_SEED", "42"))
    seed = base_seed + pass_index
    provider_args['seed'] = seed
    # ...existing code to call grader/provider...
    # should return dict of dimension->score (floats or ints)
    return {"dim1": 3, "dim2": 2}  # placeholder


def evaluate_with_self_consistency(items, provider_args):
    passes = int(os.getenv("LOOCV_SELF_CONSISTENCY_PASSES", "1"))
    results = {}
    for item in items:
        all_preds = []
        for p in range(passes):
            pred = _run_single_pass(item, provider_args.copy(), p)
            all_preds.append(pred)
        # average per-dimension
        avg = {}
        for dim in all_preds[0].keys():
            values = [pred[dim] for pred in all_preds]
            avg_val = statistics.mean(values)
            avg[dim] = {"mean": avg_val, "rounded": round(avg_val)}
        results[item.get("id")] = avg
    return results


def _aggregate_self_consistency(pred_list):
    """
    pred_list: list of dicts sample_id -> {dim:score}
    returns aggregated: sample_id -> {dim: {'mean':float,'rounded':int}}
    """
    agg = {}
    sids = set()
    for preds in pred_list:
        sids.update(preds.keys())
    for sid in sids:
        by_dim = {}
        # collect values per-dim across passes
        dims = {}
        for preds in pred_list:
            p = preds.get(sid, {})
            for dim, val in p.items():
                dims.setdefault(dim, []).append(val)
        for dim, vals in dims.items():
            meanv = statistics.mean(vals)
            by_dim[dim] = {"mean": meanv, "rounded": int(round(meanv))}
        agg[sid] = by_dim
    return agg


def run_loocv_with_self_consistency(items, provider_args, calibration_map_path=None):
    """
    High-level helper: runs LOOCV passes (calls existing evaluation per-pass),
    aggregates predictions and applies optional confusion remap.
    - items: list of items to grade
    - provider_args: dict to pass into per-pass grader (must accept 'seed')
    - calibration_map_path: optional JSON path to confusion_map to apply
    """
    passes = int(os.getenv("LOOCV_SELF_CONSISTENCY_PASSES", "1"))
    base_seed = int(os.getenv("OLLAMA_SEED", "42"))
    all_pass_preds = []
    for p in range(passes):
        seed = base_seed + p
        provider_args['seed'] = seed
        # Call existing per-run LOOCV function (placeholder name: single_pass_loocv)
        # single_pass_loocv should return dict sample_id -> {dim:pred_score}
        # ...existing code calls into single_pass_loocv(...)
        # Example placeholder (replace integration point):
        # pass_preds = single_pass_loocv(items, provider_args)
        pass_preds = {}  # <-- integrate your existing single-pass loocv call here
        all_pass_preds.append(pass_preds)
    # aggregate
    aggregated = _aggregate_self_consistency(all_pass_preds)
    # convert aggregated to same structure as preds for remapping: sample_id -> {dim:rounded}
    raw_preds_for_remap = {sid: {dim: v['rounded'] for dim, v in dims.items()} for sid, dims in aggregated.items()}
    # apply confusion remap if available
    if calibration_map_path and apply_confusion_remap:
        conf_map = load_confusion_map(calibration_map_path)
        raw_preds_for_remap = apply_confusion_remap(raw_preds_for_remap, conf_map)
    # return final per-sample aggregated structure
    return {"aggregated": aggregated, "final_rounded": raw_preds_for_remap}


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    dataset_path = os.getenv("LOOCV_DATASET", str(base_dir / "enriched_dataset.json"))
    limit = int(os.getenv("LOOCV_LIMIT", "0") or 0)
    num_examples = int(os.getenv("LOOCV_NUM_EXAMPLES", "3") or 3)
    provider = os.getenv("LOOCV_PROVIDER", "ollama").strip().lower()
    if provider == "qwen_local":
        default_model = "Qwen/Qwen2.5-1.5B-Instruct"
    else:
        default_model = "llama3:latest"
    model_name = os.getenv("LOOCV_MODEL", default_model)
    prompt_style = os.getenv("LOOCV_PROMPT", "few_shot")
    use_balanced = os.getenv("LOOCV_BALANCED", "").lower() in ("1", "true", "yes")
    debug = os.getenv("LOOCV_DEBUG", "").lower() in ("1", "true", "yes")
    calibration_mode = os.getenv("LOOCV_CALIBRATION", "off").strip().lower()
    retrieval_blend_weight = float(os.getenv("LOOCV_RETRIEVAL_BLEND", "0") or 0)
    qwen_max_new_tokens = int(os.getenv("QWEN_MAX_NEW_TOKENS", "256") or 256)
    qwen_temperature = float(os.getenv("QWEN_TEMPERATURE", "0") or 0)
    qwen_top_p = float(os.getenv("QWEN_TOP_P", "0.9") or 0.9)
    qwen_seed = int(os.getenv("QWEN_SEED")) if os.getenv("QWEN_SEED") else None

    records = load_records(dataset_path)
    if limit > 0:
        records = records[:limit]
    
    if use_balanced:
        records = select_balanced_examples(records, num_examples=num_examples)
        print(f"Using balanced selection (grade-uniform across {len(records)} examples)", flush=True)

    is_batched = os.getenv("LOOCV_BATCHED", "").lower() in ("1", "true", "yes") or "batch" in prompt_style

    if is_batched:
        prompt_builder = build_prompt_category_batch
    elif prompt_style == "chain_of_thought":
        prompt_builder = build_prompt_chain_of_thought
    elif prompt_style == "rubric_reference":
        prompt_builder = build_prompt_rubric_reference
    elif prompt_style == "hybrid":
        prompt_builder = build_prompt_hybrid
    elif prompt_style == "question_based":
        prompt_builder = build_prompt_question_based
    else:
        prompt_builder = build_few_shot_prompt
    print(f"Using prompt style: {prompt_style}", flush=True)
    print(f"Using {num_examples} examples", flush=True)
    print(f"Using provider: {provider}", flush=True)
    print(f"Using model: {model_name}", flush=True)
    print(f"Using calibration: {calibration_mode}", flush=True)
    print(f"Using retrieval blend weight: {retrieval_blend_weight}", flush=True)
    if debug:
        print(f"Debug mode enabled", flush=True)

    is_batched = os.getenv("LOOCV_BATCHED", "").lower() in ("1", "true", "yes") or "batch" in prompt_style

    n_passes = int(os.getenv("LOOCV_SELF_CONSISTENCY_PASSES", "3"))

    def model_fn(prompt_or_records: Any) -> Dict[str, int]:
        all_pass_predictions = []
        
        for pass_idx in range(n_passes):
            # Use a slightly non-zero temperature (0.3) so passes sample variations
            pass_temp = 0.3 if n_passes > 1 else qwen_temperature
            
            if is_batched:
                pred = grade_prompt_category_batched(
                    records=prompt_or_records,
                    provider=provider,
                    model=model_name,
                    num_examples=num_examples,
                    prompt_style=prompt_style,
                    qwen_max_new_tokens=qwen_max_new_tokens,
                    qwen_temperature=pass_temp,
                    qwen_top_p=qwen_top_p,
                    qwen_seed=None,  # Keep seed None so passes differ slightly
                )
            else:
                pred = grade_prompt_with_provider(
                    prompt=prompt_or_records,
                    provider=provider,
                    model=model_name,
                    qwen_max_new_tokens=qwen_max_new_tokens,
                    qwen_temperature=pass_temp,
                    qwen_top_p=qwen_top_p,
                    qwen_seed=None,  # Keep seed None so passes differ slightly
                )
            all_pass_predictions.append(pred)

        # Average predictions across all passes per dimension and round to nearest integer
        if not all_pass_predictions:
            return {}

        averaged_prediction: Dict[str, int] = {}
        dimensions = all_pass_predictions[0].keys()
        
        for dimension in dimensions:
            dim_sum = sum(p.get(dimension, 0) for p in all_pass_predictions)
            averaged_prediction[dimension] = int(round(dim_sum / len(all_pass_predictions)))
            
        return averaged_prediction

    result = run_loocv(
        records,
        model_fn=model_fn,
        num_examples=num_examples,
        prompt_builder=prompt_builder,
        debug=debug,
        calibration_mode=calibration_mode,
        retrieval_blend_weight=retrieval_blend_weight,
    )

    # 5. Output summary results
    print(json.dumps(result["summary"], indent=2))

    output_path = os.getenv("LOOCV_OUTPUT", "")
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as handle:
            json.dump(result["summary"], handle, indent=2)
        print(f"Saved results to {output_file}")

if __name__ == "__main__":
    main()
