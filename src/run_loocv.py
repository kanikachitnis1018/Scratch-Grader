import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.append(str(Path(__file__).resolve().parent))

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


def load_records(path: str) -> list[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


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

    if prompt_style == "chain_of_thought":
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

    def model_fn(prompt: str) -> Dict[str, int]:
        return grade_prompt_with_provider(
            prompt,
            provider=provider,
            model=model_name,
            qwen_max_new_tokens=qwen_max_new_tokens,
            qwen_temperature=qwen_temperature,
            qwen_top_p=qwen_top_p,
            qwen_seed=qwen_seed,
        )

    result = run_loocv(
        records,
        model_fn=model_fn,
        num_examples=num_examples,
        prompt_builder=prompt_builder,
        debug=debug,
        calibration_mode=calibration_mode,
        retrieval_blend_weight=retrieval_blend_weight,
    )
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
