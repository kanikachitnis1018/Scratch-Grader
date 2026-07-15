import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.append(str(Path(__file__).resolve().parent))

try:
    from .loocv import run_loocv
    from .ollama_grader import grade_prompt_with_ollama
    from .few_shot_prompt import build_few_shot_prompt, build_prompt_chain_of_thought, build_prompt_rubric_reference, build_prompt_hybrid
    from .balanced_selection import select_balanced_examples
except ImportError:  # pragma: no cover - supports running the file directly
    from loocv import run_loocv
    from ollama_grader import grade_prompt_with_ollama
    from few_shot_prompt import build_few_shot_prompt, build_prompt_chain_of_thought, build_prompt_rubric_reference, build_prompt_hybrid
    from balanced_selection import select_balanced_examples


def load_records(path: str) -> list[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    dataset_path = os.getenv("LOOCV_DATASET", str(base_dir / "enriched_dataset.json"))
    limit = int(os.getenv("LOOCV_LIMIT", "0") or 0)
    num_examples = int(os.getenv("LOOCV_NUM_EXAMPLES", "3") or 3)
    model_name = os.getenv("LOOCV_MODEL", "llama3:latest")
    prompt_style = os.getenv("LOOCV_PROMPT", "few_shot")
    use_balanced = os.getenv("LOOCV_BALANCED", "").lower() in ("1", "true", "yes")
    debug = os.getenv("LOOCV_DEBUG", "").lower() in ("1", "true", "yes")

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
    else:
        prompt_builder = build_few_shot_prompt
    print(f"Using prompt style: {prompt_style}", flush=True)
    print(f"Using {num_examples} examples", flush=True)
    if debug:
        print(f"Debug mode enabled", flush=True)

    def model_fn(prompt: str) -> Dict[str, int]:
        return grade_prompt_with_ollama(prompt, model=model_name)

    result = run_loocv(records, model_fn=model_fn, num_examples=num_examples, prompt_builder=prompt_builder, debug=debug)
    print(json.dumps(result["summary"], indent=2))

    output_path = os.getenv("LOOCV_OUTPUT", "")
    if output_path:
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(result["summary"], handle, indent=2)
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
