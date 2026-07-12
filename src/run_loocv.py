import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.append(str(Path(__file__).resolve().parent))

try:
    from .loocv import run_loocv
    from .ollama_grader import grade_prompt_with_ollama
    from .few_shot_prompt import build_few_shot_prompt, build_prompt_chain_of_thought
except ImportError:  # pragma: no cover - supports running the file directly
    from loocv import run_loocv
    from ollama_grader import grade_prompt_with_ollama
    from few_shot_prompt import build_few_shot_prompt, build_prompt_chain_of_thought


def load_records(path: str) -> list[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    dataset_path = os.getenv("LOOCV_DATASET", str(base_dir / "enriched_dataset.json"))
    limit = int(os.getenv("LOOCV_LIMIT", "0") or 0)
    model_name = os.getenv("LOOCV_MODEL", "llama3:latest")
    prompt_style = os.getenv("LOOCV_PROMPT", "few_shot")

    records = load_records(dataset_path)
    if limit > 0:
        records = records[:limit]

    prompt_builder = build_prompt_chain_of_thought if prompt_style == "chain_of_thought" else build_few_shot_prompt
    print(f"Using prompt style: {prompt_style}", flush=True)

    def model_fn(prompt: str) -> Dict[str, int]:
        return grade_prompt_with_ollama(prompt, model=model_name)

    result = run_loocv(records, model_fn=model_fn, prompt_builder=prompt_builder)
    print(json.dumps(result["summary"], indent=2))

    output_path = os.getenv("LOOCV_OUTPUT", "")
    if output_path:
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(result["summary"], handle, indent=2)
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
