import json
import os
import platform
import re
from pathlib import Path
from typing import Any, Dict, List

import requests

_QWEN_GENERATOR_CACHE: Dict[str, Any] = {}

try:
    from .grader import extract_project_features
    from .few_shot_prompt import build_few_shot_prompt, build_prompt_chain_of_thought, build_prompt_rubric_reference, build_prompt_hybrid, build_prompt_question_based
    from .scratch_loader import fetch_project_json
except ImportError:  # pragma: no cover - supports running the file directly
    from grader import extract_project_features
    from few_shot_prompt import build_few_shot_prompt, build_prompt_chain_of_thought, build_prompt_rubric_reference, build_prompt_hybrid, build_prompt_question_based
    from scratch_loader import fetch_project_json

try:
    from .few_shot_prompt import (
        RUBRIC_CATEGORIES,
        build_few_shot_prompt,
        build_prompt_category_batch, # <--- ADD HERE
        build_prompt_chain_of_thought,
        build_prompt_hybrid,
        build_prompt_question_based,
        build_prompt_rubric_reference,
    )
except ImportError:
    from few_shot_prompt import (
        RUBRIC_CATEGORIES,
        build_few_shot_prompt,
        build_prompt_category_batch, # <--- ADD HERE
        build_prompt_chain_of_thought,
        build_prompt_hybrid,
        build_prompt_question_based,
        build_prompt_rubric_reference,
    )


def extract_project_id_from_url(url: str) -> int:
    match = re.search(r"/projects/(\d+)", url)
    if not match:
        raise ValueError(f"Could not extract project ID from URL: {url}")
    return int(match.group(1))


def get_project_features_from_url(url: str) -> Dict[str, Any]:
    project_id = extract_project_id_from_url(url)
    project_json = fetch_project_json(project_id)
    if not project_json:
        raise ValueError(f"Could not fetch Scratch project for {url}")
    return extract_project_features(project_json)


def build_prompt_for_url(url: str, records: List[Dict[str, Any]], num_examples: int = 3, prompt_style: str = "few_shot") -> str:
    features = get_project_features_from_url(url)
    prompt_records = records[:max(1, min(num_examples, len(records) - 1))] + [{"id": None, "features": features, "grades": {}}]
    if prompt_style == "chain_of_thought":
        builder = build_prompt_chain_of_thought
    elif prompt_style == "rubric_reference":
        builder = build_prompt_rubric_reference
    elif prompt_style == "hybrid":
        builder = build_prompt_hybrid
    elif prompt_style == "question_based":
        builder = build_prompt_question_based
    else:
        builder = build_few_shot_prompt
    return builder(prompt_records, num_examples=num_examples)


def query_ollama(prompt: str, model: str = "llama3:latest") -> str:
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    options: Dict[str, Any] = {}
    if os.getenv("OLLAMA_TEMPERATURE"):
        options["temperature"] = float(os.getenv("OLLAMA_TEMPERATURE", "0") or 0)
    if os.getenv("OLLAMA_TOP_P"):
        options["top_p"] = float(os.getenv("OLLAMA_TOP_P", "0") or 0)
    if os.getenv("OLLAMA_SEED"):
        options["seed"] = int(os.getenv("OLLAMA_SEED", "0") or 0)
    if os.getenv("OLLAMA_NUM_PREDICT"):
        options["num_predict"] = int(os.getenv("OLLAMA_NUM_PREDICT", "0") or 0)
    if os.getenv("OLLAMA_REPEAT_PENALTY"):
        options["repeat_penalty"] = float(os.getenv("OLLAMA_REPEAT_PENALTY", "0") or 0)
    if options:
        payload["options"] = options

    response = requests.post(f"{ollama_base_url}/api/generate", json=payload, timeout=120)
    response.raise_for_status()
    payload_json = response.json()
    return payload_json.get("response", "")


def _get_qwen_generator(model: str):
    if model in _QWEN_GENERATOR_CACHE:
        return _QWEN_GENERATOR_CACHE[model]

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "qwen_local provider requires 'transformers' and 'torch'. "
            "Install with: pip install transformers torch"
        ) from exc

    # Safer defaults for local inference:
    # - macOS defaults to CPU to avoid common MPS/accelerate segfaults while loading larger models
    # - Linux/Windows defaults to CUDA when available, else CPU
    requested_device = os.getenv("QWEN_DEVICE", "").strip().lower()
    if requested_device in {"cpu", "cuda", "mps"}:
        runtime_device = requested_device
    else:
        if platform.system() == "Darwin":
            runtime_device = "cpu"
        elif torch.cuda.is_available():
            runtime_device = "cuda"
        else:
            runtime_device = "cpu"

    if runtime_device == "cuda":
        selected_dtype = torch.float16
    else:
        selected_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model)
    load_kwargs = {
        "dtype": selected_dtype,
        "low_cpu_mem_usage": True,
    }

    quantization_mode = os.getenv("QWEN_QUANTIZATION", "none").strip().lower()
    if quantization_mode in {"4bit", "8bit"}:
        if runtime_device != "cuda":
            print(
                f"QWEN_QUANTIZATION={quantization_mode} ignored because runtime device is '{runtime_device}'. "
                "4-bit/8-bit quantization in this path is CUDA-only.",
                flush=True,
            )
        else:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise RuntimeError(
                    "QWEN_QUANTIZATION requires bitsandbytes support. Install with: pip install bitsandbytes"
                ) from exc

            if quantization_mode == "4bit":
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=selected_dtype,
                )
                # Quantized CUDA models are already sharded/placed by HF.
                load_kwargs["device_map"] = "auto"
            elif quantization_mode == "8bit":
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                load_kwargs["device_map"] = "auto"
    try:
        qwen_model = AutoModelForCausalLM.from_pretrained(model, **load_kwargs)
    except TypeError:
        # Backward compatibility for transformers versions that still expect torch_dtype.
        legacy_kwargs = {
            "torch_dtype": selected_dtype,
            "low_cpu_mem_usage": True,
        }
        qwen_model = AutoModelForCausalLM.from_pretrained(model, **legacy_kwargs)

    if runtime_device == "cuda" and torch.cuda.is_available() and "device_map" not in load_kwargs:
        qwen_model = qwen_model.to("cuda")
    elif runtime_device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        qwen_model = qwen_model.to("mps")
    else:
        qwen_model = qwen_model.to("cpu")

    # Avoid max_length + max_new_tokens warning from default generation config.
    qwen_model.generation_config.max_length = None

    cache_entry = {"model": qwen_model, "tokenizer": tokenizer, "device": runtime_device}
    _QWEN_GENERATOR_CACHE[model] = cache_entry
    return cache_entry


def query_qwen_local(
    prompt: str,
    model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 0.9,
    seed: int | None = None,
) -> str:
    import torch

    generator = _get_qwen_generator(model)
    qwen_model = generator["model"]
    tokenizer = generator["tokenizer"]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict grading assistant. "
                "Return only a JSON object with rubric keys and integer scores."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    rendered_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered_prompt, return_tensors="pt")
    inputs = {key: value.to(qwen_model.device) for key, value in inputs.items()}

    do_sample = temperature > 0.0

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs.update({"temperature": temperature, "top_p": top_p})

    with torch.no_grad():
        generated = qwen_model.generate(**inputs, **generation_kwargs)

    if generated is None or generated.shape[0] == 0:
        return ""

    generated_tokens = generated[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def query_with_provider(
    prompt: str,
    provider: str = "ollama",
    model: str | None = None,
    qwen_max_new_tokens: int = 256,
    qwen_temperature: float = 0.0,
    qwen_top_p: float = 0.9,
    qwen_seed: int | None = None,
) -> str:
    normalized_provider = provider.strip().lower()

    if normalized_provider == "ollama":
        resolved_model = model or "llama3:latest"
        return query_ollama(prompt, model=resolved_model)

    if normalized_provider == "qwen_local":
        resolved_model = model or "Qwen/Qwen2.5-1.5B-Instruct"
        return query_qwen_local(
            prompt,
            model=resolved_model,
            max_new_tokens=qwen_max_new_tokens,
            temperature=qwen_temperature,
            top_p=qwen_top_p,
            seed=qwen_seed,
        )

    raise ValueError(f"Unsupported provider: {provider}. Use 'ollama' or 'qwen_local'.")


def parse_grades_from_response(text: str) -> Dict[str, int]:
    if not text:
        return {}

    for pattern in [
        re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE),
        re.compile(r"\{[\s\S]*\}", re.IGNORECASE),
    ]:
        for match in pattern.finditer(text):
            candidate = match.group(1).strip() if match.lastindex else match.group(0).strip()
            if not candidate.startswith("{"):
                continue
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return {key: int(value) for key, value in parsed.items() if isinstance(value, (int, float))}
            except json.JSONDecodeError:
                continue

    return {}


def grade_prompt_with_ollama(prompt: str, model: str = "llama3:latest") -> Dict[str, int]:
    response_text = query_ollama(prompt, model=model)
    return parse_grades_from_response(response_text)


def grade_prompt_with_qwen_local(
    prompt: str,
    model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 0.9,
    seed: int | None = None,
) -> Dict[str, int]:
    response_text = query_qwen_local(
        prompt,
        model=model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
    return parse_grades_from_response(response_text)


def grade_prompt_with_provider(
    prompt: str,
    provider: str = "ollama",
    model: str | None = None,
    qwen_max_new_tokens: int = 256,
    qwen_temperature: float = 0.0,
    qwen_top_p: float = 0.9,
    qwen_seed: int | None = None,
) -> Dict[str, int]:
    response_text = query_with_provider(
        prompt,
        provider=provider,
        model=model,
        qwen_max_new_tokens=qwen_max_new_tokens,
        qwen_temperature=qwen_temperature,
        qwen_top_p=qwen_top_p,
        qwen_seed=qwen_seed,
    )
    return parse_grades_from_response(response_text)

def grade_prompt_category_batched(
    records: List[Dict[str, Any]],
    provider: str = "ollama",
    model: str | None = None,
    num_examples: int = 3,
    prompt_style: str = "hybrid",  # Pass style along
    **kwargs
) -> Dict[str, int]:
    combined_grades: Dict[str, int] = {}

    for category_name in RUBRIC_CATEGORIES.keys():
        prompt = build_prompt_category_batch(
            records=records,
            category_name=category_name,
            num_examples=num_examples,
            prompt_style=prompt_style
        )
        
        response_text = query_with_provider(
            prompt,
            provider=provider,
            model=model,
            **kwargs
        )
        
        batch_grades = parse_grades_from_response(response_text)
        combined_grades.update(batch_grades)

    return combined_grades


def save_predictions(predictions: Dict[str, Any], output_path: str) -> None:
    output_path = Path(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(predictions, handle, indent=2)


def main():
    base_dir = Path(__file__).resolve().parent.parent
    dataset_path = base_dir / "enriched_dataset.json"
    output_path = base_dir / "ollama_predictions.json"
    provider = os.getenv("LOOCV_PROVIDER", "ollama").strip().lower()
    if provider == "qwen_local":
        default_model = "Qwen/Qwen2.5-1.5B-Instruct"
    else:
        default_model = "llama3:latest"
    model = os.getenv("LOOCV_MODEL", default_model)
    qwen_max_new_tokens = int(os.getenv("QWEN_MAX_NEW_TOKENS", "256") or 256)
    qwen_temperature = float(os.getenv("QWEN_TEMPERATURE", "0") or 0)
    qwen_top_p = float(os.getenv("QWEN_TOP_P", "0.9") or 0.9)
    qwen_seed = int(os.getenv("QWEN_SEED")) if os.getenv("QWEN_SEED") else None

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    url = os.getenv("SCRATCH_TEST_URL", "https://scratch.mit.edu/projects/1270822989")
    prompt = build_prompt_for_url(url, records, num_examples=3)
    print(prompt)
    print(f"\nCalling provider: {provider} ({model})...\n")
    response = query_with_provider(
        prompt,
        provider=provider,
        model=model,
        qwen_max_new_tokens=qwen_max_new_tokens,
        qwen_temperature=qwen_temperature,
        qwen_top_p=qwen_top_p,
        qwen_seed=qwen_seed,
    )
    print(response)
    save_predictions(
        {
            "url": url,
            "provider": provider,
            "model": model,
            "prompt": prompt,
            "response": response,
        },
        str(output_path),
    )


if __name__ == "__main__":
    main()
