# scratch-grader

scratch-grader is a small Python toolkit for grading Scratch projects against a 20-dimension rubric. It can:

- fetch Scratch project JSON by project ID or URL
- extract rubric-aligned features from the project structure
- build few-shot prompts for local models (Ollama or local Qwen)
- run leave-one-out cross validation over an enriched dataset
- expose a simple HTTP endpoint that returns a prompt path and parsed grades

## What the code does

The pipeline starts with raw grading data and ends with rubric predictions:

1. `src/main.py` reads an Excel sheet from `DATASET_PATH`, fetches each Scratch project, extracts features, and writes `enriched_dataset.json`.
2. `src/generate_prompt_from_test_json.py` loads the enriched dataset, fetches features for a Scratch URL, builds a prompt, and writes `few_shot_prompt.txt`.
3. `src/ollama_grader.py` routes prompts to either Ollama or local Qwen, parses the JSON-like response, and can save raw prediction output.
4. `src/loocv.py` evaluates the grader in leave-one-out mode and reports metrics such as accuracy, MAE, Cohen’s kappa, precision, recall, and F1.
5. `src/server.py` wraps prompt generation and provider-based grading (Ollama or local Qwen) in a basic HTTP server for URL-based requests.
6. `src/evaluate_stage.py` evaluates the deterministic Stage 1 -> Stage 2 scores against the human grades already stored in `enriched_dataset.json`.

The rubric logic itself lives in `src/grader.py`, which turns Scratch block JSON into a compact feature dictionary and maps spreadsheet rubric columns into rubric keys.

## Requirements

- Python 3.10+ recommended
- `pandas`
- `python-dotenv`
- `requests`
- access to a local Ollama server at `http://localhost:11434` for grading

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Setup

Create a `.env` file with the paths and settings you need:

```bash
DATASET_PATH=/path/to/your/graded_projects.xlsx
SCRATCH_TEST_URL=https://scratch.mit.edu/projects/1270822989
PORT=8000
LOOCV_MODEL=llama3:latest
```

Optional environment variables used by the code:

- `SCRATCH_API_BASE` and `SCRATCH_PROJECTS_BASE` for alternate Scratch endpoints
- `LOOCV_DATASET` to point LOOCV at a different JSON file
- `LOOCV_LIMIT` to cap the number of samples processed
- `LOOCV_NUM_EXAMPLES` to control few-shot example count
- `LOOCV_PROMPT` to choose `few_shot`, `chain_of_thought`, `rubric_reference`, `hybrid`, or `question_based`
- `LOOCV_PROVIDER` to choose `ollama` (default) or `qwen_local`
- `LOOCV_MODEL` model name for the selected provider (default: `llama3:latest` for Ollama, `Qwen/Qwen2.5-1.5B-Instruct` for qwen_local)
- `QWEN_MAX_NEW_TOKENS` (default `256`) and `QWEN_TEMPERATURE` when using `LOOCV_PROVIDER=qwen_local`
- `QWEN_TOP_P` (default `0.9`) and `QWEN_SEED` (optional) for reproducible qwen_local sampling behavior
- `QWEN_DEVICE` to force local Qwen device: `cpu`, `mps`, or `cuda` (default is safer `cpu` on macOS)
- `QWEN_QUANTIZATION` optional quantization mode for local Qwen: `none` (default), `8bit`, or `4bit` (CUDA path)
- `HF_TOKEN` (optional) to avoid Hugging Face unauthenticated rate-limit warnings
- `LOOCV_BALANCED` to enable balanced example selection
- `LOOCV_CALIBRATION` optional post-prediction calibration mode (`off` default; set `v1` to enable conservative feature-based caps)
- `LOOCV_RETRIEVAL_BLEND` optional score blending with retrieved examples (default `0`; keep `0` unless a local sweep shows improvement)
- `LOOCV_DEBUG` to print extra diagnostics
- `LOOCV_OUTPUT` to save LOOCV summary JSON
- `OLLAMA_BASE_URL` to override Ollama endpoint (default `http://localhost:11434`)
- `OLLAMA_TEMPERATURE`, `OLLAMA_TOP_P`, `OLLAMA_SEED`, `OLLAMA_NUM_PREDICT`, and `OLLAMA_REPEAT_PENALTY` to tune deterministic evaluation behavior

## How to run

Generate the enriched dataset:

```bash
python src/main.py
```

Generate a prompt for a Scratch project URL:

```bash
python src/generate_prompt_from_test_json.py
```

To use the question-based rubric variant, set `LOOCV_PROMPT=question_based` or pass `prompt_style="question_based"` to the prompt-generation helpers.

Run leave-one-out evaluation:

```bash
python src/run_loocv.py
```

Run LOOCV with Ollama explicitly:

```bash
LOOCV_PROVIDER=ollama LOOCV_MODEL=llama3:latest python src/run_loocv.py
```

Run LOOCV with local Qwen2.5-3B (Transformers backend, no Ollama):

```bash
LOOCV_PROVIDER=qwen_local LOOCV_MODEL=Qwen/Qwen2.5-1.5B-Instruct python src/run_loocv.py
```

Recommended on macOS for stability:

```bash
LOOCV_PROVIDER=qwen_local LOOCV_MODEL=Qwen/Qwen2.5-1.5B-Instruct QWEN_DEVICE=cpu python src/run_loocv.py
```

Optional CUDA quantization (Linux/NVIDIA):

```bash
LOOCV_PROVIDER=qwen_local LOOCV_MODEL=Qwen/Qwen2.5-1.5B-Instruct QWEN_DEVICE=cuda QWEN_QUANTIZATION=4bit python src/run_loocv.py
```

If using `qwen_local`, install optional dependencies first:

```bash
pip install transformers torch
```

For CUDA quantization, also install:

```bash
pip install bitsandbytes
```

Run evaluation for the Stage 1 -> Stage 2 pipeline:

```bash
python src/evaluate_stage.py
```

Start the HTTP server:

```bash
python src/server.py
```

The server accepts JSON `POST` requests with a `url` field and optional `prompt_style`, then returns a response payload containing:

- `status`
- `url`
- `prompt_path`
- `grades`

Server `POST` also supports optional model routing fields:

- `provider`: `ollama` or `qwen_local`
- `model`: model name for the selected provider
- `qwen_max_new_tokens`: max generated tokens (qwen_local only, default `256`)
- `qwen_temperature`: generation temperature (qwen_local only)
- `qwen_top_p`: nucleus sampling value (qwen_local only, default `0.9`)
- `qwen_seed`: random seed (qwen_local only, optional)

Example with Ollama:

```bash
curl -X POST http://localhost:8000 \
	-H "Content-Type: application/json" \
	-d '{
		"url": "https://scratch.mit.edu/projects/1270822989",
		"prompt_style": "hybrid",
		"provider": "ollama",
		"model": "llama3:latest"
	}'
```

Example with local Qwen:

```bash
curl -X POST http://localhost:8000 \
	-H "Content-Type: application/json" \
	-d '{
		"url": "https://scratch.mit.edu/projects/1270822989",
		"prompt_style": "hybrid",
		"provider": "qwen_local",
		"model": "Qwen/Qwen2.5-1.5B-Instruct",
		"qwen_max_new_tokens": 256,
		"qwen_temperature": 0,
		"qwen_top_p": 0.9,
		"qwen_seed": 42
	}'
```

Recommended hybrid evaluation command parity:

```bash
# Ollama hybrid + calibration (strongest recent setting)
LOOCV_PROVIDER=ollama \
LOOCV_MODEL=llama3:latest \
LOOCV_PROMPT=hybrid \
LOOCV_NUM_EXAMPLES=4 \
LOOCV_LIMIT=72 \
LOOCV_CALIBRATION=v1 \
LOOCV_RETRIEVAL_BLEND=0 \
OLLAMA_TEMPERATURE=0 \
OLLAMA_TOP_P=1 \
OLLAMA_SEED=42 \
OLLAMA_NUM_PREDICT=512 \
LOOCV_OUTPUT=loocv_ollama_hybrid_calibrated.json \
python3 src/run_loocv.py

# Qwen hybrid + calibration
LOOCV_PROVIDER=qwen_local \
LOOCV_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
LOOCV_PROMPT=hybrid \
LOOCV_NUM_EXAMPLES=3 \
LOOCV_LIMIT=72 \
LOOCV_CALIBRATION=v1 \
LOOCV_RETRIEVAL_BLEND=0 \
QWEN_DEVICE=cpu \
QWEN_MAX_NEW_TOKENS=256 \
QWEN_TEMPERATURE=0 \
QWEN_TOP_P=0.9 \
QWEN_SEED=42 \
LOOCV_OUTPUT=loocv_qwen_hybrid_calibrated.json \
python3 src/run_loocv.py
```

## Outputs

The scripts write a few files in the project root:

- `enriched_dataset.json` from `src/main.py`
- `few_shot_prompt.txt` from prompt-generation scripts and the HTTP server
- `ollama_predictions.json` from `src/ollama_grader.py`
- `loocv_results.json` or another summary file if you set `LOOCV_OUTPUT`
- `dumps/raw_projects.json`, `dumps/atomic_features.json`, and `dumps/model_scores.json` from the Stage pipeline in `src/grader.py`

## Testing

The test suite uses `unittest`. Run all tests from the project root with:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

The tests cover:

- feature extraction and dataset enrichment
- few-shot and chain-of-thought prompt building
- LOOCV metrics and held-out sample behavior
- prompt file generation
- the server response payload

## Notes

- `src/ollama_grader.py` and `src/server.py` support either Ollama or local Qwen. Use `LOOCV_PROVIDER` or request-level `provider` to switch.
- `src/main.py` overwrites `enriched_dataset.json` when it runs.
- `src/generate_prompt_from_test_json.py` and the server overwrite `few_shot_prompt.txt`.
- The code is designed to be run from the repository root so the `src` imports resolve cleanly.