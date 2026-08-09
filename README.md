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
- `LOOCV_OUTPUT` to save LOOCV summary JSON (recommended under `results/`)
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
LOOCV_NUM_EXAMPLES=7 \
LOOCV_LIMIT=72 \
LOOCV_CALIBRATION=v1 \
LOOCV_RETRIEVAL_BLEND=0 \
OLLAMA_TEMPERATURE=0 \
OLLAMA_TOP_P=1 \
OLLAMA_SEED=42 \
OLLAMA_NUM_PREDICT=512 \
LOOCV_OUTPUT=results/calibration_v1/loocv_ollama_hybrid_calibrated_v1_n7.json \
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
LOOCV_OUTPUT=results/calibration_v1/loocv_qwen_local_hybrid_calibrated_v1_n3.json \
python3 src/run_loocv.py
```

Results organization:

- Confirmed runs: `results/calibration_v1/` and `results/calibration_v2/`
- Temporary experiments: `results/tmp/`
- Archived/legacy outputs: `results/archive/`
- Optional quick-access copy of best run at root: `latest_best_run.json`

Suggested naming convention:

- `loocv_<model>_<prompt>_<calibration>_n<examples>.json`
- Example: `loocv_ollama_hybrid_calibrated_v2_n8.json`

## Outputs

The scripts write a few files in the project root and `results/`:

- `enriched_dataset.json` from `src/main.py`
- `few_shot_prompt.txt` from prompt-generation scripts and the HTTP server
- `ollama_predictions.json` from `src/ollama_grader.py`
- `results/calibration_v1/` and `results/calibration_v2/` for confirmed LOOCV summaries
- `results/tmp/` for exploratory `tmp_*.json` runs
- `results/archive/` for older run artifacts
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

## Developer Plan & Patches (concise)

This project roadmap lists prioritized, small, deterministic changes to reach the goals described in the project brief (contrastive few-shot selection, calibration v2, selective reasoning, self-consistency averaging, richer features, strict JSON outputs). Apply these changes incrementally and run unit tests after each commit.

High-level steps (apply in order)
1. Contrastive few-shot selection
   - Add a routine balanced_selection.select_contrastive_examples(enriched_dataset, n_per_group, seed)
   - Behavior: for each rubric dimension select ceil(n/2) low-score and floor(n/2) high-score examples; seed-based shuffle for determinism.
   - Files: balanced_selection.py, tests/test_balanced_selection.py

2. Calibration v2 (confusion-matrix remapping)
   - Add module calibration/confusion_map.py with build_confusion_map(true_labels, pred_labels) and apply_confusion_remap(pred_probs, confusion_map).
   - Integrate in loocv.py after raw predictions and before final scoring.
   - Files: calibration/confusion_map.py, loocv.py, tests/test_calibration_v2.py

3. Selective reasoning expansions
   - In few_shot_prompt.py, add optional reasoning blocks for "algorithms", "sequencing", "integration".
   - Expose prompt builder param selective_reasoning=['algorithms','sequencing'].
   - Ensure few_shot prompt asks the model to return strict JSON and to label sections explicitly.

4. Self-consistency averaging
   - Modify run_loocv.py to accept LOOCV_SELF_CONSISTENCY_PASSES (default 1). If >1, run the provider multiple times with different deterministic seeds and average scores per rubric (mean then round or use MAE-tuned mapping).
   - Keep seeds reproducible: use base_seed + pass_index.

5. Richer feature preprocessing
   - In scratch_loader.py add compute_max_nesting_depth(ast), detect_clone_events(ast), detect_var_initializations(ast) helpers.
   - Save computed features into atomic_features.json and enriched_dataset.json.

6. Strict JSON output enforcement
   - Enforce strict JSON response format in prompt templates and add a sanitizer in ollama_grader.py that:
     - extracts the JSON block, tries json.loads, and fails fast with parse diagnostics in debug mode.
     - If provider returns non-JSON, return a structured error object and save raw output to ollama_predictions.json for debugging.

7. Category batching and deterministic decoding
   - Ensure graders run category-by-category (4 groups of 5) when LOOCV_BATCH_BY_CATEGORY=true.
   - For Ollama, always pass OLLAMA_SEED/OLLAMA_NUM_PREDICT and temperature=0 for deterministic runs.

Per-file actionable summaries (minimal hints)

- src/balanced_selection.py
  - Summary: Add select_contrastive_examples(dataset, n_examples, seed=42, balanced=True)
  - Hint: group by dimension score, sample deterministic low/high stratified examples.
  - Tests: assert equal counts and reproducible selection with same seed.

- src/calibration/confusion_map.py (new)
  - Summary: compute per-dimension confusion matrices from LOOCV outputs and expose mapping functions to remap predicted scores -> calibrated scores.
  - Hint: store mapping as JSON mapping from predicted->remapped (or probability adjustment).

- src/few_shot_prompt.py
  - Summary: add selective_reasoning blocks and explicit "RETURN ONLY JSON" suffix; include example-driven reasoning snippets.
  - Hint: append a strict schema and "If you cannot produce valid JSON, return {'error': 'parse_failed', 'raw': <raw_text>}".

- src/run_loocv.py
  - Summary: add LOOCV_SELF_CONSISTENCY_PASSES env var handling and averaging logic.
  - Hint: for averaging use arithmetic mean across passes, then keep both mean (float) and rounded score.

- src/scratch_loader.py
  - Summary: add compute_max_nesting_depth, detect_clone_events, detect_var_initializations; integrate into atomic feature extraction pipeline.
  - Hint: small AST recursion with a max depth accumulator; mark variables seen in first several blocks as initialized.

- src/ollama_grader.py
  - Summary: add JSON extraction/sanitizer, deterministic seeding, and better error diagnostics; save raw responses to ollama_predictions.json with timestamps.
  - Hint: try to locate final { ... } block in text using rfind('{')/rfind('}') heuristics before json.loads.

- src/loocv.py
  - Summary: integrate calibration v2 hooks and optional self-consistency averaging; accept LOOCV_BATCH_BY_CATEGORY.
  - Hint: calibration hook signature: calibrated = calibration.apply(preds, mode='v2', confusion_map=cm)

- tests/*
  - Add unit tests verifying deterministic behavior (same seed → same outputs), JSON parsing for edge cases, contrastive selection reproducibility, confusion map application.

## Exact commands — run LOOCV and read accuracy

Recommended deterministic (Ollama) full run — writes summary JSON to LOOCV_OUTPUT:

```bash
# deterministic Ollama LOOCV (recommended)
LOOCV_PROVIDER=ollama \
LOOCV_MODEL=llama3:latest \
LOOCV_PROMPT=hybrid \
LOOCV_NUM_EXAMPLES=7 \
LOOCV_LIMIT=72 \
LOOCV_SELF_CONSISTENCY_PASSES=3 \
LOOCV_CALIBRATION=v2 \
OLLAMA_TEMPERATURE=0 \
OLLAMA_TOP_P=1 \
OLLAMA_SEED=42 \
OLLAMA_NUM_PREDICT=512 \
LOOCV_OUTPUT=results/loocv_ollama_accuracy.json \
python3 src/run_loocv.py
```

Quick iterative run (smaller, faster):

```bash
LOOCV_PROVIDER=ollama LOOCV_MODEL=llama3:latest LOOCV_PROMPT=hybrid LOOCV_NUM_EXAMPLES=5 LOOCV_LIMIT=20 OLLAMA_TEMPERATURE=0 OLLAMA_SEED=42 LOOCV_OUTPUT=results/loocv_quick.json python3 src/run_loocv.py
```

Inspect results (use jq):

```bash
# show overall summary
jq '.summary' results/loocv_ollama_accuracy.json

# show overall accuracy value only
jq '.summary.accuracy' results/loocv_ollama_accuracy.json

# show per-dimension accuracies
jq '.per_dimension | to_entries[] | {dimension: .key, accuracy: .value.accuracy}' results/loocv_ollama_accuracy.json
```

Notes
- Always fix OLLAMA_TEMPERATURE=0 and OLLAMA_SEED for reproducible comparisons.
- Use LOOCV_LIMIT during prompt/hyperparameter iteration to run on a subset.
- Increase LOOCV_SELF_CONSISTENCY_PASSES for more stable average predictions (slower).

Recommended deterministic commands (examples)
- LOOCV deterministic Ollama run:
  LOOCV_PROVIDER=ollama LOOCV_MODEL=llama3:latest LOOCV_PROMPT=hybrid LOOCV_NUM_EXAMPLES=7 LOOCV_SELF_CONSISTENCY_PASSES=3 LOOCV_CALIBRATION=v2 OLLAMA_SEED=42 OLLAMA_TEMPERATURE=0 OLLAMA_NUM_PREDICT=512 python3 src/run_loocv.py

Testing checklist
- run: python -m unittest discover -s tests -p "test_*.py"
- verify: results/ contains deterministic outputs; small diff allowed only when changing LOOCV_SELF_CONSISTENCY_PASSES or calibration mode.

Developer notes
- Keep changes small per commit. Each file change described above should be implemented in its own commit with test updates.

## Additional strategies to improve accuracy (actionable)

If current LOOCV accuracy is below expectations, apply the following prioritized interventions. Run small, controlled experiments (use LOOCV_LIMIT) and change one variable at a time.

1) Diagnose error modes
- Compute per-dimension confusion matrices from a baseline run to identify common confusions.
- Files: src/loocv.py, (new) src/calibration/confusion_map.py
- Quick command: run a small LOOCV then:
  jq '.per_dimension | to_entries[] | {dim:.key, confusions:.value.confusion}' results/loocv_ollama_accuracy.json

2) Contrastive few-shot & edge-case examples
- Ensure few-shot pools include low, mid, high scores and rare/edge behaviours (cloning, deep nesting).
- Files: src/balanced_selection.py, src/few_shot_prompt.py
- Tip: set LOOCV_BALANCED=true and LOOCV_NUM_EXAMPLES to an odd number (e.g., 7) to preserve majority contrast.

3) Enrich features and surface them to the prompt
- Add max nesting depth, clone lifecycle flags, variable-init counts, and custom-block signatures to atomic features.
- Files: src/scratch_loader.py, src/grader.py
- Include a one-line “Facts:” summary before the examples in the prompt (helps the model anchor).

4) Self-consistency & deterministic ensembling
- Run multiple deterministic passes (LOOCV_SELF_CONSISTENCY_PASSES=3–5) with seeds base_seed + pass_idx; average per-dimension then round.
- Files: src/run_loocv.py, src/ollama_grader.py
- Command example:
  LOOCV_SELF_CONSISTENCY_PASSES=3 OLLAMA_SEED=42 OLLAMA_TEMPERATURE=0 LOOCV_LIMIT=20 ... python3 src/run_loocv.py

5) Calibration v2: confusion-matrix remapping
- Build a per-dimension map from predicted->most-likely-true using LOOCV history; apply remap to raw preds.
- Files: src/calibration/confusion_map.py, src/loocv.py
- Use holdout folds to avoid overfitting the confusion map.

6) Reranking using parse-confidence / log-prob
- If provider exposes token-prob or log-prob, score candidate JSON parses and pick highest-confidence answer; otherwise use heuristic confidence (complete JSON + no error keys).
- Files: src/ollama_grader.py
- Save both parse and confidence in results for analysis.

7) Multi-prompt ensembled voting
- Create 3 small prompt variants (different instructions, different reasoning blocks) and aggregate via majority/average.
- Files: src/few_shot_prompt.py, src/run_loocv.py
- Deterministic seeds + temp=0 still apply to keep runs comparable.

8) Hard constraints / post-processing caps
- Apply rule-based clamps: e.g., if feature indicates "no cloning" then set cloning-related rubric dims to 0; prevent impossible high scores when features contradict them.
- Files: src/grader.py, src/loocv.py

9) Data augmentation / synthetic contrastive examples
- Create synthetic minimal projects that exemplify extreme rubric scores and include them as few-shot examples.
- Files: data/ or scripts that append to enriched_dataset.json used by prompt builders.

10) Targeted fine-tuning (last resort)
- If budget allows, fine-tune or LoRA-adapt a smaller instruction model on 1–2k labeled examples with the strict JSON output format.
- Files: new scripts for fine-tuning (outside this repo), then integrate as a provider.

Experiment recipe (small-step loop)
1. Baseline: run LOOCV_LIMIT=20 with deterministic seed and temp=0, save LOOCV_OUTPUT.
2. Add one change (e.g., contrastive examples), rerun same LOOCV_LIMIT and compare summary.accuracy.
3. Track per-dimension deltas to see where gains occur.
4. If no improvement after 3 orthogonal changes, try ensembling or calibration remap.

Quick commands (examples)
- Baseline deterministic quick run:
  LOOCV_PROVIDER=ollama LOOCV_MODEL=llama3:latest LOOCV_PROMPT=hybrid LOOCV_NUM_EXAMPLES=7 LOOCV_LIMIT=20 OLLAMA_TEMPERATURE=0 OLLAMA_SEED=42 LOOCV_OUTPUT=results/loocv_quick.json python3 src/run_loocv.py

- Self-consistency + ensembling quick trial:
  LOOCV_SELF_CONSISTENCY_PASSES=3 LOOCV_PROVIDER=ollama LOOCV_NUM_EXAMPLES=7 LOOCV_LIMIT=20 OLLAMA_TEMPERATURE=0 OLLAMA_SEED=42 LOOCV_OUTPUT=results/loocv_selfcons.json python3 src/run_loocv.py

Monitoring & prioritization
- Prioritize changes that fix high-frequency confusions first (use confusion matrix).
- Use LOOCV_LIMIT to iterate quickly.
- Keep detailed raw outputs (ollama_predictions.json) and a small CSV of failure cases to drive manual inspection and targeted few-shot construction.