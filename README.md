# scratch-grader

scratch-grader is a small Python toolkit for grading Scratch projects against a 20-dimension rubric. It can:

- fetch Scratch project JSON by project ID or URL
- extract rubric-aligned features from the project structure
- build few-shot prompts for an Ollama model
- run leave-one-out cross validation over an enriched dataset
- expose a simple HTTP endpoint that returns a prompt path and parsed grades

## What the code does

The pipeline starts with raw grading data and ends with rubric predictions:

1. `src/main.py` reads an Excel sheet from `DATASET_PATH`, fetches each Scratch project, extracts features, and writes `enriched_dataset.json`.
2. `src/generate_prompt_from_test_json.py` loads the enriched dataset, fetches features for a Scratch URL, builds a prompt, and writes `few_shot_prompt.txt`.
3. `src/ollama_grader.py` sends a prompt to a local Ollama server, parses the JSON-like response, and can save raw prediction output.
4. `src/loocv.py` evaluates the grader in leave-one-out mode and reports metrics such as accuracy, MAE, Cohen’s kappa, precision, recall, and F1.
5. `src/server.py` wraps prompt generation and Ollama grading in a basic HTTP server for URL-based requests.

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
- `LOOCV_PROMPT` to choose `few_shot`, `chain_of_thought`, `rubric_reference`, or `hybrid`
- `LOOCV_PROMPT` to choose `few_shot`, `chain_of_thought`, `rubric_reference`, `hybrid`, or `question_based`
- `LOOCV_BALANCED` to enable balanced example selection
- `LOOCV_DEBUG` to print extra diagnostics
- `LOOCV_OUTPUT` to save LOOCV summary JSON

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

Start the HTTP server:

```bash
python src/server.py
```

The server accepts JSON `POST` requests with a `url` field and optional `prompt_style`, then returns a response payload containing:

- `status`
- `url`
- `prompt_path`
- `grades`

## Outputs

The scripts write a few files in the project root:

- `enriched_dataset.json` from `src/main.py`
- `few_shot_prompt.txt` from prompt-generation scripts and the HTTP server
- `ollama_predictions.json` from `src/ollama_grader.py`
- `loocv_results.json` or another summary file if you set `LOOCV_OUTPUT`

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

- `src/ollama_grader.py` and `src/server.py` expect Ollama to be running locally.
- `src/main.py` overwrites `enriched_dataset.json` when it runs.
- `src/generate_prompt_from_test_json.py` and the server overwrite `few_shot_prompt.txt`.
- The code is designed to be run from the repository root so the `src` imports resolve cleanly.