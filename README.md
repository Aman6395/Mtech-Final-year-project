# Mtech Final Year Project: RAG with Guardrails

A Retrieval-Augmented Generation (RAG) project. It parses PDFs with Docling, chunks them,
and applies NVIDIA NeMo Guardrails (input/output checks) around the question-answering flow.

## 1. Prerequisites

| Requirement | Version / notes |
|---|---|
| Python | **3.12** (see `.python-version`; `requires-python >= 3.12`) |
| Git | to clone the repo |
| uv (recommended) | Python package manager. Install: https://docs.astral.sh/uv/getting-started/installation/ |
| Internet access | needed on first run to download Hugging Face models (e.g. `sentence-transformers/all-MiniLM-L6-v2`) and Docling models |
| Gemini API key | needed for the guardrails LLM (https://aistudio.google.com/apikey) |
| Ollama (optional) | only if you use local models such as `qwen3-embedding:0.6b` |

Note: the project pins `torch==2.2.2` and `numpy<2`, so use Python 3.12 exactly.

## 2. Clone the repository

```powershell
git clone <your-repo-url>
cd Mtech-Final-year-project
```

## 3. Install dependencies

### Option A: uv (recommended, uses the locked versions in `uv.lock`)

```powershell
uv sync
```

`uv sync` creates `.venv` with Python 3.12 (uv downloads it automatically if missing) and installs
everything from `uv.lock`, including the dev dependencies (pytest).

### Option B: pip with `requirements.txt`

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS / Linux: `python3.12 -m venv .venv` then `source .venv/bin/activate`.

`requirements.txt` is exported from `uv.lock`. Regenerate it after dependency changes with:

```powershell
uv export --no-hashes --format requirements-txt -o requirements.txt
```

## 4. Configure environment variables

```powershell
copy .env.example .env
```

Edit `.env` and put in your real Gemini API key. `.env` is git-ignored, so it is never pushed.

If guardrails cannot find a key, set it in the shell for the session:

```powershell
$env:OPENAI_API_KEY = "your-gemini-api-key"
```

## 5. Verify the setup

```powershell
uv run pytest
```

or, with the pip setup:

```powershell
pytest
```

Run the sample notebook with the `.venv` kernel: open `sample.ipynb` in VS Code or Jupyter and select
the `.venv` interpreter. `ipykernel` is already a dependency.

## 6. Project layout

```
data/raw/                  input PDFs (sample.pdf)
data/processed/            parsed Markdown and chunks.json output
src/rag_project/service/   document loading, chunking and chunk validation
src/rag_project/guardrails/ NeMo Guardrails service and config (config/config.yml)
tests/                     pytest tests
```

## Troubleshooting

- **`Python 3.12 not found`**: run `uv python install 3.12`, then `uv sync` again.
- **Torch / numpy import errors**: delete `.venv` and run `uv sync` again so the locked versions are used.
- **Hugging Face download fails**: check your internet connection or proxy. The first run downloads models into the local cache.
- **Guardrails authentication error**: check that `GEMINI_API_KEY` and `OPENAI_API_KEY` are set.
