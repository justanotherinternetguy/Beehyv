# generalresearch

A unified research-paper pipeline: **ingest PDFs → discuss papers with an AI agent swarm → generate runnable code repos**.

```
PDF
 └─[ingest]→  cleaned JSON
                 ├─[discuss]→  expert Q&A (agent swarm)
                 └─[codegen]→  generated code repository
```

---

## Directory layout

```
generalresearch/
├── run.py                  # Master CLI (ingest / discuss / codegen)
├── Makefile                # Shorthand commands
├── requirements.txt        # Combined Python dependencies
│
├── papers/                 # Input PDFs (4 pre-loaded test papers)
│   ├── bert.pdf
│   ├── attention_is_all_you_need.pdf
│   ├── vision_transformer.pdf
│   └── og_attention.pdf
│
├── data/
│   ├── raw_json/           # Raw S2ORC JSON output from Grobid
│   └── cleaned_json/       # Cleaned JSONs consumed by discuss + codegen
│       └── BERT_cleaned.json   # Pre-processed BERT (ready to use immediately)
│
├── outputs/                # Generated code repos land here
│
├── ingestion/              # PDF → S2ORC JSON (s2orc-doc2json)
│   ├── doc2json/           # Core conversion library
│   ├── setup.py            # Install with: pip install -e ingestion/
│   └── scripts/
│       ├── setup_grobid.sh # Download Grobid (one-time)
│       └── ingest_pdf.sh   # Internal script called by run.py ingest
│
├── paper2code/             # JSON → code generation pipeline
│   ├── codes/              # Pipeline stages (0_pdf_process through 4_debugging)
│   └── prompts/            # LLM prompt templates
│
└── agentswarm/             # Expert agent Q&A over papers
    ├── cli.py, orchestrator.py, expert.py
    ├── retriever.py, blackboard.py
    ├── llm.py, paper_loader.py
    └── __init__.py
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install -e ingestion/        # installs doc2json package
```

### 2. Set API key(s)

```bash
export OPENROUTER_API_KEY="sk-or-..."   # required for discuss + codegen
# OR for OpenAI directly:
export OPENAI_API_KEY="sk-..."
```

### 3. Discuss a paper (no ingestion needed — BERT is pre-loaded)

```bash
python run.py discuss "What is masked language modeling?"
```

Or with Make:
```bash
make discuss QUESTION="What is masked language modeling?"
```

### 4. Ingest a new PDF

Requires Docker. Pull the image once:
```bash
make setup-grobid
# or: sudo docker pull grobid/grobid:0.9.0-crf
```

Start Grobid in a separate terminal (keep it running while ingesting):
```bash
make start-grobid
# or: sudo docker run --rm --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.9.0-crf
```

Then ingest any PDF:
```bash
python run.py ingest papers/bert.pdf
# or
make ingest PAPER=bert
```

Outputs:
- `data/raw_json/bert.json`
- `data/cleaned_json/bert_cleaned.json`

### 5. Generate a code repo from a paper

```bash
python run.py codegen data/cleaned_json/bert_cleaned.json
# or
make codegen JSON=data/cleaned_json/bert_cleaned.json
```

With a local vLLM backend instead of OpenRouter:
```bash
python run.py codegen data/cleaned_json/bert_cleaned.json --local
make codegen-local JSON=data/cleaned_json/bert_cleaned.json
```

Generated repo lands in `outputs/<paper_name>_repo/`.

---

## Command reference

### `python run.py ingest <PDF> [-o OUTPUT_DIR]`

Converts a PDF to a cleaned JSON file ready for `discuss` or `codegen`.

| Arg | Description |
|-----|-------------|
| `pdf` | Path to input PDF |
| `-o / --output` | Output dir (default: `data/cleaned_json/`) |

### `python run.py discuss "<QUESTION>" [--papers FILE ...]`

Runs a moderated panel of LLM paper-expert agents that answer questions grounded in evidence from the papers.

| Arg | Description |
|-----|-------------|
| `question` | The question to ask |
| `--papers` | One or more `*_cleaned.json` files (default: BERT) |
| `--max-agents` | Max experts per question (default: 5) |
| `--top-k` | Evidence chunks per expert (default: 4) |
| `--critique-rounds` | Rounds of cross-critique (default: 1) |
| `--model` | OpenRouter model override |

### `python run.py codegen <CLEANED_JSON> [--name NAME] [--model MODEL] [--local]`

Runs the full paper→code generation pipeline: planning → analysis → coding.

| Arg | Description |
|-----|-------------|
| `cleaned_json` | Path to `*_cleaned.json` |
| `--name` | Output name (default: JSON stem) |
| `--model` | Model ID override |
| `--local` | Use local vLLM backend (DeepSeek-Coder) |

---

## Full pipeline walkthrough

```bash
# 1. One-time: pull Grobid Docker image
make setup-grobid

# 2. Start Grobid (in a separate terminal, keep it running)
make start-grobid

# 3. Ingest all test papers
make ingest-all

# 4. Ask the agent swarm a question across all ingested papers
python run.py discuss "How do transformers handle positional encoding?" \
    --papers data/cleaned_json/*.json \
    --critique-rounds 2

# 5. Generate code for the attention paper
make codegen JSON=data/cleaned_json/attention_is_all_you_need_cleaned.json
```

---

## How each component works

### ingestion (s2orc-doc2json)
Sends the PDF to a **Grobid** server running in Docker (`grobid/grobid:0.9.0-crf`, port 8070), which parses it into TEI-XML. The `doc2json` library then converts TEI-XML into a structured **S2ORC JSON** with body text, sections, citations, figures, and equations. `0_pdf_process.py` strips noisy metadata (cite spans, eq spans, etc.) to produce a **cleaned JSON** for downstream use.

### agentswarm
Each `*_cleaned.json` becomes one **PaperExpertAgent**. When you ask a question:
1. Agents score relevance to the question (BM25 keyword matching).
2. Selected agents retrieve top-k evidence chunks and compose answers grounded only in their paper.
3. Agents critique each other's claims.
4. An orchestrator synthesizes a final answer with consensus points, disagreements, and citations.

### paper2code
A four-stage LLM pipeline:
1. **Planning** — LLM reads the paper and generates an overview, software design, and task list.
2. **Config extraction** — hyperparameters are pulled into `config.yaml`.
3. **Analysis** — each file in the task list gets detailed logic specs.
4. **Coding** — LLM generates each file with full context from prior files and specs.

---

## Environment variables

| Variable | Used by | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | discuss, codegen | OpenRouter API key |
| `OPENAI_API_KEY` | codegen (OpenAI path) | OpenAI API key |
| `GROBID_DIR` | ingest | Grobid install path (default: `~/grobid-0.7.3`) |
