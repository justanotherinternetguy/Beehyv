# generalresearch — unified Makefile
# ─────────────────────────────────────────────────────────────────────────────
# Prerequisites:
#   export OPENROUTER_API_KEY="sk-or-..."   (for discuss + codegen via OpenRouter)
#   export OPENAI_API_KEY="..."             (for codegen via OpenAI directly)
# ─────────────────────────────────────────────────────────────────────────────

PYTHON   := python3
PAPER    ?= bert
PDF      ?= papers/$(PAPER).pdf
JSON     ?= data/cleaned_json/$(PAPER)_cleaned.json
QUESTION ?= "What is the main contribution of this paper?"
MODEL    ?= tencent/hy3-preview:free

.PHONY: help setup setup-grobid start-grobid ingest discuss codegen codegen-local clean

help:
	@echo ""
	@echo "  generalresearch — PDF papers → discussion + code"
	@echo ""
	@echo "  Setup:"
	@echo "    make setup           Install Python dependencies"
	@echo "    make setup-grobid    Pull the Grobid Docker image (one-time)"
	@echo "    make start-grobid    Run Grobid container (keep open in a separate terminal)"
	@echo ""
	@echo "  Core pipeline:"
	@echo "    make ingest   PDF=papers/bert.pdf"
	@echo "        Convert PDF → S2ORC JSON → cleaned JSON"
	@echo ""
	@echo "    make discuss  QUESTION=\"What is masked language modeling?\""
	@echo "        Run agent swarm Q&A over all cleaned JSONs in data/cleaned_json/"
	@echo ""
	@echo "    make codegen  JSON=data/cleaned_json/bert_cleaned.json"
	@echo "        Generate code repo from a cleaned JSON paper (OpenRouter)"
	@echo ""
	@echo "    make codegen-local  JSON=data/cleaned_json/bert_cleaned.json"
	@echo "        Generate code repo using local vLLM DeepSeek backend"
	@echo ""
	@echo "  Utilities:"
	@echo "    make clean           Remove generated outputs"
	@echo ""
	@echo "  Variables:"
	@echo "    PAPER    Paper stem name (default: bert)"
	@echo "    PDF      Path to input PDF (default: papers/\$(PAPER).pdf)"
	@echo "    JSON     Cleaned JSON path (default: data/cleaned_json/\$(PAPER)_cleaned.json)"
	@echo "    QUESTION Question for discuss (default: see above)"
	@echo "    MODEL    OpenRouter model (default: $(MODEL))"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────

setup:
	pip install openai tqdm tiktoken transformers
	pip install -e ingestion/
	@echo ""
	@echo "Python dependencies installed."
	@echo "For local vLLM support: pip install vllm"
	@echo "For PDF ingestion: pull image with 'make setup-grobid', then run 'make start-grobid'"

setup-grobid:
	bash ingestion/scripts/setup_grobid.sh

start-grobid:
	bash ingestion/scripts/run_grobid.sh

# ── Pipeline steps ────────────────────────────────────────────────────────────

ingest:
	@echo "=== Ingesting: $(PDF) ==="
	$(PYTHON) run.py ingest $(PDF)

discuss:
	@echo "=== Discussing: $(QUESTION) ==="
	$(PYTHON) run.py discuss $(QUESTION) \
		--papers $(shell ls data/cleaned_json/*.json 2>/dev/null | tr '\n' ' ')

codegen:
	@echo "=== Generating code from: $(JSON) ==="
	$(PYTHON) run.py codegen $(JSON) --model $(MODEL)

codegen-local:
	@echo "=== Generating code (local vLLM) from: $(JSON) ==="
	$(PYTHON) run.py codegen $(JSON) --local

# ── Shortcuts for test papers ─────────────────────────────────────────────────

ingest-all:
	@for pdf in papers/*.pdf; do \
		echo ""; \
		echo "=== Ingesting $$pdf ==="; \
		$(PYTHON) run.py ingest "$$pdf" || echo "Failed: $$pdf"; \
	done

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	rm -rf outputs/
	rm -rf data/raw_json/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned outputs and cache."
