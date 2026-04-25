# Beehyv / generalresearch Documentation

This repository is a research-paper automation pipeline. Its core flow is:

```text
PDF papers
  -> ingestion/doc2json + Grobid
  -> raw S2ORC-style JSON
  -> cleaned JSON
  -> agent swarm discussion / brainstorming
  -> paper-to-code planning, analysis, code generation, evaluation, reproduction
```

The repo currently contains three major systems:

- `run.py`: the top-level CLI that ties the main workflows together.
- `agentswarm/`: a paper-grounded multi-agent discussion and brainstorming system.
- `paper2code/codes/`: a numbered LLM pipeline that turns a cleaned paper JSON into generated source code and reproduction artifacts.
- `ingestion/`: a bundled `doc2json`/S2ORC conversion package for PDF, LaTeX, JATS, and SPP-style inputs, with the top-level project using the PDF/Grobid path.

There are also preloaded paper PDFs in `papers/`, raw and cleaned JSONs in `data/`, generated artifacts in `outputs/`, prompt templates in `paper2code/prompts/`, and tests for the agent swarm in `agentswarm/test_agentswarm.py`.

## Repository Layout

```text
.
+-- run.py
+-- Makefile
+-- README.md
+-- HELP.md
+-- requirements.txt
+-- agentswarm/
+-- paper2code/
|   +-- codes/
|   +-- examples/
|   +-- prompts/
+-- ingestion/
|   +-- scripts/
|   +-- doc2json/
+-- papers/
+-- data/
|   +-- raw_json/
|   +-- cleaned_json/
+-- outputs/
```

Important note: `README.md` currently contains unresolved merge-conflict markers. `HELP.md` has the complete project-level quickstart text without the conflict markers.

## Runtime Dependencies

Top-level `requirements.txt` includes:

- LLM and codegen: `openai`, `tiktoken`, `transformers`, `tqdm`.
- PDF ingestion: `beautifulsoup4`, `requests`, `lxml`, `python-magic`, `latex2mathml`, `Flask`, `itsdangerous`, `boto3`.
- Optional local model backend: `vllm`.

The `agentswarm` package intentionally uses only Python stdlib plus this repo's own modules. The LLM call is made through `urllib` directly.

Environment variables used by the code:

- `OPENROUTER_API_KEY`: required by `agentswarm` and the OpenRouter-backed `paper2code` scripts.
- `OPENAI_API_KEY`: mentioned by docs as an alternative, but most current `paper2code` scripts instantiate `OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])`.

## Top-Level CLI: `run.py`

`run.py` is the main entry point. It exposes four subcommands: `ingest`, `discuss`, `brainstorm`, and `codegen`.

### `python run.py ingest <PDF> [-o OUTPUT_DIR]`

Purpose: Convert a PDF into cleaned JSON.

Flow:

1. Validates that the PDF path exists.
2. Defaults output to `data/cleaned_json/`.
3. Calls `ingestion/scripts/ingest_pdf.sh`.
4. The script checks that Grobid is alive at `localhost:8070`.
5. It runs `ingestion/doc2json/grobid2json/process_pdf.py` to create `data/raw_json/<paper>.json`.
6. It runs `paper2code/codes/0_pdf_process.py` to remove noisy metadata and write `<paper>_cleaned.json`.

### `python run.py discuss "<QUESTION>" [--papers ...]`

Purpose: Ask a question over one or more cleaned paper JSON files.

Flow:

1. Builds an `agentswarm` CLI argument list.
2. Defaults papers to `data/cleaned_json/BERT_cleaned.json`.
3. Passes `--max-agents`, `--top-k`, `--critique-rounds`, model, logging, and streaming options through to `agentswarm.cli`.
4. Prints the final synthesis, consensus, disagreements, and citations.

### `python run.py brainstorm "<AREA>" [--papers ...]`

Purpose: Generate future research directions by combining ideas across papers.

Flow:

1. Uses the same paper loading, retriever, LLM client, and agent construction as discussion.
2. Runs `BrainstormOrchestrator`.
3. Prints seed ideas, cross-pollinated ideas, and a final research agenda.

### `python run.py codegen <CLEANED_JSON> [--name NAME] [--model MODEL] [--local]`

Purpose: Generate a code repository from a cleaned paper JSON.

Flow:

1. Validates the cleaned JSON path.
2. Derives `paper_name` from the JSON stem unless `--name` is given.
3. Creates:
   - `outputs/<paper_name>/` for planning, analysis, cost, and raw LLM artifacts.
   - `outputs/<paper_name>_repo/` for generated source files.
4. Chooses script variants:
   - OpenRouter path: `1_planning.py`, `2_analyzing.py`, `3_coding.py`.
   - Local vLLM path: `1_planning_llm.py`, `2_analyzing_llm.py`, `3_coding_llm.py`.
5. Runs stages in order:
   - planning
   - config extraction
   - analysis
   - coding
6. Copies `outputs/<paper_name>/planning_config.yaml` into the generated repo as `config.yaml`.

## Makefile

`Makefile` wraps common commands:

- `make setup`: installs partial Python dependencies and editable `ingestion/`.
- `make setup-grobid`: pulls `grobid/grobid:0.9.0-crf`.
- `make start-grobid`: starts the Grobid Docker container on port `8070`.
- `make ingest PDF=papers/bert.pdf`: runs `run.py ingest`.
- `make ingest-all`: ingests every PDF in `papers/`.
- `make discuss QUESTION="..."`: asks over all JSON files in `data/cleaned_json/`.
- `make codegen JSON=...`: runs OpenRouter-backed codegen.
- `make codegen-local JSON=...`: runs local-vLLM codegen.
- `make clean`: removes outputs, raw JSON, and cache directories.

## Data Model and Artifacts

### Paper Inputs

`papers/` contains PDF inputs:

- `attention_is_all_you_need.pdf`
- `bert.pdf`
- `og_attention.pdf`
- `vision_transformer.pdf`

### Raw JSON

`data/raw_json/*.json` is S2ORC-style JSON generated from PDF parsing. It includes metadata, body paragraphs, citation spans, reference spans, equation spans, bibliography entries, and reference entries.

### Cleaned JSON

`data/cleaned_json/*_cleaned.json` is produced by `0_pdf_process.py`. Cleaning recursively removes keys that are noisy or large for downstream LLM use:

- `cite_spans`
- `ref_spans`
- `eq_spans`
- `authors`
- `bib_entries`
- `year`
- `venue`
- `identifiers`
- `_pdf_hash`
- `header`

The cleaned JSON remains structured around `pdf_parse.abstract`, `pdf_parse.body_text`, `pdf_parse.back_matter`, and `pdf_parse.ref_entries`.

### Codegen Outputs

For a paper named `foo`, codegen writes:

- `outputs/foo/planning_response.json`: raw completion responses from planning.
- `outputs/foo/planning_trajectories.json`: conversation history across planning prompts.
- `outputs/foo/planning_config.yaml`: extracted training/configuration YAML.
- `outputs/foo/planning_artifacts/`: human-readable plan/design/config files.
- `outputs/foo/analyzing_artifacts/`: per-file logic analysis text.
- `outputs/foo/*_simple_analysis_response.json`: raw per-file analysis responses.
- `outputs/foo/*_simple_analysis_trajectories.json`: per-file analysis prompt histories.
- `outputs/foo/coding_artifacts/`: raw coding responses.
- `outputs/foo/accumulated_cost.json`: accumulated estimated model cost.
- `outputs/foo/cost_info.log`: token and cost logs.
- `outputs/foo_repo/`: generated source repository.

## `agentswarm/` Architecture

The agent swarm provides paper-grounded Q&A and brainstorming.

### Package Exports

`agentswarm/__init__.py` re-exports the public API:

- Data state: `Blackboard`, `Claim`, `Critique`, `Evidence`, `Synthesis`.
- Brainstorming state: `BrainstormBlackboard`, `ResearchIdea`, `CrossPollinatedIdea`.
- Agents and orchestrators: `PaperExpertAgent`, `SwarmOrchestrator`, `BrainstormOrchestrator`.
- Loading/retrieval/LLM/logging helpers: `load_paper`, `load_papers`, `KeywordRetriever`, `OpenRouterLLM`, `SwarmLogger`.

### Paper Loading: `paper_loader.py`

Core dataclasses:

- `PaperChunk`: one searchable text unit with `chunk_id`, `paper_id`, `paper_title`, `section`, `sec_num`, `text`, and `source`.
- `Paper`: loaded paper with `paper_id`, `title`, `abstract`, source `path`, and `chunks`.

`load_paper(path)` reads a cleaned JSON file and normalizes it into chunks:

- abstract blocks from `pdf_parse.abstract`
- body paragraphs from `pdf_parse.body_text`
- back matter from `pdf_parse.back_matter`
- figure/table/reference entries from `pdf_parse.ref_entries`

If no structured chunks exist but top-level abstract text exists, it creates one fallback abstract chunk.

`load_papers(paths)` preserves input order and calls `load_paper` for each file.

### Retrieval: `retriever.py`

`KeywordRetriever` is a dependency-free BM25-style retriever.

How it works:

1. Flattens all paper chunks into one corpus.
2. Tokenizes text with regex `TOKEN_RE = r"[A-Za-z0-9][A-Za-z0-9_\\-]*"`.
3. Builds per-chunk term frequencies and corpus document frequencies.
4. Scores query/chunk pairs with BM25-like weighting using `k1 = 1.5` and `b = 0.75`.
5. Can scope search to a single `paper_id`.
6. Returns either raw `RetrievalResult` objects or `Evidence` dataclasses.

### Shared State: `blackboard.py`

The discussion pipeline uses auditable dataclasses:

- `Evidence`: retrieved paper chunk plus a computed citation label.
- `Claim`: one expert's answer to a question.
- `Critique`: one expert's response to another expert's claim.
- `Synthesis`: final answer, consensus list, disagreement list, and citations.
- `Blackboard`: mutable session state containing question, selected agents, claims, critiques, synthesis, and timestamp.

The orchestrator appends claims and critiques to the blackboard and sets the final synthesis.

### LLM Client: `llm.py`

`OpenRouterLLM` is a minimal stdlib OpenRouter client.

Features:

- Default model: `nvidia/nemotron-3-super-120b-a12b:free`.
- Reads API key from `OPENROUTER_API_KEY`.
- Uses `urllib.request` rather than the OpenAI SDK.
- Supports blocking completions via `complete(messages)`.
- Supports SSE streaming via `complete_stream(messages, on_token)`.
- Parses OpenRouter chat-completion responses and raises explicit runtime errors for HTTP, URL, empty-response, and malformed-response failures.

### Logging: `log.py`

`SwarmLogger` writes progress to stderr and optionally to a file.

Features:

- Colorful TTY output with automatic color disabling for non-TTY streams.
- Phase headers and completion lines.
- Agent start/done timing.
- Token streaming through `on_token`.
- Optional timestamped file logging via Python `logging`.

### Expert Agent: `expert.py`

`PaperExpertAgent` scopes one agent to one paper.

Main methods:

- `relevance(question)`: retrieves one evidence chunk from its paper and returns the top score.
- `answer(question)`: retrieves `top_k` chunks, asks the LLM to answer using only that evidence, and returns a `Claim`.
- `critique(question, target)`: retrieves evidence for the question plus target claim and returns a `Critique`.
- `propose_research(area, id_counter)`: asks the LLM for 2-3 structured future-research seed ideas grounded in the paper.
- `cross_pollinate(area, seed, id_counter)`: asks the LLM for one hybrid idea combining this paper with another paper's seed.
- `summarize_position(claim)`: formats a claim with a citation.

Confidence is heuristic: no evidence gives `0.0`; otherwise confidence is `min(0.95, 0.35 + top_score / 20)` rounded to two decimals.

### Q&A Orchestration: `orchestrator.py`

`SwarmOrchestrator` coordinates a discussion.

Flow:

1. Builds a fresh `Blackboard(question=...)`.
2. Selects agents by descending `agent.relevance(question)`.
3. Keeps agents with positive relevance, or the single top-ranked agent if none score above zero.
4. Caps participation at `max_agents`.
5. Asks each selected expert to answer.
6. Runs `critique_rounds` of all selected experts critiquing every other expert's claims.
7. Synthesizes a final `Synthesis`.

The current synthesis is deterministic and simple: it concatenates expert positions and critiques rather than making another LLM call. Consensus and disagreement text are fixed templates plus the single-agent caveat.

### Brainstorming: `brainstorm.py`

Brainstorming has its own dataclasses:

- `ResearchIdea`: a seed direction from one paper.
- `CrossPollinatedIdea`: a hybrid direction generated by one paper expert after reading another paper's seed.
- `BrainstormBlackboard`: area, seeds, cross-pollinations, agenda, timestamp.

Flow:

1. Select top `max_agents` by relevance to the research area.
2. Seed round: each expert proposes 2-3 ideas from its own paper.
3. Cross-pollination: each expert reads every other expert's seeds and proposes hybrid ideas. This repeats for `cross_pollinate_rounds`.
4. Final agenda: the orchestrator calls the LLM with all seed and hybrid ideas and asks for 3-5 actionable research directions.

Parsing helpers expect structured model output delimited by:

- `---IDEA---` / `---END---`
- `TEXT:`
- `GROUNDING:`
- `GAP:`
- `---CROSSPOLLINATE---`
- `CONNECTION:`

If seed parsing fails, the whole response is reduced to one fallback idea.

### CLI: `agentswarm/cli.py`

The package CLI supports:

- `agentswarm discuss <question>`
- `agentswarm brainstorm <area>`

Shared flags:

- `--papers`
- `--model`
- `--top-k`
- `--log-file`
- `--no-stream`

Discussion-specific flags:

- `--max-agents`
- `--critique-rounds`

Brainstorm-specific flags:

- `--max-agents`
- `--cp-rounds`

### Tests: `test_agentswarm.py`

The tests cover:

- Loading BERT cleaned JSON into chunks.
- Keyword retrieval for masked language modeling.
- Orchestrator execution with a stub LLM.

## `paper2code/codes/` Architecture

`paper2code/codes` is a collection of numbered scripts. Most scripts execute immediately at module import after parsing arguments, so they are meant to be run as CLI scripts, not imported as libraries.

There are two backend families:

- OpenRouter/OpenAI SDK scripts: `1_planning.py`, `2_analyzing.py`, `3_coding.py`.
- Local vLLM scripts: `1_planning_llm.py`, `2_analyzing_llm.py`, `3_coding_llm.py`.

The top-level `run.py codegen` path only invokes stages 1, 1.1, 2, and 3. Stages 1.2, 3.1, 4, 5, and `eval.py` are utility/extension stages that are present but not part of the default `run.py codegen` pipeline.

### Shared Utilities: `utils.py`

Important helpers:

- `extract_planning(path)`: loads `planning_trajectories.json`, keeps assistant turns, strips `</think>`, and returns the first three assistant messages.
- `content_to_json(...)`: attempts multiple regex-based cleanup strategies to parse model output wrapped in `[CONTENT]...[/CONTENT]` into JSON.
- `extract_code_from_content(...)`: extracts the first fenced code block.
- `extract_code_from_content2(...)`: extracts a fenced Python code block.
- `format_json_data(data)`: converts a JSON object into a text artifact with section dividers.
- `cal_cost(response_json, model_name)`: estimates cost from token usage for a hard-coded table of OpenAI model prices.
- `print_log_cost(...)`: prints and appends token/cost info to `cost_info.log`.
- `num_tokens_from_messages(...)`: estimates chat tokens using `tiktoken`.
- `read_all_files(directory, allowed_ext, is_print=True)`: recursively reads code/text files while skipping hidden paths, files without extensions, and oversized files.
- `read_python_files(directory)`: recursively reads `.py` files.
- `extract_json_from_string(text)`: extracts fenced JSON.
- `get_now_str()`: timestamp string for result filenames.

### Stage 0: `0_pdf_process.py`

Purpose: clean raw S2ORC JSON.

`remove_spans(data)` recursively removes noisy metadata keys from dictionaries and lists. `main(args)` loads `--input_json_path`, writes cleaned JSON to `--output_json_path`, and prints the saved path.

### Stage 1: Planning: `1_planning.py`

Purpose: use an LLM to build a reproduction plan, software architecture, task list, and config.

Inputs:

- `--paper_name`
- `--gpt_version`
- `--paper_format JSON|LaTeX`
- `--pdf_json_path`
- `--pdf_latex_path`
- `--output_dir`

Flow:

1. Loads the paper as JSON or LaTeX.
2. Creates an OpenAI SDK client pointed at OpenRouter.
3. Builds four prompt steps:
   - overall reproduction plan
   - architecture design with file list and Mermaid diagrams
   - logic/task breakdown with package requirements
   - `config.yaml` generation
4. Appends each prompt and assistant response to `trajectories`.
5. Saves:
   - `planning_response.json`
   - `planning_trajectories.json`
   - `accumulated_cost.json`
   - `cost_info.log`

Caveat: `api_call(msg, gpt_version)` ignores its `gpt_version` parameter and currently calls model `"tencent/hy3-preview:free"` directly.

### Stage 1 Local: `1_planning_llm.py`

Purpose: local-vLLM version of planning.

Differences:

- Uses `transformers.AutoTokenizer` and `vllm.LLM`.
- Supports Qwen-specific Yarn rope scaling and DeepSeek-specific stop token setup.
- Does not calculate OpenRouter/OpenAI costs.
- Saves `planning_response.json` and `planning_trajectories.json`.

### Stage 1.1: `1.1_extract_config.py`

Purpose: extract usable artifacts from planning trajectories.

Flow:

1. Reads `planning_trajectories.json`.
2. Takes turn index `8` as the YAML-producing response.
3. Extracts fenced `yaml` and writes `planning_config.yaml`.
4. Calls `extract_planning` to obtain the first three assistant planning outputs.
5. Parses architecture and task JSON.
6. Writes:
   - `planning_artifacts/1.1_overall_plan.txt`
   - `planning_artifacts/1.2_arch_design.txt`
   - `planning_artifacts/1.3_logic_design.txt`
   - `planning_artifacts/1.4_config.yaml`

Caveat: this stage assumes the planning trajectory has the same turn layout created by `1_planning.py`/`1_planning_llm.py`, especially the YAML at index `8`.

### Stage 1.2: `1.2_rag_config.py`

Purpose: refine generated config model/dataset names into Hugging Face IDs.

Flow:

1. Loads `planning_config.yaml`.
2. Asks OpenRouter to identify model and dataset names as a JSON list.
3. Optionally uses `huggingface_hub.HfApi` to search model IDs by downloads.
4. Replaces detected names in the config with refined names.
5. Backs up the original config to `planning_config.yaml.bak`.

Caveats:

- The CLI accepts `--gpt_version`, but the actual OpenRouter call is hard-coded to `"tencent/hy3-preview:free"`.
- It searches Hugging Face models only, even though the prompt mentions datasets too.

### Stage 2: Analysis: `2_analyzing.py`

Purpose: generate detailed per-file logic analysis before code generation.

Inputs are the paper, planning config, planning trajectories, and task list.

Flow:

1. Loads paper JSON or LaTeX.
2. Loads `planning_config.yaml`.
3. Extracts planning context with `extract_planning`.
4. Reads `task_list.json` if present; otherwise parses the third planning output.
5. Extracts `Task list` and `Logic Analysis`.
6. For each file in the task list except `config.yaml`:
   - prompts the LLM for detailed logic analysis
   - writes text to `analyzing_artifacts/<file>_simple_analysis.txt`
   - writes raw response and trajectories to output root
   - updates accumulated cost

Caveat: `api_call` also hard-codes `"tencent/hy3-preview:free"`.

### Stage 2 Local: `2_analyzing_llm.py`

Purpose: local-vLLM version of analysis.

It mirrors `2_analyzing.py` but uses local model generation and skips cost accounting.

### Stage 3: Coding: `3_coding.py`

Purpose: generate each source file in dependency/task order.

Flow:

1. Loads paper content, config, and planning context.
2. Parses the task list from planning output.
3. Loads each file's detailed analysis from stage 2.
4. Maintains `done_file_lst` and `done_file_dict`.
5. For each todo file except `config.yaml`:
   - includes already generated code files in the prompt
   - asks the LLM to write only the current file
   - extracts fenced code from the response
   - writes raw response to `coding_artifacts/<file>_coding.txt`
   - writes code to `output_repo_dir/<file>`
   - updates accumulated cost

The generated `config.yaml` is copied by `run.py`, not produced directly by this stage.

Caveat: `api_call` hard-codes `"tencent/hy3-preview:free"`.

### Stage 3 Local: `3_coding_llm.py`

Purpose: local-vLLM version of coding.

It mirrors `3_coding.py`, but uses local model generation. If normal fenced-code extraction fails, it falls back to Python-specific code extraction and then to the raw response.

Potential issue: the local version loads detailed logic analysis from `*_simple_analysis_trajectories.json` using `detailed_logic_analysis_trajectories[0]['content']`, which is the system prompt rather than the assistant analysis in the trajectories written by stage 2 local. This may reduce analysis quality for local coding.

### Stage 3.1: `3.1_coding_sh.py`

Purpose: generate a `reproduce.sh` runner for the generated repo.

Flow:

1. Loads generated Python files from `output_repo_dir`.
2. Loads config and planning context.
3. Prompts OpenRouter to write a self-contained Bash script that installs dependencies and runs the project.
4. Extracts code and writes `output_repo_dir/reproduce.sh`.

This stage is not invoked by default from `run.py codegen`.

### Stage 4: `4_debugging.py`

Purpose: ask an LLM to patch a generated repo based on an execution error log.

Core behavior:

1. Reads `--error_file_name`.
2. Loads planning trajectories and task list from `--output_dir`.
3. Reads generated Python files and optionally `config.yaml`/`reproduce.sh`.
4. Prompts OpenRouter to output SEARCH/REPLACE blocks.
5. `parse_and_apply_changes` applies each replacement against files in the debug repo and writes `.NNN.bak` backups.

Known issue: `parse_args()` does not define `--output_repo_dir`, but the script later reads `args.output_repo_dir`. As written, the script will raise an attribute error unless this argument is added.

### Stage 5: `5_reproduce.py`

Purpose: generate and run a reproduction experiment around a generated repo.

Flow:

1. Loads paper JSON, planning artifacts, config, and generated code.
2. Asks an LLM to extract dataset/evaluation/model/training information as JSON.
3. Falls back to hard-coded Transformer/WMT2014 defaults if extraction fails.
4. Saves `reproduce_dataset_info.json`.
5. Asks an LLM to generate a complete `reproduce.py`.
6. Falls back to `reproduce_template.py` if generation fails.
7. Writes `requirements.txt` and `run_reproduce.sh` into the reproduction directory.
8. Installs required packages with pip.
9. Runs `reproduce.py`.
10. Reads `reproduce_results.json` if present and writes `reproduce_summary.json`.

This stage is relatively specialized around Transformer-style sequence-to-sequence reproduction. The fallback template is specifically for "Attention Is All You Need".

### Reproduction Template: `reproduce_template.py`

Purpose: built-in fallback reproduction script for a scaled-down Transformer.

Main features:

- Downloads a Hugging Face translation dataset.
- Trains or reuses a SentencePiece BPE tokenizer.
- Defines PyTorch dataset/collate utilities.
- Implements:
  - sinusoidal positional encoding
  - multi-head attention
  - feed-forward network
  - encoder layer
  - decoder layer
  - encoder-decoder Transformer
  - label smoothing loss
  - Noam-style learning-rate schedule
  - beam search
- Trains for demo-scale steps and dataset size.
- Evaluates with BLEU and writes `reproduce_results.json`.

### Evaluation: `eval.py`

Purpose: use an LLM evaluator to score generated code correctness.

Modes:

- `ref_free`: evaluate paper + generated repo only.
- `ref_based`: evaluate paper + generated repo + gold/reference repo.

Flow:

1. Loads paper JSON.
2. Reads generated code. In `--papercoder` mode, it uses planning task order and includes `config.yaml`.
3. Loads the selected prompt template from `paper2code/prompts/`.
4. Optionally includes gold code for reference-based evaluation.
5. Counts tokens and exits if above 128k.
6. Requests `generated_n` independent model evaluations.
7. Parses JSON responses with `score` and `critique_list`.
8. Averages valid scores.
9. Writes an evaluation JSON into `eval_result_dir`.
10. Logs cost.

Caveats:

- The request model is hard-coded to `"tencent/hy3-preview:free"` despite a `--gpt_version` argument.
- `avg_score = sum(all_scores) / len(all_scores)` will fail if all responses are invalid.
- The output key is misspelled as `"scroe_lst"`.

## `ingestion/` Architecture

The top-level project uses the PDF path, but the bundled `doc2json` package supports multiple source formats.

### Shell Scripts

`ingestion/scripts/setup_grobid.sh` pulls `grobid/grobid:0.9.0-crf`.

`ingestion/scripts/run_grobid.sh` runs Grobid in Docker:

```text
sudo docker run --rm --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.9.0-crf
```

`ingestion/scripts/ingest_pdf.sh` is the top-level ingest worker:

1. Validates the PDF file.
2. Resolves repo paths.
3. Checks Grobid `/api/isalive`.
4. Runs `grobid2json/process_pdf.py`.
5. Runs `0_pdf_process.py`.
6. Prints raw and cleaned output paths.

### PDF/Grobid Path

`ingestion/doc2json/grobid2json/process_pdf.py` has two main entry points:

- `process_pdf_stream(...)`: sends in-memory PDF bytes to Grobid and converts the returned TEI soup to S2ORC JSON.
- `process_pdf_file(...)`: sends a local PDF to Grobid, writes a `.tei.xml` temp file, converts it to a `Paper`, and writes release JSON.

`ingestion/doc2json/grobid2json/grobid/grobid_client.py` handles the Grobid HTTP API:

- Default server: `localhost:8070`.
- Supports `processFulltextDocument`, citation parsing, header-name parsing, and affiliation parsing.
- Writes failed PDF IDs to `failed.log` on non-200 responses.
- Retries on HTTP 503 after sleeping.

`ingestion/doc2json/grobid2json/tei_to_json.py` converts TEI XML into the internal S2ORC `Paper` object.

Major responsibilities:

- Normalize Grobid object IDs like bibliography, figure, table, and equation references.
- Parse bibliography entries.
- Extract formulas.
- Convert TEI tables to HTML table markup.
- Extract figures and tables into `ref_entries`.
- Detect bracket-style citations.
- Convert notes to paragraphs.
- Process paragraphs into text plus citation/reference/equation spans.
- Extract abstract, body text, and back matter.
- Return a `Paper` object via `convert_tei_xml_soup_to_s2orc_json` or `convert_tei_xml_file_to_s2orc_json`.

### S2ORC Data Classes: `ingestion/doc2json/s2orc.py`

Core classes:

- `ReferenceEntry`: figure/table/footnote/section/equation reference payloads.
- `BibliographyEntry`: parsed citation metadata.
- `Affiliation`: author affiliation.
- `Author`: author name, affiliation, and email.
- `Metadata`: title, authors, year, venue, identifiers.
- `Paragraph`: text, citation spans, reference spans, equation spans, section info.
- `Paper`: complete parsed paper with metadata, abstract, body, back matter, bibliography, and references.

`Paper.release_json(doc_type="pdf")` emits the release format used by downstream code:

```text
{
  "paper_id": ...,
  "header": ...,
  "title": ...,
  "authors": ...,
  "abstract": "...",
  "pdf_parse": {
    "abstract": [...],
    "body_text": [...],
    "back_matter": [...],
    "bib_entries": {...},
    "ref_entries": {...}
  }
}
```

`load_s2orc(paper_dict)` loads either older `grobid_parse` layouts or current `pdf_parse`/body-text layouts into a `Paper`.

### Flask Utility

`ingestion/doc2json/flask/app.py` exposes a simple upload service on port `8080`.

Supported inputs:

- `.pdf`: processed by `process_pdf_stream`.
- `.gz`: processed as LaTeX via `process_tex_stream`.
- `.nxml`: processed as JATS via `process_jats_stream`.
- `/upload_url?url=...`: downloads a PDF URL and returns JSON.

### LaTeX Path

`ingestion/doc2json/tex2json/` handles gzipped LaTeX sources.

Key files:

- `process_tex.py`: stream/file wrapper for LaTeX processing.
- `tex_to_xml.py`: extracts LaTeX archives, normalizes LaTeX, converts normalized LaTeX to XML, then to S2ORC JSON.
- `xml_to_json.py`: the main XML-to-S2ORC converter. It processes authors, bibliography entries, reference tokens, lists, paragraphs, equations, footnotes, figures, tables, sections, abstract, and body text.

The LaTeX path can call Grobid for bibliography and author parsing.

### JATS Path

`ingestion/doc2json/jats2json/` handles JATS/NXML.

Key files:

- `process_jats.py`: stream/file wrapper.
- `jats_to_json.py`: converts JATS XML into S2ORC-style JSON.
- `pmc_utils/front_tag_utils.py`: parses journal metadata, IDs, title, categories, dates, funding, authors, affiliations, and abstract.
- `pmc_utils/back_tag_utils.py`: parses back-matter bibliography entries.
- `pmc_utils/all_tag_utils.py`: parses paragraphs, spans, formulas, xrefs, and section recursion.
- `pmc_utils/extract_utils.py`: extracts figure, table, and supplement blobs.

### SPP Path

`ingestion/doc2json/spp2json/` supports SPP conversion.

Key files:

- `process_pdf.py`: local PDF processing wrapper for SPP output.
- `spp/spp_client.py`: SPP service client.
- `spp/spp_json_to_s2orc_json.py`: converts SPP JSON to the shared S2ORC `Paper` representation.

### Utility Modules

`ingestion/doc2json/utils/` contains shared parsing helpers:

- `grobid_util.py`: metadata extraction from Grobid XML, bibliography parsing, title/authors/venue/pages/IDs helpers.
- `citation_util.py`: citation span utilities and cleanup for Grobid authors.
- `refspan_util.py`: replaces reference span placeholders and updates indices.
- `latex_util.py`: LaTeX normalization, math removal, and LaTeX-to-XML helpers.
- `soup_utils.py`: BeautifulSoup tag manipulation helpers.

## Feature Workflows

### Feature: PDF Ingestion

User command:

```bash
python run.py ingest papers/bert.pdf
```

Implementation path:

```text
run.py cmd_ingest
  -> ingestion/scripts/ingest_pdf.sh
    -> curl Grobid /api/isalive
    -> ingestion/doc2json/grobid2json/process_pdf.py
      -> GrobidClient.process_pdf
      -> convert_tei_xml_file_to_s2orc_json
      -> Paper.release_json
    -> paper2code/codes/0_pdf_process.py
      -> remove_spans
```

Outputs:

- `data/raw_json/bert.json`
- `data/cleaned_json/bert_cleaned.json`

### Feature: Paper Q&A

User command:

```bash
python run.py discuss "What is masked language modeling?"
```

Implementation path:

```text
run.py cmd_discuss
  -> agentswarm.cli cmd_discuss
    -> load_papers
    -> KeywordRetriever
    -> PaperExpertAgent per paper
    -> SwarmOrchestrator.run
      -> select_agents
      -> PaperExpertAgent.answer
      -> PaperExpertAgent.critique
      -> synthesize
```

Key behavior:

- Evidence is keyword/BM25 retrieved, not embedding retrieved.
- Expert answers are LLM-generated and constrained by prompt to retrieved paper evidence.
- Final synthesis is deterministic concatenation of claims/critiques.

### Feature: Research Brainstorming

User command:

```bash
python run.py brainstorm "efficient transformers" --papers data/cleaned_json/*.json
```

Implementation path:

```text
run.py cmd_brainstorm
  -> agentswarm.cli cmd_brainstorm
    -> BrainstormOrchestrator.run
      -> select relevant paper experts
      -> propose_research seed round
      -> cross_pollinate with other experts' seeds
      -> LLM synthesize agenda
```

Outputs are printed to stdout rather than written by default.

### Feature: Paper-to-Code Generation

User command:

```bash
python run.py codegen data/cleaned_json/bert_cleaned.json
```

Implementation path:

```text
run.py cmd_codegen
  -> 1_planning.py
  -> 1.1_extract_config.py
  -> 2_analyzing.py
  -> 3_coding.py
  -> copy planning_config.yaml to generated repo
```

The generated code is only as reliable as the LLM output. The pipeline saves raw prompts/responses extensively so failures can be inspected.

### Feature: Local-vLLM Code Generation

User command:

```bash
python run.py codegen data/cleaned_json/bert_cleaned.json --local
```

Implementation path:

```text
run.py cmd_codegen
  -> 1_planning_llm.py
  -> 1.1_extract_config.py
  -> 2_analyzing_llm.py
  -> 3_coding_llm.py
```

The local path requires `vllm`, `transformers`, model weights, and suitable GPU resources.

### Feature: Config Refinement

Manual command:

```bash
python paper2code/codes/1.2_rag_config.py --output_dir outputs/<paper>
```

This attempts to replace model/dataset strings in `planning_config.yaml` with concrete Hugging Face IDs.

### Feature: Generated Repo Runner Script

Manual command:

```bash
python paper2code/codes/3.1_coding_sh.py \
  --paper_name <paper> \
  --gpt_version <model> \
  --pdf_json_path data/cleaned_json/<paper>_cleaned.json \
  --output_dir outputs/<paper> \
  --output_repo_dir outputs/<paper>_repo
```

This creates `outputs/<paper>_repo/reproduce.sh`.

### Feature: Generated Repo Debugging

Intended manual command shape:

```bash
python paper2code/codes/4_debugging.py \
  --error_file_name error.txt \
  --output_dir outputs/<paper> \
  --paper_name <paper> \
  --save_num 1 \
  --output_repo_dir outputs/<paper>_repo
```

However, the script currently does not declare `--output_repo_dir`, so this feature needs a small parser fix before it can run.

### Feature: Reproduction Run

Manual command:

```bash
python paper2code/codes/5_reproduce.py \
  --paper_name <paper> \
  --pdf_json_path data/cleaned_json/<paper>_cleaned.json \
  --output_dir outputs/<paper> \
  --output_repo_dir outputs/<paper>_repo
```

This generates a reproduction directory, writes `reproduce.py`, installs dependencies, runs the reproduction, and saves summary JSON if results are produced.

### Feature: LLM Evaluation

Manual command:

```bash
python paper2code/codes/eval.py \
  --paper_name <paper> \
  --pdf_json_path data/cleaned_json/<paper>_cleaned.json \
  --data_dir paper2code \
  --output_dir outputs/<paper> \
  --target_repo_dir outputs/<paper>_repo \
  --eval_result_dir outputs/eval \
  --eval_type ref_free \
  --generated_n 8 \
  --papercoder
```

This asks an LLM to rate generated code correctness on a 1-5 scale and write a JSON result.

## Current Code Caveats and Maintenance Notes

- `README.md` has unresolved merge-conflict markers. `HELP.md` appears to contain the intended full documentation.
- Several `paper2code` scripts accept a model argument but hard-code `"tencent/hy3-preview:free"` in the actual API call.
- `4_debugging.py` references `args.output_repo_dir` without declaring that CLI argument.
- Some scripts are not import-safe because they parse args and execute at module top level.
- The OpenRouter-backed scripts require `OPENROUTER_API_KEY` and will fail immediately if it is missing.
- Cost calculation in `utils.py` is based on a static model-price table and may not reflect OpenRouter pricing.
- `1.1_extract_config.py` assumes a fixed planning trajectory index for YAML extraction.
- `eval.py` can divide by zero if every model evaluation response fails JSON parsing.
- `eval.py` writes `"scroe_lst"` instead of `"score_lst"`.
- `paper2code` code extraction relies on fenced code blocks. If the model returns malformed fences, scripts fall back to raw content in some stages.
- `make clean` removes all `outputs/` and `data/raw_json/`; use carefully if generated artifacts matter.

## How to Extend the System

To add a new paper-processing feature, prefer using cleaned JSON as the integration boundary. `paper_loader.load_paper` already normalizes cleaned S2ORC into chunks, and `KeywordRetriever` works over any loaded `Paper`.

To add a new agent behavior, extend `PaperExpertAgent` with a prompt method and add an orchestrator method or new orchestrator class. Keep shared state as dataclasses, following `Blackboard` and `BrainstormBlackboard`.

To add a new codegen stage, follow the existing artifact pattern:

1. Read `outputs/<paper>/planning_trajectories.json`.
2. Read `outputs/<paper>/planning_config.yaml`.
3. Read/write raw response artifacts into a stage-specific directory.
4. Keep generated repo changes inside `outputs/<paper>_repo/`.

To make `paper2code` easier to maintain, the highest-impact refactor would be to extract shared argument parsing, model calling, task-list parsing, and artifact writing into reusable functions instead of duplicating them across numbered scripts.
