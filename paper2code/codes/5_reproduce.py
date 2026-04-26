"""
Stage 5: Reproduction
Reads planning artifacts to understand datasets and evaluation, then:
1. Identifies datasets and evaluation metrics from the paper
2. Downloads/prepares the dataset (via HuggingFace or recreates locally)
3. Generates a complete self-contained reproduction runner script
4. Executes it and compares results against paper's reported numbers
"""

import json
import os
import sys
import argparse
import subprocess
import re

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    read_python_files,
    read_all_files,
    extract_planning,
    content_to_json,
    extract_code_from_content,
    extract_code_from_content2,
    print_response,
    load_accumulated_cost,
    save_accumulated_cost,
    get_now_str,
    get_llm_client_and_model,
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TEMPLATE_PATH = os.path.join(_THIS_DIR, "reproduce_template.py")


def _builtin_reproduce_template(info: dict, max_train_steps: int, dataset_size: int) -> str:
    """Copy reproduce_template.py to the reproduce dir with substituted defaults."""
    ds = info.get("primary_dataset", {})
    hf_id = ds.get("huggingface_id", "wmt14")
    hf_cfg = ds.get("huggingface_config", "de-en")
    src_lang = ds.get("source_lang", "en")
    tgt_lang = ds.get("target_lang", "de")
    vocab_size = ds.get("vocab_size", 8000)
    paper_bleu = info.get("evaluation", {}).get("paper_results", {}).get("base_model", 27.3)

    if not os.path.exists(_TEMPLATE_PATH):
        raise FileNotFoundError(f"reproduce_template.py not found at {_TEMPLATE_PATH}")

    with open(_TEMPLATE_PATH) as f:
        code = f.read()

    # Patch default argument values so the script has correct defaults baked in
    replacements = {
        'default="wmt14"': f'default="{hf_id}"',
        'default="de-en"': f'default="{hf_cfg}"',
        'default="en"': f'default="{src_lang}"',
        'default="de"': f'default="{tgt_lang}"',
        "default=8000": f"default={vocab_size}",
        "default=27.3": f"default={paper_bleu}",
        "default=500)": f"default={max_train_steps})",
        "default=5000)": f"default={dataset_size})",
    }
    for old, new in replacements.items():
        code = code.replace(old, new, 1)
    return code


parser = argparse.ArgumentParser()
parser.add_argument("--paper_name", type=str, required=True)
parser.add_argument("--gpt_version", type=str, default="tencent/hy3-preview:free")
parser.add_argument("--pdf_json_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--output_repo_dir", type=str, required=True)
parser.add_argument("--reproduce_dir", type=str, default="")
parser.add_argument(
    "--max_train_steps",
    type=int,
    default=500,
    help="Steps for demo reproduction (much fewer than paper's full run)",
)
parser.add_argument(
    "--dataset_size",
    type=int,
    default=5000,
    help="Number of sentence pairs to use (subset for fast demo)",
)
args = parser.parse_args()

client, gpt_version = get_llm_client_and_model(args.gpt_version)

paper_name = args.paper_name
output_dir = os.path.abspath(args.output_dir)
output_repo_dir = os.path.abspath(args.output_repo_dir)
_default_reproduce = os.path.join(os.path.dirname(output_dir), f"{paper_name}_reproduce")
reproduce_dir = os.path.abspath(args.reproduce_dir or _default_reproduce)
max_train_steps = args.max_train_steps
dataset_size = args.dataset_size

os.makedirs(reproduce_dir, exist_ok=True)

accumulated_cost_file = os.path.join(output_dir, "accumulated_cost.json")
total_accumulated_cost = load_accumulated_cost(accumulated_cost_file)

# ──────────────────────────────────────────────────────────────────────────────
# Load paper content
# ──────────────────────────────────────────────────────────────────────────────
with open(args.pdf_json_path) as f:
    paper_content = json.load(f)

# ──────────────────────────────────────────────────────────────────────────────
# Load existing planning and generated code artifacts
# ──────────────────────────────────────────────────────────────────────────────
planning_artifacts_dir = os.path.join(output_dir, "planning_artifacts")
overall_plan = ""
arch_design = ""
config_yaml = ""

for fname in ["1.1_overall_plan.txt", "1.2_arch_design.txt", "1.3_logic_design.txt"]:
    fpath = os.path.join(planning_artifacts_dir, fname)
    if os.path.exists(fpath):
        with open(fpath) as f:
            content = f.read()
        if "overall_plan" in fname:
            overall_plan = content
        elif "arch_design" in fname:
            arch_design = content

config_yaml_path = os.path.join(output_dir, "planning_config.yaml")
if os.path.exists(config_yaml_path):
    with open(config_yaml_path) as f:
        config_yaml = f.read()

generated_code = read_all_files(
    output_repo_dir,
    allowed_ext=[".py", ".yaml", ".yml", ".md", ".sh", ".txt"],
    is_print=False,
)
generated_code_str = ""
for fname, code in generated_code.items():
    ext = os.path.splitext(fname)[1]
    lang = "python" if ext == ".py" else "yaml" if ext in (".yaml", ".yml") else "bash" if ext == ".sh" else "text"
    generated_code_str += f"```{lang}\n## File: {fname}\n{code}\n```\n\n"


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: Ask the LLM to extract dataset and evaluation info from the paper
# ──────────────────────────────────────────────────────────────────────────────
print("\n------- Step 1: Extract Dataset & Evaluation Info -------")

extract_msg = [
    {
        "role": "system",
        "content": (
            "You are an expert ML researcher. You will analyze a research paper and its reproduction plan "
            "to identify exactly what datasets are needed and how the code should be evaluated. "
            "Be precise and actionable."
        ),
    },
    {
        "role": "user",
        "content": f"""## Paper
{paper_content}

## Reproduction Plan
{overall_plan}

## Config
{config_yaml}

## Task
Analyze the paper and plan above. Extract the following information in JSON format:

{{
  "primary_dataset": {{
    "name": "dataset name (e.g. WMT 2014 EN-DE)",
    "huggingface_id": "HuggingFace dataset id if available (e.g. wmt14, with config en-de)",
    "huggingface_config": "config name for HuggingFace dataset (e.g. de-en)",
    "source_lang": "source language code (e.g. en)",
    "target_lang": "target language code (e.g. de)",
    "train_split": "train split name",
    "val_split": "validation split name",
    "test_split": "test split name",
    "tokenizer_type": "BPE or WordPiece",
    "vocab_size": 37000,
    "shared_vocab": true
  }},
  "evaluation": {{
    "metric": "BLEU or F1",
    "tool": "sacrebleu or evalb",
    "paper_results": {{
      "base_model": 27.3,
      "description": "newstest2014 BLEU for base EN-DE model"
    }}
  }},
  "model_config": {{
    "d_model": 512,
    "num_heads": 8,
    "num_encoder_layers": 6,
    "num_decoder_layers": 6,
    "d_ff": 2048,
    "dropout": 0.1,
    "max_seq_len": 512
  }},
  "training": {{
    "optimizer": "Adam",
    "warmup_steps": 4000,
    "label_smoothing": 0.1,
    "beam_size": 4,
    "length_penalty_alpha": 0.6
  }},
  "required_packages": ["list of pip packages needed"]
}}

Return ONLY the JSON object, no markdown fences.
""",
    },
]

_FALLBACK_INFO = None  # filled below if needed

try:
    print(f"[LLM] backend={os.environ.get('LLM_BACKEND', 'openrouter').lower()} model={gpt_version}", flush=True)
    extract_response = client.chat.completions.create(
        model=gpt_version,
        messages=extract_msg,
        temperature=0.1,
    )
    extract_text = extract_response.choices[0].message.content or ""
    print(extract_text[:500])
    if "</think>" in extract_text:
        extract_text = extract_text.split("</think>")[-1].strip()
    json_match = re.search(r"\{[\s\S]*\}", extract_text)
    dataset_eval_info = json.loads(json_match.group() if json_match else extract_text)
    print("[INFO] Dataset/eval info extracted via LLM.")
except Exception as e:
    print(f"[WARNING] LLM extraction failed ({e}). Using built-in defaults.")
    # Fallback defaults for the Transformer / WMT2014 paper
    dataset_eval_info = {
        "primary_dataset": {
            "name": "WMT 2014 EN-DE",
            "huggingface_id": "wmt14",
            "huggingface_config": "de-en",
            "source_lang": "en",
            "target_lang": "de",
            "train_split": "train",
            "val_split": "validation",
            "test_split": "test",
            "tokenizer_type": "BPE",
            "vocab_size": 37000,
            "shared_vocab": True,
        },
        "evaluation": {
            "metric": "BLEU",
            "tool": "sacrebleu",
            "paper_results": {"base_model": 27.3, "description": "newstest2014 BLEU"},
        },
        "model_config": {
            "d_model": 512,
            "num_heads": 8,
            "num_encoder_layers": 6,
            "num_decoder_layers": 6,
            "d_ff": 2048,
            "dropout": 0.1,
            "max_seq_len": 512,
        },
        "training": {
            "optimizer": "Adam",
            "warmup_steps": 4000,
            "label_smoothing": 0.1,
            "beam_size": 4,
            "length_penalty_alpha": 0.6,
        },
        "required_packages": [
            "torch",
            "datasets",
            "sentencepiece",
            "sacrebleu",
            "tqdm",
        ],
    }

# Save extracted info
info_path = os.path.join(output_dir, "reproduce_dataset_info.json")
with open(info_path, "w") as f:
    json.dump(dataset_eval_info, f, indent=2)
print(f"\n[INFO] Dataset/eval info saved to {info_path}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: Generate the reproduction runner script
# ──────────────────────────────────────────────────────────────────────────────
print("\n------- Step 2: Generate Reproduction Runner -------")

reproduce_msg = [
    {
        "role": "system",
        "content": (
            "You are an expert ML engineer. Generate a complete, self-contained Python script "
            "that reproduces the paper's key experiment. The script must:\n"
            "1. Install required packages if missing\n"
            "2. Download the dataset from HuggingFace datasets\n"
            "3. Train/tokenize using sentencepiece\n"
            "4. Implement and train the model\n"
            "5. Evaluate and compare against paper results\n"
            "Write clean, runnable Python code. No placeholders. No 'TODO' comments."
        ),
    },
    {
        "role": "user",
        "content": f"""## Paper
{paper_content}

## Reproduction Plan
{overall_plan}

## Dataset and Evaluation Info
{json.dumps(dataset_eval_info, indent=2)}

## Already Generated Code
{generated_code_str}

## Configuration YAML
{config_yaml}

## Task
Generate a complete self-contained Python script called `reproduce.py` that reproduces the paper's experiment.

### Script requirements:
1. **Dataset**: Use `datasets` library (HuggingFace) to download `{dataset_eval_info.get('primary_dataset', {}).get('huggingface_id', 'wmt14')}` with config `{dataset_eval_info.get('primary_dataset', {}).get('huggingface_config', 'de-en')}`.
   - Use only {dataset_size} training examples for this demo run (fast iteration)
   - Use the full validation/test set for evaluation

2. **Tokenizer**: Train a SentencePiece BPE tokenizer with shared vocab of {dataset_eval_info.get('primary_dataset', {}).get('vocab_size', 37000)} tokens on the training data.
   - Save tokenizer to `tokenizer.model` in the reproduce directory
   - Skip training if it already exists

3. **Model**: Implement the full Transformer architecture from scratch using PyTorch:
   - Multi-head self-attention, encoder-decoder attention, feed-forward layers
   - Sinusoidal positional encoding
   - For this DEMO run, use SCALED DOWN hyperparams:
     * d_model=256, num_heads=4, num_layers=3, d_ff=512, dropout=0.1
     * (Full paper uses d_model=512, 8 heads, 6 layers)
   - This lets us verify correctness cheaply; scale up for full reproduction

4. **Training**:
   - Adam optimizer with beta1=0.9, beta2=0.98, eps=1e-9
   - Warmup + inverse sqrt LR schedule (warmup_steps=400 for demo, 4000 for full)
   - Label smoothing epsilon=0.1
   - Train for {max_train_steps} steps (paper uses 100K for base)
   - Use batch size of ~32 sentence pairs (paper uses ~25K tokens per batch)
   - Save checkpoint every 100 steps

5. **Evaluation**:
   - Use sacrebleu to compute BLEU on the test set
   - Run beam search with beam_size=4, length_penalty_alpha=0.6
   - Print comparison table showing:
     * Paper's reported score (from dataset_eval_info)
     * Our demo score (expected to be lower due to fewer steps/smaller model)
     * Extrapolated estimate if scaled to full training

6. **Output**: Save results to `reproduce_results.json` with:
   - bleu_score, training_steps, model_params, dataset_size, paper_target_bleu
   - comparison summary

The script should be robust: handle GPU/CPU, handle missing packages with clear error messages, and print clear progress.

Write ONLY the Python code, wrapped in ```python ... ``` fences.
""",
    },
]

reproduce_code = ""
try:
    print(f"[LLM] backend={os.environ.get('LLM_BACKEND', 'openrouter').lower()} model={gpt_version}", flush=True)
    reproduce_response = client.chat.completions.create(
        model=gpt_version,
        messages=reproduce_msg,
        temperature=0.2,
        max_tokens=8192,
    )
    reproduce_text = reproduce_response.choices[0].message.content or ""
    print_response(reproduce_response.model_dump())

    if reproduce_text:
        if "</think>" in reproduce_text:
            reproduce_text = reproduce_text.split("</think>")[-1].strip()
        reproduce_code = extract_code_from_content2(reproduce_text)
        if not reproduce_code:
            reproduce_code = extract_code_from_content(reproduce_text)
        if not reproduce_code and "import" in reproduce_text:
            reproduce_code = reproduce_text

        # Save raw LLM response for debugging
        raw_path = os.path.join(output_dir, "reproduce_response.txt")
        with open(raw_path, "w") as f:
            f.write(reproduce_text)
except Exception as e:
    print(f"[WARNING] LLM code generation failed ({e}).")

if not reproduce_code:
    print("[INFO] Using built-in reproduction template.")
    reproduce_code = _builtin_reproduce_template(
        dataset_eval_info, max_train_steps, dataset_size
    )

# Save the reproduction script
reproduce_script_path = os.path.join(reproduce_dir, "reproduce.py")
with open(reproduce_script_path, "w") as f:
    f.write(reproduce_code)
print(f"\n[INFO] Reproduction script saved to {reproduce_script_path}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: Generate requirements.txt for the reproduction
# ──────────────────────────────────────────────────────────────────────────────
required_packages = dataset_eval_info.get("required_packages", [
    "torch", "datasets", "sentencepiece", "sacrebleu", "tqdm"
])
# Always include these essentials
essentials = ["torch", "datasets", "sentencepiece", "sacrebleu", "tqdm"]
for pkg in essentials:
    if pkg not in required_packages:
        required_packages.append(pkg)

req_path = os.path.join(reproduce_dir, "requirements.txt")
with open(req_path, "w") as f:
    f.write("\n".join(required_packages) + "\n")
print(f"[INFO] requirements.txt saved to {req_path}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: Generate a runner shell script
# ──────────────────────────────────────────────────────────────────────────────
runner_sh = f"""#!/bin/bash
# Reproduction runner for {paper_name}
# Generated by paper2code Stage 5

set -e

REPRODUCE_DIR="{reproduce_dir}"
cd "$REPRODUCE_DIR"

echo "=== Installing dependencies ==="
pip install -q -r requirements.txt

echo ""
echo "=== Running reproduction ==="
echo "Dataset size: {dataset_size} training pairs (demo subset)"
echo "Training steps: {max_train_steps} (paper uses 100K)"
echo ""
python reproduce.py \\
    --reproduce_dir "$REPRODUCE_DIR" \\
    --max_train_steps {max_train_steps} \\
    --dataset_size {dataset_size}

echo ""
echo "=== Results ==="
if [ -f "$REPRODUCE_DIR/reproduce_results.json" ]; then
    python -c "
import json
with open('$REPRODUCE_DIR/reproduce_results.json') as f:
    r = json.load(f)
print('BLEU Score:', r.get('bleu_score', 'N/A'))
print('Paper Target:', r.get('paper_target_bleu', 'N/A'))
print('Training Steps:', r.get('training_steps', 'N/A'))
print('Dataset Size:', r.get('dataset_size', 'N/A'))
"
fi
"""

runner_path = os.path.join(reproduce_dir, "run_reproduce.sh")
with open(runner_path, "w") as f:
    f.write(runner_sh)
os.chmod(runner_path, 0o755)
print(f"[INFO] Runner script saved to {runner_path}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5: Execute the reproduction script
# ──────────────────────────────────────────────────────────────────────────────
print("\n------- Step 5: Installing Dependencies & Running Reproduction -------")

# Install packages
print("[INFO] Installing required packages...")
install_cmd = [sys.executable, "-m", "pip", "install", "-q"] + required_packages
try:
    result = subprocess.run(install_cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"[WARNING] Some packages failed to install:\n{result.stderr[:500]}")
    else:
        print("[INFO] Packages installed successfully.")
except subprocess.TimeoutExpired:
    print("[WARNING] Package installation timed out.")

# Run the reproduction script
print(f"\n[INFO] Running reproduce.py (steps={max_train_steps}, dataset_size={dataset_size})...")
print("[INFO] This may take a few minutes...\n")

run_cmd = [
    sys.executable, "-u",  # unbuffered so output isn't scrambled
    reproduce_script_path,
    "--reproduce_dir", reproduce_dir,
    "--max_train_steps", str(max_train_steps),
    "--dataset_size", str(dataset_size),
]

try:
    result = subprocess.run(
        run_cmd,
        capture_output=False,  # stream to stdout/stderr
        text=True,
        timeout=3600,  # 1 hour max
        cwd=reproduce_dir,
    )
    if result.returncode != 0:
        print(f"\n[ERROR] Reproduction script failed with code {result.returncode}")
        # Try to generate a debug-friendly error message
        print("[INFO] Checking for reproduce_results.json anyway...")
except subprocess.TimeoutExpired:
    print("\n[WARNING] Reproduction timed out after 1 hour.")
except Exception as e:
    print(f"\n[ERROR] Failed to run reproduction: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6: Read and display results
# ──────────────────────────────────────────────────────────────────────────────
print("\n------- Step 6: Results Comparison -------")

results_path = os.path.join(reproduce_dir, "reproduce_results.json")
if os.path.exists(results_path):
    with open(results_path) as f:
        results = json.load(f)

    paper_target = results.get("paper_target_bleu", "N/A")
    our_bleu = results.get("bleu_score", "N/A")
    steps = results.get("training_steps", max_train_steps)
    ds_size = results.get("dataset_size", dataset_size)
    model_params = results.get("model_params", "N/A")

    print("\n" + "=" * 60)
    print(f"  REPRODUCTION RESULTS: {paper_name}")
    print("=" * 60)
    print(f"  Paper Target BLEU  : {paper_target}")
    print(f"  Our Demo BLEU      : {our_bleu}")
    print(f"  Training Steps     : {steps} / 100,000 (paper)")
    print(f"  Dataset Size       : {ds_size:,} / 4,500,000 (paper)")
    print(f"  Model Parameters   : {model_params}")
    print("=" * 60)
    if isinstance(our_bleu, (int, float)) and isinstance(paper_target, (int, float)):
        ratio = our_bleu / paper_target if paper_target > 0 else 0
        print(f"  Performance ratio  : {ratio:.1%} of paper score")
        print(f"  (expected lower due to {100_000 // steps}x fewer steps and smaller dataset)")
    print("=" * 60)

    # Save a summary
    summary = {
        "paper_name": paper_name,
        "timestamp": get_now_str(),
        "paper_target_bleu": paper_target,
        "our_demo_bleu": our_bleu,
        "training_steps_used": steps,
        "paper_training_steps": 100000,
        "dataset_size_used": ds_size,
        "paper_dataset_size": 4500000,
        "model_params": model_params,
        "status": "success" if isinstance(our_bleu, (int, float)) else "partial",
    }
    summary_path = os.path.join(output_dir, "reproduce_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[INFO] Summary saved to {summary_path}")
else:
    print("[WARNING] reproduce_results.json not found. The reproduction script may have failed.")
    print(f"[INFO] Check the reproduce directory: {reproduce_dir}")
    print("[INFO] You can run manually: python reproduce.py")

print("\n[DONE] Stage 5 (Reproduction) complete.")
