#!/usr/bin/env bash
# Convert a research paper PDF → raw S2ORC JSON → cleaned JSON.
#
# Usage:
#   bash ingestion/scripts/ingest_pdf.sh <PDF_PATH> [OUTPUT_DIR]
#
# Prerequisites:
#   - Grobid must be running: make start-grobid  (in a separate terminal)
#   - Python env: pip install -e ingestion/

set -euo pipefail

PDF_PATH="${1:-}"
if [[ -z "$PDF_PATH" ]]; then
    echo "Usage: bash ingestion/scripts/ingest_pdf.sh <PDF_PATH> [OUTPUT_DIR]" >&2
    exit 1
fi
[[ -f "$PDF_PATH" ]] || { echo "Error: PDF not found: $PDF_PATH" >&2; exit 1; }

PDF_PATH="$(realpath "$PDF_PATH")"
PAPER_NAME="$(basename "$PDF_PATH" .pdf)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
INGESTION_ROOT="$REPO_ROOT/ingestion"

OUTPUT_DIR="${2:-$REPO_ROOT/data/cleaned_json}"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"
mkdir -p "$OUTPUT_DIR"

RAW_DIR="$REPO_ROOT/data/raw_json"
mkdir -p "$RAW_DIR"

RAW_JSON_PATH="$RAW_DIR/${PAPER_NAME}.json"
CLEANED_JSON_PATH="$OUTPUT_DIR/${PAPER_NAME}_cleaned.json"

# ── Check Grobid is running ───────────────────────────────────────────────────

GROBID_PORT=8070

if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${GROBID_PORT}/api/isalive" 2>/dev/null | grep -q "200"; then
    echo "Grobid ready on port ${GROBID_PORT}."
else
    echo "Error: Grobid is not running on port ${GROBID_PORT}." >&2
    echo "Start it first (in a separate terminal):" >&2
    echo "  make start-grobid" >&2
    echo "  # or: sudo docker run --rm --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.9.0-crf" >&2
    exit 1
fi

# ── PDF → raw JSON ────────────────────────────────────────────────────────────

TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT

echo ""
echo "=== Step 1/2: PDF → S2ORC JSON ==="
python "$INGESTION_ROOT/doc2json/grobid2json/process_pdf.py" \
    -i "$PDF_PATH" \
    -t "$TEMP_DIR" \
    -o "$RAW_DIR"

[[ -f "$RAW_JSON_PATH" ]] || { echo "Error: expected $RAW_JSON_PATH not produced." >&2; exit 1; }
echo "Raw JSON: $RAW_JSON_PATH"

# ── raw JSON → cleaned JSON ───────────────────────────────────────────────────

echo ""
echo "=== Step 2/2: Cleaning JSON ==="
python "$REPO_ROOT/paper2code/codes/0_pdf_process.py" \
    --input_json_path "$RAW_JSON_PATH" \
    --output_json_path "$CLEANED_JSON_PATH"

echo ""
echo "Done."
echo "  Raw JSON    : $RAW_JSON_PATH"
echo "  Cleaned JSON: $CLEANED_JSON_PATH"
