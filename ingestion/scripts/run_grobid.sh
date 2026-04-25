#!/usr/bin/env bash
# Start the Grobid server via Docker (runs in foreground; Ctrl-C to stop).
# Usage: bash ingestion/scripts/run_grobid.sh

echo "Starting Grobid 0.9.0-crf on http://localhost:8070 ..."
echo "Press Ctrl-C to stop."
echo ""

sudo docker run --rm --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.9.0-crf
