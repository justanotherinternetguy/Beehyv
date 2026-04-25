#!/usr/bin/env bash
# Pull the Grobid Docker image (one-time setup).
# Usage: bash ingestion/scripts/setup_grobid.sh

echo "Pulling Grobid 0.9.0-crf Docker image ..."
sudo docker pull grobid/grobid:0.9.0-crf
echo ""
echo "Done. Start Grobid with:"
echo "  bash ingestion/scripts/run_grobid.sh"
echo "  make start-grobid"
