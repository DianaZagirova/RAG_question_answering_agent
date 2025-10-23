#!/bin/bash
# Run LLM Voter to combine RAG and full-text evaluation results

cd "$(dirname "$0")"

python scripts/llm_voter.py "$@"
