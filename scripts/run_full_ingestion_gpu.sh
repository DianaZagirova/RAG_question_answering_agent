#!/bin/bash
# Run full ingestion with GPU acceleration on the least-used GPU

# Use GPU 1 (has most free memory: ~21 GB free)
export CUDA_VISIBLE_DEVICES=1

echo "============================================================"
echo "Running ingestion with GPU acceleration"
echo "============================================================"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Run with validated-only flag
python scripts/run_full_ingestion.py --validated-only "$@"
