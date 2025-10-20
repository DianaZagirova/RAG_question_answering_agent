#!/usr/bin/env python3
"""
Test GPU usage for embedding generation.
"""
import torch
import os
from sentence_transformers import SentenceTransformer
import time

print("="*70)
print("GPU Configuration Test")
print("="*70)

# Check CUDA availability
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
        mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
        mem_free = mem_total - mem_reserved
        print(f"    Total: {mem_total:.1f} GB, Free: {mem_free:.1f} GB, Used: {mem_reserved:.1f} GB")

# Check environment variable
cuda_device = os.getenv('CUDA_VISIBLE_DEVICES', 'not set')
print(f"\nCUDA_VISIBLE_DEVICES: {cuda_device}")

# Recommend best GPU based on free memory
if torch.cuda.is_available():
    print("\n" + "="*70)
    print("GPU Recommendation")
    print("="*70)
    
    best_gpu = None
    max_free = 0
    for i in range(torch.cuda.device_count()):
        mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
        mem_free = mem_total - mem_reserved
        if mem_free > max_free:
            max_free = mem_free
            best_gpu = i
    
    print(f"\nBest GPU: {best_gpu} ({torch.cuda.get_device_name(best_gpu)})")
    print(f"Free memory: {max_free:.1f} GB")
    print(f"\nTo use this GPU, set in .env:")
    print(f"  CUDA_DEVICE={best_gpu}")
    print(f"\nOr run with:")
    print(f"  CUDA_VISIBLE_DEVICES={best_gpu} python scripts/run_full_ingestion.py --validated-only")

# Test embedding speed
print("\n" + "="*70)
print("Embedding Speed Test")
print("="*70)

model_name = 'sentence-transformers/all-mpnet-base-v2'
print(f"\nLoading model: {model_name}")

# Test on CPU
print("\n1. CPU Test:")
model_cpu = SentenceTransformer(model_name, device='cpu')
test_texts = ["This is a test sentence for embedding generation."] * 100

start = time.time()
embeddings_cpu = model_cpu.encode(test_texts, batch_size=32, show_progress_bar=False)
cpu_time = time.time() - start
print(f"   100 texts: {cpu_time:.3f}s ({cpu_time/100*1000:.1f}ms per text)")

# Test on GPU if available
if torch.cuda.is_available():
    print("\n2. GPU Test:")
    device = f'cuda:{best_gpu}' if best_gpu is not None else 'cuda:0'
    model_gpu = SentenceTransformer(model_name, device=device)
    
    start = time.time()
    embeddings_gpu = model_gpu.encode(test_texts, batch_size=64, show_progress_bar=False)
    gpu_time = time.time() - start
    print(f"   100 texts: {gpu_time:.3f}s ({gpu_time/100*1000:.1f}ms per text)")
    
    speedup = cpu_time / gpu_time
    print(f"\n   Speedup: {speedup:.1f}x faster on GPU")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print(f"For 18,442 papers (~25 chunks each = 461,050 chunks):")
    print(f"  CPU time: ~{461050 * cpu_time / 100 / 3600:.1f} hours")
    print(f"  GPU time: ~{461050 * gpu_time / 100 / 3600:.1f} hours")
    print(f"  Time saved: ~{(461050 * cpu_time / 100 - 461050 * gpu_time / 100) / 3600:.1f} hours")
else:
    print("\n⚠️  GPU not available - will run on CPU (much slower)")

print("\n" + "="*70)
