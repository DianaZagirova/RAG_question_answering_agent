"""
Benchmark script to estimate ingestion time on CPU vs GPU.
Tests embedding generation speed which is the bottleneck.
"""
import os
import sys
from pathlib import Path
import time
from datetime import timedelta

# Set CPU mode for testing
os.environ['CUDA_VISIBLE_DEVICES'] = ''

sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer


def benchmark_embeddings(device='cpu', n_samples=50):
    """
    Benchmark embedding generation speed.
    
    Args:
        device: 'cpu' or 'cuda'
        n_samples: Number of text samples to test
    """
    print(f"\n{'='*70}")
    print(f"BENCHMARK: {device.upper()}")
    print(f"{'='*70}\n")
    
    # Create sample texts (similar to paper chunks)
    sample_texts = [
        "This is a sample scientific text about aging and longevity. " * 20
        for _ in range(n_samples)
    ]
    
    print(f"Loading model on {device}...")
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    if device == 'cuda':
        model = model.to('cuda')
    print(f"✓ Model loaded\n")
    
    print(f"Generating embeddings for {n_samples} chunks...")
    start = time.time()
    
    embeddings = model.encode(
        sample_texts,
        batch_size=32 if device == 'cuda' else 8,
        show_progress_bar=False
    )
    
    elapsed = time.time() - start
    
    print(f"✓ Completed in {elapsed:.2f} seconds")
    print(f"  Rate: {n_samples / elapsed:.2f} chunks/second")
    print(f"  Per chunk: {elapsed / n_samples * 1000:.1f} ms")
    
    return n_samples / elapsed


def estimate_full_ingestion(chunks_per_second, total_papers=43637, chunks_per_paper=25.5):
    """Estimate full ingestion time."""
    total_chunks = total_papers * chunks_per_paper
    total_seconds = total_chunks / chunks_per_second
    
    return {
        'total_papers': total_papers,
        'total_chunks': int(total_chunks),
        'chunks_per_second': chunks_per_second,
        'total_seconds': total_seconds,
        'total_time': timedelta(seconds=int(total_seconds)),
        'hours': total_seconds / 3600,
        'days': total_seconds / 86400
    }


def main():
    print("\n" + "="*70)
    print("INGESTION TIME ESTIMATION")
    print("="*70)
    
    print("\nThis will benchmark embedding generation speed on CPU.")
    print("GPU benchmark requires manually running with CUDA_VISIBLE_DEVICES=3")
    
    # CPU Benchmark
    print("\n" + "="*70)
    print("Testing CPU Performance...")
    print("="*70)
    
    cpu_rate = benchmark_embeddings(device='cpu', n_samples=50)
    
    print("\n" + "="*70)
    print("FULL INGESTION ESTIMATES")
    print("="*70)
    
    # CPU Estimate
    print("\n📊 CPU Ingestion Estimate:")
    cpu_est = estimate_full_ingestion(cpu_rate, total_papers=43637)
    print(f"  Total papers: {cpu_est['total_papers']:,}")
    print(f"  Total chunks: {cpu_est['total_chunks']:,}")
    print(f"  Processing rate: {cpu_est['chunks_per_second']:.1f} chunks/second")
    print(f"  Estimated time: {cpu_est['total_time']}")
    print(f"  Hours: {cpu_est['hours']:.1f}")
    print(f"  Days: {cpu_est['days']:.1f}")
    
    # GPU Estimate (based on previous test: 143.5 papers/min = ~60 chunks/sec)
    print("\n📊 GPU Ingestion Estimate (from previous test):")
    # Previous test: 100 papers in 41 seconds = 143.5 papers/min
    # 100 papers created 2549 chunks = 25.49 chunks/paper
    # 2549 chunks in 41 seconds = 62.2 chunks/second
    gpu_chunks_per_sec = 62.2
    gpu_est = estimate_full_ingestion(gpu_chunks_per_sec, total_papers=43637)
    print(f"  Total papers: {gpu_est['total_papers']:,}")
    print(f"  Total chunks: {gpu_est['total_chunks']:,}")
    print(f"  Processing rate: {gpu_est['chunks_per_second']:.1f} chunks/second")
    print(f"  Estimated time: {gpu_est['total_time']}")
    print(f"  Hours: {gpu_est['hours']:.1f}")
    
    # Comparison
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    speedup = cpu_est['total_seconds'] / gpu_est['total_seconds']
    time_saved = cpu_est['total_seconds'] - gpu_est['total_seconds']
    
    print(f"\n⚡ GPU Speedup: {speedup:.1f}x faster than CPU")
    print(f"⏱️  Time saved: {timedelta(seconds=int(time_saved))}")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if cpu_est['hours'] > 48:
        print("\n⚠️  CPU ingestion would take over 2 days!")
        print("   Strongly recommend using GPU 3:")
        print("   CUDA_VISIBLE_DEVICES=3 python scripts/run_full_ingestion.py --reset")
    elif cpu_est['hours'] > 24:
        print("\n⚠️  CPU ingestion would take over 1 day.")
        print("   GPU recommended:")
        print("   CUDA_VISIBLE_DEVICES=3 python scripts/run_full_ingestion.py --reset")
    else:
        print("\n✓ CPU ingestion is feasible but GPU is still faster.")
    
    print("\n💡 Options:")
    print("  1. Use GPU 3 (recommended):")
    print("     CUDA_VISIBLE_DEVICES=3 python scripts/run_full_ingestion.py --reset")
    print("\n  2. Use CPU (if GPU busy):")
    print("     CUDA_VISIBLE_DEVICES='' python scripts/run_full_ingestion.py --reset")
    print("\n  3. Process in batches (to free GPU periodically):")
    print("     CUDA_VISIBLE_DEVICES=3 python scripts/run_full_ingestion.py --limit 10000 --reset")
    print("     # Wait, then:")
    print("     CUDA_VISIBLE_DEVICES=3 python scripts/run_full_ingestion.py --limit 20000")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
