"""
Measure how long a single prediction takes.

Reports the breakdown:
    - Cold start: first prediction (includes one-time CUDA/cuDNN init)
    - Warm latency: typical time after warmup (this is the realistic number)

Usage:
    python scripts/benchmark_predict.py
    python scripts/benchmark_predict.py --runs 200 --image path/to/mri.jpg
"""

import argparse
import statistics
import time
from pathlib import Path

from src.inference.predictor import BrainTumorPredictor


DEFAULT_IMAGE = "data/raw/archive/Testing/glioma/Te-gl_1.jpg"
DEFAULT_MODEL = "outputs/models/best_model.pth"


def main():
    parser = argparse.ArgumentParser(description="Benchmark single-image prediction latency")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to model checkpoint")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Sample image path")
    parser.add_argument("--runs", type=int, default=50, help="Number of timed runs after warmup")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup runs (not counted)")
    args = parser.parse_args()

    if not Path(args.image).exists():
        raise SystemExit(f"Sample image not found: {args.image}")

    print("Loading model...")
    load_start = time.perf_counter()
    predictor = BrainTumorPredictor(args.model)
    load_ms = (time.perf_counter() - load_start) * 1000
    print(f"Model load time: {load_ms:.0f} ms\n")

    # Cold-start measurement (includes lazy CUDA/cuDNN initialization)
    print("Cold start (first prediction)...")
    cold_start = time.perf_counter()
    predictor.predict(args.image)
    cold_ms = (time.perf_counter() - cold_start) * 1000
    print(f"Cold-start latency: {cold_ms:.1f} ms\n")

    # Warmup
    print(f"Warming up ({args.warmup} runs)...")
    for _ in range(args.warmup):
        predictor.predict(args.image)

    # Timed runs
    print(f"Running {args.runs} timed predictions...\n")
    timings_ms = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        predictor.predict(args.image)
        timings_ms.append((time.perf_counter() - t0) * 1000)

    timings_ms.sort()
    n = len(timings_ms)
    p50 = timings_ms[n // 2]
    p95 = timings_ms[int(n * 0.95)]
    p99 = timings_ms[min(int(n * 0.99), n - 1)]

    print("=" * 50)
    print(f"  Sample image:   {args.image}")
    print(f"  Device:         {predictor.device}")
    print(f"  Runs:           {n}")
    print("-" * 50)
    print(f"  min            {min(timings_ms):8.1f} ms")
    print(f"  median (p50)   {p50:8.1f} ms     <-- typical")
    print(f"  mean           {statistics.mean(timings_ms):8.1f} ms")
    print(f"  p95            {p95:8.1f} ms     <-- 95% of requests faster than this")
    print(f"  p99            {p99:8.1f} ms")
    print(f"  max            {max(timings_ms):8.1f} ms")
    print("=" * 50)
    print(f"\n  Throughput: {1000 / statistics.mean(timings_ms):.1f} predictions/sec")


if __name__ == "__main__":
    main()
