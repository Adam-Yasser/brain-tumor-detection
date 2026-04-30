"""
Measure end-to-end API latency (what the .NET backend will actually experience).

Unlike benchmark_predict.py which only times the model, this script sends
real HTTP requests to a running API and measures the full round trip.

Usage:
    # 1) start the API in another terminal:  python scripts/run_api.py --port 8765
    # 2) then run this:                       python scripts/benchmark_api.py
"""

import argparse
import statistics
import time
from pathlib import Path

import requests


DEFAULT_URL = "http://localhost:8765"
DEFAULT_IMAGE = "data/raw/archive/Testing/glioma/Te-gl_1.jpg"


def main():
    parser = argparse.ArgumentParser(description="Benchmark the API end-to-end")
    parser.add_argument("--url", default=DEFAULT_URL, help="API base URL")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Sample image to upload")
    parser.add_argument("--runs", type=int, default=50, help="Timed runs after warmup")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup runs")
    args = parser.parse_args()

    if not Path(args.image).exists():
        raise SystemExit(f"Sample image not found: {args.image}")

    health = requests.get(f"{args.url}/health", timeout=5).json()
    print(f"Health: {health}\n")

    image_bytes = Path(args.image).read_bytes()

    print(f"Warming up ({args.warmup} runs)...")
    for _ in range(args.warmup):
        requests.post(
            f"{args.url}/predict",
            files={"file": ("mri.jpg", image_bytes, "image/jpeg")},
            timeout=30,
        )

    print(f"Running {args.runs} timed API calls...\n")
    timings_ms = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        r = requests.post(
            f"{args.url}/predict",
            files={"file": ("mri.jpg", image_bytes, "image/jpeg")},
            timeout=30,
        )
        timings_ms.append((time.perf_counter() - t0) * 1000)
        r.raise_for_status()

    timings_ms.sort()
    n = len(timings_ms)
    p50 = timings_ms[n // 2]
    p95 = timings_ms[int(n * 0.95)]
    p99 = timings_ms[min(int(n * 0.99), n - 1)]

    print("=" * 50)
    print(f"  URL:            {args.url}")
    print(f"  Image size:     {len(image_bytes) / 1024:.1f} KB")
    print(f"  Runs:           {n}")
    print("-" * 50)
    print(f"  min            {min(timings_ms):8.1f} ms")
    print(f"  median (p50)   {p50:8.1f} ms     <-- typical")
    print(f"  mean           {statistics.mean(timings_ms):8.1f} ms")
    print(f"  p95            {p95:8.1f} ms")
    print(f"  p99            {p99:8.1f} ms")
    print(f"  max            {max(timings_ms):8.1f} ms")
    print("=" * 50)
    print(f"\n  Throughput: {1000 / statistics.mean(timings_ms):.1f} requests/sec (sequential)")


if __name__ == "__main__":
    main()
