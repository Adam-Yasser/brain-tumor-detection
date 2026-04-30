"""
Launch the Brain Tumor Detection API.

Usage:
    python scripts/run_api.py                  # default: 0.0.0.0:8000
    python scripts/run_api.py --port 9000      # custom port
    python scripts/run_api.py --reload         # auto-reload on code change (dev)

Environment variables:
    MODEL_PATH  — path to the .pth checkpoint (default: outputs/models/best_model.pth)
"""

import argparse

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run the Brain Tumor Detection API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code change")
    args = parser.parse_args()

    print(f"Starting API on http://{args.host}:{args.port}")
    print(f"Interactive docs: http://{args.host}:{args.port}/docs")

    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
