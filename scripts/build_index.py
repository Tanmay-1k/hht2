"""
Usage:
    python scripts/build_index.py --langs hi en --num-per-lang 200

Builds (or rebuilds) the FAISS index + metadata sidecar from
ai4bharat/MSMARCO-XI using the chunk strategy configured in .env
(CHUNK_STRATEGY). Run this once before starting the API server, or let
the server auto-build it on first startup (see app/main.py).
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion.indexer import build_index  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", default=["hi"], help="MSMARCO-XI language configs to load")
    parser.add_argument("--num-per-lang", type=int, default=200)
    args = parser.parse_args()

    build_index(languages=args.langs, num_docs_per_lang=args.num_per_lang)


if __name__ == "__main__":
    main()
