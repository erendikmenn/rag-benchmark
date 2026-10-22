"""Explicit sealed-ranking CPU derivation; no model or inference entry point."""
import sys

sys.dont_write_bytecode = True

from rag_benchmark.industrial_none_derivation_v1 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
