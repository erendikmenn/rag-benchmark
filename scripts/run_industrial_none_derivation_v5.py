"""Explicit CPU-only Industrial None V5 entry point."""
import sys

sys.dont_write_bytecode = True

from rag_benchmark.industrial_none_derivation_v5 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
