"""Explicit CPU-only Industrial None V2 entry point."""
import sys

sys.dont_write_bytecode = True

from rag_benchmark.industrial_none_derivation_v2 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
