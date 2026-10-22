"""Explicit private industrial S/J entry point; invoke with Python -B."""
import sys

sys.dont_write_bytecode = True

from rag_benchmark.industrial_mps_sj_v1 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
