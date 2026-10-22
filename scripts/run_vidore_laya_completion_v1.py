"""Entry point for an independently admitted sealed-ranking Laya delta."""
import sys

sys.dont_write_bytecode = True

from rag_benchmark.vidore_laya_completion_v1 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
