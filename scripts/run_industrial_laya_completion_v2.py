"""Prepare and preflight the separately admitted Industrial Laya V2 parent."""
import sys

sys.dont_write_bytecode = True

from rag_benchmark.industrial_laya_completion_v2 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
