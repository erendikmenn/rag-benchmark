"""Entry point for the new first-run bounded Industrial Laya V3 stage."""
import sys

sys.dont_write_bytecode = True

from rag_benchmark.industrial_laya_bounded_v5 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
