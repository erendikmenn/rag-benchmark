"""Entry point for the separately admitted fresh bounded Industrial Laya stage."""
import sys

sys.dont_write_bytecode = True

from rag_benchmark.industrial_laya_bounded_v1 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
