"""Entry point for the separately admitted bounded Laya recovery."""
import sys

sys.dont_write_bytecode = True

from rag_benchmark.vidore_laya_recovery_v2 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
