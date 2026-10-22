"""Entry point for an explicitly reviewed ViDoRe cache-only manifest."""
import sys

sys.dont_write_bytecode = True

from rag_benchmark.vidore_cache_recovery_v1 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
