# Running the multimodal suite locally

Run commands from the repository root. The measured machine is an M4 Max with 128 GB unified memory. This is an execution target, not a speed guarantee. Install Python 3.11–3.13 and `uv` first.

```bash
uv sync --locked --extra models --extra multimodal
```

## Channels and results

| Channel | Input representation | Retriever |
|---|---|---|
| B | Published OCR, cleaned code, or a separately prepared source-only text view | BM25 |
| G | Same text view | BGE-M3 dense |
| E | Same text view | EmbeddingGemma 2 text |
| N | Original image, waveform, sampled video, or composed image query | EmbeddingGemma 2 native |
| S | Original media | SigLIP2 for photos; ColQwen2.5 token MaxSim for pages; CLAP for audio; CLIP frame pooling for video |
| J | Original media together with its fixed source-only text representation | EmbeddingGemma 2 joint |

A requested channel set evaluates every nonempty subset, using equal-weight RRF for combinations. `--candidate-k 20,50,100` applies to reranking, and `--budget-modes per_channel,total` selects the two candidate-budget conditions. The primary reported reranking condition is K=50 with a per-channel retrieval budget of 100. `--rerankers none,laya_text,bge_reranker_text,gemma4_relevance` requests all ranking conditions. An unavailable channel is reported explicitly and cannot produce a score.

All models use pinned revisions and local files during inference. Acquisition is separate. The repository's MIT license covers its code; model and dataset licenses remain those of their publishers. Raw media and model weights are not redistributed here.

## Prepare data and models

```bash
uv run --locked --extra models --extra multimodal rag-multimodal prepare xm3600
uv run --locked --extra models --extra multimodal rag-multimodal prepare vidore --collection computer_science
uv run --locked --extra models --extra multimodal rag-multimodal prepare clotho
uv run --locked --extra models --extra multimodal rag-multimodal prepare clotho-additional
uv run --locked --extra models --extra multimodal rag-multimodal prepare fleurs
uv run --locked --extra models --extra multimodal rag-multimodal prepare msrvtt
uv run --locked --extra models --extra multimodal rag-multimodal prepare codesearchnet --language python
```

Other code languages are `ruby`, `javascript`, `java`, `go`, and `php`. Each prepared manifest pins its source and validates the full gallery, query and label counts. CIRR's official media requires its publisher's access process; annotations alone do not make a runnable evaluation.

```bash
uv run --locked --extra models --extra multimodal rag-benchmark prepare-models --only bge,embeddinggemma,laya
uv run --locked --extra models --extra multimodal rag-multimodal preflight --download --models embeddinggemma_native,embeddinggemma_joint,siglip2,clap,clip_video,bge_reranker
uv run --locked --extra models --extra multimodal python -m rag_benchmark.multimodal_colqwen --download --device mps
```

Preflight uses synthetic fixtures to check backend execution. Its passing status is not an accuracy measurement. Dedicated ColQwen loading is required because token MaxSim is not a single-vector cosine search.

## Native retrieval

```bash
uv run --locked --extra models --extra multimodal rag-multimodal run \
  --dataset data/multimodal/xm3600-tr \
  --run-dir runs/multimodal/xm3600-tr-native-specialist \
  --channels N,S --specialist-text-overflow truncate_to_model_limit
```

This runs all 7,233 queries by default. `--limit` creates an explicitly partial diagnostic run. SigLIP2's native 64-token limit clips 56 of these Turkish queries under the explicit policy above. The default overflow policy is `error`; policy, original lengths and clipping counts are recorded. The EG2 branch retains its own full supported input.

For documents, the published OCR already provides a text view:

```bash
uv run --locked --extra models --extra multimodal rag-multimodal run \
  --dataset data/multimodal/vidore-v3-computer_science-en \
  --run-dir runs/multimodal/vidore-cs-all-retrieval \
  --channels B,G,E,N,S,J --rerankers none
```

Native media batch size defaults to 1; dense text batch size defaults to 8. The full source content is validated against model limits. Overlength code functions require a separately identified chunk-aggregation experiment; silent truncation is never the main protocol.

## Source-only text views

Whisper transcribes original Turkish audio; reference transcripts are never its input:

```bash
uv run --locked --extra models --extra multimodal rag-multimodal asr \
  data/multimodal/fleurs-tr_tr-test data/multimodal/fleurs-tr_tr-whisper \
  --download --device mps --dtype float32 --score-wer
```

Gemma 4 E4B descriptions and relevance scoring use a separately verified local multimodal server. This is distinct from the earlier text-only pilot's Gemma 4 26B-A4B answer generator.

```bash
uv run --locked --extra models --extra multimodal rag-multimodal prepare-generation --include-runtime
uv run --locked --extra models --extra multimodal rag-multimodal serve
```

Keep that terminal open. From another terminal:

```bash
uv run --locked --extra models --extra multimodal rag-multimodal describe \
  --dataset data/multimodal/xm3600-tr \
  --output data/multimodal/xm3600-tr-gemma-described --caption-language tr
```

Descriptions are generated once from source media alone. The default `--caption-language auto` follows the declared dataset language, falling back to English only when none is declared; explicit `en`, `tr` and `fr` are supported. The description cache is reused across matching language views; model, prompt and source hashes are checked. A dataset view is published only after every required description is complete. Gold query captions, answers and reference transcripts are excluded. Image queries with an edit instruction need a source-only description of the reference image for text-only methods.

Then run `B,G,E,N,S,J` on the described view. Add the requested rerankers as a separate run. `gemma4_relevance` needs the local server; it provides candidate relevance scores, not ground-truth correctness labels. Do not start multiple active GPU inference workers. Context shifting is disabled, and incomplete responses are rejected.

## Publish measured results

```bash
uv run --locked --extra models --extra multimodal rag-multimodal export \
  --run-dir runs/multimodal/vidore-cs-all-retrieval \
  --output reports/multimodal/vidore-cs-all-retrieval
```

`report.json` and `matrix.csv` distinguish completed, failed, unsupported and planned conditions. A successful orchestration exit does not imply every family completed: inspect these statuses. Public exports contain aggregate metrics and configuration; raw queries, media, prompts and local exception logs remain local. Hit@K, Recall@K and generated-answer correctness are different measurements.

Reusing a run directory resumes valid scores. Changing data, model settings, prompt or precision changes identities. Cached rows are not counted as fresh inference latency. Freeze prepared manifests before starting a run, including metadata fields.

For JavaScript, select the separately identified `--code-overlength shared_segments_max` protocol. Both encoders share lossless source boundaries; retrieval and reranker scores are aggregated at the original function ID. For measured same-run differences, run `rag-multimodal analyze --dataset DATASET --run-dir RUN --output REPORT`. The analysis checks frozen identities and query coverage before computing paired source-group intervals; insufficient source groups produce descriptive differences without invented confidence intervals.
