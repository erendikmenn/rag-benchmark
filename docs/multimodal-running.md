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

Native media batch size defaults to 1; dense text batch size defaults to 8. `--bge-reranker-batch-size N` selects a separate batch size only for the BGE reranker; omitted, it retains the native/specialist batch setting. This leaves G/E retrieval and other ranking adapters unchanged. An explicit change is recorded in BGE adapter identity, so batch-1 scores cannot be silently relabelled as batch-8 inference. Validate memory and quality for a new batch condition before comparing its speed; the flag itself is not a measured speedup. The full source content is validated against model limits. Overlength code functions require a separately identified chunk-aggregation experiment; silent truncation is never the main protocol.

## Source-only text views

Whisper transcribes original Turkish audio; reference transcripts are never its input:

```bash
uv run --locked --extra models --extra multimodal rag-multimodal asr \
  data/multimodal/fleurs-tr_tr-test data/multimodal/fleurs-tr_tr-whisper \
  --download --device mps --dtype float32 --score-wer
```

Whisper uses an explicit static KV cache with compilation disabled (`greedy-static-kv-synchronous-stop-v1`). This avoids the installed Transformers MPS deferred-stop cleanup bug while retaining greedy decoding and native timestamp-based processing of recordings longer than 30 seconds. The cache policy is recorded in adapter identity, usage and dataset provenance; transcripts from the previous dynamic-cache identity are not reused. Short/long real CPU and MPS checks have passed. Backend checks do not measure transcription accuracy.

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

Multimodal BM25 uses `positive_scores_only_v1`: only candidates with a positive lexical score are retained. Unmatched queries yield an empty list, and fewer than the configured budget are kept when appropriate. The adapter specification records this policy; caches from the older zero-score-eligible policy are incompatible.

For Gemma relevance on an audio collection containing sources longer than its 30-second input limit, explicitly select `--audio-overlength source_windows_max`. The wrapper retains the entire resampled waveform in contiguous windows, repeats complete provided source text in every window, scores each window, and takes the maximum actual score at the original source ID. It preserves negative scores and does not combine evidence across audio windows. Native short recordings are unchanged. This is a separately identified ranking protocol; without the flag, overlong audio raises an error instead of being clipped. The FLEURS collection contains one 36.84-second source requiring two windows. Source coverage and real model-backend health are reported separately from retrieval accuracy.

For code reranking, use the independent `--code-rerank-overlength source_chunks_max` flag. It limits every reranker source chunk to 4,096 characters, also checking both text-embedding tokenizers, and preserves the complete function through deterministic source-order chunks. The original function receives the maximum actual chunk relevance score. The default text retrieval vectors stay unchanged; JavaScript can separately require `--code-overlength shared_segments_max` for retrieval. A function fitting an embedding tokenizer can still exceed Laya's serialized state limit, so retrieval and reranker input budgets are validated separately. Each base reranker retains its strict prompt/query context checks; this option never enables silent truncation.

### Cached embedding dimension sweep

After a full native run finishes, compare EG2 vector dimensions without new encoder inference:

```bash
.venv/bin/python -m rag_benchmark.multimodal_cli dimensions \
  --dataset data/multimodal/xm3600-tr \
  --base-run runs/multimodal/xm3600-tr-native-specialist \
  --output reports/multimodal-dimensions/xm3600-tr
```

The sweep first reproduces the original 768-dimensional rankings and every stored query metric. It validates the frozen dataset, adapter identity, complete ordered vector blocks, ranking cache and SQLite results. Missing or inconsistent native evidence produces an unavailable report. Specialist fusion is included only when its independent evidence also validates.

The first 128/256/512 FP32 coordinates are copied and L2-normalized, then searched over the **entire corpus**. N+S is recalculated with S fixed and the original candidate budgets. Source inputs, weights and original inference precision remain fixed. Prefix normalization is mathematically equivalent to the adapter's dimension reduction, with possible small FP32 rounding differences; bitwise equivalence to a fresh direct-dimension model call is not claimed.

The 128/256/512 dimensions reduce raw FP32 N-vector storage by 83.3% / 66.7% / 33.3%. These figures describe vector payloads, not total process memory, fused-index size or speed. No new encoder latency is inferred from cached results. Legacy caches may lack original per-vector checksums; the report records newly observed array and ordered-ID hashes and verifies the unchanged 768 baseline instead of inventing historical checksum metadata.

### Video sampling parameter experiments

`run` accepts `--video-fps`, `--video-max-frames` and `--video-vision-budget`. The defaults are the published baseline: 1 fps, at most 16 frames and 140 EG2 soft tokens per frame. Native N, joint J and CLIP S use the same sampling FPS and frame cap; the soft-token setting governs EG2. Default adapter identities remain unchanged. Sampling changes create separate identities and must not overwrite the main baseline.

```bash
.venv/bin/python -m rag_benchmark.multimodal_cli run \
  --dataset data/multimodal/msrvtt-1k-a \
  --run-dir runs/ablations/msrvtt-1k-a-fps2-frames32-vision140 \
  --channels N,S --rerankers none \
  --video-fps 2 --video-max-frames 32 --video-vision-budget 140 \
  --specialist-text-overflow truncate_to_model_limit
```

The [nine-condition execution manifest](../configs/multimodal-video-ablation-jobs.json) fixes FPS at 0.5/1/2 and the frame cap at 8/16/32. It contains planned commands, not measured results. Run these through the sole GPU owner and export to `reports/ablations/video`; they are outside the main-matrix completion denominator. Its baseline entry reuses the original validated run/cache. All conditions are visual-only; audio is excluded. Raw MSR-VTT candidates have no source-only text for J, so J requires a separately prepared description view.

### Document answer-evaluation preparation

The source-verified preparation module runs on CPU without a model call:

```bash
.venv/bin/python -m rag_benchmark.multimodal_qa \
  --dataset data/multimodal/vidore-v3-computer_science-en \
  --raw-queries data/multimodal/raw/vidore-v3-computer_science/queries/test-00000-of-00001.parquet \
  --output work/qa/computer-science-readiness.json
```

It binds raw query-parquet hashes to the frozen manifest, verifies prepared-file hashes and checks every selected-language question and canonical answer against the published source. Positive source labels must reference the frozen corpus. The [eight-collection audit](../reports/multimodal-qa-readiness.json) covers all 2,419 questions; it publishes counts, protocol and hashes only. Raw questions, references and predictions remain local.

This audit is a preparation result, not measured answer quality. The separate resumable runner below freezes the local generator, prompt and decoding settings, validates source media before use, and saves predictions before evaluation. `generation_query()` exposes only the question, language and ID. `select_evidence()` defines closed-book, oracle and actual retrieved top-five conditions for text or images. Oracle is a labelled diagnostic using up to five positive sources, not an ordinary retrieval result. Missing evidence fails explicitly instead of silently dropping a question or source. Text availability differs from image availability: the CS corpus contains two empty OCR fields.

Answer EM and token F1 measure lexical overlap with the canonical reference. Normalization uses Unicode NFKC, language-aware Turkish case handling, punctuation-to-space and whitespace tokenization. Uncertified `raw_answers` are not treated as equivalent reference aliases. Citation source-set precision/recall/hit compare cited source IDs with positive qrels; they do not measure whether the answer's claims are supported. Semantic correctness and citation entailment remain unmeasured and require separate calibrated assessment. This evaluation is outside the 18,460-cell retrieval matrix and the earlier RAGTurk pilot.

### Resumable document QA execution

`multimodal_qa_runner` evaluates one completed retrieval cell per run. Its implementation and adversarial cache checks have passed CPU tests with a fake transport. The first six real one-question health cases passed four and rejected two text-evidence outputs with invalid citation IDs. This verifies neither full-split accuracy nor semantic answer correctness. Use the sole GPU owner and verified local Gemma server described above; do not run it alongside the active suite's model job.

```bash
.venv/bin/python -m rag_benchmark.multimodal_qa_runner \
  --dataset data/multimodal/vidore-v3-computer_science-en \
  --raw-queries data/multimodal/raw/vidore-v3-computer_science/queries/test-00000-of-00001.parquet \
  --run-dir runs/multimodal-qa/cs-bm25 \
  --retrieval-run runs/multimodal/vidore-v3-computer_science-en-text-rerankers \
  --retrieval-cell f317bfabf1550840c0cb0ddd17404adb1fe64dde95083e1c45b85bb9cc4548e3 \
  --max-new 6
```

The example binds the actual BM25/per-channel retrieval cell. The runner registers closed-book, oracle and retrieved conditions in both text and image representations. `--max-new` limits attempted tasks; six attempts do not cover all six condition groups on a multi-question collection. Omit it to process all registered tasks after a successful backend check. Missing retrieval configuration, empty retrieved evidence and evidence exceeding the explicit text cap remain visible as unsupported tasks. Missing selected source text fails that task. Evidence is never silently clipped or replaced with reference answers.

Generation uses pinned Gemma 4 E4B Q4 weights/projector, temperature 0, seed 42, thinking disabled and 384 reserved output tokens. The instruction requires the question's language and JSON answer/citations. Up to five complete source texts must fit a 24,000-character cap; images follow the fixed 1,120-pixel maximum side. The backend separately enforces its verified context and normal completion. The earlier text pilot's 26B-A4B generator is a different condition.

Raw requests, answers and errors are stored only in a git-ignored run directory. A run lock prevents duplicate owners. Source, config, verified runtime and full ranking hashes bind resume behavior. Resuming revalidates cached requests, predictions, citation IDs and recomputed metrics; corrupted completed rows become failed and are excluded, without automatic model retries. Ordinary failed attempts require explicit `--retry-failed`. `aggregate.json` reports completed-only means together with planned/completed/failed/unsupported counts; it never labels these lexical metrics as semantic accuracy. Full retrieval-variant generation and calibrated semantic assessment remain pending.

### Reverse retrieval data preparation

The CPU-only builder prepares separate media-query/reference-text-gallery tasks:

```bash
.venv/bin/python -m rag_benchmark.multimodal_reverse_data \
  --parent data/multimodal/xm3600-tr \
  --output data/ablations/xm3600-tr-reverse-reference-gallery-v1
```

The output must be new and empty. Parent and derived record hashes and actual media bytes are validated; frozen parent records are not modified. The [prepared-view audit](../reports/multimodal-reverse-readiness.json) records:

| Task | Media queries | Text gallery entries | Positive query–text labels |
|---|---:|---:|---:|
| XM3600 Turkish image → caption | 3,600 | 7,233 | 7,233 |
| XM3600 English image → caption | 3,600 | 7,200 | 7,200 |
| FLEURS audio → normalized reference transcript | 743 | 329 | 743 |

Each image's annotated captions are positive; distinct caption IDs are preserved even when wording repeats. FLEURS uses the frozen normalized transcript groups. Reverse queries contain only source media, while human reference text is the explicitly intended search gallery. This is a separate task and must not be relabelled as source-only ASR or generated descriptions.

These manifests have **`runtime_ready=false`**. The current forward engine rejects their separate track names; a reverse registry and a compatible media-query/text-gallery adapter are still required. No reverse model inference or retrieval quality has been measured. Direct audio queries and source-only Whisper queries must remain separate future conditions. These prepared views contribute zero cells to the main forward matrix.


### Plan document QA across completed retrieval conditions

Build the execution manifest without starting a model:

```bash
.venv/bin/python -m rag_benchmark.multimodal_qa_plan \
  --output work/continuation/document-qa-execution-plan.json \
  --shared-generation-cache .cache/qa-generations
```

The planner prepares one `closed_book oracle` × `text image` control job per frozen collection, then one `retrieved` × `text image` job per exact completed retrieval identity. Export duplicates collapse; missing actual ranking stores, failed/partial cells and smoke runs are excluded or retained as pending. Different measured configurations can have the same primary method/budget/K label and stay separate here. The commands pass expected dataset identity and retrieval report SHA256; changed inputs fail before server preflight.

Use `--conditions` and `--representations` to select a subset explicitly. Selection is part of the runner's frozen configuration and requires a separate run directory if it changes. The manifest does not execute commands, launch a server or prove QA accuracy. A queue owner must verify that the model slot is free and run the local pinned generator before executing a job; never send these jobs to the server currently serving another experiment. The [current aggregate plan](../reports/multimodal-qa-execution-plan-summary.json) records 409,148 planned tasks and zero generated answers. Controls are scheduled once. With `--shared-generation-cache`, identical full requests can reuse one response across cells after verifying source bytes, source IDs/order, runtime, model, prompt and decoding settings. Each task still independently validates citations and computes metrics against its own references. No real saved-call count or speedup has been measured. Regenerate the plan after new retrieval results complete.


### Separate reverse retrieval

The intended-reference-gallery preparation is a distinct task: a source image searches annotated captions, or source audio searches normalized reference transcripts. The query never contains the target text. These reference galleries are explicitly labelled as such; they are not source-only ASR or generated descriptions.

The dedicated runner connects pinned EG2 native media queries to the same pinned EG2 text-document encoder (768 dimensions, matching revision/device/dtype, `title: none | text: ` document prefix). Media-only queries consume no textual query prefix. It verifies the exact frozen parent transformation and every media checksum before encoding. Query media vectors must come from a verified completed forward N run; there is no fresh-media fallback. The baseline model/configuration, vector order and coverage, and all stored forward rankings, scores and metrics must match. Only the intended reference-text gallery needs new encoding. The runner searches the complete text gallery with normalized cosine, and records multi-positive metrics. Cache blocks, diagnostics, scores and metrics are revalidated on resume. Missing runtime audio counts remain unavailable. The frozen prepared manifest's `runtime_ready=false` still prevents accidental use in the forward engine; this separate runner does not alter that manifest or the forward gold-free policy.

With the single model slot available, an explicit future run is:

```bash
.venv/bin/python -m rag_benchmark.multimodal_reverse_runner \
  --dataset data/ablations/xm3600-tr-reverse-reference-gallery-v1 \
  --parent data/multimodal/xm3600-tr \
  --run-dir runs/ablations/xm3600-tr-reverse-eg2 \
  --media-baseline-run runs/multimodal/xm3600-tr-native-specialist \
  --media-baseline-cache .cache/multimodal
```

[Three planned commands and frozen source hashes](../configs/multimodal-reverse-jobs.json) cover Turkish/English images and Turkish speech. Raw rankings/vectors stay in ignored run directories; optional `--output` writes an aggregate report only. Before scheduling, verify the manifest hashes listed in that plan. No real reverse model inference has run yet: fake-encoder CPU tests and complete read-only validation of actual forward caches passed for 3,600 TR images, the same 3,600 EN images and 743 speech recordings. EN explicitly reuses the TR media baseline only after matching every ID and source byte. No new model inference was used for those checks. Specialist reverse retrieval, Whisper-query comparisons, confidence intervals and answer generation remain separate pending work. These jobs contribute zero to the primary forward-matrix denominator.

### Explicit description output budgets

The default source-only caption budget remains 192 output tokens. A normal-stop response is mandatory; reaching the cap never produces an accepted description. For a separately identified recovery, the single queue owner can select `describe --max-output-tokens 512`. This setting changes the generator/cache identity, retains prompt-plus-output context validation and keeps the old partial cache intact. It does not validate description meaning or guarantee that every response will fit. Do not relabel the completed 192-token Clotho measurements or mix their descriptions with a different-budget generator.

### Code query-prefix comparison without repeating document inference

The separate code-prefix runner validates an ordinary completed E baseline and reuses its document vectors (or the exact shared JavaScript chunks). It recomputes the old rankings, scores and metrics before allowing any new query inference. Missing or incompatible caches produce an unavailable result; documents are never silently re-encoded. The pinned model configuration verifies `task: code retrieval | query: ` and the unchanged `title: none | text: ` document prefix. Query settings inherit the baseline; an explicit batch override is separately recorded.

```bash
.venv/bin/python -m rag_benchmark.multimodal_code_prefix_ablation \
  --dataset data/multimodal/codesearchnet-python-test \
  --baseline-run runs/multimodal/codesearchnet-python-test-dense-fusions \
  --baseline-cache .cache/multimodal \
  --run-dir runs/ablations/codesearchnet-python-code-query-prefix \
  --verify-baseline-only
```

This command is a CPU cache check. Removing `--verify-baseline-only` authorizes new query encoding and must be done only by the GPU queue owner. [Six planned language jobs](../configs/multimodal-code-prefix-jobs.json) preserve source/report hashes and separate output directories. Real CPU verification passed on all six languages: 183,295 functions and 183,398 cached document vectors (JavaScript uses 14,084 vectors for 13,981 functions). No code-prefix query inference or accuracy result has been measured yet. These experiments belong outside the main forward-matrix denominator.

The shared generation cache is opt-in and requires a private, git-ignored directory (0700; database/lock 0600). References never enter its generation key or raw response store. A checksum mismatch, interrupted running entry or changed request/media/runtime/configuration fails closed; even `--retry-failed` cannot regenerate an integrity failure. Explicit retry is allowed only for a recorded transport exception with unchanged inputs. Reports distinguish new transport-call attempts (including failures), shared responses and local completed-task cache hits; historical response usage is not fresh inference latency.


## Staged BGE sparse and ColBERT core

`multimodal_bge_extra.py` implements the separately tested sparse-head formula and exact blocked mean-MaxSim scoring, plus an injected-backend extraction/cache core. It does not add a production model backend, CLI or measured benchmark. Main G remains BGE dense. The sparse mode uses maximum positive trained-head weight per non-special token ID; ColBERT drops CLS, retains EOS and averages maximum token similarities, including negative values. JavaScript source scores take the maximum score of its existing lossless chunks.

Head projection, normalization, scoring and saved token vectors are explicitly FP32. A future backend must show that its actual CLS output matches the verified dense baseline under the strict tolerance; real MPS/BF16 compatibility is unmeasured and must not be asserted from CPU fakes. The core rejects truncated or non-right-padded input, changed inputs/runtime, incompatible supplied CLS output, nonfinite calculations and corrupted/interrupted caches. The separate `VerifiedGBaseline` reader now validates original G source/query rows, adapter config, complete rankings and all 17 metrics; process-level private cache locking and the real token backend remain pending. Use `verified_inputs(role)` immediately before extraction to recheck frozen source files and protected state. Its public summary strips local vector paths. Both trained heads must pass pinned SHA checks; random replacement heads are forbidden.

All six code galleries remain in scope. Local complete-input token accounting predicts 116.703 GiB of FP32 token payload, excluding metadata and temporary storage; this is not a RAM or speed measurement. Ruby is the first planned bounded full-gallery validation. A single future hidden-state pass must produce both extra heads; no separate fresh dense-baseline pass is planned. Candidate-only ColBERT ranking would be a distinct reranking experiment. The formulas are bound to [official FlagEmbedding commit fd1a2bd](https://github.com/FlagOpen/FlagEmbedding/tree/fd1a2bdf69488ffebe0327999d4400d8c8058a0b), with file hashes in the module. No sparse/ColBERT quality result is claimed.

The citation prompt contract `qa-exact-source-id-citations-v2` supplies an exact ordered JSON array of allowed source IDs in the trusted system instruction. Citation strings must match one complete ID; titles, descriptions and URLs cannot replace IDs. The first real health attempt had four passing and two rejected samples; v2 has passed CPU regressions but a real retry is pending. Preserve the old integrity-blocked requests, regenerate source/runner-bound execution plans and use new run directories for the new contract. See [health diagnosis](../reports/multimodal-qa-runtime-health.json); this is an execution check, not a semantic correctness score.
