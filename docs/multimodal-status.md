# Multimodal execution status

Updated: 2026-10-07 02:59 Europe/Istanbul.

**Full-split evaluation has started.** The first published result is the [ViDoRe computer-science BM25 baseline](../reports/multimodal/vidore-v3-computer_science-en-bm25/report.md): 215 queries, 1,360 candidate pages, Hit@5 0.9209, Recall@5 0.5353 and nDCG@10 0.6334. It measures retrieval, not generated-answer correctness.

The authorized [plan](multimodal-plan.tr.md) covers seven tracks, 321 retrieval subsets and 1,284 method families. The [registry](../configs/multimodal-variants.csv) describes planned scope, not completed experiments. Each actual run publishes completed, planned, unsupported and failed cells separately.

| Track | Verified prepared data | Execution / published evaluation |
|---|---|---|
| Photo | XM3600: 3,600 images, 7,233 TR and 7,200 EN queries | TR EG2 / SigLIP2 / fusion Hit@5: 84.64% / 59.88% / 76.28%; EN: 83.75% / 76.89% / 82.92%; both full comparisons complete |
| Visual document | All eight ViDoRe V3 collections: 19,252 pages, 2,419 base queries | Positive-only BM25 complete on all eight / 2,419 queries; CS native-image measured; dense/joint/ColQwen queued |
| Environmental audio | Clotho: 1,045 clips / 5,225 captions; separate additional-relevance protocol: 1,037 queries / 3,116 positives | Original full protocol complete: EG2 / CLAP / fusion Hit@5 11.75% / 37.42% / 26.47%; additional-relevance run active |
| Turkish speech | FLEURS: 743 recordings / 329 unique normalized transcript queries / 743 positives | Native preflight and real CPU Whisper transcription passed; full native/ASR views queued |
| Video | 1,000 videos / 1,000 prescribed 1K-A queries; 884 clips have audio | Native video and CLIP frame preflight passed; full evaluation queued |
| Code | All six languages: 183,295 functions / 52,561 queries | Positive-only full BM25 complete for all six languages / 52,561 queries; dense comparisons queued |
| Composed image query | CIRR official annotations inspected; official media needs the publisher's access process | Native joint-input preflight passed; full official gallery unavailable |

## Dataset and model audit notes

- The pinned current ViDoRe energy snapshot has 2,225 pages, four fewer than the paper's 2,229. The verified eight-collection total is consequently 19,252. No replacement records are invented.
- Clotho's pinned additional-relevance file contains 1,037 unique queries. It is evaluated as a separate protocol; it does not silently replace the original 5,225-caption evaluation.
- CodeSearchNet source comments and Python standalone docstrings are removed with syntax parsing, independently of query text. Runtime string literals are preserved. Full expected gallery/query counts and source hashes are required.
- Specialist query limits differ: SigLIP2 64, CLIP 77 and CLAP 512 tokens. Strict mode errors on overflow; explicitly selected truncation is separately identified and counted. XM3600 TR has 56 queries longer than SigLIP2's limit; English has none. No input loss is hidden.
- ColQwen uses token-level MaxSim, not an averaged single vector. Its real-weight CPU and MPS preflights passed. The initial dense-interface diagnostic is superseded by the dedicated token-retrieval adapter.
- Local EG2 image/audio/video/joint and specialist health checks are diagnostics, not benchmark scores. A 31-second EG2 audio diagnostic processed the entire waveform without clipping.
- ASR transcripts and generated descriptions use source media only. Gold query captions, transcripts and answers never become candidate descriptions. Human/reference transcripts remain evaluation data.

## Execution and continuation

Independent dataset downloads, tests and CPU baselines run concurrently. One heavy MPS job owns the GPU slot; a second model server or GPU worker must not start until it is released. Current retrieval process and logs are local under `work/overnight/`. Inspect them before resuming; completed inference is cached and must not be charged as fresh model latency.

The durable local queue is `work/overnight/continue_suite.py`, with journal `work/overnight/suite-state.json`, top-level log `work/overnight/suite.log` and per-job logs under `work/overnight/suite-logs/`. It owns an exclusive GPU lock and an idle-sleep assertion. The first worker finished; the continuation queue is now active. It stops launching new jobs at 09:00 and records deferred work. Do not start a second queue while its PID is active. The separate CPU worker is `work/overnight/run_bm25.py`, with log `bm25-positive.log`. It finished all six code and eight PDF baselines under the positive-only lexical policy (54,980 queries); no CPU rerun is needed without a changed protocol. The earlier `finish_bm25.py` chain was cancelled; do not restart it.

The next steps are full native/specialist retrieval on ready datasets, source-only ASR/caption views, compatible B/G/E/N/S/J fusions, and the three optional rerankers. Parameter sweeps and answer-generation evaluation remain pending. The complete family/budget grid contains substantially more work than a single retrieval run; no unmeasured completion time or accuracy is promised.

Raw datasets, prompts, media, keys and local caches stay out of Git. Publish only aggregate reports and tested source changes in atomic commits. Refresh English/Turkish READMEs after meaningful completed milestones. Overnight continuation checks are scheduled through 09:00 Europe/Istanbul on 2026-10-07.

## Completed audits and current limitations

- [Dataset audit](../reports/multimodal-dataset-audit.json): 20 complete protocol views. Totals across views overlap; they are not independent unique-source counts. All raw source files remain frozen.
- [English photo comparison](../reports/multimodal/xm3600-en-native-specialist/paired-comparisons.md): 7,200 queries, 3,600 source groups, 5,000 paired bootstrap samples. Fusion did not improve EG2 in this setting.
- The original Turkish photo run used an earlier manifest schema. Its exact manifest is preserved alongside the report, and every record/media hash was verified identical. The final-schema Turkish rerun and SigLIP2/fusion comparison are complete, including source-group confidence intervals.
- Multimodal BM25 now keeps only positive lexical matches (`positive_scores_only_v1`), including empty results for unmatched queries; it never fills the budget with zero-score candidates. This change invalidates old B-channel/fusion caches. Legacy baseline reports remain historical until explicitly refreshed; their original measurements are not silently relabelled.
- The initial PDF worker predated the explicit-empty-OCR fix, so its B/G/E/J coverage flags are historical limitations of that process. The separate BM25 baseline retains all 1,360 pages, including two empty OCR fields; corrected full combinations are queued.
- Two JavaScript functions exceed both text encoder limits. The separately named shared-segmentation protocol preserves every source byte and aggregates scores at the original function ID. It does not remove those functions.
- Caption language follows the declared dataset language (TR photos receive Turkish descriptions); source-only generation never sees the query. Explicit language changes create different cache identities.
- A five-query Gemma relevance cell smoke is queued before full LLM ranking. It is technical validation only, not a full-split accuracy result. Full LLM ranking, remaining model/parameter variants, answer generation and CIRR are not declared complete.

- The explicit long-audio Gemma reranking protocol preserves all 589,440 samples of the 36.84-second FLEURS source in 30 + 6.84-second windows. Its two actual CPU HTTP relevance calls passed. This is backend health, not a retrieval accuracy result; the full speech ranking job is separately labelled `gemma-audio-windows-full`.
- Code text rerankers and Gemma ranking have separate queue jobs, so local Gemma server startup cannot block Laya/BGE execution.
