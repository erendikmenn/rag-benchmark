# Multimodal execution status

Updated: 2026-10-07 08:00 Europe/Istanbul.

**Full-split evaluation has started.** The first published result is the [ViDoRe computer-science BM25 baseline](../reports/multimodal/vidore-v3-computer_science-en-bm25/report.md): 215 queries, 1,360 candidate pages, Hit@5 0.9209, Recall@5 0.5353 and nDCG@10 0.6334. It measures retrieval, not generated-answer correctness.

The authorized [plan](multimodal-plan.tr.md) covers seven tracks, 321 retrieval subsets and 1,284 method families. The [registry](../configs/multimodal-variants.csv) describes planned scope, not completed experiments. Each actual run publishes completed, planned, unsupported and failed cells separately.

| Track | Verified prepared data | Execution / published evaluation |
|---|---|---|
| Photo | XM3600: 3,600 images, 7,233 TR and 7,200 EN queries | TR EG2 / SigLIP2 / fusion Hit@5: 84.64% / 59.88% / 76.28%; EN: 83.75% / 76.89% / 82.92%; both full comparisons complete |
| Visual document | All eight ViDoRe V3 collections: 19,252 pages, 2,419 base queries | Positive-only BM25 complete on all eight / 2,419 queries; CS retrieval + Laya + BGE complete: 126 + 378 + 378 = 882 cells; full Gemma and other dense document collections pending |
| Environmental audio | Clotho: 1,045 clips / 5,225 captions; separate additional-relevance protocol: 1,037 queries / 3,116 positives | Original full protocol complete: EG2 / CLAP / fusion Hit@5 11.75% / 37.42% / 26.47%; additional-relevance full run also complete (separate query/label protocol) |
| Turkish speech | FLEURS: 743 recordings / 329 unique normalized transcript queries / 743 positives | Native full retrieval complete; source-only Whisper ASR complete on all 743 recordings (WER 7.29%); all 31 B/G/E/N/J subsets × two budgets complete (62 cells); speech rerankers pending |
| Video | 1,000 videos / 1,000 prescribed 1K-A queries; 884 clips have audio | Full visual-frame comparison complete: EG2 / CLIP / fusion Hit@5 75.00% / 53.80% / 67.10%; audio excluded from this condition |
| Code | All six languages: 183,295 functions / 52,561 queries | Positive-only full BM25 complete for all six languages / 52,561 queries; Python and Ruby full B/G/E comparisons complete (16,179 queries, seven methods × two budgets); Go dense retrieval running; Java/PHP/JavaScript queued |
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

The next steps are full native/specialist retrieval on ready datasets, source-only ASR/caption views, compatible B/G/E/N/S/J fusions, and the three optional rerankers. The first parameter sweep is complete on seven native dataset views: 128/256/512/768 dimensions with validated cached vectors and full-corpus search (104 dimension/method/budget cells). Remaining representation/parameter sweeps and answer-generation evaluation are pending. The complete family/budget grid contains substantially more work than a single retrieval run; no unmeasured completion time or accuracy is promised.

Raw datasets, prompts, media, keys and local caches stay out of Git. Publish only aggregate reports and tested source changes in atomic commits. Refresh English/Turkish READMEs after meaningful completed milestones. Overnight continuation checks are scheduled through 09:00 Europe/Istanbul on 2026-10-07.

## Completed audits and current limitations

- [Dataset audit](../reports/multimodal-dataset-audit.json): 20 complete protocol views. Totals across views overlap; they are not independent unique-source counts. All raw source files remain frozen.
- [English photo comparison](../reports/multimodal/xm3600-en-native-specialist/paired-comparisons.md): 7,200 queries, 3,600 source groups, 5,000 paired bootstrap samples. Fusion did not improve EG2 in this setting.
- The original Turkish photo run used an earlier manifest schema. Its exact manifest is preserved alongside the report, and every record/media hash was verified identical. The final-schema Turkish rerun and SigLIP2/fusion comparison are complete, including source-group confidence intervals.
- Multimodal BM25 now keeps only positive lexical matches (`positive_scores_only_v1`), including empty results for unmatched queries; it never fills the budget with zero-score candidates. This change invalidates old B-channel/fusion caches. Legacy baseline reports remain historical until explicitly refreshed; their original measurements are not silently relabelled.
- The initial PDF worker predated the explicit-empty-OCR fix, so its B/G/E/J coverage flags are historical limitations of that process. The separate BM25 baseline retains all 1,360 pages, including two empty OCR fields; corrected CS full retrieval combinations are now complete.
- Two JavaScript functions exceed both text encoder limits. The separately named shared-segmentation protocol preserves every source byte and aggregates scores at the original function ID. It does not remove those functions.
- Caption language follows the declared dataset language (TR photos receive Turkish descriptions); source-only generation never sees the query. Explicit language changes create different cache identities.
- A five-query Gemma relevance check completed: 63 retrieval baselines + 63 Gemma K=20 cells, 278 unique fresh score pairs, no failed cells. It is technical validation only, not a full-split accuracy result. Full LLM ranking, remaining model/parameter variants, answer generation and CIRR are not declared complete.

- The explicit long-audio Gemma reranking protocol preserves all 589,440 samples of the 36.84-second FLEURS source in 30 + 6.84-second windows. Its two actual CPU HTTP relevance calls passed. This is backend health, not a retrieval accuracy result; the full speech ranking job is separately labelled `gemma-audio-windows-full`.
- Code text rerankers and Gemma ranking have separate queue jobs, so local Gemma server startup cannot block Laya/BGE execution.

- The reranker-only code protocol keeps all 117,266,851 source bytes in 184,231 chunks across 183,295 functions. CPU tokenization of every chunk with each language's longest query found no Laya/BGE context overflow. Gemma retains its own strict runtime context validation.
- Clotho additional relevance has one connected source group containing 86.5% of queries. Its paired differences remain descriptive; confidence intervals are withheld under the explicit conservative majority-group reporting rule.
- Latest implementation validation: 342 tests passed with the complete model dependencies; the minimal dependency environment passed 307 tests with 22 optional-dependency skips. These are software checks, separate from benchmark accuracy.

- [Cached dimension sweep](../reports/multimodal-dimensions/README.md): unchanged 768-dimensional source rankings and stored per-query metrics reproduced on TR/EN photos, both Clotho protocols, Turkish speech, video and CS document images. Prefix128/256/512 vectors were searched over every candidate; S was held fixed for validated N+S fusions. New encoder inference calls: zero. Raw vector storage ratios are not process-memory or speed claims.

- [Video comparison](../reports/multimodal/msrvtt-1k-a-native-specialist/paired-comparisons.md): full 1K-A gallery and prescribed 1,000 queries complete. Both adapters use 1 fps / at most 16 visual frames; no soundtrack is consumed. EG2 beats the CLIP frame-pooling baseline by 21.20 percentage points at Hit@5 (95% paired interval: 18.20–24.20).

- [Python/Ruby dense retrieval](../reports/multimodal/code-dense-summary.md): all 28 method/budget cells complete. EG2 / BGE-M3 / BM25 Hit@5 is 84.46% / 62.18% / 38.93% for Python and 86.28% / 68.20% / 45.60% for Ruby. Equal-weight RRF did not improve standalone EG2; code-generation correctness is not measured.

- [CS document comparison](../reports/multimodal/vidore-v3-computer_science-en-all-retrieval/report.md): 126 completed retrieval cells. Highest observed primary Hit@5 is BM25 + native EG2 + ColQwen (98.14%); standalone ColQwen has the highest Recall@5 (65.34%) and nDCG@10 (0.7585). All 215 queries connect into one source-document group, so paired confidence intervals are withheld. These are descriptive test-set comparisons.

- [Complete Laya document grid](../reports/multimodal/laya-document-summary.md): 63 retrieval subsets × K=20/50/100 × two budgets = 378 full-query cells. At primary K=50, all 63 matched contrasts lose Hit@5, Recall@5 and nDCG@10 against the same retrieval without Laya. ColQwen Hit@5 changes from 97.21% to 54.42%; Recall@5 from 65.34% to 20.43%. Laya only sees source page text. All 215 queries are in one connected document group, so these are descriptive differences with no confidence interval or generated-answer accuracy claim. The publication is a retained completed-Laya milestone; BGE is now available in the full comparison below, while full Gemma remains pending. All 504 included cells' metrics were independently reproduced from stored rankings and qrels.

- The original Whisper ASR attempt stopped in Transformers' MPS deferred-stop cleanup (`EncoderDecoderCache.layers` missing). A CPU-only reproducer confirms the cache handling failure independently of source audio. Per-call static KV caching now selects synchronous stopping, retaining greedy decoding, EOS checks and native long-form processing. Regression tests cover the actual cache-selection branch and short/long greedy output equivalence. The separately identified `whisper-source-only-v2` protocol does not reuse old dynamic-cache transcripts.
- [Actual Whisper CPU health checks](../reports/multimodal-whisper-cpu-preflight.json) and [MPS checks](../reports/multimodal-whisper-mps-preflight.json) passed on 16.92- and 36.84-second sources: all 270,720 and 589,440 samples were retained, with one and two decoding segments respectively. These checks verify execution and source coverage. Full ASR quality is now measured separately in the speech report below.

- [Complete document reranker comparison](../reports/multimodal/document-reranker-summary.md): 882 cells, with 189,630 stored rankings and 3,223,710 metric values independently reproduced. At primary K=50, BGE improves nDCG@10 in 60/63 same-retrieval comparisons and exceeds Laya in all 63, but standalone ColQwen still has higher nDCG@10 than every BGE K=50 method. The earlier Laya milestone overlaps this full report and is not additional independent evidence. Confidence intervals remain withheld for the one connected source group.

- Recovery at 07:35 Europe/Istanbul: after CS reranking completed, new GPU jobs failed to reach macOS `MTLCompilerService`; the previous queue ended at 06:32 with remaining jobs failed or deferred. Those states do not count as completed comparisons. A fresh MPS arithmetic check and both actual Whisper checks passed; the managed queue was restarted. A new infrastructure-error guard stops the queue on fresh Metal-service failures and preserves completed work instead of cascading through other jobs. Remaining code retrieval now precedes expensive speech reranking and generated-description jobs; the 09:00 launch cutoff is unchanged. Failed export snapshots are retained locally, with no raw logs published.

- [Full Turkish speech ASR and retrieval](../reports/multimodal/speech-asr-summary.md): all 743 recordings transcribed, no empty transcripts; corpus WER 965 / 13,233 = 7.29% (716 substitutions, 126 deletions, 123 insertions). All 62 retrieval cells cover the same 329 transcript queries and 743 candidates. B/G/E searching Whisper text and J using audio + Whisper text each achieve Hit@1, Recall@5 and nDCG@10 of 1.0. This transcript-match task is saturated for those text methods; it is not a paraphrase or question-answering evaluation. The original native task is repeated in this derived view and is not independent additional evidence. ASR completed at 07:49 and retrieval at 07:53 Europe/Istanbul; Go dense retrieval followed. Full speech reranking remains pending.
