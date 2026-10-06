# Multimodal execution status

Updated: 2026-10-07 02:15 Europe/Istanbul.

**Full-split evaluation has started.** The first published result is the [ViDoRe computer-science BM25 baseline](../reports/multimodal/vidore-cs-bm25/report.md): 215 queries, 1,360 candidate pages, Hit@5 0.9209, Recall@5 0.5353 and nDCG@10 0.6334. It measures retrieval, not generated-answer correctness.

The authorized [plan](multimodal-plan.tr.md) covers seven tracks, 321 retrieval subsets and 1,284 method families. The [registry](../configs/multimodal-variants.csv) describes planned scope, not completed experiments. Each actual run publishes completed, planned, unsupported and failed cells separately.

| Track | Verified prepared data | Execution / published evaluation |
|---|---|---|
| Photo | XM3600: 3,600 images, 7,233 TR and 7,200 EN queries | Full native EG2 TR run active; EN and SigLIP2 queued |
| Visual document | All eight ViDoRe V3 collections: 19,252 pages, 2,419 base queries | Computer-science BM25 full split published; dense/native/joint/ColQwen queued |
| Environmental audio | Clotho: 1,045 clips / 5,225 captions; separate additional-relevance protocol: 1,037 queries / 3,116 positives | EG2 and CLAP native preflight passed; full retrieval queued |
| Turkish speech | FLEURS: 743 recordings / 329 unique normalized transcript queries / 743 positives | Native preflight and real CPU Whisper transcription passed; full native/ASR views queued |
| Video | Official MSR-VTT video archive downloading | Native video and CLIP frame preflight passed; dataset evaluation pending |
| Code | Python 43,827 candidates / 14,918 queries; Ruby 4,360 / 1,261; Go 28,120 / 8,122 | Remaining language downloads/imports and full retrieval queued |
| Composed image query | CIRR official annotations inspected; official media needs the publisher's access process | Native joint-input preflight passed; full official gallery unavailable |

## Dataset and model audit notes

- The pinned current ViDoRe energy snapshot has 2,225 pages, four fewer than the paper's 2,229. The verified eight-collection total is consequently 19,252. No replacement records are invented.
- Clotho's pinned additional-relevance file contains 1,037 unique queries. It is evaluated as a separate protocol; it does not silently replace the original 5,225-caption evaluation.
- CodeSearchNet source comments and Python standalone docstrings are removed with syntax parsing, independently of query text. Runtime string literals are preserved. Full expected gallery/query counts and source hashes are required.
- Specialist query limits differ: SigLIP2 64, CLIP 77 and CLAP 512 tokens. Strict mode errors on overflow; explicitly selected truncation is separately identified and counted. XM3600 TR has 56 queries longer than SigLIP2's limit; English has none. No input loss is hidden.
- ColQwen uses token-level MaxSim, not an averaged single vector. Its real-weight CPU preflight passed; the MPS preflight remains queued. The initial dense-interface diagnostic is superseded by the dedicated token-retrieval adapter.
- Local EG2 image/audio/video/joint and specialist health checks are diagnostics, not benchmark scores. A 31-second EG2 audio diagnostic processed the entire waveform without clipping.
- ASR transcripts and generated descriptions use source media only. Gold query captions, transcripts and answers never become candidate descriptions. Human/reference transcripts remain evaluation data.

## Execution and continuation

Independent dataset downloads, tests and CPU baselines run concurrently. One heavy MPS job owns the GPU slot; a second model server or GPU worker must not start until it is released. Current retrieval process and logs are local under `work/overnight/`. Inspect them before resuming; completed inference is cached and must not be charged as fresh model latency.

The next steps are full native/specialist retrieval on ready datasets, source-only ASR/caption views, compatible B/G/E/N/S/J fusions, and the three optional rerankers. Parameter sweeps and answer-generation evaluation remain pending. The complete family/budget grid contains substantially more work than a single retrieval run; no unmeasured completion time or accuracy is promised.

Raw datasets, prompts, media, keys and local caches stay out of Git. Publish only aggregate reports and tested source changes in atomic commits. Refresh English/Turkish READMEs after meaningful completed milestones. Overnight continuation checks are scheduled through 09:00 Europe/Istanbul on 2026-10-07.
