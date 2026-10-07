# FLEURS Turkish: full Whisper ASR and retrieval

Whisper transcribed **743/743 recordings** with no empty outputs. Independently recomputed corpus WER is **7.2924%**: **965 edits / 13,233 reference words** (716 substitutions, 126 deletions, 123 insertions). References were used only after source-only transcription. **1 − WER is not semantic or answer accuracy.**

Retrieval completed **62 cells: all 31 nonempty B/G/E/N/J combinations × two candidate budgets**, with the full **329 transcript-group queries and 743-recording gallery**. B/G/E use Whisper text; N uses audio only; J combines audio and Whisper text. No reranker is used.

The primary table below uses the per-channel budget of 100 and RRF k=60. These are exact-transcript-group queries: official normalized transcripts are the queries, and source-only ASR supplies candidate text. They are not independently written paraphrases or QA questions. High retrieval scores can coexist with ASR word errors. Hit@5 requires one matching recording; Recall@5 measures all recordings belonging to the query's transcript group.

| Channels | Hit@1 | Hit@5 | Recall@5 | nDCG@10 |
|---|---:|---:|---:|---:|
| B | 100.00% | 100.00% | 100.00% | 1.000000 |
| G | 100.00% | 100.00% | 100.00% | 1.000000 |
| E | 100.00% | 100.00% | 100.00% | 1.000000 |
| N | 99.70% | 100.00% | 99.54% | 0.997026 |
| J | 100.00% | 100.00% | 100.00% | 1.000000 |
| B+G | 100.00% | 100.00% | 100.00% | 1.000000 |
| B+E | 100.00% | 100.00% | 100.00% | 1.000000 |
| B+N | 100.00% | 100.00% | 100.00% | 1.000000 |
| B+J | 100.00% | 100.00% | 100.00% | 1.000000 |
| G+E | 100.00% | 100.00% | 100.00% | 1.000000 |
| G+N | 100.00% | 100.00% | 99.90% | 0.999795 |
| G+J | 100.00% | 100.00% | 100.00% | 1.000000 |
| E+N | 100.00% | 100.00% | 99.90% | 0.999509 |
| E+J | 100.00% | 100.00% | 100.00% | 1.000000 |
| N+J | 100.00% | 100.00% | 100.00% | 0.999470 |
| B+G+E | 100.00% | 100.00% | 100.00% | 1.000000 |
| B+G+N | 100.00% | 100.00% | 100.00% | 1.000000 |
| B+G+J | 100.00% | 100.00% | 100.00% | 1.000000 |
| B+E+N | 100.00% | 100.00% | 100.00% | 1.000000 |
| B+E+J | 100.00% | 100.00% | 100.00% | 1.000000 |
| B+N+J | 100.00% | 100.00% | 100.00% | 1.000000 |
| G+E+N | 100.00% | 100.00% | 100.00% | 1.000000 |
| G+E+J | 100.00% | 100.00% | 100.00% | 1.000000 |
| G+N+J | 100.00% | 100.00% | 100.00% | 1.000000 |
| E+N+J | 100.00% | 100.00% | 100.00% | 1.000000 |
| B+G+E+N | 100.00% | 100.00% | 100.00% | 1.000000 |
| B+G+E+J | 100.00% | 100.00% | 100.00% | 1.000000 |
| B+G+N+J | 100.00% | 100.00% | 100.00% | 1.000000 |
| B+E+N+J | 100.00% | 100.00% | 100.00% | 1.000000 |
| G+E+N+J | 100.00% | 100.00% | 100.00% | 1.000000 |
| B+G+E+N+J | 100.00% | 100.00% | 100.00% | 1.000000 |

B = BM25; G = BGE-M3 text; E = EmbeddingGemma2 text; N = EmbeddingGemma2 audio; J = EmbeddingGemma2 audio + Whisper transcript. The JSON also includes all 31 total-budget conditions and all 17 recorded metrics per cell.

The native N condition matches every metric of the earlier original-audio run in both budgets. Query records, qrels and media bytes are identical. This run nevertheless **freshly encoded all 743 audio candidates** (74.52 seconds of recorded encoding time) under the ASR-view identity. It is the same logical benchmark repeated, not an independent benchmark or a cached encoding.

Whisper checkpoint `openai/whisper-large-v3-turbo@41f01f3fe87f28c78e2fbf8b568835947dd65ed9` ran on MPS float32 with Turkish transcription, no query/reference prompt, greedy decoding, static KV caching and compilation disabled. All **150,083,520 audio samples** were verified against saved cache usage and source WAV headers. The longest recording is 36.84 seconds; 1 recording uses native long-form processing. No source waveform was truncated. The recorded 816.21 seconds cover synchronized model generation, not preprocessing or total pipeline wall time.

WER normalization applies NFKC, Turkish I/İ case mapping, case folding, punctuation-to-space and whitespace tokenization. WER is word-weighted over 743 recordings; retrieval averages 329 transcript-group queries. These denominators and tasks differ.

The published paired analysis verifies **329 transcript source groups**. J−N nDCG@10 is +0.002974 with a 95% paired source-group interval [0.000886, 0.005585] (8 query wins, 0 losses, 321 ties). This is an unadjusted exploratory comparison on a task with near-perfect results. Shared speakers or topics can create dependencies beyond transcript groups; the result does not establish general speech-semantic or paraphrase robustness.

The CPU audit independently recomputed WER, verified file/media hashes and all ASR cache identities, and recomputed **20,398 stored query rankings / 346,766 metric values** across the 62 retrieval cells. No new model inference was used for this summary. Raw audio, transcripts, reference text, query text and local paths are excluded.

[Retrieval report](fleurs-tr_tr-whisper-all-retrieval/report.md) · [Paired analysis](fleurs-tr_tr-whisper-all-retrieval/paired-comparisons.md) · [Exact aggregate summary and provenance hashes](speech-asr-summary.json)
