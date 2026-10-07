# Reverse media-to-reference retrieval

Three full-gallery EG2 protocols completed. Queries contain only the original media. The candidate gallery intentionally contains reference captions/transcripts (`gold_fields_used=true`); this is not source-only ASR or a generated-answer evaluation. These **three extra protocol jobs do not add to the 18,460-cell main matrix**.

| Query → target | Queries | Gallery texts | Hit@1 | Hit@5 | Recall@5 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| [TR image → reference caption](xm3600-tr-reverse-eg2.json) | 3,600 | 7,233 | 63.53% | 87.42% | 70.23% | 0.6894 |
| [EN image → reference caption](xm3600-en-reverse-eg2.json) | 3,600 | 7,200 | 66.75% | 89.14% | 73.85% | 0.7231 |
| [TR audio → reference transcript](fleurs-tr_tr-test-reverse-eg2.json) | 743 | 329 | 99.46% | 100.00% | 100.00% | 0.9977 |

Hit@5 asks whether any labelled target appears in the first five; Recall@5 measures the share of all labelled targets recovered. Images can have multiple captions. The speech gallery contains one normalized reference text per transcript group, so its Hit@5 and Recall@5 coincide.

The encoder is pinned EmbeddingGemma 2, revision `914f7f89142e33e77833254d9c9b90c3cef7303b`, 768 dimensions, MPS bfloat16. Existing media vectors were accepted only after exact source bytes, preprocessing, cache seals, IDs, forward rankings and metrics were validated. Media queries consume no text prefix; gallery text uses `title: none | text: `. Full-gallery normalized dot products rank the targets.

| Protocol | Newly encoded gallery texts | Gallery encoding seconds | Reused query-media rows | Newly encoded media rows |
|---|---:|---:|---:|---:|
| TR image → reference caption | 7,233 | 29.957 | 3,600 | 0 |
| EN image → reference caption | 7,200 | 28.533 | 3,600 | 0 |
| TR audio → reference transcript | 329 | 5.256 | 743 | 0 |

Encoding time covers fresh text rows only, not end-to-end request latency; zero fresh media rows does not imply that computing media embeddings costs zero. All newly encoded gallery text tokens were retained. Historical FLEURS media-cache audio sample diagnostics remain unavailable, so no new sample-retention measurement is inferred.

Independent CPU reconstruction verified **7,943 query rankings and 135,031 metric values**, including full-gallery cosine sorting and 17 independent metric formulas. It verifies cached arithmetic and provenance without re-running the encoders. TR/EN reuse the same 3,600 images; speech has 329 transcript groups with possible speaker/topic dependencies. No confidence intervals or combined cross-task accuracy are reported. Specialist reverse models and Whisper-as-query comparisons remain pending.

[Machine-readable summary](summary.json) · [Independent audit](independent-audit.json)
