# Clotho source-only descriptions

All **1,045 source clips** have nonempty English descriptions from the pinned local Gemma 4 E4B Q4 generator. Generation completed on 7 October 2026 at 14:10 Europe/Istanbul. The derived gallery retains the original **5,225 caption queries and 5,225 positive source relations**. Every source and derived media checksum, description-cache identity, text/provenance mapping and source manifest was checked. A second CPU reviewer independently confirmed the checks.

| Recorded check | Result |
|---|---:|
| Source clips and nonempty descriptions | 1,045 / 1,045 |
| Distinct description text hashes | 737 |
| Most frequent identical description | 23 clips |
| Normal generation stop | 1,045 / 1,045 |
| Prompt tokens per request | 454–829 |
| Output tokens per request | 2–108 |
| Reserved output / context limit | 192 / 8,192 tokens |
| Source duration range | 15.003–30.000 seconds |

Requests receive only the source audio through the generic English description prompt; query captions, relevance labels and reference answers are excluded. The cache records 27,216 output tokens and 1,124.95 seconds summed across successful requests. Those are historical request measurements, not fresh audit inference or total job wall time. The audit made zero model calls.

Repeated descriptions are a quality observation, not an identity failure. No human judgement of caption correctness has been collected. Historical generation caches record token/context usage but do not record sample counts for each input: sample-retention diagnostics remain unavailable. Source durations and the implementation's complete-audio/resample/overlength-rejection contract were checked separately; they do not create retrospective per-request measurements.

Full BM25/BGE/EG2/native/CLAP/joint retrieval on this description gallery is currently running; no new retrieval score or primary-matrix completion is claimed by this preparation milestone. [Aggregate provenance and audit](clotho-description-audit.json).
