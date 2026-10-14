# Colab throughput measurements — 14 October 2026

These are actual bounded CUDA samples, with source and measurement hashes in the [aggregate JSON](colab-throughput-measurements.json). They add no accuracy result or main-matrix cell. The GPU for the timing samples was NVIDIA A100-SXM4-80GB; source commit was `98a981ed292b49f519ec2e6555cc1ab0cd929976`.

## Observed rates

| Work | Actual sample | Warm measured time | Interpretation |
|---|---:|---:|---|
| BGE reranker, Physics | 480 pair attempts | 7.114 s | 67.47 pairs/s |
| BGE reranker, JS / Ruby / Python | 240 attempts each | 5.937 / 3.470 / 3.554 s | 40.43 / 69.17 / 67.52 source pairs/s |
| Laya, Physics | 480 pair attempts | 5.480 s | 87.59 pairs/s |
| Laya, JS / Ruby / Python | 240 attempts each | 17.768 / 5.427 / 11.945 s | 13.51 / 44.23 / 20.09 source pairs/s |
| Gemma relevance, Physics | 12 valid pairs | 10.234 s | 1.17 pairs/s |
| Gemma relevance, Ruby / Python | 4 valid pairs each | 1.512 / 1.413 s | 2.65 / 2.83 pairs/s; very small samples |
| QA, one server slot | 20 attempts, 15 valid, 5 rejected | 135.241 s including first cold request | Not a successful QA completion rate |

Source-pair and underlying model-call counts differ for JavaScript's `source_chunks_max` policy. The JavaScript Gemma case stopped after three valid pairs and a strict failure; it supplies no full-case throughput. Ruby/Python whole-source rates cannot be applied to all languages or long chunks. Repeated top20/top100 sample candidates were fresh timing calls, not score-cache hits. Model loading, transport preflight and per-shape warmup are recorded separately.

The generation server used the pinned Gemma Q4 model and projector on actual CUDA. An initial wrapper incorrectly replaced the inherited NVIDIA library path and caused CPU fallback before QA requests; the actual reported sample used the corrected environment and verified CUDA layers, CUDA projector and owned GPU process. CPU fallback attempts are not included in the CUDA rates.

## Measured BGE batching candidate

The same six queries and 600 actual candidate pairs were scored under three explicit batch identities:

| Batch | Warm time | Throughput | Speedup versus batch1 |
|---:|---:|---:|---:|
| 1 | 8.887 s | 67.52 pairs/s | 1.00× |
| 8 | 2.188 s | 274.24 pairs/s | 4.06× |
| 32 | 2.159 s | 277.91 pairs/s | 4.12× |

This is a measured speedup for this sample, not the whole project. Maximum absolute score drift was 0.078125. Full rankings differed for all six queries at batch8 and batch32. Each condition has a separate adapter identity; existing batch1 results/cache are not overwritten or relabeled. No quality equivalence or significance is claimed.

## Concrete workload arithmetic and overnight target

The current QA plan has 540,842 request-eligible task evaluations but **122,110 exact request keys** after matching evidence order, media and fixed runtime/config. The 418,732 duplicate opportunities are not yet measured response-cache hits. Warm single-slot latencies weighted by five request shapes yield **171.4 A100 hours of attempted generation** for these keys. This includes the cost of failed attempts, excludes cold setup and later repair/retry, and is **not an ETA for successful completion**. A valid-only extrapolation would select out the expensive failures and is not used. Representative request-shape or census-mix parallel throughput is required before replacing this serial shape-weighted planning scenario.

The known full-grid candidate union is **18,239,347 source pairs per reranker**, including **13,155,926 code pairs**. K20/50/100, retrieval subsets and both budgets already share pair scores within one unified run. These are full-grid opportunities, not certified fresh eligible remaining calls: existing score keys, unsupported sources, CUDA versus MPS identities and missing retrieval channels still matter. The 5,554,788 conditional future QA task evaluations have no exact-key census yet and are not included in the 122,110 keys.

**Target: 15 October 08:00 Europe/Istanbul.** The immediate parallel workload is the three document retrievals and their complete BGE/Laya grids, with QA/Gemma throughput and strict-runtime gates on A100. Each full document rerank job targets 882 cells: 126 reused none cells plus 378 BGE and 378 Laya cells. It reads sealed completed retrieval rankings and forbids fresh retrieval fallback. Source/hash/metric audit and durable checkpoint transfer are required before publication. This operational target does not redefine the broader research scope or certify that it fits overnight.

Physics retrieval finished in **20m 7s** on A100; Pharmaceuticals finished in **31m 20s** on L4. Both wall times include checkpoint work and are for different datasets. Independent full-result acceptance remains pending. Pharmaceuticals subsequently lost its runtime before reranking, with no full durable backup yet confirmed; the cause is unknown. Industrial continues independently. Colab transport interruption and replacement preparation must be included in any wall-clock plan.

A whole-research finish time is not established by these data. Remaining full QA, multimodal Gemma scoring, representation/parameter extensions, external CIRR media access and human semantic calibration are separate work. The previous unmeasured eleven-day estimate remains withdrawn. More hardware does not resolve missing access or human labels.

For a concrete first-wave planning scenario, six top100 channels give at most 181,200 / 218,400 / 169,800 source-pair opportunities per reranker for Physics / Pharmaceuticals / Industrial. Applying the measured Physics A100 means for batch1 BGE and Laya yields **1.321 / 1.592 / 1.238 A100 GPU-hours**, or **4.150 GPU-hours combined**. This assumes the Physics pair-cost distribution for all three datasets; it is not a runtime upper bound, measured L4 rate, remaining-work count, or wall-clock ETA. Model preparation, CPU bookkeeping/fusion, checkpoint transfer and independent audit are excluded.

## Actual eight-slot generation measurements

The owned A100 server exposed eight actual 8,192-token slots (65,536 total context). Production prompts, QA schema, seed, temperature and output limits were retained; this is a separate runtime candidate, with no prompt/response-cache hits.

- **QA:** 18 steady requests in **31.791 seconds**, **0.566 attempts/s**; 14 passed and four failed with recorded `finish_reason=length` at the unchanged 384-output-token cap. Including the two excluded warmups: 20 attempts, 15 valid, five failed. Independent matching verified the same 18 request hashes, evidence bytes/order and input shapes as the serial sample. Replacing its first cold call with the actually measured warm repeat of that same request gives **73.534 seconds** serial and a **2.313×** mixed-sample candidate speedup. Including the cold call would misleadingly inflate that factor to 4.084×.
- **Gemma code relevance:** 32 Ruby and 32 Python source candidates, one actual query each, completed in **21.800 seconds**, **2.936 pairs/s**, **64/64 valid**. Both warmups also passed. Output cap remains 32; 15,074 prompt tokens and 1,024 generated tokens were observed. This candidate lacks a matching serial measurement of these exact 64 pairs, so no serial speedup is claimed. No JavaScript segmentation or other-language rate is inferred.

Applying the observed QA sample mix to all 122,110 current exact keys gives **59.908 A100 attempt-hours**. It is a small mixed-sample planning scenario, not evidence-shape-weighted and not a successful-completion ETA. The 171.4-hour serial shape-weighted scenario remains a separate calculation: dividing it by eight slots or by the 2.313× mixed-sample factor is not justified. Overlapping per-request durations cannot supply per-shape saturated throughput.

As a scale check only, applying 2.936 pairs/s to all 13,155,926 prospective code pairs yields roughly **1,245 A100 hours**. This is **not a full-code finish forecast**: it samples only two queries and does not reconcile eligible fresh pairs, cached scores, JavaScript chunks or other-language costs. It does demonstrate why counting reporting cells or assuming eight slots yield eightfold throughput is inadequate. The observations do not support a complete whole-research overnight promise. QA strict failures and external/human prerequisites remain open.

The full Physics checkpoint (1,038,274,560 bytes) was recovered to the Mac with its SHA verified. Independent CPU audit then passed all **126 conditions, 38,052 rankings and 646,884 metric values**. Main-matrix acceptance/publication is a separate step; this timing update changes no accuracy cells.
