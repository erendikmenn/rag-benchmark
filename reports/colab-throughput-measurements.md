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

The current QA plan has 540,842 request-eligible task evaluations but **122,110 exact request keys** after matching evidence order, media and fixed runtime/config. The 418,732 duplicate opportunities are not yet measured response-cache hits. Warm single-slot latencies weighted by five request shapes yield **171.4 A100 hours of attempted generation** for these keys. This includes the cost of failed attempts, excludes cold setup and later repair/retry, and is **not an ETA for successful completion**. A valid-only extrapolation would select out the expensive failures and is not used. Eight-slot throughput must be measured before replacing this serial planning scenario.

The known full-grid candidate union is **18,239,347 source pairs per reranker**, including **13,155,926 code pairs**. K20/50/100, retrieval subsets and both budgets already share pair scores within one unified run. These are full-grid opportunities, not certified fresh eligible remaining calls: existing score keys, unsupported sources, CUDA versus MPS identities and missing retrieval channels still matter. The 5,554,788 conditional future QA task evaluations have no exact-key census yet and are not included in the 122,110 keys.

**Target: 15 October 08:00 Europe/Istanbul.** The immediate parallel workload is the three document retrievals and their complete BGE/Laya grids, with QA/Gemma throughput and strict-runtime gates on A100. Each full document rerank job targets 882 cells: 126 reused none cells plus 378 BGE and 378 Laya cells. It reads sealed completed retrieval rankings and forbids fresh retrieval fallback. Source/hash/metric audit and durable checkpoint transfer are required before publication. This operational target does not redefine the broader research scope or certify that it fits overnight.

Physics retrieval finished in **20m 7s** on A100; Pharmaceuticals finished in **31m 20s** on L4. Both wall times include checkpoint work and are for different datasets. Independent full-result acceptance remains pending. Pharmaceuticals subsequently lost its runtime before reranking, with no full durable backup yet confirmed; the cause is unknown. Industrial continues independently. Colab transport interruption and replacement preparation must be included in any wall-clock plan.

A whole-research finish time is not established by these data. Remaining full QA, multimodal Gemma scoring, representation/parameter extensions, external CIRR media access and human semantic calibration are separate work. The previous unmeasured eleven-day estimate remains withdrawn. More hardware does not resolve missing access or human labels.

For a concrete first-wave planning scenario, six top100 channels give at most 181,200 / 218,400 / 169,800 source-pair opportunities per reranker for Physics / Pharmaceuticals / Industrial. Applying the measured Physics A100 means for batch1 BGE and Laya yields **1.321 / 1.592 / 1.238 A100 GPU-hours**, or **4.150 GPU-hours combined**. This assumes the Physics pair-cost distribution for all three datasets; it is not a runtime upper bound, measured L4 rate, remaining-work count, or wall-clock ETA. Model preparation, CPU bookkeeping/fusion, checkpoint transfer and independent audit are excluded.
