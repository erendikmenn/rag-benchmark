# Jev pilot results and review notes

All ten local RAG variants were evaluated on the same 20 development questions. OpenRouter served `typesafe/jev-1.13-20260917` for all 319 unique requests, representing 400 logical judgments over 200 saved answers. No main-run requests failed or remain pending. The final test was not used.

**These are exploratory Jev judgments, not human-verified answer accuracy.** Every variant has a denominator of 20; repeated answers across ten variants do not create 200 independent questions. All variants use the same Gemma 4 answer model.

| Retrieval | Laya | Token F1 | Jev correct | Jev partial | Jev incorrect | Jev abstained |
|---|---|---:|---:|---:|---:|---:|
| BM25 | Off | 0.5100 | 11/20 (55%) | 7 | 0 | 2 |
| BM25 | On | 0.4085 | 9/20 (45%) | 5 | 1 | 5 |
| BGE-M3 | Off | 0.4661 | 12/20 (60%) | 7 | 0 | 1 |
| BGE-M3 | On | 0.4409 | 9/20 (45%) | 7 | 0 | 4 |
| EmbeddingGemma 2 | Off | 0.4592 | 14/20 (70%) | 4 | 0 | 2 |
| EmbeddingGemma 2 | On | 0.4112 | 8/20 (40%) | 6 | 1 | 5 |
| BM25 + BGE-M3 | Off | 0.5209 | 13/20 (65%) | 6 | 0 | 1 |
| BM25 + BGE-M3 | On | 0.4126 | 8/20 (40%) | 7 | 0 | 5 |
| BM25 + EmbeddingGemma 2 | Off | 0.5326 | 14/20 (70%) | 6 | 0 | 0 |
| BM25 + EmbeddingGemma 2 | On | 0.3986 | 7/20 (35%) | 7 | 1 | 5 |

For BM25 + EmbeddingGemma 2 without Laya, F1 remains **0.5326**, while Jev labels **14/20 correct (70%)** and **6/20 partial**. These metrics measure different things. Six partial labels do not mean six wholly false answers, and they receive no credit in the strict fully-correct rate. EmbeddingGemma 2 alone also receives 14/20 correct; the pilot does not establish that either path generally wins.

Across all ten variants there are 105 correct, 62 partial, 3 incorrect and 30 abstained labels. The pooled 105/200 (52.5%) combines different systems; it is not the deployment success probability of any one system. There are zero `unjudgeable` labels, which does not prove all source material is unambiguous.

Grounding labels total 176 supported, 4 partly supported and 20 abstained. All 105 fully-correct labels also receive supported labels, so the reported joint rate equals the correct rate. **Do not interpret this as a verified hallucination rate:** a concrete grounding error was found below. Correctness and grounding also disagree on some abstentions, another reason to calibrate the judge.

Confidence below 0.7 flags 58 correctness judgments and 47 grounding judgments for review; none is silently removed. The threshold is diagnostic, not a calibrated acceptance rule. For BM25 + EmbeddingGemma 2, three correctness labels and zero grounding labels fall below that threshold.

## Source checks and judge limitations

A separate assistant inspected the 20 BM25 + EmbeddingGemma 2 outputs before viewing Jev labels. This is a qualitative assistant audit, not human calibration; its provisional labels do not replace the frozen Jev results.

- **Different wording can lower F1:** `wikipedia:beiktanavrupakupalar_ee589386#q0000` has F1 0.25, but the answer correctly gives second place. Jev labels it correct/supported, consistent with the source check.
- **Missing requested information:** `wikipedia:2008grammydlleri_e2939adc#q0000` supplies one of two requested songs. Jev labels it partial/supported. The missing gold list was absent from the five retrieved contexts.
- **A real grounding miss:** `web:01964chipcomtr_0c5b701a#q0001` assigns Snapdragon 820 / Adreno 530 to Z5. The supplied Z5 passage `web:00558maxicepcom_dccef6ea#c0001` says 810 / 430. Jev labels correctness partial (confidence 0.42) but grounding supported (0.79), missing the explicit contradiction. Its supported labels therefore cannot certify absence of factual errors.
- **Interpretation needs calibration:** some long-form partial labels depend on which reference details are essential. The blind assistant audit disagreed with several such judgments, and flagged additional chronology/list-format ambiguities. No human agreement rate has been measured.

The separate [synthetic Turkish controls](../semantic-controls-tr/report.md) matched 10/12 intended labels, with an uncertainty-handling failure. The rubric was retained unchanged; these controls are not included in any RAG denominator.

## Usage and provenance

The main benchmark evaluation used **761,303 input tokens** and **20,523 output tokens**, with OpenRouter-reported cost **US$0.031974726** (about 3.2 US cents). This covers its 319 successful requests only; controls and connection diagnostics are separate. The local RAG generation was not rerun.

Source run fingerprint: `045252d34c401a7286e7c03cea4c7059b283219a02ad1a618d1aa8cb0830ba10`.
Evaluation bundle fingerprint: `c23741a72bec24bf52a2790a126582f3426f428971332965f481ccda4cd1a579`.
Rubric fingerprint: `f01c641ac31d93ecb2638a57a22b711c04b62e7f303a3185cb4935260f2df9fe`.
Judge code SHA-256: `360cd7fae93c05c27baeaedadb4af054e6aa69a4dcf2786c2f9c67f409be9eaf`.

[Machine-readable summary](summary.json) · [Per-answer labels and confidence](per-answer-judgments.jsonl) · [Generated count report](report.md) · [Evaluation protocol](../../docs/semantic-evaluation.md)
