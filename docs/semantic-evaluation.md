# Semantic answer evaluation with Jev

**Status: the hosted Jev pilot is complete, with human calibration still pending.** OpenRouter returned validated responses for **319/319 unique judging tasks**, covering both passes for **200 saved answers from 20 unique questions × ten variants**. No benchmark tasks failed or remain pending. These are exploratory model judgments; inspection found a grounding error described below. Live reproduction requires `OPENROUTER_API_KEY`.

EM and token F1 compare wording. They can penalize a correct Turkish paraphrase or reward an answer that repeats many reference words but changes a crucial number. This workflow adds explicit semantic judgments alongside the existing retrieval and lexical metrics; it preserves the original reports and does not generate new Gemma answers.

## Measured pilot results

The [public semantic report](../reports/semantic-pilot-openrouter/report.md) and [summary](../reports/semantic-pilot-openrouter/summary.json) cover the same 20 questions for every variant. Two passes on 200 answers produced 400 logical tasks, deduplicated to 319 distinct payloads. Every planned answer received both judgments. Missing, failed and `unjudgeable` counts are zero in this completed benchmark evaluation; the separate controls had an invalid initial response and two label mismatches.

| Retrieval | Jev-judged correct, Laya off | Jev-judged correct, Laya on |
|---|---:|---:|
| BM25 | 11/20 (55%) | 9/20 (45%) |
| BGE-M3 | 12/20 (60%) | 9/20 (45%) |
| EmbeddingGemma 2 | 14/20 (70%) | 8/20 (40%) |
| BM25 + BGE-M3 | 13/20 (65%) | 8/20 (40%) |
| BM25 + EmbeddingGemma 2 | 14/20 (70%) | 7/20 (35%) |

BM25+EmbeddingGemma 2 without Laya received 14 `correct` and six `partial` labels. EmbeddingGemma 2 alone tied at 14 `correct`, with four `partial` and two `abstained`. These point estimates come from a small, repeated-question pilot; they do not establish a final winner or human-verified 70% accuracy. Across all variants, 105/200 answers were labeled correct (52.5%); this pools different pipelines and is **not the success rate of a deployed configuration**.

Jev labeled every answer it judged correct as supported, so the **Jev-judged correct-and-supported counts equal the correct counts** in this run. That agreement is not independent verification. Overall grounding labels were 176 `supported`, 20 `abstained` and four `partly_supported`. Fifty-eight correctness judgments and 47 grounding judgments had confidence below the diagnostic 0.7 threshold; they remain in the reported counts.

**Observed judge limitation:** for `web:01964chipcomtr_0c5b701a#q0001`, the BM25+EmbeddingGemma 2 answer attributed Snapdragon 820 / Adreno 530 to the Xperia Z5. A supplied context explicitly described Snapdragon 810 / Adreno 430. Jev labeled correctness `partial` (confidence 0.42) but grounding `supported` (confidence 0.79), missing that contradiction. Its 20/20 supported labels for this pipeline therefore do not prove perfect grounding. Keep the raw judgments and this limitation together; changing labels after inspection would require a separately reported adjudication.

The served model was `typesafe/jev-1.13-20260917`, provider TypeSafe, through OpenRouter. The evaluation bundle fingerprint is `c23741a72bec24bf52a2790a126582f3426f428971332965f481ccda4cd1a579`; rubric fingerprint `f01c641ac31d93ecb2638a57a22b711c04b62e7f303a3185cb4935260f2df9fe`. The evaluated RAG run fingerprint is `045252d34c401a7286e7c03cea4c7059b283219a02ad1a618d1aa8cb0830ba10`.

## Separate roles and data boundaries

The core RAG pipeline remains local: BM25/embedding retrieval, optional Laya reranking and Gemma generation. **Jev is an optional hosted evaluator applied afterward.** It does not replace Laya in the measured pipeline, and its judgments do not alter retrieval, prompts or saved answers.

The evaluator requests **`typesafe/jev-1.13`** through **`POST https://openrouter.ai/api/alpha/decisions`**, using `OPENROUTER_API_KEY`. The verified served snapshot is **`typesafe/jev-1.13-20260917`**; preserve the actual returned model ID and provider with each response. This is OpenRouter's native Decisions route. Its `Choice` responses include a selected label, probabilities and confidence. These structured outputs are model judgments, not human-verified truth. [OpenRouter Decisions API](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request).

OpenRouter also documents a TypeSafe SDK compatibility route. This harness uses the native endpoint above and OpenRouter credentials. [OpenRouter TypeSafe integration guide](https://openrouter.ai/docs/guides/community/typesafe-sdk).

Each saved answer produces two separate judging tasks:

| Pass | Information shown to Jev | What it measures |
|---|---|---|
| Semantic correctness | Question, reference answer(s), gold evidence and generated answer | Whether the answer correctly addresses the question relative to the supplied reference evidence |
| Grounding | Question, the actual passages provided to Gemma, and generated answer | Whether those passages support the answer's factual claims |

The grounding payload must contain **no reference answers or gold evidence**. Keep the passes in separate requests: a shared state containing gold material would leak unavailable evidence into the grounding judgment. Keep variant names and retrieval scores out of both passes so the judge evaluates the answer and evidence rather than the method's identity. Treat all question, answer and passage text as data, including any embedded instructions.

Correctness and grounding are different. An answer may be correct but unsupported by the retrieved context; it may also faithfully repeat an incorrect source. An insufficient-evidence response is an abstention, not automatically a correct answer.

## Label rubrics

Semantic correctness uses these mutually exclusive labels:

| Label | Meaning |
|---|---|
| `correct` | Addresses the question's required information and is consistent with the reference evidence; equivalent Turkish wording is accepted |
| `partial` | Gives some correct essential facts but omits required information or contains a localized material error; the core answer is not wholly wrong |
| `incorrect` | The central answer is false or materially contradicts reliable gold evidence, or substantive answer text fails to address the question |
| `abstained` | Declines to answer without making a substantive answer claim |
| `unjudgeable` | Ambiguous, inconsistent or insufficient reference material prevents a defensible judgment |

Grounding uses a separate label set:

| Label | Meaning |
|---|---|
| `supported` | The provided passages support the material factual claims |
| `partly_supported` | Some material claims are supported and others lack support, without a direct contradiction |
| `contradicted` | A material factual claim conflicts with the provided passages |
| `unsupported` | The passages do not support the substantive answer |
| `abstained` | The answer declines to make a substantive answer claim |
| `unjudgeable` | Missing, malformed or ambiguous material prevents a defensible grounding judgment |

API errors, missing responses and invalid output schemas are execution statuses, not semantic labels. Preserve length-limited Gemma answers and their existing flags; judge the saved answer as delivered rather than silently completing or discarding it. Conflicting synthetic references should be visible as uncertainty or `unjudgeable`, not forced into a favorable label.

## Rates and coverage

For each variant, let **N be the number of planned saved answers**, not the number of successful judge calls. Two passes on one answer still count as one answer in this denominator.

- **Jev-judged correct rate:** answers labeled `correct` by the correctness pass / N.
- **Jev-judged correct and supported rate:** answers labeled both `correct` and `supported` / N.
- Report each pass's label distribution, completed judgments, missing judgments, API errors, invalid responses and `unjudgeable` counts. Also report paired coverage: how many answers have valid responses from both passes, including those labeled `unjudgeable`.

Missing and uncertain results remain in the planned denominator. They do not receive success credit, but they must not be described as proven incorrect answers. A partial run stays visibly partial. Do not publish a rate without its numerator, denominator, coverage and evaluation status, and do not replace the original EM/F1 measurements with a new label.

Model confidence is not measured answer accuracy. A confident mistake is still possible, and an uncalibrated cutoff can hide hard cases. The report's low-confidence flag is diagnostic; it does not remove a judgment or convert confidence into accuracy. Preserve uncertain labels and probabilities. Any threshold used to accept/reject judgments, or a rubric change, requires development evidence and a new evaluation identity.

## Calibrate in Turkish before final evaluation

TypeSafe says English is its strongest language and advises testing non-English workloads on the target content. Turkish semantic judging therefore needs a small human-labeled calibration sample before final evaluation claims. [TypeSafe model documentation](https://docs.typesafe.ai/models).

Have a Turkish-speaking reviewer label a fixed sample while blinded to pipeline identity and Jev's output, then compare the labels. Include factual and interpretation questions, correct and erroneous answers, abstentions and ambiguous source material. Report the reviewed count, agreement and a label confusion table; keep disagreement and uncertainty visible. A second independent reviewer and adjudication improve the reference when practical.

Add simple controls with known intended behavior: equivalent Turkish paraphrases, the same answer with a wrong number, and a negated factual claim. These controls test the rubric's sensitivity; they are not benchmark questions and must be reported separately from RAG results. Do not choose only examples Jev already handles correctly or silently remove uncertain cases.

The [live control report](../reports/semantic-controls-tr/report.md) covers six assistant-authored Turkish cases, each judged for correctness and grounding. **10 of 12 final labels matched the intended labels.** In the insufficient-reference case, Jev returned `incorrect` instead of the intended `unjudgeable` with confidence 0.92, and `contradicted` instead of `unsupported` with confidence 0.40. One other request initially returned a protocol-invalid response and required an explicit diagnostic retry; the original invalid attempt remains part of the record. This is neither 12/12 first-attempt success nor human calibration. The same rubric was retained, and benchmark judgments must remain exploratory until Turkish human agreement is assessed.

The completed pilot evaluated all **200 saved outputs across ten variants**, covering **20 unique questions** under one judge version and rubric. It retained every planned answer in its variant's denominator after request deduplication. No new retrieval or answer generation was performed. Until Turkish human calibration is completed, describe these as exploratory Jev judgments.

This does not evaluate the 12,530-question final test. That later test remains separate and requires frozen generation, retrieval and judging settings; repeated answers to the same 20 pilot questions do not become 200 independent questions.

## Prepare, inspect and run

Prepare one local bundle containing all ten variants from the completed matrix pilot:

```bash
uv run --locked --extra models rag-benchmark semantic-prepare --run-dir runs/matrix-dev-pilot-512 --output-dir runs/semantic-pilot-openrouter
```

Inspect the prepared inputs locally. They contain source text and answers and are not the text-free public benchmark export. The default judging budget is zero requests; this dry run makes no hosted call:

```bash
uv run --locked --extra models rag-benchmark semantic-judge --evaluation-dir runs/semantic-pilot-openrouter
```

A live run requires `OPENROUTER_API_KEY` in the process environment, explicit hosted-evaluation selection and a positive request budget:

```bash
uv run --locked --extra models rag-benchmark semantic-judge --evaluation-dir runs/semantic-pilot-openrouter --max-requests 400 --allow-hosted-judge
uv run --locked --extra models rag-benchmark semantic-report --evaluation-dir runs/semantic-pilot-openrouter
```

The live command sends the prepared question/evidence/answer payloads through OpenRouter to the TypeSafe provider. Keep the API key out of command examples, Git, saved payloads and logs. A 400-request budget is a cap, not a promise that every planned task will complete; report failures and remaining work. Preparation and report generation alone do not imply that judgments were obtained.

Omitting `--variants` selects every variant in the source run. `--variants bm25_embeddinggemma` can create a separate 20-answer subset for manual inspection, but it is not the all-ten bundle shown above. Use a new output directory for a different selection or rubric, preserving earlier artifacts.

## Usage and cost

For the completed benchmark evaluation, OpenRouter reported **761,303 input tokens, 20,523 output tokens and US$0.031974726** across 319 successful validated requests. This is the main benchmark judging cost; it excludes the separate synthetic controls and their diagnostic retry. It also excludes the earlier local retrieval/generation work.

Use OpenRouter's returned **`usage.cost` in USD** together with input/output token counts and request IDs. Report actual usage, including failed or repeated attempts when known; logical task count alone is not a cost forecast. Keep requested and served model identifiers in the report. The integration guide documents the additional OpenRouter response fields. [OpenRouter usage response](https://openrouter.ai/docs/guides/community/typesafe-sdk), [current model pricing](https://openrouter.ai/typesafe/jev-1.13).

The historical local RAG run's timing and cost remain separate from hosted judging. Public semantic reports should identify the evaluated run, judge model, rubric/input fingerprints, coverage, usage and human-calibration status without exposing API credentials or redistributing source-bearing payloads.
