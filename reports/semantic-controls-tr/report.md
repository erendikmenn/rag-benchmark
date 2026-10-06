# Turkish Jev behavior controls

Six assistant-authored synthetic cases were evaluated with separate correctness and grounding requests through OpenRouter. **10/12 final judgments matched the intended labels. These controls are not a human calibration set or a benchmark accuracy estimate.**

| Case | Intended correctness / grounding | Observed correctness / grounding |
|---|---|---|
| Equivalent longer answer | correct / supported | correct / supported |
| Wrong number | incorrect / contradicted | incorrect / contradicted |
| Reversed negation | incorrect / contradicted | incorrect / contradicted |
| Incomplete list | partial / supported | partial / supported |
| Correct fact absent from retrieved context | correct / unsupported | correct / unsupported |
| Reference evidence cannot establish the fact | unjudgeable / unsupported | incorrect / contradicted |

For the final case, the source says library opening hours have not been announced and the candidate claims 09:00. The evidence cannot determine the actual opening hour. Jev nevertheless chose `incorrect` with confidence 0.92 and `contradicted` with confidence 0.40. This shows why confidence is not empirical accuracy and why unsupported statements must be distinguished from disproven statements.

One initial response failed protocol validation. Its exact invalid field was not retained; a subsequent explicit retry passed. There were 13 control requests for 12 final judgments. The full states, expected labels, unchanged returned probabilities and known usage are in [results.json](results.json). Their rubric fingerprint is recorded there. One earlier connection smoke request and the invalid response have unrecorded usage and are excluded from the reported control cost.

The rubric was not tuned after seeing these failures. The RAG pilot uses this rubric as an exploratory judge, with human calibration still pending.
