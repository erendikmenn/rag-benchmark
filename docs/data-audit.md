# RAGTurk release audit

Audited on 2026-10-06. This benchmark uses the available public snapshot, **not the full dataset reported in the paper**. All models receive the same 37,511-chunk corpus. There are 2,000 development questions and 12,530 held-out test questions after structural validation.

## Provenance and licensing

- Dataset: [metunlp/ragturk on Hugging Face](https://huggingface.co/datasets/metunlp/ragturk).
- Immutable revision: `86d35335d01b30869ab8eb392e380921fa1e2185`.
- Authors' [repository](https://github.com/metunlp/ragturk) and [paper](https://aclanthology.org/2026.sigturk-1.15.pdf).
- Dataset license: **CC BY-NC-SA 4.0**, as declared upstream. This is separate from this repository's MIT software license. The noncommercial condition means this is not unrestricted open data. Underlying web and Wikipedia material retains its applicable rights and attribution requirements.
- Raw and normalized records remain under ignored `data/`; this repository publishes import code, synthetic test fixtures, and aggregate audit results, not the source dataset or its text.

## Reported versus available data

| Unit | Paper | Imported public snapshot |
| --- | ---: | ---: |
| Articles | 11,196 | 8,104 usable; 8,105 discovered |
| Original chunks | 58,289 | 37,511 |
| Question–answer records | 20,459 | 14,530 valid; 14,534 parseable/detected rows |
| Development questions | — | 2,000 |
| Test questions | — | 12,530 |

The Git snapshot contains 24,062 files. The importer selects 18,735 data files, totaling 74,509,793 bytes (74.51 MB decimal): JSON articles and the three Markdown files needed to recover web articles, chunks, and questions. Counts refer to records, not files, vectors, retrieval pairs, or generated responses.

The published directory names are misleading: `formal_5k/dataset/json/` contains 2,790 Wikipedia articles, while `formal_5k/dataset/raw/articles/` contains 5,315 web articles. `informal_6k` does not contain the missing usable dataset records in this revision. We do not infer source type from those directory names.

| Source | Usable articles | Chunks | Questions |
| --- | ---: | ---: | ---: |
| Wikipedia | 2,790 | 24,103 | 5,579 |
| Web | 5,314 | 13,408 | 8,951 |
| Total | 8,104 | 37,511 | 14,530 |

This revision cannot support the earlier proposed 18,000-question final test set. The reproducible final test set has 12,530 questions; ten end-to-end variants would produce 125,300 test answers.

## Import and rejection policy

JSON chunks are preserved as published. Web chunks are recovered by slicing the published `article.md` body using the original character ranges in `chunks_index.md`. The importer verifies body length, range bounds, and declared chunk count. It does not rechunk articles, generate labels, scrape missing pages, or invent replacement content. Unescaped pipes in chunk headings/previews are handled by locating the unambiguous chunk ID and character range.

Every chunk ID is qualified by source and article, such as `web:<article_id>#<chunk_id>`. Questions contain the original question, reference answers, original gold chunk references mapped to global IDs, article, category, and source. All gold IDs must resolve to nonempty chunks. Answer and relevance fields are evaluation labels; they must not enter the retrieval query or generation prompt.

All exclusions and partial records are recorded in local `rejections.jsonl`:

| Issue | Count | Treatment |
| --- | ---: | --- |
| Missing chunk-index companion file | 1 article | Excluded because original chunks cannot be recovered |
| Missing question companion file | 8 articles | Chunks retained in corpus; no questions fabricated |
| Malformed question table with ambiguous cells | 4 question rows | Excluded; no guessing at question/answer boundaries |
| Declared question count differs from actual rows | 1,665 articles | Actual valid rows counted; metadata discrepancy reported |

No parsed chunk or otherwise valid question was discarded to reach a target dataset size. 8,092 articles have at least one valid question. The manifest records the complete counts, rejection reasons, selected raw-file inventory digest, prepared-file digests, revision, and split parameters.

## Development/test separation

Seed 42 assigns entire connected groups to development or test. Articles are joined when they share normalized exact question text or normalized exact **labeled evidence** text. Normalization uses Unicode NFC and whitespace folding. Articles, repeated questions, and shared labeled evidence therefore remain together, including transitive matches. Unreferenced repeated boilerplate does not merge otherwise unrelated articles.

Both splits search the same complete corpus. Corpus availability is part of the retrieval task; test question/answer labels must remain outside tuning. This is a custom benchmark split of released synthetic data, not an official hidden test set or a claim that model pretraining never included it.

| Split | Questions | Articles with questions | Factual | Interpretation | Web | Wikipedia |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Development | 2,000 | 1,122 | 1,170 | 830 | 1,255 | 745 |
| Test | 12,530 | 6,970 | 7,284 | 5,246 | 7,696 | 4,834 |

The full prepared files were checked: zero cross-split article overlap, zero normalized exact question overlap, zero normalized exact gold-evidence overlap, unique corpus/question IDs, and all gold references present. There are 8,058 grouping components; the largest contains eight questions. This does **not** guarantee separation of semantic paraphrases, near-duplicate documents, or model-pretraining contamination.

## Language and label quality

Web article metadata declares `lang: en` for all 5,314 imported articles, despite visibly Turkish content. A seed-42 sample of 40 web questions, answers, and their gold passages, spanning both question categories and varied domains, was checked; all sampled content was Turkish. This is a limited content check, not exhaustive language certification. The importer records the conflicting metadata and does not discard Turkish text based on that field.

The release also declares `gemini-2.5-flash` for Wikipedia article generation and `gpt-5-mini` for web articles; these fields differ from the generation process described in the paper. We preserve and report what the actual snapshot says rather than assuming the paper's pipeline produced every released record.

Question–answer pairs and relevance labels are synthetic, not independently human-verified ground truth. Spot checks include context-dependent questions and questions targeting page boilerplate. Structural validity does not establish answer correctness, unique answerability across the corpus, or complete relevance judgments. Report factual/interpretation and web/Wikipedia results separately; a human-reviewed sample is needed before making strong quality claims. Reference-answer exact match alone is particularly limited for interpretation questions.

## Reproduction and integrity

From the installed repository environment:

```sh
python -m rag_benchmark.data --output-dir data/ragturk --dev-size 2000 --seed 42 --revision 86d35335d01b30869ab8eb392e380921fa1e2185
```

The first preparation needs network access and Git. Git transfers the many small upstream files in one pack instead of making thousands of individual HTTP resolve requests. An existing pinned raw cache is SHA-256 verified and reused offline. Normalization and benchmark loading require no network. `load_dataset` verifies the requested prepared question file and corpus against the saved digests and validates gold references.

Aggregate snapshot fingerprint: `71289eaa5e6b4d6f5c8285ab0336fdc3568c1828972645d9b4456337b8326d1c`.

| Prepared file | Bytes | SHA-256 |
| --- | ---: | --- |
| `corpus.jsonl` | 32,103,953 | `83d6ce53f63467787f7cc52f693bdb847cb3caa12939914634fce3f2a1eeae7b` |
| `questions.dev.jsonl` | 1,357,482 | `cde642d95acbdb25af01fa138bef671e9ff9d543e28d94fd4ca76447e0f3fbb4` |
| `questions.test.jsonl` | 8,526,139 | `11984075dfc86a6b4106eea282e868866c566a77dcd0fe132bd1c7170419c0a1` |
| `rejections.jsonl` | 2,072 | `2581db8ae7359d7d6860249128a20aae372aa0e3469eb9e5a0d998f5f2fbf98b` |

Selected raw-file inventory SHA-256: `980a4510fb6f73b476391b7cfe1b17d1e9817d751b730cc0be35c13e5861e871`. Raw file digests provide reproducibility and detect local changes; they are not a digital signature from the dataset authors.
