# Local model contracts

The core RAG pipeline never calls hosted inference. Downloading weights is an explicit preparation operation (`fetch_model`), using immutable Hugging Face commit IDs. RAG inference only loads those local snapshots. Missing models, unsupported architectures, invalid vectors, truncated passages and silent device fallbacks fail the run; there is no substitute model or fabricated ranking. The optional [Jev semantic evaluation](semantic-evaluation.md) is a separate hosted step over saved outputs and requires explicit selection; it does not change the local retrieval or generation pipeline.

## Models

| Role | Model | Revision | Representation |
|---|---|---|---|
| Dense retrieval | `BAAI/bge-m3` | `5617a9f61b028005a4858fdac845db406aefb181` | Native 1024 dimensions; dense only |
| Dense retrieval | `google/embeddinggemma-2` | `914f7f89142e33e77833254d9c9b90c3cef7303b` | Native 768 dimensions; text-only encoder loading |
| Evidence reranking | `convaiinnovations/laya-multilingual` | `1720e3e3357cfe1e281542e223f8273b0890ca34` | Per-passage probability of useful evidence |
| Answer generation | `google/gemma-4-26B-A4B-it-qat-q4_0-gguf` | See run configuration | Same local Gemma checkpoint in every variant |

Model metadata and availability were verified on 2026-10-06. Availability is not evidence of Turkish retrieval quality. BGE-M3 is MIT licensed; Google models and Laya are published under Apache-2.0; retain each downloaded model's license and attribution when redistributing.

## Retrieval

`RetrievalIndex(corpus, cache_dir, embeddings=None)` accepts dictionaries with `id`, `text`, and optional `title`. `build(method)` prepares indexes. `search(question, method, top_k=50, cache_query=True)` returns original metadata plus `score` and one-based `rank`.

Methods: `bm25`, `bge`, `embeddinggemma`, `bm25_bge`, `bm25_embeddinggemma`, `bge_embeddinggemma`, `bm25_bge_embeddinggemma`. Fusion uses RRF with k=60. It combines ranks, not incompatible raw scores. Main dense search is exact dot product over L2-normalized vectors, without approximate-index recall effects.

BM25 uses Unicode word tokens, Turkish dotted/dotless-I lowercase, no English stopword list, and no stemming. BGE receives raw questions and title plus body. EmbeddingGemma receives `task: search result | query: ...` and `title: TITLE | text: BODY` (title `none` if absent). The corpus text is otherwise identical. Inputs exceeding either model's token budget are rejected before inference instead of silently truncated.

The EmbeddingGemma formatting follows [Google's SentenceTransformers RAG example](https://ai.google.dev/gemma/docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers). The pinned checkpoint's native mean pooling includes prompt tokens; its 768-dimensional output and normalization modules are retained. After manually formatting each input, the adapter passes `prompt=""` to prevent SentenceTransformers from adding a second prefix. A separate question-answering prompt exists upstream, but Google's RAG example recommends the retrieval prompt used here.

Both embedders accept only `float32` or `bfloat16`. EmbeddingGemma 2 must not run in float16. Native dimensions remain unequal by design; BGE is not arbitrarily truncated. Stored/search vectors are normalized FP32. Cache identities include corpus content/order, model ID/revision, formatter version, precision, device and relevant package versions. A query cache can be bypassed with `cache_query=False` for resident latency measurement.

EmbeddingGemma 2's SentenceTransformers processor requires the official `chat_template.jinja` and image processor dependencies even for this text-only workload. Preparation includes the template; the models dependency group includes the required image extra. No image inputs or vision tower are used here. Dense corpus builds checkpoint every 256 documents and resume completed blocks after interruption.

## Laya

`LayaReranker(config).rerank(question, candidates)` returns sorted copies of candidates, preserving `retrieval_score` and adding `laya_score`. The SDK loads the explicit multilingual checkpoint, not the automatic language router. Every candidate is a separate `{query, passage}` state with a fixed yes/no evidence question. This is a disclosed benchmark prompt, not a claim that Laya is an official Jev model or a pretrained specialist reranker.

The default is reranking without filtering (`threshold=None`). Set a finite threshold in [0,1] only if the experiment protocol calls for filtering; it may remove every candidate. SDK execution uses eager FP32 and disables the library's shape-dependent MPS autocast. Requested device changes and runtime CPU fallback are rejected. `max_len=8192` overrides the checkpoint's shorter default; returned truncation flags fail the run.

An optional `backend='http'` speaks Laya's `/v1/systemone` protocol only over loopback and checks `/health` for the exact multilingual revision. The SDK path is preferred for complete device/precision control; external URLs and HTTP redirects are not allowed.

## Gemma generation

`LocalGenerator(config).generate(question, contexts)` uses llama.cpp's local `/v1/chat/completions` endpoint. The serving alias (`model`) is distinct from the artifact identity. Configure `model_id`, immutable `revision`, absolute `model_path`, and the GGUF `model_sha256`. The adapter checks the server's `/props` absolute path and `/v1/models` alias, then verifies the GGUF checksum once before its first answer. Startup verification is excluded from the reported answer latency.

`preflight()` performs this verification before a run manifest is created and returns the actual stable server properties, including reported build information, default generation settings/context size, chat template, and model metadata. Their fingerprint joins the generator cache identity. Volatile model creation timestamps and server state are excluded; an absent server build string is explicitly recorded as unreported.

The fixed Turkish system instruction requires passage-grounded answers with source IDs and abstention if evidence is insufficient. Context is serialized as JSON data. Temperature, seed, output budget and thinking configuration are fixed across variants. `last_usage` preserves token counts, elapsed time, finish reason and available server timings. A length-limited completion remains explicitly marked `finish_reason='length'`; downstream evaluation must report it.

Only loopback addresses are accepted. Proxy environment variables are ignored and redirects are disabled. An OpenAI-compatible local protocol does not use an OpenAI service or API key. The server must be started separately with the specified local GGUF and alias. Prompt cache is disabled by default for clean resident-query timing.

## Sources

- [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3)
- [EmbeddingGemma 2 model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2)
- [Laya SDK](https://github.com/NandhaKishorM/laya)
- [Laya HTTP protocol](https://github.com/NandhaKishorM/laya/blob/main/docs/http-api.md)
- [Official Gemma QAT checkpoint](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf)
