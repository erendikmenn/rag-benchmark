# MSR-VTT 1K-A video description integrity audit

CPU audit passed for all 1,000 English source-only descriptions. Each saved caption matches its ordered source ID, actual video-byte digest, exact generator/cache identity and derived text/provenance. All records report normal stop, output budget 512 and context capacity 8,192. Prompt tokens range from 1,017 to 1,581; completion tokens from 29 to 446.

All 1,000 source and derived video files match. The 1,000 queries and 1,000 relevance mappings are unchanged. Query files are byte-identical; the dataset writer reserializes qrel rows, so qrel file hashes differ while parsed relevance mappings match exactly. Media asset inventories are byte-identical.

The declared visual sampling policy is 1 frame per second, at most 16 chronologically sampled frames, source-duration limit 60 seconds and image maximum side 1,120 pixels. Audio is excluded. Historical cache records do not contain per-call frame-retention diagnostics. This audit verifies saved provenance and execution integrity; it does not measure full temporal coverage or human caption correctness.

Recorded successful requests total 120,955 completion tokens and approximately 3,982.72 seconds. This is historical request usage, not fresh inference or total job wall time. The audit made zero model/server calls and adds zero main-matrix cells. Retrieval quality is assessed separately.

The companion JSON contains source/derived manifest, corpus, media inventory, query/qrel, generator, implementation and cache inventory hashes without raw captions, source IDs or absolute paths.
