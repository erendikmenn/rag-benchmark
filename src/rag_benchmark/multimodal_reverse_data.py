"""Prepare explicitly separate media-query/reference-text-gallery protocols on CPU.

These reference galleries are intentional task inputs, not source-only captions
or ASR. The current forward engine cannot run these manifests unchanged.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from .multimodal_data import _link_asset, normalize_transcript, read_jsonl, sha256, validate_dataset

PROTOCOL = 'reverse-reference-gallery-v1'
PREREQUISITES = [
    'Dedicated reverse track/registry and intended-reference-gallery provenance policy.',
    'Native image/audio query encoder paired with a compatible text-gallery encoder; current EG2 native document encoder requires media.',
    'Separate direct-media and source-only Whisper-query conditions; prepared reference transcripts are gallery inputs only.',
]


def reverse_records(corpus, queries, qrels, *, kind, references=None):
    source_by_id = {row['id']: row for row in corpus}
    query_by_id = {row['id']: row for row in queries}
    if len(source_by_id) != len(corpus) or len(query_by_id) != len(queries):
        raise ValueError('Duplicate parent source/query identifiers')
    by_source = defaultdict(set)
    sources_by_caption = defaultdict(set)
    for row in qrels:
        if row['query_id'] not in query_by_id or row['corpus_id'] not in source_by_id:
            raise ValueError('Parent relevance reference missing')
        if row['relevance'] > 0:
            by_source[row['corpus_id']].add(row['query_id'])
            sources_by_caption[row['query_id']].add(row['corpus_id'])
    if set(by_source) != set(source_by_id):
        raise ValueError('Every original media source needs a reverse positive')
    gallery, reversed_queries, reversed_qrels = [], [], []
    if kind == 'image_to_caption':
        for row in queries:
            positives = sorted(sources_by_caption[row['id']])
            if len(positives) != 1 or not isinstance(row.get('text'), str) or not row['text'].strip():
                raise ValueError('Each caption must reference exactly one source image')
            gallery.append({'id': row['id'], 'text': row['text'], 'metadata': {
                'group_id': positives[0], 'language': row.get('metadata', {}).get('language'),
                'gallery_role': 'intended_reference_caption', 'parent_query_id': row['id']}})
    elif kind == 'audio_to_reference_transcript':
        reference_by_source = {row['corpus_id']: row for row in references or []}
        if len(reference_by_source) != len(references or []) or set(reference_by_source) != set(source_by_id):
            raise ValueError('Reference transcripts must cover every audio source exactly once')
        gallery_texts = {}
        for identifier, row in source_by_id.items():
            reference = reference_by_source[identifier]
            text = normalize_transcript(reference.get('normalized_transcript', reference.get('transcript', '')))
            if not text or len(by_source[identifier]) != 1:
                raise ValueError('Audio source needs one nonempty normalized transcript group')
            group = next(iter(by_source[identifier]))
            if normalize_transcript(query_by_id[group]['text']) != text or row.get('metadata', {}).get('group_id') != group:
                raise ValueError('Normalized reference transcript differs from frozen parent group')
            if group in gallery_texts and gallery_texts[group] != text:
                raise ValueError('Conflicting text inside normalized transcript group')
            gallery_texts[group] = text
        if len(set(gallery_texts.values())) != len(gallery_texts):
            raise ValueError('Identical normalized transcript appears in multiple groups')
        for group, text in sorted(gallery_texts.items()):
            gallery.append({'id': group, 'text': text, 'metadata': {'group_id': group,
                'language': 'tr', 'gallery_role': 'intended_normalized_reference_transcript'}})
    else:
        raise ValueError('Unknown reverse protocol')
    modality = 'image' if kind == 'image_to_caption' else 'audio'
    for row in corpus:
        if set(row.get('media', {})) != {modality}:
            raise ValueError('Reverse query source has unexpected media mapping')
        identifier = row['id']
        group = identifier if modality == 'image' else next(iter(by_source[identifier]))
        reversed_queries.append({'id': identifier, 'media': dict(row['media']), 'metadata': {
            'group_id': group, 'query_role': 'source_media_only', 'parent_corpus_id': identifier}})
        reversed_qrels.extend({'query_id': identifier, 'corpus_id': key, 'relevance': 1.0}
                             for key in sorted(by_source[identifier]))
    return gallery, reversed_queries, reversed_qrels


def _write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows))


def validate_reverse(destination):
    destination = Path(destination)
    manifest = json.loads((destination / 'dataset.json').read_text())
    if manifest.get('protocol') != PROTOCOL or manifest.get('runtime_ready') is not False:
        raise ValueError('Reverse protocol must retain explicit runtime prerequisites')
    identity = manifest.get('identity')
    unsigned = {key: value for key, value in manifest.items() if key != 'identity'}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest() != identity:
        raise ValueError('Reverse manifest identity mismatch')
    for name, expected in manifest['files'].items():
        path = destination / name
        if path.stat().st_size != expected['size_bytes'] or sha256(path) != expected['sha256']:
            raise ValueError('Derived file fingerprint mismatch')
    gallery, queries, qrels = [read_jsonl(destination / f'{name}.jsonl') for name in ('corpus', 'queries', 'qrels')]
    gallery_ids, query_ids = {row['id'] for row in gallery}, {row['id'] for row in queries}
    if len(gallery_ids) != len(gallery) or len(query_ids) != len(queries):
        raise ValueError('Duplicate reverse gallery/query identifiers')
    pairs = {(row['query_id'], row['corpus_id']) for row in qrels}
    if len(pairs) != len(qrels) or {key[0] for key in pairs} != query_ids or {key[1] for key in pairs} != gallery_ids:
        raise ValueError('Reverse qrels have duplicate, missing or unused references')
    gallery_by_id = {row['id']: row for row in gallery}
    query_by_id = {row['id']: row for row in queries}
    if any(not isinstance(row.get('text'), str) or not row['text'].strip() or row.get('media') for row in gallery):
        raise ValueError('Reverse gallery must contain complete intended reference text only')
    if any('text' in row or not row.get('media') for row in queries):
        raise ValueError('Reverse queries must contain source media without reference text')
    if any(query_by_id[row['query_id']]['metadata']['group_id'] !=
           gallery_by_id[row['corpus_id']]['metadata']['group_id'] for row in qrels):
        raise ValueError('Reverse positive label crosses source/transcript groups')
    if any(row['relevance'] != 1.0 for row in qrels):
        raise ValueError('Reverse task labels must retain binary positives')
    inventory = {row['path']: row for row in read_jsonl(destination / 'assets.jsonl')}
    referenced = {path for row in queries for path in row['media'].values()}
    if referenced != set(inventory):
        raise ValueError('Reverse query media inventory mismatch')
    for relative, entry in inventory.items():
        path = (destination / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(destination.resolve()):
            raise ValueError('Reverse media escapes derived directory')
        if path.stat().st_size != entry['size_bytes'] or sha256(path) != entry['sha256']:
            raise ValueError('Reverse media hash mismatch')
    actual = {'corpus': len(gallery), 'queries': len(queries), 'qrels': len(qrels), 'media_files': len(inventory)}
    if actual != manifest['counts'] or actual != manifest['expected_counts']:
        raise ValueError('Reverse counts differ from protocol expectation')
    if len(queries) != manifest['parent_counts']['corpus'] or len(qrels) != manifest['parent_counts']['positive_qrels']:
        raise ValueError('Reverse source/positive coverage incomplete')
    return manifest


def prepare_reverse(parent, output):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    if output == parent or output.is_relative_to(parent) or parent.is_relative_to(output):
        raise ValueError('Derived output must be separate from the frozen parent directory')
    original = validate_dataset(parent, verify_assets=True)
    if original['track'] == 'photo' and original['id'] in {'xm3600-tr', 'xm3600-en'}:
        kind = 'image_to_caption'
    elif original['track'] == 'speech' and original['id'] == 'fleurs-tr_tr-test':
        kind = 'audio_to_reference_transcript'
    else:
        raise ValueError('Only frozen XM3600 TR/EN and original FLEURS reference protocols are supported')
    parent_manifest_hash = sha256(parent / 'dataset.json')
    corpus, queries, qrels = [read_jsonl(parent / f'{name}.jsonl') for name in ('corpus', 'queries', 'qrels')]
    references = read_jsonl(parent / 'evaluation_transcripts.jsonl') if kind.startswith('audio') else None
    gallery, reversed_queries, reversed_qrels = reverse_records(corpus, queries, qrels, kind=kind, references=references)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Derived output must be empty; never overwrite a prepared reverse snapshot')
    output.mkdir(parents=True, exist_ok=True)
    inventory = {}
    for row in reversed_queries:
        for relative in row['media'].values():
            _link_asset(parent / relative, output / relative)
            inventory[relative] = {'path': relative, 'size_bytes': (output / relative).stat().st_size,
                                   'sha256': sha256(output / relative)}
    parent_inventory = {row['path']: row for row in read_jsonl(parent / 'assets.jsonl')}
    if inventory != parent_inventory:
        raise ValueError('Derived media mapping or hash differs from frozen parent inventory')
    for name, rows in [('corpus', gallery), ('queries', reversed_queries), ('qrels', reversed_qrels),
                       ('assets', [inventory[key] for key in sorted(inventory)])]:
        _write_jsonl(output / f'{name}.jsonl', rows)
    counts = {'corpus': len(gallery), 'queries': len(reversed_queries), 'qrels': len(reversed_qrels), 'media_files': len(inventory)}
    expected_counts = {'corpus': len(queries), 'queries': len(corpus),
        'qrels': sum(row['relevance'] > 0 for row in qrels), 'media_files': original['counts']['media_files']}
    if counts != expected_counts:
        raise ValueError('Derived reverse counts do not preserve frozen parent coverage')
    files = {name: {'sha256': sha256(output / name), 'size_bytes': (output / name).stat().st_size}
             for name in ('corpus.jsonl', 'queries.jsonl', 'qrels.jsonl', 'assets.jsonl')}
    manifest = {'schema_version': 1, 'id': original['id'] + '-reverse-reference-gallery-v1',
        'track': 'reverse_' + kind, 'protocol': PROTOCOL, 'kind': kind, 'status': 'prepared_requires_reverse_engine',
        'runtime_ready': False, 'runtime_prerequisites': PREREQUISITES,
        'revision': original['revision'], 'parent_id': original['id'], 'parent_manifest_sha256': parent_manifest_hash,
        'parent_files': original['files'], 'sources': original['sources'], 'license': original['license'],
        'parent_counts': {'corpus': len(corpus), 'positive_qrels': sum(row['relevance'] > 0 for row in qrels)},
        'counts': counts, 'expected_counts': expected_counts, 'files': files, 'main_matrix_denominator_contribution': 0,
        'gallery_provenance': {'source': 'intended_reference_text_gallery', 'gold_fields_used': True,
            'role': 'The reference text is the retrieval target in this separate reverse protocol; not a source-only generated view.'},
        'query_provenance': {'source': 'frozen_parent_media', 'reference_text_used_in_query': False},
        'duplicate_caption_policy': 'Preserve distinct annotated caption IDs even when text repeats; positives are same-image captions only.',
        'normalized_transcript_policy': 'Use pinned parent NFKC/casefold/whitespace normalization and one gallery entry per verified transcript group.'}
    manifest['identity'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (output / 'dataset.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    validate_reverse(output)
    if any(sha256(parent / name) != spec['sha256'] for name, spec in original['files'].items()):
        raise ValueError('Parent record files changed during preparation')
    if sha256(parent / 'dataset.json') != parent_manifest_hash:
        raise ValueError('Parent manifest changed during preparation')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = prepare_reverse(args.parent, args.output)
    print(json.dumps({key: manifest[key] for key in ('id', 'status', 'runtime_ready', 'counts', 'identity')}))


if __name__ == '__main__':
    main()
