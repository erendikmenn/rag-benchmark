"""Separate reverse EG2 cosine retrieval; explicit CLI inference, no forward-policy changes.

Prepared reverse manifests remain immutable and runtime_ready=false describes the
forward runner, not this explicit bridge. Actual reverse model execution/accuracy
has not been measured. Reference captions/transcripts are intended target gallery.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import fcntl
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import numpy as np

from .models import MODEL_DEFAULTS, stable_hash
from .multimodal import (aggregate_metrics, cached_encode, exact_dot_rankings, graded_metrics,
    encoding_cache_identity, load_dataset, _channel_identity, _merge_diagnostics, _validate_vectors)
from .multimodal_data import read_jsonl, sha256, validate_dataset
from .multimodal_reverse_data import reverse_records, validate_reverse

VERSION = 'separate-reverse-eg2-cosine-v1'
GALLERY_PREFIX = 'title: none | text: '
QUERY_PREFIX = 'task: search result | query: '


def implementation_hashes():
    root = Path(__file__).parent
    return {name: sha256(root / name) for name in (
        'multimodal_reverse_runner.py', 'multimodal_reverse_data.py', 'multimodal.py',
        'multimodal_models.py', 'models.py')}


@dataclass
class ReverseDataset:
    root: Path
    manifest: dict
    corpus: list
    queries: list
    qrels: dict
    identity: str

    def model_item(self, item, *, text_only=False):
        # This separate task permits intended reference gallery text. It never
        # calls or overrides the forward source-only candidate-text guard.
        if item.get('media'):
            if text_only or 'text' in item:
                raise ValueError('Reverse media query cannot contain reference text')
            return {'id': item['id'], 'media': {key: str((self.root / relative).resolve())
                    for key, relative in item['media'].items()}}
        return {'id': item['id'], 'text': item['text'], 'media': {}}


def verified_reverse(dataset_dir, parent_dir):
    root, parent = Path(dataset_dir).resolve(), Path(parent_dir).resolve()
    manifest = validate_reverse(root)
    original = validate_dataset(parent, verify_assets=True)
    if manifest['parent_manifest_sha256'] != sha256(parent / 'dataset.json') or manifest['parent_id'] != original['id']:
        raise ValueError('Reverse parent manifest fingerprint differs')
    if manifest['parent_files'] != original['files'] or manifest['sources'] != original['sources']:
        raise ValueError('Reverse parent source binding differs')
    if manifest['gallery_provenance'].get('gold_fields_used') is not True or manifest['gallery_provenance'].get('source') != 'intended_reference_text_gallery':
        raise ValueError('Reverse protocol requires honest intended-reference-gallery provenance')
    if manifest['query_provenance'] != {'source': 'frozen_parent_media', 'reference_text_used_in_query': False}:
        raise ValueError('Reverse query provenance must remain source media only')
    actual = [read_jsonl(root / f'{name}.jsonl') for name in ('corpus', 'queries', 'qrels')]
    source = [read_jsonl(parent / f'{name}.jsonl') for name in ('corpus', 'queries', 'qrels')]
    references = read_jsonl(parent / 'evaluation_transcripts.jsonl') if manifest['kind'].startswith('audio') else None
    expected = reverse_records(*source, kind=manifest['kind'], references=references)
    if tuple(actual) != expected or read_jsonl(root / 'assets.jsonl') != read_jsonl(parent / 'assets.jsonl'):
        raise ValueError('Reverse records/media differ from exact frozen source transformation')
    qrels = defaultdict(dict)
    for row in actual[2]:
        qrels[row['query_id']][row['corpus_id']] = row['relevance']
    return ReverseDataset(root, manifest, actual[0], actual[1], dict(qrels), manifest['identity'])


class ReverseTextGalleryEncoder:
    """Pinned E document encoder with explicit full-text token coverage diagnostics."""
    def __init__(self, config):
        from .multimodal import TextEmbeddingAdapter
        self.base = TextEmbeddingAdapter('embeddinggemma', config)
        self.config = dict(self.base.embedder.config)
        self.dimension = self.base.dimension
        self.prompt_identity = {'document': GALLERY_PREFIX}
        self.identity = stable_hash({'reverse_gallery': VERSION, 'base': self.base.identity,
            'document_prefix': GALLERY_PREFIX, 'implementation': implementation_hashes()})

    def encode(self, items, role):
        if role != 'document' or any(item.get('media') or not item.get('text') for item in items):
            raise ValueError('Reverse gallery encoding requires nonempty text-only documents')
        model = self.base.embedder._load()
        texts = [self.base.embedder.format_document(item) for item in items]
        lengths = [len(ids) for ids in model.tokenizer(texts, truncation=False, padding=False)['input_ids']]
        if max(lengths, default=0) > model.max_seq_length:
            raise ValueError('Reverse reference gallery exceeds native formatted token limit; no truncation')
        values = self.base.encode(items, role='document')
        self.last_usage = {'context_limit': model.max_seq_length, 'text_overflow_policy': 'error',
            'truncated_text_items': 0, 'items': [{'id': item['id'], 'modalities': ['text'],
                'original_tokens': count, 'retained_tokens': count} for item, count in zip(items, lengths, strict=True)]}
        return values

    def unload(self):
        self.base.unload()


class ReverseEG2Bridge:
    def __init__(self, *, device='mps', dtype='bfloat16', query_batch_size=1, gallery_batch_size=8,
                 query_adapter=None, gallery_adapter=None):
        from .multimodal_models import EmbeddingGemma2Adapter
        common = {'device': device, 'dtype': dtype, 'dimension': 768, 'max_length': 8192}
        self.query = query_adapter or EmbeddingGemma2Adapter({**common, 'batch_size': query_batch_size})
        self.gallery = gallery_adapter or ReverseTextGalleryEncoder({**common, 'batch_size': gallery_batch_size})
        self.validate()
        if self.query.config['device'] != device or self.query.config['dtype'] != dtype:
            raise ValueError('Injected adapter device/dtype differ from requested bridge condition')
        self.contract = {'family': 'google/embeddinggemma-2', 'revision': MODEL_DEFAULTS['embeddinggemma']['revision'],
            'dimension': 768, 'device': device, 'dtype': dtype, 'gallery_document_prefix': GALLERY_PREFIX,
            'query_condition': 'native_media_only; existing native adapter consumes no textual query prefix',
            'configured_native_query_prefix': QUERY_PREFIX, 'cosine': 'normalized full-gallery dot product',
            'query_adapter_identity': self.query.identity, 'gallery_adapter_identity': self.gallery.identity,
            'implementation': implementation_hashes()}
        self.identity = stable_hash(self.contract)
        self._seal = stable_hash(self._settings())

    def _settings(self):
        return {'query_config': self.query.config, 'gallery_config': self.gallery.config,
            'query_identity': self.query.identity, 'gallery_identity': self.gallery.identity,
            'query_dimension': self.query.dimension, 'gallery_dimension': self.gallery.dimension}

    def validate(self):
        expected = MODEL_DEFAULTS['embeddinggemma']
        for adapter in (self.query, self.gallery):
            if not isinstance(adapter.identity, str) or not adapter.identity:
                raise ValueError('Reverse encoders need independent immutable identities')
            if any(adapter.config.get(key) != expected[key] for key in ('model_id', 'revision', 'dimension')) or adapter.dimension != 768:
                raise ValueError('Reverse encoders must share exact pinned EG2 model/revision/native dimension')
            if adapter.config.get('local_files_only', True) is not True:
                raise ValueError('Reverse inference must remain offline')
        for field in ('device', 'dtype', 'max_length'):
            if self.query.config.get(field, 8192) != self.gallery.config.get(field, 8192):
                raise ValueError('Reverse encoder device/dtype/context mismatch')
        if self.query.config.get('query_prompt') != QUERY_PREFIX or self.query.config.get('document_prompt') != GALLERY_PREFIX:
            raise ValueError('Native prefix differs from pinned bridge protocol')
        if getattr(self.gallery, 'prompt_identity', {}).get('document') != GALLERY_PREFIX:
            raise ValueError('Text-gallery document prefix differs from bridge protocol')
        if self.query.config.get('mode') != 'native':
            raise ValueError('Reverse query bridge requires the actual native media encoder')
        if hasattr(self, '_seal') and stable_hash(self._settings()) != self._seal:
            raise ValueError('Reverse bridge configuration changed after identity was frozen')


def verified_cached_encode(dataset, adapter, role, cache_dir, block_size):
    key = stable_hash({'encoding': encoding_cache_identity(dataset, adapter, role), 'text_only': False})
    path = Path(cache_dir) / 'vectors' / key
    previous = set(path.glob('*.npy'))
    for block in previous:
        proof = block.with_suffix('.checksums.json')
        if not proof.is_file():
            raise ValueError('Reverse cached vector block lacks checksum proof')
        stored = json.loads(proof.read_text())
        if stored != {'vectors': sha256(block), 'diagnostics': sha256(block.with_suffix('.usage.json'))}:
            raise ValueError('Reverse cached vector/diagnostic checksum mismatch')
    try:
        return cached_encode(dataset, adapter, role, cache_dir, block_size=block_size)
    finally:
        # Seal successful fresh blocks even if a later encoder block fails, so
        # interruption recovery reuses only integrity-checked completed inputs.
        for block in set(path.glob('*.npy')) - previous:
            diagnostics = block.with_suffix('.usage.json')
            if diagnostics.is_file():
                block.with_suffix('.checksums.json').write_text(json.dumps({
                    'vectors': sha256(block), 'diagnostics': sha256(diagnostics)}))


def _public_usage(usage, *, audio_required=False):
    diagnostics = usage.get('adapter_diagnostics', {})
    fields = ('status', 'total_rows', 'missing_rows', 'rows', 'reported_rows', 'expanded_tokens_max', 'original_tokens_total',
              'retained_tokens_total', 'truncated_text_items', 'observed_truncated_text_items',
              'processing_counts_observed', 'processing_settings')
    records = diagnostics.get('items', [])
    total = diagnostics.get('total_rows')
    complete_audio = (audio_required and type(total) is int and total > 0 and len(records) == total
        and all(type(item.get(key)) is int and item[key] >= 0
            for item in records for key in ('source_audio_samples', 'retained_audio_samples')))
    audio = {'status': 'complete' if complete_audio else 'unavailable' if audio_required else 'not_applicable',
        'required_query_rows': total if audio_required else 0,
        'source_audio_samples_total': sum(item['source_audio_samples'] for item in records) if complete_audio else None,
        'retained_audio_samples_total': sum(item['retained_audio_samples'] for item in records) if complete_audio else None,
        'scope': 'actual per-item diagnostics only; missing sample counts are unknown, never zero'}
    return {**{key: usage.get(key) for key in ('identity', 'role', 'cache_rows', 'new_rows', 'inference_seconds')},
        'diagnostics': {key: diagnostics[key] for key in fields if key in diagnostics},
        'audio_sample_coverage': audio, 'cache_reads_are_inference': False}


def _verify_ranked_result(result, query_id, corpus_ids, labels, expected):
    if result != expected or result.get('query_id') != query_id:
        raise ValueError('Cached reverse ranking differs from current frozen vectors/labels')
    ranking = result['ranking']
    ids = [row['id'] for row in ranking]
    if len(ids) != len(set(ids)) or not set(ids) <= corpus_ids:
        raise ValueError('Cached reverse result has invalid candidate IDs')
    if any(type(row['score']) not in (int, float) or not math.isfinite(row['score']) for row in ranking):
        raise ValueError('Cached reverse scores are invalid')
    if result['metrics'] != graded_metrics(ids, labels):
        raise ValueError('Cached reverse metrics differ from original qrels')


def _baseline_native_adapter(config):
    from .multimodal_models import EmbeddingGemma2Adapter
    return EmbeddingGemma2Adapter(config)


def verified_media_baseline(dataset, parent_dir, bridge, baseline_run, cache_dir, baseline_parent=None):
    """Read-only proof of parent N vectors, never a fresh-media encoding fallback.

    The pinned native implementation ignores document text and consumes no text
    for a textless media query. Equal config/adapter identity binds preprocessing,
    packages, model revision, device/dtype and role-independent media preparation.
    Actual media hashes bind the decoded inputs. Saved forward dot scores verify
    both cached matrices against completed original measurements.
    """
    if baseline_run is None or cache_dir is None:
        raise ValueError('Verified media baseline run and cache are required')
    parent = load_dataset(baseline_parent or parent_dir)
    validate_dataset(parent.root, verify_assets=True)
    report_path = Path(baseline_run) / 'report.json'
    report_hash = sha256(report_path)
    report = json.loads(report_path.read_text())
    spec = report['adapter_specs']['channels']['N']
    adapter = _baseline_native_adapter(spec['config'])
    if (spec['class'] != 'rag_benchmark.multimodal_models.EmbeddingGemma2Adapter'
            or adapter.identity != spec['identity'] or adapter.identity != bridge.query.identity
            or any(adapter.config.get(key) != value for key, value in spec['config'].items())
            or adapter.config != bridge.query.config or adapter.config['mode'] != 'native'):
        raise ValueError('Media baseline native model/config/implementation differs')
    if report['dataset']['identity'] != parent.identity or report['scope'] != 'frozen_collection':
        raise ValueError('Media baseline source or scope differs')
    query_ids = [item['id'] for item in parent.queries]
    corpus_ids = [item['id'] for item in parent.corpus]
    if report['configuration']['query_ids'] != query_ids or report['evaluated_query_count'] != len(query_ids):
        raise ValueError('Media baseline query coverage differs')
    channel = report['channels']['N']
    if channel['status'] != 'completed' or channel['identity'] != _channel_identity(parent, 'N', {'N': adapter}):
        raise ValueError('Media baseline channel identity differs')
    fingerprints = {str(report_path.resolve()): report_hash}
    matrices, diagnostics = {}, {}
    for role, items in (('document', parent.corpus), ('query', parent.queries)):
        key = stable_hash({'encoding': encoding_cache_identity(parent, adapter, role), 'text_only': False})
        if channel.get(role + '_encoding', {}).get('identity', key) != key:
            raise ValueError('Media baseline encoding identity differs')
        directory = Path(cache_dir) / 'vectors' / key
        blocks, records, cursor = [], [], 0
        for path in sorted(directory.glob('*.npy')):
            start, stop = map(int, path.stem.split('-'))
            if start != cursor or stop <= start or stop > len(items):
                raise ValueError('Media baseline block coverage differs')
            usage_path = path.with_suffix('.usage.json')
            usage = json.loads(usage_path.read_text())
            if usage.get('rows') != stop-start or [item['id'] for item in usage.get('items', [])] != [item['id'] for item in items[start:stop]]:
                raise ValueError('Media baseline diagnostic source order differs')
            values = np.load(path, allow_pickle=False)
            _validate_vectors(values, stop-start, 768)
            if not np.allclose(np.linalg.norm(values, axis=1), 1, rtol=0, atol=1e-5):
                raise ValueError('Media baseline vectors are not unit normalized')
            blocks.append(values)
            records.append(usage)
            cursor = stop
            fingerprints[str(path.resolve())] = sha256(path)
            fingerprints[str(usage_path.resolve())] = sha256(usage_path)
        if cursor != len(items):
            raise ValueError('Media baseline vector cache is missing rows')
        matrices[role] = np.concatenate(blocks)
        diagnostics[role] = _merge_diagnostics(records, len(items))
    cells = [cell for cell in report['cells'] if cell['status'] == 'completed'
        and cell['variant_id'].split('__')[1:] == ['n', 'none'] and cell['budget_mode'] == 'per_channel']
    if len(cells) != 1:
        raise ValueError('Media baseline completed native cell is unavailable')
    cell = cells[0]
    expected_identity = stable_hash({**report['configuration'], 'variant': cell['variant_id'],
        'mode': cell['budget_mode'], 'candidate_k': None, 'channel_identities': {'N': channel['identity']},
        'reranker': None, 'reranker_prompts': None})
    if cell['identity'] != expected_identity or cell['completed_queries'] != len(query_ids):
        raise ValueError('Media baseline native cell identity/coverage differs')
    rankings = exact_dot_rankings(matrices['document'], matrices['query'], corpus_ids,
        top_k=report['configuration']['channel_budget'])
    measured = []
    database_path = Path(baseline_run) / 'progress.sqlite3'
    with sqlite3.connect(database_path.resolve().as_uri() + '?mode=ro', uri=True) as database:
        stored = {qid: json.loads(payload) for qid, payload in database.execute(
            'SELECT query_id,payload FROM results WHERE cell=?', (cell['identity'],))}
    if set(stored) != set(query_ids):
        raise ValueError('Media baseline stored query coverage differs')
    maximum_score_delta = 0.0
    for qid, ranking in zip(query_ids, rankings, strict=True):
        result = stored[qid]
        metrics = graded_metrics([item['id'] for item in ranking], parent.qrels[qid])
        saved = result['ranking']
        if (result.get('query_id') != qid or len(saved) != len(ranking)
                or result['metrics'] != metrics
                or any(old.get('id') != new['id'] or type(old.get('rank')) is not int or old['rank'] != new['rank']
                    or type(old.get('score')) not in (int, float) or not math.isfinite(old['score'])
                    or not math.isclose(old['score'], new['score'], rel_tol=0, abs_tol=2e-6)
                    for old, new in zip(saved, ranking, strict=True))):
            raise ValueError('Media baseline vectors disagree with measured rankings/scores/metrics')
        maximum_score_delta = max(maximum_score_delta, max(
            (abs(old['score'] - new['score']) for old, new in zip(saved, ranking, strict=True)), default=0))
        measured.append(metrics)
    if aggregate_metrics(measured) != cell['metrics']:
        raise ValueError('Media baseline aggregate metrics differ')
    mapping = {item['id']: index for index, item in enumerate(parent.corpus)}
    indices = []
    for query in dataset.queries:
        if query.get('text') or query['id'] not in mapping:
            raise ValueError('Reverse media query source mapping differs')
        original = parent.corpus[mapping[query['id']]]
        if set(original['media']) != set(query['media']) or set(query['media']) not in ({'image'}, {'audio'}):
            raise ValueError('Reverse reuse supports only identical image/audio media inputs')
        for modality, relative in query['media'].items():
            if sha256(dataset.root / relative) != sha256(parent.root / original['media'][modality]):
                raise ValueError('Reverse media input bytes differ')
        indices.append(mapping[query['id']])
    if any(sha256(Path(path)) != value for path, value in fingerprints.items()):
        raise ValueError('Media baseline files changed during verification')
    proof = {'source_report_sha256': report_hash, 'source_dataset_identity': parent.identity,
        'native_adapter_identity': adapter.identity, 'native_cell_identity': cell['identity'],
        'source_cache_sha256': stable_hash(sorted(fingerprints.values())),
        'input_equivalence': 'same pinned native preprocessing; identical media bytes; no consumed text/prefix',
        'stored_forward_results_verified': len(query_ids), 'stored_forward_results_sha256': stable_hash(stored),
        'score_absolute_tolerance': 2e-6, 'maximum_observed_score_delta': maximum_score_delta,
        'score_tolerance_reason': 'float32 dot-product accumulation across batch layouts; exact IDs/ranks/metrics required'}
    return matrices['document'][indices], {'identity': stable_hash(proof), 'role': 'query',
        'cache_rows': len(indices), 'new_rows': 0, 'inference_seconds': None,
        'adapter_diagnostics': diagnostics['document'], 'media_baseline': proof}


def run_reverse(dataset_dir, parent_dir, run_dir, *, bridge=None, top_k=100, block_size=128,
                media_baseline_run=None, media_baseline_cache=None, media_baseline_parent=None):
    if type(top_k) is not int or top_k < 20:
        raise ValueError('Reverse stored top_k must cover all reported cutoffs (at least20)')
    dataset = verified_reverse(dataset_dir, parent_dir)
    run = Path(run_dir).resolve()
    if subprocess.run(['git', 'check-ignore', '--quiet', str(run)], capture_output=True).returncode != 0:
        raise ValueError('Reverse raw rankings/vectors must stay in a git-ignored run directory')
    bridge = bridge or ReverseEG2Bridge()
    bridge.validate()
    try:
        queries, query_usage = verified_media_baseline(dataset, parent_dir, bridge,
            media_baseline_run, media_baseline_cache, media_baseline_parent)
    except (ValueError, KeyError, OSError) as error:
        return {'dataset': dataset.manifest['id'], 'status': 'unavailable', 'completed_queries': 0,
            'planned_queries': len(dataset.queries), 'reason': 'Verified matching forward media vectors unavailable',
            'unavailability_type': type(error).__name__, 'new_model_calls': 0,
            'main_matrix_denominator_contribution': 0}
    seal = stable_hash({'version': VERSION, 'dataset': dataset.identity, 'bridge': bridge.identity,
        'top_k': top_k, 'media_baseline': query_usage['media_baseline'], 'implementation': implementation_hashes()})
    run.mkdir(parents=True, exist_ok=True)
    with (run / 'reverse.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with sqlite3.connect(run / 'progress.sqlite3') as database:
            database.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT)')
            database.execute('CREATE TABLE IF NOT EXISTS results (query_id TEXT PRIMARY KEY,payload TEXT)')
            old = database.execute("SELECT value FROM meta WHERE key='seal'").fetchone()
            if old and old[0] != seal:
                raise ValueError('Reverse source/model/config changed; use a separate run directory')
            database.execute("INSERT OR IGNORE INTO meta VALUES ('seal',?)", (seal,))
            database.commit()
            report = {'schema_version': 1, 'protocol': VERSION, 'dataset': dataset.manifest['id'],
                'dataset_identity': dataset.identity, 'status': 'running', 'planned_queries': len(dataset.queries),
                'full_gallery_candidates': len(dataset.corpus), 'top_k': min(top_k, len(dataset.corpus)),
                'bridge': bridge.contract, 'run_identity': seal, 'completed_queries': 0,
                'source_only_queries': True, 'gallery': 'intended reference caption/transcript targets',
                'main_matrix_denominator_contribution': 0, 'new_model_calls': None,
                'grouping': {'source_groups': len({row['metadata']['group_id'] for row in dataset.queries}),
                    'unit': 'original image' if dataset.manifest['kind'].startswith('image') else 'verified normalized transcript group',
                    'intervals': 'not computed; same-source dependence explicit, speaker/topic dependencies may remain'},
                'specialist_reverse_support': 'pending; no SigLIP/CLAP score claimed',
                'generated_answer_accuracy': None}
            try:
                documents, gallery_usage = verified_cached_encode(dataset, bridge.gallery, 'document', run / 'cache', block_size)
                bridge.gallery.unload()
                bridge.validate()
                rankings = exact_dot_rankings(documents, queries, [row['id'] for row in dataset.corpus], top_k=min(top_k, len(dataset.corpus)))
                current_ids = {row['id'] for row in dataset.queries}
                stored_ids = {row[0] for row in database.execute('SELECT query_id FROM results')}
                if not stored_ids <= current_ids:
                    raise ValueError('Reverse cache contains unexpected query rows')
                metrics, hits = [], 0
                for query, ranking in zip(dataset.queries, rankings, strict=True):
                    qid = query['id']
                    values = graded_metrics([row['id'] for row in ranking], dataset.qrels[qid])
                    result = {'query_id': qid, 'ranking': ranking, 'metrics': values}
                    old = database.execute('SELECT payload FROM results WHERE query_id=?', (qid,)).fetchone()
                    if old:
                        _verify_ranked_result(json.loads(old[0]), qid, {row['id'] for row in dataset.corpus}, dataset.qrels[qid], result)
                        hits += 1
                    else:
                        with database:
                            database.execute('INSERT INTO results VALUES (?,?)', (qid, json.dumps(result)))
                    metrics.append(values)
                # Detect accidental manifest/record edits before publishing aggregate results.
                validate_reverse(dataset.root)
                report.update(status='completed', completed_queries=len(metrics), metrics=aggregate_metrics(metrics),
                    result_cache_hits=hits, gallery_encoding=_public_usage(gallery_usage), query_encoding=_public_usage(query_usage, audio_required=dataset.manifest['kind'].startswith('audio')),
                    latency_scope='fresh encoding rows only; cached encodings/results are not inference')
                report['media_baseline'] = query_usage['media_baseline']
            except Exception as error:
                report.update(status='failed', reason='Reverse runtime failed; see ignored failures.jsonl')
                with (run / 'failures.jsonl').open('a') as handle:
                    handle.write(json.dumps({'error_type': type(error).__name__, 'error': str(error)}) + '\n')
            finally:
                bridge.gallery.unload()
                bridge.query.unload()
            (run / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--media-baseline-run', type=Path, required=True)
    parser.add_argument('--media-baseline-cache', type=Path, required=True)
    parser.add_argument('--media-baseline-parent', type=Path,
        help='Frozen source of baseline run; permits cross-language reuse only with matching media IDs and bytes')
    parser.add_argument('--device', choices=('cpu', 'mps'), default='mps')
    parser.add_argument('--dtype', choices=('float32', 'bfloat16'), default='bfloat16')
    parser.add_argument('--output', type=Path, help='Optional aggregate-only report; raw cache stays in ignored run-dir')
    args = parser.parse_args(argv)
    report = run_reverse(args.dataset, args.parent, args.run_dir, bridge=ReverseEG2Bridge(device=args.device, dtype=args.dtype),
        media_baseline_run=args.media_baseline_run, media_baseline_cache=args.media_baseline_cache,
        media_baseline_parent=args.media_baseline_parent)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('dataset', 'status', 'completed_queries', 'planned_queries')}))
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
