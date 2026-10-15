"""Pinned source-only composed-image captions; no retrieval, evaluation or DB calls."""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time

ROOT = Path(__file__).absolute().parents[1]
PROTOCOL = 'composed-source-caption-en-rp11-v1'
COUNTS = {'corpus': 39826, 'queries': 30037}
SERVER_SHA = '1bf5602a2be50097c44282a6ecc63234d94ee8f96d39fd45fed6eff4e74b6b41'
SAMPLER_SHA = '50464f0925e52ad77f6137325eca8f102afe7c520899289aa7f9a15cde057f31'


def require(ok, code):
    if not ok:
        raise ValueError(code)


def seal(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe(path, root=ROOT):
    p = Path(path).absolute()
    require(not any(x.is_symlink() for x in (p, *p.parents)), 'symlink_path')
    p = p.resolve()
    require(p.is_relative_to(root.resolve()), 'path_escape')
    return p


def write_once(path, value):
    with Path(path).open('x') as stream:
        os.chmod(path, 0o600)
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read(path):
    return json.loads(Path(path).read_bytes())


def validate_source(dataset, counts=COUNTS):
    require(dataset.track == 'composed_image', 'composed_image_source')
    require(len(dataset.corpus) == counts['corpus'] and len(dataset.queries) == counts['queries'], 'full_source_counts')
    require(len({row['id'] for row in dataset.queries}) == counts['queries'], 'unique_query_ids')
    documents = {row['id']: row for row in dataset.corpus}
    require(len(documents) == counts['corpus'], 'unique_gallery_ids')
    for row in dataset.corpus:
        require(not row.get('text') and set(row.get('media', {})) == {'image'}, 'gallery_pixels_only')
    for query in dataset.queries:
        reference = str(query['metadata'].get('reference_id', query['metadata'].get('reference_image_id')))
        require(reference in documents and set(query.get('media', {})) == {'image'}, 'gallery_reference_required')
        require(isinstance(query.get('text'), str), 'verbatim_transition_required')
        if not query['text'].strip():
            require(query['metadata'].get('transition_text_empty') is True, 'empty_transition_flag_required')
        require(sha(dataset.root / query['media']['image']) == sha(dataset.root / documents[reference]['media']['image']), 'reference_image_binding')
    if counts == COUNTS:
        validate_official_source(dataset)


def validate_official_source(dataset):
    from rag_benchmark.lasco_data import ANNOTATION_SHA256, GALLERY_SHA256, REVISION
    manifest = dataset.manifest
    require(manifest['id'] == 'lasco-validation' and manifest['revision'] == REVISION, 'official_lasco_identity')
    require(manifest.get('metadata', {}).get('annotation_revision') == REVISION and manifest['metadata'].get('split') == 'validation', 'official_lasco_split')
    sources = manifest['sources']
    by_name = {row['file']: row for row in sources}
    require(len(sources) == len(by_name) == 3 and set(by_name) == {'lasco_val.json', 'lasco_val_corpus.json', 'val2014.zip'}, 'official_source_entries')
    require(by_name['lasco_val.json']['sha256'] == ANNOTATION_SHA256 and by_name['lasco_val_corpus.json']['sha256'] == GALLERY_SHA256, 'official_annotation_hashes')
    require(all(type(row.get('size_bytes')) is int and row['size_bytes'] > 0 for row in sources), 'official_source_sizes')
    require(re.fullmatch('[a-f0-9]{64}', by_name['val2014.zip']['sha256']) is not None, 'official_media_archive_hash')
    require(sum(not query['text'].strip() for query in dataset.queries) == 6 and manifest['metadata'].get('empty_transition_count') == 6, 'six_official_empty_transitions')


def source_item(dataset, row):
    require(not row.get('text') and set(row.get('media', {})) == {'image'}, 'gallery_pixels_only')
    p = safe(dataset.root / row['media']['image'], dataset.root)
    return {'text': '', 'media': {'image': str(p)}}


def payload(item, config):
    from rag_benchmark.multimodal_generation import LocalMultimodalGenerator, media_content
    require(set(item) == {'text', 'media'} and item['text'] == '' and set(item['media']) == {'image'}, 'caption_input_allowlist')
    generator = LocalMultimodalGenerator(config)
    return {'model': generator.config['model'], 'messages': [
        {'role': 'system', 'content': generator.caption_system},
        {'role': 'user', 'content': media_content(item, generator.config)}],
        'temperature': 0.0, 'seed': 42, 'stream': False, 'max_tokens': 512,
        'chat_template_kwargs': {'enable_thinking': False}, 'cache_prompt': bool(config['cache_prompt']),
        'n_keep': -1, 'repeat_penalty': 1.1}


def strict(response, context):
    require(type(context) is int and context == 8192, 'runtime_context')
    require(response['status_code'] == 200, 'http_status')
    require(hashlib.sha256(response['body'].encode()).hexdigest() == response['body_sha256'], 'raw_body_hash')
    body = json.loads(response['body'])
    require(len(body.get('choices', [])) == 1, 'single_choice')
    choice = body['choices'][0]
    usage = body.get('usage', {})
    require(not any(v.get('truncated') or v.get('tokens_dropped', 0) for v in [body, choice, usage]), 'truncated_input')
    require(type(usage.get('prompt_tokens')) is int and 0 <= usage['prompt_tokens'] <= context - 512, 'context_reservation')
    require(type(usage.get('completion_tokens')) is int and 0 <= usage['completion_tokens'] <= 512, 'completion_usage')
    require(choice.get('finish_reason') == 'stop', 'normal_stop')
    text = choice.get('message', {}).get('content')
    require(isinstance(text, str) and bool(text.strip()), 'nonempty_caption')
    previous, length = None, 0
    for word in re.findall(r'\w+', text.casefold()):
        length = length + 1 if word == previous else 1
        require(length <= 20, 'pathological_repetition')
        previous = word
    return text


def verify(spec):
    for path, digest in spec['files'].items():
        require(sha(path) == digest, 'source_or_artifact_changed')


def prepare(dataset_path, destination, output, server, config, runtime_path, sampler_path):
    from rag_benchmark.multimodal import load_dataset
    from rag_benchmark.multimodal_generation import E4B_CONFIG, LocalMultimodalGenerator
    from rag_benchmark.multimodal_cli import generation_server_command
    dataset = load_dataset(dataset_path)
    validate_source(dataset)
    destination, output = safe(destination), safe(output)
    require(not destination.exists() and not output.exists(), 'fresh_destination_and_output')
    require(output.is_relative_to(ROOT / 'work') or output.is_relative_to(ROOT / 'runs'), 'private_run_directory')
    config = dict(config)
    for key in ['model_path', 'projector_path']:
        if key in config:
            config[key] = str(safe(config[key] if Path(config[key]).is_absolute() else ROOT / config[key]))
    generator = LocalMultimodalGenerator(config)
    config = generator.config
    require(config['caption_language'] == 'en' and config['max_tokens'] == 512 and config['temperature'] == 0
            and config['seed'] == 42 and config['enable_thinking'] is False and config['image_max_side'] == 1120, 'fixed_english_policy')
    require(config['base_url'] == 'http://127.0.0.1:8081/v1', 'owned_loopback_endpoint')
    require(all(config[k] == E4B_CONFIG[k] for k in ['model_id', 'revision', 'model_sha256', 'projector_sha256']), 'reviewed_model_artifacts')
    require(sha(server) == SERVER_SHA, 'reviewed_server_artifact')
    require(sha(sampler_path) == SAMPLER_SHA, 'reviewed_sampler_proof')
    runtime = read(runtime_path)
    require(type(runtime['context_limit']) is int and runtime['context_limit'] == 8192, 'runtime_context')
    require(all(sha(config[prefix + '_path']) == config[prefix + '_sha256'] for prefix in ['model', 'projector']), 'actual_model_artifact_hash')
    command = generation_server_command(safe(server), config)  # Pins model/projector; no process start.
    paths = {Path(__file__).absolute(), *sorted((ROOT / 'src/rag_benchmark').glob('*.py')), safe(server),
             safe(runtime_path), safe(sampler_path), safe(config['model_path']), safe(config['projector_path']), dataset.root / 'dataset.json'}
    paths.update(dataset.root / name for name in dataset.manifest['files'])
    paths.update(dataset.root / relative for row in dataset.corpus + dataset.queries for relative in row.get('media', {}).values())
    files = {str(p): sha(p) for p in sorted(paths)}
    return {'protocol': PROTOCOL, 'source': str(dataset.root), 'source_identity': dataset.identity,
            'destination': str(destination), 'output': str(output), 'counts': COUNTS, 'config': config,
            'runtime_proof': runtime, 'sampler_proof_sha256': sha(sampler_path), 'files': files,
            'server_command': command, 'request_policy': {'repeat_penalty': 1.1, 'max_tokens': 512},
            'benchmark_runs': 0, 'database_writes': False}


def prepare_rows(dataset, spec, directory, transport, *, payload_builder=payload, max_new=None, read_only=False):
    """Only accepted, hash-bound rows can resume; any claimed incomplete row blocks."""
    records, new = [], 0
    binding = seal(spec)
    controls = copy.deepcopy({k: spec[k] for k in ['config', 'runtime_proof', 'protocol']})
    control_sha = seal(controls)
    # Full spec once per boundary; immutable execution controls are cheap per row.
    def control_guard():
        require(seal({k: spec[k] for k in controls}) == control_sha, 'execution_control_mutation')
    control_guard()
    rows = directory / 'rows'
    if read_only:
        require(rows.is_dir(), 'completed_rows_missing')
    else:
        rows.mkdir(mode=0o700, exist_ok=True)
    for index, row in enumerate(dataset.corpus):
        control_guard()
        original_row = copy.deepcopy(row)
        row_sha = seal(row)
        item = source_item(dataset, row)
        asset = sha(item['media']['image'])
        callback_item, callback_config = copy.deepcopy(item), copy.deepcopy(controls['config'])
        request = payload_builder(callback_item, callback_config)
        require(callback_item == item and callback_config == controls['config'], 'callback_input_mutation')
        control_guard()
        require(seal(row) == row_sha and source_item(dataset, row) == item and sha(item['media']['image']) == asset, 'row_or_asset_mutation')
        request_sha = seal(request)
        path = rows / f'{index:09d}'
        claim = {'source_id': row['id'], 'asset_sha256': asset, 'request_sha256': request_sha,
                 'spec_sha256': binding, 'runtime_sha256': seal(controls['runtime_proof'])}
        if path.exists():
            require({p.name for p in path.iterdir()} == {'claim.json', 'raw-response.json', 'accepted.json'}, 'claimed_incomplete_no_retry')
            require(read(path / 'claim.json') == claim, 'claim_binding')
            response, accepted = read(path / 'raw-response.json'), read(path / 'accepted.json')
            require(accepted == {**claim, 'text': strict(response, controls['runtime_proof']['context_limit']), 'raw_response_sha256': seal(response)}, 'accepted_receipt_binding')
        else:
            require(not read_only, 'completed_row_missing')
            if max_new is not None and new >= max_new:
                require(seal(spec) == binding, 'full_spec_mutation')
                return records, False
            path.mkdir(mode=0o700)
            write_once(path / 'claim.json', claim)
            try:
                original = seal(request)
                response = transport(request)
                write_once(path / 'raw-response.json', response)  # Persist before acceptance/strict parsing.
                control_guard()
                require(seal(row) == row_sha and row == original_row and source_item(dataset, row) == item, 'source_row_mutation')
                require(seal(request) == original, 'request_mutation')
                text = strict(response, controls['runtime_proof']['context_limit'])
                require(sha(item['media']['image']) == asset, 'source_changed_during_request')
                accepted = {**claim, 'text': text, 'raw_response_sha256': seal(response)}
                write_once(path / 'accepted.json', accepted)
                new += 1
            except BaseException as exc:
                write_once(path / 'failed.json', {'status': 'failed_requires_review', 'error_type': type(exc).__name__})
                raise
        control_guard()
        require(seal(row) == row_sha and sha(item['media']['image']) == asset, 'source_row_or_asset_mutation')
        records.append(accepted)
    require(seal(spec) == binding, 'full_spec_mutation')
    return records, True


def publication_rows(dataset, accepted):
    require(len(accepted) == len(dataset.corpus), 'complete_gallery_captions')
    corpus, by_id = [], {}
    for original, receipt in zip(dataset.corpus, accepted, strict=True):
        require(original['id'] == receipt['source_id'], 'caption_source_order')
        require(sha(dataset.root / original['media']['image']) == receipt['asset_sha256'], 'published_image_hash')
        by_id[original['id']] = receipt
        corpus.append({'id': original['id'], 'text': receipt['text'], 'media': original['media'],
                       'metadata': {'text_provenance': {'source': 'generated_from_source_only', 'query_independent': True,
                         'gold_fields_used': False, 'caption_receipt_sha256': seal(receipt), 'spec_sha256': receipt['spec_sha256']}}})
    queries = []
    for original in dataset.queries:
        reference = str(original['metadata'].get('reference_id', original['metadata'].get('reference_image_id')))
        receipt = by_id[reference]
        require(sha(dataset.root / original['media']['image']) == receipt['asset_sha256'], 'query_reference_image')
        surrogate = 'Reference: ' + receipt['text'] + '\nRequested change: ' + original['text']
        metadata = {k: original['metadata'][k] for k in ['reference_id', 'reference_image_id', 'exclude_ids', 'group_id', 'language', 'transition_text_empty'] if k in original['metadata']}
        metadata['transition_text_empty'] = not bool(original['text'].strip())
        metadata['text_surrogate'] = {'text': surrogate, 'provenance': {'source': 'reference_description_and_instruction',
             'gold_fields_used': False, 'gallery_caption_query_independent': True, 'reference_id': reference,
             'reference_asset_sha256': receipt['asset_sha256'], 'caption_receipt_sha256': seal(receipt),
             'transition_sha256': seal(original['text']), 'format': 'reference-requested-change-v1'}}
        queries.append({'id': original['id'], 'text': original['text'], 'media': original['media'], 'metadata': metadata})
    return corpus, queries


def publish(dataset, spec, accepted):
    from rag_benchmark.multimodal_data import write_dataset
    from rag_benchmark.multimodal import load_dataset
    require(len(accepted) == spec['counts']['corpus'], 'full_gallery_required')
    corpus, queries = publication_rows(dataset, accepted)
    destination = Path(spec['destination'])
    stage = destination.with_name(destination.name + '.preparing')
    require(not destination.exists() and not stage.exists(), 'fresh_derived_view')
    stage.mkdir(mode=0o700)
    for row in corpus + queries:
        for relative in row['media'].values():
            source, target = safe(dataset.root / relative, dataset.root), safe(stage / relative, stage)
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if not target.exists():
                os.link(source, target)
    qrels = [json.loads(line) for line in (dataset.root / 'qrels.jsonl').read_text().splitlines() if line.strip()]
    write_dataset(stage, dataset_id=dataset.manifest['id'] + '-source-caption-en-rp11-v1', track='composed_image',
        revision=seal({'source': dataset.identity, 'spec': seal(spec)}), corpus=corpus, queries=queries, qrels=qrels,
        sources=dataset.manifest.get('sources', []), license=dataset.manifest.get('license', 'upstream'),
        text_source='generated_from_source_only', expected_counts=spec['counts'],
        metadata={'source_dataset_identity': dataset.identity, 'caption_policy_sha256': seal(spec),
                  'split': dataset.manifest.get('split', 'evaluation')})
    loaded = load_dataset(stage)
    require(loaded.qrels == dataset.qrels, 'qrels_changed')
    loaded.require_text()
    verify(spec)
    from rag_benchmark.lasco_data import publish_directory
    publish_directory(stage, destination)
    return loaded.identity


@contextmanager
def deadline(seconds):
    previous = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('absolute_http_deadline')))
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def assert_idle():
    text = subprocess.check_output(['ps', '-axo', 'pid=,command='], text=True)
    for line in text.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit() or int(parts[0]) == os.getpid():
            continue
        require(not re.search(r'(?:^|/|\s)(?:llama-server|continue_suite\.py|prepare_(?:multimodal|ragturk)_\w+\.py)(?:\s|$)', parts[1]) and ' -m rag_benchmark.' not in parts[1], 'another_model_owner')


def stop_owned(server):
    if server is not None and server.poll() is None:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()  # Only this invocation's Popen child.
            server.wait(timeout=30)


def execute(spec, *, resume=False, max_new=None):
    from rag_benchmark.multimodal import load_dataset
    from rag_benchmark.multimodal_generation import LocalMultimodalGenerator
    import httpx
    require(spec['protocol'] == PROTOCOL and spec['counts'] == COUNTS, 'reviewed_protocol')
    verify(spec)
    dataset = load_dataset(spec['source'])
    require(dataset.identity == spec['source_identity'], 'source_identity')
    validate_source(dataset)
    output = safe(spec['output'])
    if output.exists():
        require(resume and read(output / 'pins.json') == spec, 'explicit_same_pins_resume_required')
        sessions = sorted(output.glob('session-*'))
        last = read(sessions[-1] / 'result.json')
        require(last['status'] in {'paused_completed_rows', 'completed'} and last['owned_server_closed'] is True, 'failed_session_no_retry')
        if last['status'] == 'completed':
            require(load_dataset(spec['destination']).identity == last['derived_identity'], 'published_identity_changed')
            prepare_rows(dataset, spec, output, None, read_only=True)
            return last
    else:
        require(not resume, 'resume_missing_run')
        output.mkdir(parents=True, mode=0o700)
        write_once(output / 'pins.json', spec)
        sessions = []
    session = output / f'session-{len(sessions):06d}'
    session.mkdir(mode=0o700)
    server, result = None, {'status': 'failed_requires_review', 'benchmark_runs': 0}
    lock = (ROOT / 'work/overnight/gpu.lock').open('a+b')
    old_term = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('owned_termination')))
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_idle()
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 8081))
        with (session / 'server.private.log').open('x') as log:
            server = subprocess.Popen(spec['server_command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            write_once(session / 'owned-server.json', {'pid': server.pid, 'argv': spec['server_command']})
            started = time.monotonic()
            while True:
                require(server.poll() is None and time.monotonic() - started < 180, 'server_startup')
                try:
                    with httpx.Client(timeout=2, trust_env=False, follow_redirects=False) as client:
                        if client.get('http://127.0.0.1:8081/health').status_code == 200:
                            break
                except httpx.RequestError:
                    pass
                time.sleep(1)
            with deadline(180):
                actual = LocalMultimodalGenerator(spec['config']).preflight()
            require(seal(actual) == seal(spec['runtime_proof']), 'runtime_mismatch')
            write_once(session / 'runtime-proof.json', actual)
            def transport(request):
                require(server.poll() is None, 'owned_server_exited')
                with deadline(240), httpx.Client(timeout=httpx.Timeout(connect=5, read=180, write=30, pool=5), trust_env=False, follow_redirects=False) as client:
                    with client.stream('POST', 'http://127.0.0.1:8081/v1/chat/completions', json=request) as response:
                        chunks, size = [], 0
                        for chunk in response.iter_bytes():
                            size += len(chunk)
                            require(size <= 2 * 1024 * 1024, 'response_limit')
                            chunks.append(chunk)
                        raw = b''.join(chunks)
                return {'status_code': response.status_code, 'body': raw.decode('utf-8'), 'body_sha256': hashlib.sha256(raw).hexdigest()}
            accepted, complete = prepare_rows(dataset, spec, output, transport, max_new=max_new)
            verify(spec)
            result['completed_rows'] = len(accepted)
            if complete:
                result['derived_identity'] = publish(dataset, spec, accepted)
                result['status'] = 'completed'
            else:
                result['status'] = 'paused_completed_rows'
    finally:
        old_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            stop_owned(server)
        finally:
            lock.close()
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, old_term)
            result['owned_server_closed'] = server is None or server.poll() is not None
            write_once(session / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--execute', action='store_true')
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--spec-sha256')
    for name in ['dataset', 'destination', 'output', 'server', 'config', 'runtime-proof', 'sampler-proof']:
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-new', type=int)
    args = parser.parse_args()
    os.umask(0o077)
    os.chdir(ROOT)
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    require(args.max_new is None or args.max_new > 0, 'positive_max_new')
    if args.prepare:
        require(all(getattr(args, n) is not None for n in ['dataset', 'destination', 'output', 'server', 'config', 'runtime_proof', 'sampler_proof']), 'prepare_arguments')
        spec = prepare(args.dataset, args.destination, args.output, args.server, read(args.config), args.runtime_proof, args.sampler_proof)
        write_once(safe(args.spec), spec)
        print(json.dumps({'status': 'prepared_not_executed', 'spec_sha256': sha(args.spec)}))
    else:
        raw = args.spec.read_bytes()
        require(args.spec_sha256 and hashlib.sha256(raw).hexdigest() == args.spec_sha256, 'external_spec_hash')
        result = execute(json.loads(raw), resume=args.resume, max_new=args.max_new)
        print(json.dumps({k: v for k, v in result.items() if k in {'status', 'completed_rows', 'owned_server_closed'}}))


if __name__ == '__main__':
    main()
