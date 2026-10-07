import hashlib
import json

import pytest

from rag_benchmark.multimodal_data import write_dataset
from rag_benchmark.multimodal_reverse_data import prepare_reverse, reverse_records, validate_reverse


def parent(tmp_path, *, speech=False):
    path = tmp_path / 'parent'
    path.mkdir()
    kind = 'audio' if speech else 'image'
    for name in ('a', 'b'):
        (path / f'{name}.bin').write_bytes(name.encode() * 16)
    corpus = [{'id': name, 'media': {kind: f'{name}.bin'}, 'metadata': {'group_id': 'g'}} for name in ('a', 'b')]
    if speech:
        queries = [{'id': 'g', 'text': 'hello world', 'metadata': {'group_id': 'g'}}]
        qrels = [{'query_id': 'g', 'corpus_id': name, 'relevance': 1.0} for name in ('a', 'b')]
        references = [{'corpus_id': name, 'normalized_transcript': ' HELLO  world '} for name in ('a', 'b')]
        (path / 'evaluation_transcripts.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in references))
    else:
        queries = [{'id': str(i), 'text': 'same duplicated caption', 'metadata': {'language': 'tr'}} for i in (1, 2, 3)]
        qrels = [{'query_id': str(i), 'corpus_id': 'a' if i < 3 else 'b', 'relevance': 1.0} for i in (1, 2, 3)]
    write_dataset(path, dataset_id='fleurs-tr_tr-test' if speech else 'xm3600-tr',
                  track='speech' if speech else 'photo', revision='a' * 40, corpus=corpus,
                  queries=queries, qrels=qrels, sources=[], license='fixture', text_source='absent_requires_generation')
    return path


def rows(path, name):
    return [json.loads(line) for line in (path / f'{name}.jsonl').read_text().splitlines()]


def test_image_reverse_keeps_duplicate_captions_and_same_image_all_positives(tmp_path):
    source = parent(tmp_path)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}
    output = tmp_path / 'reverse'
    manifest = prepare_reverse(source, output)
    assert manifest['counts'] == {'corpus': 3, 'queries': 2, 'qrels': 3, 'media_files': 2}
    assert len({row['id'] for row in rows(output, 'corpus')}) == 3
    assert len({row['text'] for row in rows(output, 'corpus')}) == 1
    positive = {row['corpus_id'] for row in rows(output, 'qrels') if row['query_id'] == 'a'}
    from rag_benchmark.multimodal import graded_metrics
    assert graded_metrics(['1'], {key: 1.0 for key in positive})['recall@5'] == 0.5
    assert graded_metrics(['1', '2'], {key: 1.0 for key in positive})['recall@5'] == 1.0
    assert positive == {'1', '2'}  # duplicate text attached to b remains negative for image a
    assert all('text' not in row for row in rows(output, 'queries'))
    assert manifest['gallery_provenance']['gold_fields_used'] is True
    assert manifest['runtime_ready'] is False and manifest['main_matrix_denominator_contribution'] == 0
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}
    assert validate_reverse(output)['identity'] == manifest['identity']
    from rag_benchmark.multimodal import load_dataset
    with pytest.raises(ValueError, match='canonical track'):
        load_dataset(output)


def test_audio_reverse_deduplicates_normalized_transcripts_retains_every_recording(tmp_path):
    source = parent(tmp_path, speech=True)
    output = tmp_path / 'reverse'
    manifest = prepare_reverse(source, output)
    assert manifest['counts'] == {'corpus': 1, 'queries': 2, 'qrels': 2, 'media_files': 2}
    assert rows(output, 'corpus')[0]['text'] == 'hello world'
    assert {row['metadata']['group_id'] for row in rows(output, 'queries')} == {'g'}
    assert all(row['corpus_id'] == 'g' for row in rows(output, 'qrels'))
    assert all(set(row['media']) == {'audio'} and 'text' not in row for row in rows(output, 'queries'))


@pytest.mark.parametrize('damage', ['record', 'media', 'source_mapping'])
def test_parent_hash_or_media_damage_fails_before_derivation(tmp_path, damage):
    source = parent(tmp_path)
    if damage == 'record':
        (source / 'queries.jsonl').write_text('[]\n')
    elif damage == 'media':
        (source / 'a.bin').write_bytes(b'changed')
    else:
        (source / 'b.bin').unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare_reverse(source, tmp_path / 'reverse')
    assert not (tmp_path / 'reverse').exists()


def test_normalized_reference_mismatch_or_missing_reference_fails(tmp_path):
    source = parent(tmp_path, speech=True)
    corpus, queries, qrels = [rows(source, name) for name in ('corpus', 'queries', 'qrels')]
    references = rows(source, 'evaluation_transcripts')
    references[0]['normalized_transcript'] = 'different'
    with pytest.raises(ValueError, match='differs'):
        reverse_records(corpus, queries, qrels, kind='audio_to_reference_transcript', references=references)
    with pytest.raises(ValueError, match='cover every'):
        reverse_records(corpus, queries, qrels, kind='audio_to_reference_transcript', references=references[:1])


def test_derived_snapshot_hash_damage_and_overwrite_are_rejected(tmp_path):
    source = parent(tmp_path)
    output = tmp_path / 'reverse'
    prepare_reverse(source, output)
    with pytest.raises(ValueError, match='empty'):
        prepare_reverse(source, output)
    (output / 'qrels.jsonl').write_text('')
    with pytest.raises(ValueError, match='fingerprint'):
        validate_reverse(output)


def test_parent_media_without_positive_is_rejected():
    with pytest.raises(ValueError, match='Every original'):
        reverse_records([{'id': 'a', 'media': {'image': 'a'}}], [{'id': 'q', 'text': 'caption'}], [], kind='image_to_caption')


def test_caption_associated_with_multiple_images_is_rejected():
    corpus = [{'id': key, 'media': {'image': key}} for key in ('a', 'b')]
    qrels = [{'query_id': 'q', 'corpus_id': key, 'relevance': 1.0} for key in ('a', 'b')]
    with pytest.raises(ValueError, match='exactly one'):
        reverse_records(corpus, [{'id': 'q', 'text': 'caption'}], qrels, kind='image_to_caption')


def test_wrong_source_modality_is_rejected():
    with pytest.raises(ValueError, match='unexpected media'):
        reverse_records([{'id': 'a', 'media': {'audio': 'a'}}], [{'id': 'q', 'text': 'caption'}],
                        [{'query_id': 'q', 'corpus_id': 'a', 'relevance': 1.0}], kind='image_to_caption')


def test_manifest_identity_detects_changed_protocol_metadata(tmp_path):
    source = parent(tmp_path)
    output = tmp_path / 'reverse'
    prepare_reverse(source, output)
    manifest = json.loads((output / 'dataset.json').read_text())
    manifest['main_matrix_denominator_contribution'] = 1
    (output / 'dataset.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='manifest identity'):
        validate_reverse(output)
