"""Video ablation wiring, identities and timestamp sampling without model loading."""
from types import SimpleNamespace

import pytest

from rag_benchmark.multimodal_cli import main
from rag_benchmark.multimodal_models import make_multimodal_adapter, sample_video
from rag_benchmark.multimodal_specialists import SpecialistAdapter


def capture(tmp_path, monkeypatch, extra=()):
    captured = {}
    monkeypatch.setattr('rag_benchmark.multimodal.load_dataset', lambda _: SimpleNamespace(track='video'))
    def run(*args, **kwargs):
        captured.update(kwargs['adapters'])
        return {'scope': 'partial', 'evaluated_query_count': 0}
    monkeypatch.setattr('rag_benchmark.multimodal.run_matrix', run)
    assert main(['run', '--dataset', str(tmp_path), '--run-dir', str(tmp_path / 'run'),
                 '--channels', 'N,J,S', *extra]) == 0
    return captured


def test_default_adapter_config_and_identity_equal_historical_defaults(tmp_path, monkeypatch):
    current = capture(tmp_path, monkeypatch)
    common = {'device': 'mps', 'dtype': 'bfloat16', 'batch_size': 1}
    native = {**common, 'dimension': 768, 'vision_budget': 280}
    historical = {channel: make_multimodal_adapter(channel, native) for channel in ('N', 'J')}
    historical['S'] = SpecialistAdapter('clip_video', {**common, 'text_overflow_policy': 'error'})
    for channel in historical:
        assert current[channel].config == historical[channel].config
        assert current[channel].identity == historical[channel].identity
        assert current[channel]._model is None


@pytest.mark.parametrize('fps,frames,budget', [(0.5, 8, 70), (1, 16, 140), (2, 32, 1120)])
def test_all_video_adapters_receive_same_sampling(tmp_path, monkeypatch, fps, frames, budget):
    current = capture(tmp_path, monkeypatch, ['--video-fps', str(fps), '--video-max-frames', str(frames),
                                           '--video-vision-budget', str(budget)])
    for adapter in current.values():
        assert adapter.config['video_fps'] == fps
        assert adapter.config['video_max_frames'] == frames
        assert adapter.config['video_vision_budget'] == budget
        assert adapter._model is None


@pytest.mark.parametrize('flag,value', [('--video-fps', '0'), ('--video-fps', '-1'),
    ('--video-fps', 'nan'), ('--video-fps', 'inf'), ('--video-max-frames', '0'),
    ('--video-max-frames', '17'), ('--video-vision-budget', '141')])
def test_invalid_video_parameters_fail_before_dataset_loading(tmp_path, flag, value):
    with pytest.raises(SystemExit) as exc:
        main(['run', '--dataset', str(tmp_path), '--run-dir', str(tmp_path / 'run'), flag, value])
    assert exc.value.code == 2


def test_timestamp_fixture_keeps_identical_native_specialist_frame_indices(tmp_path, monkeypatch):
    av = pytest.importorskip('av')
    import numpy as np
    path = tmp_path / 'fixture.mp4'
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('mpeg4', rate=4)
        stream.width = stream.height = 16
        stream.pix_fmt = 'yuv420p'
        for index in range(81):
            frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), index, dtype=np.uint8), format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    adapters = capture(tmp_path, monkeypatch)
    outputs = [sample_video(path, adapter.config['video_fps'], adapter.config['video_max_frames'])
               for adapter in adapters.values()]
    assert all(output[1]['frame_indices'] == outputs[0][1]['frame_indices'] for output in outputs)
    historical_frames, historical_usage = sample_video(path, 1.0, 16)
    assert outputs[0][1] == historical_usage
    assert np.array_equal(outputs[0][0], historical_frames)
    assert len(historical_usage['frame_indices']) == 16
    assert historical_usage['audio_included'] is False
