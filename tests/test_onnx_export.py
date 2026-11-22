"""Validate delivered ONNX graphs, metadata, and PyTorch/ORT parity.

The shipped web bundle is the single whole-sky target model (512x288).
The legacy texture model remains in checkpoints/ for reference only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from nca4plot.export_onnx import DEFAULT_STEPS, load_model
from nca4plot.model import ExportCA, make_random_masks, parameter_count


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'checkpoints' / 'nca_starry_model.pt'
TARGET_CHECKPOINT = ROOT / 'checkpoints' / 'nca_target_model.pt'
WEB_MODEL_DIR = ROOT / 'web' / 'models'


def test_legacy_texture_onnx_metadata() -> None:
    meta_path = ROOT / 'checkpoints' / 'nca_starry_meta.json'
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        sizes = {int(s): int(meta['steps_per_inference'][str(s)]) for s in meta['sizes']}
        expected_params = str(meta['parameter_count'])
    else:
        sizes = {128: 4, 256: 4, 512: 2}
        expected_params = str(parameter_count(load_model(CHECKPOINT)))
    for size, steps in sizes.items():
        path = ROOT / 'checkpoints' / f'nca_starry_{size}.onnx'
        assert path.exists(), f'missing {path}'
        model = onnx.load(path.as_posix())
        onnx.checker.check_model(model)
        props = {entry.key: entry.value for entry in model.metadata_props}
        assert props['size'] == str(size)
        assert props['steps_per_inference'] == str(steps)
        assert props['parameter_count'] == expected_params
        assert 'CC BY 4.0' in props['license']
        # Legacy texture model is no longer shipped in the web bundle;
        # if a copy is present it must still be byte-identical.
        web_copy = WEB_MODEL_DIR / path.name
        if web_copy.exists():
            assert path.read_bytes() == web_copy.read_bytes()


def test_legacy_texture_256_pytorch_onnx_parity() -> None:
    ca = load_model(CHECKPOINT)
    wrapper = ExportCA(ca, DEFAULT_STEPS[256]).eval()
    size = 256
    generator = torch.Generator().manual_seed(20261119)
    state = torch.randn(1, ca.chn, size, size, generator=generator) * 0.05
    masks = make_random_masks(DEFAULT_STEPS[size], size, size, generator=generator)
    with torch.no_grad():
        expected = wrapper(state, masks).numpy()

    session = ort.InferenceSession(
        (ROOT / 'checkpoints' / 'nca_starry_256.onnx').as_posix(),
        providers=['CPUExecutionProvider'],
    )
    actual = session.run(
        ['state_out'],
        {
            'state': state.numpy().astype(np.float32),
            'masks': masks.numpy().astype(np.float32),
        },
    )[0]
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


def test_target_web_bundle_metadata_and_parity() -> None:
    meta = json.loads((WEB_MODEL_DIR / 'nca_target_meta.json').read_text(encoding='utf-8'))
    assert meta['kind'] == 'target'
    assert (meta['width'], meta['height']) == (512, 288)
    assert meta['chn'] == 16
    assert meta['parameter_count'] == 10368
    assert meta['steps_per_inference'] == 2

    onnx_path = WEB_MODEL_DIR / meta['model']
    assert onnx_path.exists()
    model = onnx.load(onnx_path.as_posix())
    onnx.checker.check_model(model)
    props = {entry.key: entry.value for entry in model.metadata_props}
    assert props['size'] == '512x288'
    assert props['steps_per_inference'] == '2'
    assert props['parameter_count'] == '10368'
    assert 'CC BY 4.0' in props['license']
    assert onnx_path.read_bytes() == (ROOT / 'checkpoints' / onnx_path.name).read_bytes()

    seed_path = WEB_MODEL_DIR / meta['seed_image']
    assert seed_path.exists()
    target_path = ROOT / 'data' / 'target_sky_512x288.png'
    assert meta['target_sha256'] == hashlib.sha256(target_path.read_bytes()).hexdigest()

    ca = load_model(TARGET_CHECKPOINT)
    ca.life_gating = True
    assert parameter_count(ca) == 10368
    wrapper = ExportCA(ca, 2).eval()
    generator = torch.Generator().manual_seed(20261120)
    state = torch.rand(1, ca.chn, 288, 512, generator=generator)
    masks = make_random_masks(2, 288, 512, generator=generator)
    with torch.no_grad():
        expected = wrapper(state, masks).numpy()
    session = ort.InferenceSession(onnx_path.as_posix(), providers=['CPUExecutionProvider'])
    actual = session.run(
        ['state_out'],
        {
            'state': state.numpy().astype(np.float32),
            'masks': masks.numpy().astype(np.float32),
        },
    )[0]
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)
