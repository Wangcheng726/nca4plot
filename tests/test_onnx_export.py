"""Validate delivered ONNX graphs, metadata, and PyTorch/ORT parity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from nca4plot.export_onnx import DEFAULT_STEPS, load_model
from nca4plot.model import ExportCA, make_random_masks


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'checkpoints' / 'nca_starry_model.pt'
WEB_MODEL_DIR = ROOT / 'web' / 'models'


def test_all_onnx_sizes_exist_and_have_complete_metadata() -> None:
    for size, steps in DEFAULT_STEPS.items():
        path = ROOT / 'checkpoints' / f'nca_starry_{size}.onnx'
        assert path.exists(), f'missing {path}'
        model = onnx.load(path.as_posix())
        onnx.checker.check_model(model)
        props = {entry.key: entry.value for entry in model.metadata_props}
        assert props['size'] == str(size)
        assert props['steps_per_inference'] == str(steps)
        assert props['parameter_count'] == '5856'
        assert 'CC BY 4.0' in props['license']
        assert path.read_bytes() == (WEB_MODEL_DIR / path.name).read_bytes()


def test_256_pytorch_onnx_parity() -> None:
    ca = load_model(CHECKPOINT)
    wrapper = ExportCA(ca, DEFAULT_STEPS[256]).eval()
    size = 256
    generator = torch.Generator().manual_seed(20261119)
    state = torch.randn(1, 12, size, size, generator=generator) * 0.05
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
