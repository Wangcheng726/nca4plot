"""Tests for genuine all-channel erasure and continued inference."""

from __future__ import annotations

from pathlib import Path

import torch

from nca4plot.model import CA, set_seed
from nca4plot.rollout import erase_disc, load_ca


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'checkpoints' / 'nca_starry_model.pt'


def test_erase_disc_clears_every_channel_at_center() -> None:
    state = torch.ones(1, 12, 33, 33)
    erased = erase_disc(state, 16.0, 16.0, 4.0, feather=1.0)
    assert torch.count_nonzero(erased[..., 16, 16]) == 0
    assert torch.allclose(erased[..., 0, 0], torch.ones(12))


def test_delivery_checkpoint_loads_and_infers_after_erase() -> None:
    assert MODEL.exists(), 'run export_onnx before the test suite'
    set_seed(12)
    ca = load_ca(MODEL, torch.device('cpu'))
    assert isinstance(ca, CA)
    state = ca.seed(1, 32, device='cpu')
    with torch.no_grad():
        for _ in range(12):
            mask = (torch.rand(1, 1, 32, 32) < 0.5).float()
            state = ca.step(state, mask)
        assert torch.count_nonzero(state) > 0
        before = state.clone()
        state = erase_disc(state, 16.0, 16.0, 5.0, feather=1.5)
        assert torch.count_nonzero(state[..., 16, 16]) == 0
        assert not torch.equal(state, before)
        mask = torch.ones(1, 1, 32, 32)
        next_state = ca.step(state, mask)
    assert next_state.shape == state.shape
    assert torch.isfinite(next_state).all()
    assert torch.count_nonzero(next_state) > 0
