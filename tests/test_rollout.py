"""Genuine all-channel erasure and continued inference on the plant model."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from nca4plot.model import set_seed
from nca4plot.rollout import erase_disc, load_ca
from nca4plot.train_plant import make_seed

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "checkpoints" / "plant_A" / "plant_latest.pt"


def test_erase_disc_clears_every_channel_at_center() -> None:
    state = torch.ones(1, 12, 33, 33)
    erased = erase_disc(state, 16.0, 16.0, 4.0, feather=1.0)
    assert torch.count_nonzero(erased[..., 16, 16]) == 0
    assert torch.allclose(erased[..., 0, 0], torch.ones(12))


@pytest.mark.skipif(not MODEL.exists(), reason="plant checkpoint not trained yet")
def test_plant_checkpoint_loads_and_infers_after_erase() -> None:
    set_seed(12)
    ca = load_ca(MODEL, torch.device("cpu"))
    assert ca.life_gating is True
    state = make_seed(ca.chn, 40, 40, 20, 32, torch.device("cpu"))
    with torch.no_grad():
        for _ in range(12):
            mask = (torch.rand(1, 1, 40, 40) < 0.5).float()
            state = ca.step(state, mask)
        assert torch.count_nonzero(state) > 0
        before = state.clone()
        state = erase_disc(state, 20.0, 20.0, 5.0, feather=1.5)
        assert torch.count_nonzero(state[..., 20, 20]) == 0
        assert not torch.equal(state, before)
        next_state = ca.step(state, torch.ones(1, 1, 40, 40))
    assert next_state.shape == state.shape
    assert torch.isfinite(next_state).all()
