"""Model architecture, tensor-shape, and RGB calibration tests."""

from __future__ import annotations

import numpy as np
import torch

from nca4plot.model import CA, ExportCA, parameter_count, to_rgb
from nca4plot.utils import state_to_uint8


def test_parameter_count_is_exact() -> None:
    ca = CA()
    assert parameter_count(ca) == 5856


def test_production_configuration_parameter_count() -> None:
    # The delivered sky models use 16 channels / 128 hidden units.
    ca = CA(chn=16, hidden_n=128)
    assert parameter_count(ca) == 10368


def test_step_preserves_shape_and_is_finite_on_square_input() -> None:
    torch.manual_seed(4)
    ca = CA().eval()
    state = torch.randn(1, 12, 9, 9) * 0.05
    mask = torch.ones(1, 1, 9, 9)
    with torch.no_grad():
        out = ca.step(state, mask)
    assert out.shape == state.shape
    assert torch.isfinite(out).all()


def test_one_by_one_and_non_square_inputs_support_circular_padding() -> None:
    ca = CA().eval()
    with torch.no_grad():
        singleton = ca.step(torch.zeros(1, 12, 1, 1), torch.ones(1, 1, 1, 1))
        rectangular = ca.step(torch.zeros(1, 12, 7, 11), torch.ones(1, 1, 7, 11))
    assert singleton.shape == (1, 12, 1, 1)
    assert rectangular.shape == (1, 12, 7, 11)
    assert torch.isfinite(singleton).all()
    assert torch.isfinite(rectangular).all()


def test_seed_is_all_zero_and_exactly_twelve_channels() -> None:
    ca = CA()
    state = ca.seed(2, 16, device='cpu')
    assert state.shape == (2, 12, 16, 16)
    assert torch.count_nonzero(state) == 0


def test_multistep_export_wrapper_uses_each_mask() -> None:
    torch.manual_seed(9)
    ca = CA().eval()
    wrapper = ExportCA(ca, 3).eval()
    state = torch.randn(1, 12, 6, 6) * 0.02
    masks = torch.zeros(3, 1, 6, 6)
    masks[0] = 1.0
    with torch.no_grad():
        out = wrapper(state, masks)
    assert out.shape == state.shape
    assert torch.isfinite(out).all()


def test_rgb_state_uses_plus_half_offset() -> None:
    state = torch.zeros(1, 12, 2, 2)
    state[0, 0] = 1.0
    state[0, 1] = -1.0
    state[0, 2] = 0.0
    rgb = to_rgb(state)
    assert torch.allclose(rgb[0, :, 0, 0], torch.tensor([1.5, -0.5, 0.5]))
    pixels = state_to_uint8(state)
    assert pixels.shape == (2, 2, 3)
    assert pixels.dtype == np.uint8
    np.testing.assert_array_equal(pixels[0, 0], np.array([255, 0, 128], dtype=np.uint8))
