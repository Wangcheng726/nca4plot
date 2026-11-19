"""Minimal texture NCA matching the official Distill / Google Research notebook."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn


IDENTITY = torch.tensor(
    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
)
SOBEL_X = torch.tensor(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
)
LAPLACIAN = torch.tensor(
    [[1.0, 2.0, 1.0], [2.0, -12.0, 2.0], [1.0, 2.0, 1.0]]
)


def perception_filters(device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return the four fixed per-channel perception kernels, shape [4, 3, 3]."""
    return torch.stack([IDENTITY, SOBEL_X, SOBEL_X.T, LAPLACIAN]).to(device=device, dtype=dtype)


def circular_pad_3x3(x: torch.Tensor) -> torch.Tensor:
    """Circular padding implemented with Cat/Slice so its ONNX graph is portable."""
    # width/height may be as small as 1; the repeated edge slices still work.
    x = torch.cat((x[..., -1:, :], x, x[..., :1, :]), dim=-2)
    x = torch.cat((x[..., :, -1:], x, x[..., :, :1]), dim=-1)
    return x


def perchannel_conv(x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    """Apply the same bank of filters independently to every channel."""
    b, ch, h, w = x.shape
    y = x.reshape(b * ch, 1, h, w)
    y = circular_pad_3x3(y)
    y = F.conv2d(y, filters[:, None])
    return y.reshape(b, -1, h, w)


class CA(nn.Module):
    """The 5,856-parameter texture neural cellular automaton."""

    def __init__(self, chn: int = 12, hidden_n: int = 96):
        super().__init__()
        self.chn = int(chn)
        self.hidden_n = int(hidden_n)
        self.w1 = nn.Conv2d(chn * 4, hidden_n, 1)
        self.w2 = nn.Conv2d(hidden_n, chn, 1, bias=False)
        self.w2.weight.data.zero_()

    def perceive(self, x: torch.Tensor) -> torch.Tensor:
        return perchannel_conv(x, perception_filters(x.device, x.dtype))

    def step(self, x: torch.Tensor, update_mask: torch.Tensor) -> torch.Tensor:
        """One update with an externally supplied per-cell [B,1,H,W] mask."""
        y = self.w2(torch.relu(self.w1(self.perceive(x))))
        return x + y * update_mask

    def forward(self, x: torch.Tensor, update_rate: float = 0.5) -> torch.Tensor:
        b, c, h, w = x.shape
        update_mask = (torch.rand(b, 1, h, w, device=x.device, dtype=x.dtype) + update_rate).floor()
        return self.step(x, update_mask)

    def seed(self, n: int, sz: int = 128, device: torch.device | str | None = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(n, self.chn, sz, sz, device=device)


class ExportCA(nn.Module):
    """Fixed-size ONNX wrapper: K externally masked CA updates in one inference call."""

    def __init__(self, ca: CA, steps: int):
        super().__init__()
        self.ca = ca
        self.steps = int(steps)

    def forward(self, state: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        x = state
        for k in range(self.steps):
            x = self.ca.step(x, masks[k : k + 1])
        return x


def to_rgb(state: torch.Tensor) -> torch.Tensor:
    """Convert the first three CA channels to [0,1] RGB."""
    return state[:, :3] + 0.5


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def finite_or_raise(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"{name} contains NaN or Inf")


def rolled_state_norm(state: torch.Tensor) -> float:
    return float(state.float().square().mean().sqrt().item())


def make_random_masks(
    steps: int,
    height: int,
    width: int,
    *,
    probability: float = 0.5,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Convenience helper for tests/export; the browser makes masks in JS."""
    return (
        torch.rand(steps, 1, height, width, generator=generator, device=device) < probability
    ).to(torch.float32)
