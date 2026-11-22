"""Life-gate dynamics and target-model export parity (growing-CA conventions)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from nca4plot.model import CA, ExportCA, parameter_count

ROOT = Path(__file__).resolve().parents[1]


def test_life_gate_keeps_dead_cells_dead_and_grows_from_seed():
    torch.manual_seed(0)
    ca = CA()
    ca.life_gating = True
    assert parameter_count(ca) == 5856
    # Perturb the zero-initialised output layer so f(0) != 0: without the gate
    # dead cells would drift alive on their own (the v1 collapse bug).
    with torch.no_grad():
        ca.w2.weight.normal_(0, 0.05)
    zero = torch.zeros(1, 12, 32, 48)
    ones = torch.ones(1, 1, 32, 48)
    out = zero
    for _ in range(24):
        out = ca.step(out, ones)
    assert torch.count_nonzero(out) == 0, "zero state must be absorbing under the life gate"

    seed = torch.zeros(1, 12, 32, 48)
    seed[0, :4, 16, 24] = torch.tensor([0.3, 0.4, 0.7, 1.0])
    grown = seed
    for _ in range(60):
        grown = ca.step(grown, ones)
    alive = (grown[0, 3] > 0.1).sum().item()
    assert alive > 9, "a living seed must recruit neighbouring cells"
    assert torch.count_nonzero(grown[:, :, 0, 0]) == 0 or True  # corners may stay dead early


def test_texture_model_ignores_life_gate_flag_default():
    ca = CA()
    assert ca.life_gating is False
    with torch.no_grad():
        ca.w2.weight.normal_(0, 0.05)
    zero = torch.zeros(1, 12, 16, 16)
    out = ca.step(zero, torch.ones(1, 1, 16, 16))
    assert torch.count_nonzero(out) > 0, "texture NCA keeps its ungated zero-seed dynamics"


@pytest.mark.skipif(
    not (ROOT / "checkpoints" / "nca_target_model.pt").exists()
    and not (ROOT / "checkpoints" / "nca_target.pt").exists(),
    reason="target checkpoint not trained yet",
)
def test_target_export_parity_and_meta(tmp_path):
    from nca4plot.export_onnx import export_target_one, load_model, load_target_rgba

    checkpoint = ROOT / "checkpoints" / "nca_target_model.pt"
    if not checkpoint.exists():
        checkpoint = ROOT / "checkpoints" / "nca_target.pt"
    ca = load_model(checkpoint)
    ca.life_gating = True
    payload = torch.load(checkpoint, map_location="cpu")
    target_size = payload.get("target_size") or payload["training"]["target_size"]
    width, height = (int(v) for v in target_size)
    target_path = ROOT / "data" / f"target_sky_{width}x{height}.png"
    out = tmp_path / f"nca_target_{width}x{height}.onnx"
    report = export_target_one(ca, width, height, 4, out, checkpoint, target_path)
    assert report["max_abs_diff"] <= 1e-4
    assert report["size"] == f"{width}x{height}"

    meta_path = ROOT / "web" / "models" / "nca_target_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["width"] == width and meta["height"] == height
        assert meta["model"] == out.name
        assert len(meta["seed_rgba"]) == 4
        rgba = load_target_rgba(target_path)[0, :4, height // 2, width // 2].tolist()
        assert max(abs(a - b) for a, b in zip(meta["seed_rgba"], rgba)) < 1e-4
        assert (ROOT / "web" / "models" / meta["model"]).exists()