"""Plant NCA: life gate, official-recipe helpers, checkpoint + web bundle parity."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from nca4plot.model import CA, ExportCA, parameter_count
from nca4plot.train_plant import circle_masks, load_target_rgba, make_seed, rgba_uint8

ROOT = Path(__file__).resolve().parents[1]
PLANT_CKPT = ROOT / "checkpoints" / "plant_A" / "plant_latest.pt"
WEB_MODELS = ROOT / "web" / "models"


def test_life_gate_keeps_dead_cells_dead_and_grows_from_seed():
    torch.manual_seed(0)
    ca = CA()
    ca.life_gating = True
    with torch.no_grad():
        ca.w2.weight.normal_(0, 0.05)  # f(0) != 0: gate must still absorb zero
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
    assert (grown[0, 3] > 0.1).sum().item() > 9, "a living seed must recruit neighbours"


def test_official_plant_configuration_parameter_count():
    ca = CA(16, 128, 8.0, kernels=3, bias2=True)
    assert parameter_count(ca) == 8336
    assert "w2.bias" in ca.state_dict()
    assert ca.perceive(torch.zeros(1, 16, 8, 8)).shape == (1, 48, 8, 8)


def test_target_is_premultiplied_and_seed_uses_official_channels(tmp_path):
    img = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    img.putpixel((1, 2), (255, 0, 0, 128))
    p = tmp_path / "t.png"
    img.save(p)
    t, w, h = load_target_rgba(p, None, 0, torch.device("cpu"))
    assert (w, h) == (4, 4)
    assert torch.allclose(t[0, :3, 2, 1], torch.tensor([128 / 255, 0.0, 0.0]), atol=1e-6)
    s = make_seed(16, 4, 4, 1, 2, torch.device("cpu"))
    assert float(s[0, 3:, 2, 1].min()) == 1.0
    assert float(s[0, :3].abs().sum()) == 0.0
    assert torch.count_nonzero(s).item() == 13  # alpha + 12 hidden channels


def test_circle_masks_are_hard_binary_discs():
    m = circle_masks(4, 32, 32, torch.device("cpu"))
    assert m.shape == (4, 1, 32, 32)
    assert set(m.unique().tolist()) <= {0.0, 1.0}
    assert float(m.sum()) > 0


def test_rgba_uint8_composites_over_white():
    s = torch.zeros(1, 16, 2, 2)
    s[0, 3] = 0.5
    s[0, 0] = 0.25  # premultiplied red
    px = rgba_uint8(s)
    assert px.shape == (2, 2, 3) and px.dtype == np.uint8
    assert px[0, 0, 0] == 191  # 1 - 0.5 + 0.25 = 0.75
    assert px[0, 0, 1] == 128


@pytest.mark.skipif(not PLANT_CKPT.exists(), reason="plant checkpoint not trained yet")
def test_plant_checkpoint_architecture():
    payload = torch.load(PLANT_CKPT, map_location="cpu")
    assert payload["architecture"]["kernels"] == 3
    assert payload["architecture"]["bias2"] is True
    assert payload["life_gating"] is True
    assert payload["parameter_count"] == 8336
    assert payload["seed_xy"] == [40, 64]


@pytest.mark.skipif(not (WEB_MODELS / "plant_meta_80.json").exists(),
                    reason="plant web bundle not exported yet")
def test_plant_web_bundle_meta_and_living_parity():
    import onnx
    import onnxruntime as ort

    from nca4plot.export_onnx import load_model

    meta = json.loads((WEB_MODELS / "plant_meta_80.json").read_text(encoding="utf-8"))
    assert meta["seed_mode"] == "single_cell"
    assert meta["guard"] is False
    assert meta["chn"] == 16 and meta["parameter_count"] == 8336
    assert meta["steps_per_inference"] == 4
    onnx_path = WEB_MODELS / meta["model"]
    assert onnx_path.exists()
    assert (WEB_MODELS / meta["target_image"]).exists()
    model = onnx.load(onnx_path.as_posix())
    onnx.checker.check_model(model)
    props = {e.key: e.value for e in model.metadata_props}
    assert props["size"] == "80x80"
    assert "CC BY 4.0" in props["license"]

    ca = load_model(PLANT_CKPT)
    wrapper = ExportCA(ca, meta["steps_per_inference"]).eval()
    w, h = meta["width"], meta["height"]
    sx, sy = meta["seed_xy"]
    seed = torch.zeros(1, ca.chn, h, w)
    seed[0, 3:, sy, sx] = 1.0
    gen = torch.Generator().manual_seed(7)
    with torch.no_grad():
        x = seed.clone()
        for _ in range(48):
            x = ca.step(x, (torch.rand(1, 1, h, w, generator=gen) < 0.5).float())
        masks = (torch.rand(meta["steps_per_inference"], 1, h, w, generator=gen) < 0.5).float()
        expected = wrapper(x, masks).numpy()
    sess = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
    actual = sess.run(["state_out"], {"state": x.numpy(), "masks": masks.numpy()})[0]
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-4)
