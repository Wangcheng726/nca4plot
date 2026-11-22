"""Minimal plant UI: static contract checks (browser checks live in scripts/)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
UI_FILES = ("index.html", "styles.css", "js/app.js")


def test_ui_files_exist():
    for f in (*UI_FILES, "js/nca.js"):
        assert (WEB / f).exists(), f"missing {f}"


def test_ui_is_minimal():
    total = sum(len((WEB / f).read_text(encoding="utf-8").splitlines()) for f in UI_FILES)
    assert total < 400, f"UI grew to {total} lines; keep it minimal"


def test_no_starry_references_in_ui():
    for f in ("index.html", "js/app.js"):
        s = (WEB / f).read_text(encoding="utf-8").lower()
        assert "starry" not in s and "星空" not in s


def test_engine_supports_single_cell_seed():
    s = (WEB / "js/nca.js").read_text(encoding="utf-8")
    assert "single_cell" in s and "seed_xy" in s
    assert "this.stateData[c * this.n2 + idx] = 1.0" in s
    assert "this.guardEnabled && this.satFraction" in s


@pytest.mark.skipif(not (WEB / "models" / "plant_meta_80.json").exists(),
                    reason="plant web bundle not exported yet")
def test_meta_contract():
    meta = json.loads((WEB / "models" / "plant_meta_80.json").read_text(encoding="utf-8"))
    for key in ("kind", "width", "height", "model", "steps_per_inference", "chn",
                "parameter_count", "seed_mode", "seed_xy", "guard", "target_image"):
        assert key in meta, f"meta missing {key}"
    assert meta["kind"] == "target"
    assert (WEB / "models" / meta["model"]).exists()
    assert (WEB / "models" / meta["target_image"]).exists()
