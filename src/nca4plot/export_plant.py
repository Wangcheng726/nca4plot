"""Export a plant checkpoint for the minimal web demo.

Writes web/models/<name>_<W>.onnx + <name>_meta_<W>.json + target preview PNG.
Parity is checked on BOTH the single-cell seed and a living 96-step state
(a zero-state check is vacuous for life-gated models).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image

from .export_onnx import load_model, load_payload, sha256_file
from .model import ExportCA, parameter_count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--web-dir", type=Path, default=Path("web/models"))
    ap.add_argument("--name", default="plant")
    ap.add_argument("--steps", type=int, default=4)
    args = ap.parse_args()

    payload = load_payload(args.checkpoint)
    ca = load_model(args.checkpoint)  # CPU, eval, life_gating from payload
    w, h = (int(v) for v in payload["target_size"])
    sx, sy = (int(v) for v in payload["seed_xy"])
    wrapper = ExportCA(ca, args.steps).eval()

    seed = torch.zeros(1, ca.chn, h, w, dtype=torch.float32)
    seed[0, 3:, sy, sx] = 1.0
    gen = torch.Generator().manual_seed(20261122)
    with torch.no_grad():
        x = seed.clone()
        for _ in range(96):
            m = (torch.rand(1, 1, h, w, generator=gen) < 0.5).float()
            x = ca.step(x, m)
        living = x.clone()
        masks = (torch.rand(args.steps, 1, h, w, generator=gen) < 0.5).float()
        expected = [wrapper(s, masks).numpy() for s in (seed, living)]

    args.web_dir.mkdir(parents=True, exist_ok=True)
    out = args.web_dir / f"{args.name}_{w}.onnx"
    torch.onnx.export(
        wrapper, (seed, masks), out.as_posix(),
        export_params=True, opset_version=19, do_constant_folding=True,
        input_names=["state", "masks"], output_names=["state_out"], dynamo=False,
    )
    model = onnx.load(out)
    onnx.checker.check_model(model)
    props = {
        "title": "Resilience: a plant that grows back (growing-CA recipe, Distill 2020)",
        "size": f"{w}x{h}",
        "steps_per_inference": str(args.steps),
        "state_layout": "NCHW float32; RGBA premultiplied in [0,1] + hidden; zero = dead",
        "mask_layout": "K x 1 x H x W, values 0 or 1, applied per cell",
        "seed": f"single cell ({sx},{sy}), channels[3:]=1",
        "training_checkpoint": args.checkpoint.name,
        "training_step": str(payload.get("step", 0)),
        "license": "Code MIT; weights CC BY 4.0; target artwork original (course project)",
    }
    for key, value in props.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(model, out)

    sess = ort.InferenceSession(out.as_posix(), providers=["CPUExecutionProvider"])
    parity = []
    for label, state, exp in (("seed", seed, expected[0]), ("living", living, expected[1])):
        act = sess.run(["state_out"], {"state": state.numpy(), "masks": masks.numpy()})[0]
        diff = np.abs(act - exp)
        max_abs = float(diff.max())
        if max_abs > 1e-4:
            raise AssertionError(f"ONNX parity failed on {label}: max abs diff={max_abs}")
        parity.append({"state": label, "max_abs_diff": max_abs, "mean_abs_diff": float(diff.mean())})

    # Target preview: original RGBA composited over white (what the CA aims at).
    target_path = Path(payload["target"])
    img = Image.open(target_path).convert("RGBA")
    if img.size != (w, h):
        raise AssertionError(f"target {img.size} != {(w, h)}")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.alpha_composite(img)
    preview = args.web_dir / f"{args.name}_target_{w}.png"
    bg.convert("RGB").save(preview)

    meta = {
        "kind": "target",
        "width": w,
        "height": h,
        "model": out.name,
        "steps_per_inference": args.steps,
        "chn": ca.chn,
        "hidden_n": ca.hidden_n,
        "kernel_scale": ca.kernel_scale,
        "kernels": ca.kernels,
        "parameter_count": parameter_count(ca),
        "seed_mode": "single_cell",
        "seed_xy": [sx, sy],
        "guard": False,
        "target_image": preview.name,
        "target_sha256": payload.get("target_sha256"),
        "source_checkpoint": args.checkpoint.name,
        "source_step": int(payload.get("step", 0)),
        "state_layout": "RGBA premultiplied in [0,1] + hidden; zero state = dead",
    }
    meta_path = args.web_dir / f"{args.name}_meta_{w}.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    report = {
        "source_checkpoint": args.checkpoint.as_posix(),
        "source_sha256": sha256_file(args.checkpoint),
        "source_step": int(payload.get("step", 0)),
        "onnx": {"path": out.as_posix(), "bytes": out.stat().st_size,
                 "sha256": sha256_file(out), "parity": parity},
        "meta": meta_path.as_posix(),
        "target_preview": preview.as_posix(),
    }
    report_path = args.checkpoint.parent / "export_plant_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
