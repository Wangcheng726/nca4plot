"""Export fixed-size NCA graphs for ONNX Runtime Web and JSON weights.

Two kinds share the same small fully-convolutional architecture:
  texture  - square grid, RGB+0.5 offset, infinite drift (persistent mode);
  target   - rectangular grid, RGBA in [0,1], life-gated regrowth of a fixed
             painting (regenerate / growing modes).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image

from .model import CA, ExportCA, parameter_count


DEFAULT_STEPS = {128: 4, 256: 4, 384: 3, 512: 2}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_payload(checkpoint: Path) -> dict:
    payload = torch.load(checkpoint, map_location="cpu")
    if "model_state_dict" not in payload:
        raise KeyError(f"{checkpoint} does not contain model_state_dict")
    return payload


def load_model(checkpoint: Path, device: torch.device = torch.device("cpu")) -> CA:
    payload = load_payload(checkpoint)
    w1 = payload["model_state_dict"]["w1.weight"]
    hidden_n, in_ch = int(w1.shape[0]), int(w1.shape[1])
    arch = payload.get("architecture", {})
    ca = CA(in_ch // 4, hidden_n, kernel_scale=float(arch.get("kernel_scale", 1.0)))
    ca.load_state_dict(payload["model_state_dict"])
    ca.life_gating = bool(payload.get("life_gating", False))
    ca.to(device).eval()
    return ca


def save_delivery_checkpoint(ca: CA, source: Path, out_path: Path, source_payload: dict) -> dict:
    """Save a small inference/retraining checkpoint without optimizer state."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    training = {
        "step": int(source_payload.get("step", 0)),
        "seed": source_payload.get("seed"),
        "exemplar": source_payload.get("exemplar"),
        "exemplar_sha256": source_payload.get("exemplar_sha256"),
        "target": source_payload.get("target"),
        "target_size": source_payload.get("target_size"),
        "target_sha256": source_payload.get("target_sha256"),
        "life_gating": bool(source_payload.get("life_gating", False)),
        "training_config": source_payload.get("training_config"),
    }
    payload = {
        "format_version": 2,
        "architecture": {"chn": ca.chn, "hidden_n": ca.hidden_n,
                         "kernel_scale": ca.kernel_scale},
        "parameter_count": parameter_count(ca),
        "model_state_dict": {k: v.detach().cpu() for k, v in ca.state_dict().items()},
        "training": training,
        "source_full_checkpoint": source.name,
        "license": "Weights CC BY 4.0; code MIT; source artwork public domain",
    }
    torch.save(payload, out_path)
    return {
        "path": out_path.as_posix(),
        "bytes": out_path.stat().st_size,
        "sha256": sha256_file(out_path),
        "step": training["step"],
    }


def export_one(ca: CA, size: int, steps: int, out_path: Path, checkpoint: Path) -> dict:
    wrapper = ExportCA(ca, steps).eval()
    state = torch.zeros(1, ca.chn, size, size, dtype=torch.float32)
    gen = torch.Generator().manual_seed(20261119 + size)
    masks = (torch.rand(steps, 1, size, size, generator=gen) < 0.5).float()
    with torch.no_grad():
        expected = wrapper(state, masks).cpu().numpy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (state, masks),
        out_path.as_posix(),
        export_params=True,
        opset_version=19,
        do_constant_folding=True,
        input_names=["state", "masks"],
        output_names=["state_out"],
        dynamo=False,
    )

    model = onnx.load(out_path)
    onnx.checker.check_model(model)
    props = {
        "title": "nca4plot Starry Night texture NCA",
        "architecture": f"texture NCA {ca.chn}ch / {ca.hidden_n} hidden / four fixed kernels",
        "parameter_count": str(parameter_count(ca)),
        "size": str(size),
        "steps_per_inference": str(steps),
        "state_layout": "NCHW float32, first 3 channels RGB offset by +0.5",
        "mask_layout": "K x 1 x H x W, values 0 or 1, applied per cell",
        "training_checkpoint": checkpoint.name,
        "training_exemplar": "Starry Night sky crops (public domain source)",
        "license": "Code MIT; weights CC BY 4.0; source artwork public domain",
    }
    for key, value in props.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(model, out_path)

    sess = ort.InferenceSession(out_path.as_posix(), providers=["CPUExecutionProvider"])
    actual = sess.run(
        ["state_out"],
        {
            "state": state.numpy().astype(np.float32),
            "masks": masks.numpy().astype(np.float32),
        },
    )[0]
    diff = np.abs(actual - expected)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    if max_abs > 1e-4:
        raise AssertionError(f"ONNX parity failed for size {size}: max abs diff={max_abs}")
    return {
        "size": size,
        "steps": steps,
        "path": out_path.as_posix(),
        "bytes": out_path.stat().st_size,
        "sha256": sha256_file(out_path),
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
    }


def load_target_rgba(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGBA")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def export_target_one(
    ca: CA, width: int, height: int, steps: int, out_path: Path,
    checkpoint: Path, target_path: Path,
) -> dict:
    """Rectangular life-gated graph; parity checked on seed AND living states."""
    wrapper = ExportCA(ca, steps).eval()
    target = load_target_rgba(target_path)
    if tuple(target.shape[-2:]) != (height, width):
        raise AssertionError(f"target {target.shape} != {width}x{height}")
    seed = torch.zeros(1, ca.chn, height, width, dtype=torch.float32)
    seed[0, :4, height // 2, width // 2] = target[0, :4, height // 2, width // 2]
    gen = torch.Generator().manual_seed(20261120)
    masks = (torch.rand(steps, 1, height, width, generator=gen) < 0.5).float()
    living = torch.zeros(1, ca.chn, height, width, dtype=torch.float32)
    living[0, :4] = target[0, :4]
    inputs = [seed, living]
    with torch.no_grad():
        expected = [wrapper(x, masks).cpu().numpy() for x in inputs]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, (seed, masks), out_path.as_posix(),
        export_params=True, opset_version=19, do_constant_folding=True,
        input_names=["state", "masks"], output_names=["state_out"], dynamo=False,
    )
    model = onnx.load(out_path)
    onnx.checker.check_model(model)
    props = {
        "title": "nca4plot Starry Night target NCA (growing-CA style)",
        "architecture": f"target NCA {ca.chn}ch (RGBA+{ca.chn - 4} hidden) / {ca.hidden_n} hidden / four fixed kernels",
        "parameter_count": str(parameter_count(ca)),
        "kind": "target",
        "size": f"{width}x{height}",
        "steps_per_inference": str(steps),
        "state_layout": "NCHW float32; first 4 channels RGBA in [0,1] (no offset); zero = dead",
        "life_gate": "per step: x *= (maxpool3x3(alpha)>0.1) before AND after update",
        "mask_layout": "K x 1 x H x W, values 0 or 1, applied per cell",
        "training_checkpoint": checkpoint.name,
        "training_target": "whole Starry Night sky crop (public domain source)",
        "target_sha256": sha256_file(target_path),
        "license": "Code MIT; weights CC BY 4.0; source artwork public domain",
    }
    for key, value in props.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(model, out_path)

    sess = ort.InferenceSession(out_path.as_posix(), providers=["CPUExecutionProvider"])
    diffs = []
    for x, exp in zip(inputs, expected):
        actual = sess.run(["state_out"], {
            "state": x.numpy().astype(np.float32),
            "masks": masks.numpy().astype(np.float32),
        })[0]
        diffs.append(np.abs(actual - exp))
    max_abs = float(max(d.max() for d in diffs))
    mean_abs = float(max(d.mean() for d in diffs))
    if max_abs > 1e-4:
        raise AssertionError(f"target ONNX parity failed: max abs diff={max_abs}")
    return {
        "kind": "target",
        "size": f"{width}x{height}",
        "steps": steps,
        "path": out_path.as_posix(),
        "bytes": out_path.stat().st_size,
        "sha256": sha256_file(out_path),
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "parity_inputs": ["single_seed", "full_target_rgba"],
    }


def export_json_weights(ca: CA, out_path: Path) -> dict:
    tensors = {}
    for name, tensor in ca.state_dict().items():
        arr = tensor.detach().cpu().contiguous().numpy()
        tensors[name] = {"shape": list(arr.shape), "data": arr.reshape(-1).astype(float).tolist()}
    payload = {
        "format": "nca4plot-weights-v1",
        "parameter_count": parameter_count(ca),
        "architecture": {"chn": ca.chn, "hidden_n": ca.hidden_n,
                         "kernel_scale": ca.kernel_scale},
        "license": "CC BY 4.0",
        "tensors": tensors,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "path": out_path.as_posix(),
        "bytes": out_path.stat().st_size,
        "sha256": sha256_file(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/nca_starry.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--model-out", type=Path, default=Path("checkpoints/nca_starry_model.pt"))
    parser.add_argument("--web-dir", type=Path, default=Path("web/models"))
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--kind", choices=["texture", "target"], default="texture")
    parser.add_argument("--target-steps", type=int, default=4)
    parser.add_argument("--primary-target", default="256x144",
                        help="WxH whose export also writes the legacy un-suffixed aliases")
    args = parser.parse_args()

    source_payload = load_payload(args.checkpoint)
    ca = load_model(args.checkpoint)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    primary_wh = tuple(int(v) for v in args.primary_target.lower().split("x"))

    if args.kind == "target":
        ca.life_gating = True
        width, height = (int(v) for v in source_payload["target_size"])
        target_path = Path(source_payload.get("target") or "")
        if not target_path.exists():
            target_path = Path(f"data/target_sky_{width}x{height}.png")
        target_path = target_path.resolve()
        if args.model_out == Path("checkpoints/nca_starry_model.pt"):
            args.model_out = Path("checkpoints/nca_target_model.pt")
        model_report = save_delivery_checkpoint(ca, args.checkpoint, args.model_out, source_payload)
        print(json.dumps({"event": "model_checkpoint", **model_report}, ensure_ascii=False), flush=True)
        out = args.out_dir / f"nca_target_{width}x{height}.onnx"
        report = export_target_one(ca, width, height, args.target_steps, out, args.checkpoint, target_path)
        web_reports = []
        if args.web_dir is not None:
            args.web_dir.mkdir(parents=True, exist_ok=True)
            web_path = args.web_dir / out.name
            shutil.copy2(out, web_path)
            web_reports.append({"path": web_path.as_posix(), "bytes": web_path.stat().st_size})
            seed_image_name = f"nca_target_seed_{width}x{height}.png"
            shutil.copy2(target_path, args.web_dir / seed_image_name)
            web_reports.append({"path": (args.web_dir / seed_image_name).as_posix(),
                                "bytes": (args.web_dir / seed_image_name).stat().st_size})
            rgba = load_target_rgba(target_path)[0, :4, height // 2, width // 2].tolist()
            meta = {
                "kind": "target",
                "width": width,
                "height": height,
                "model": out.name,
                "steps_per_inference": args.target_steps,
                "seed_rgba": [round(v, 6) for v in rgba],
                "seed_image": seed_image_name,
                "chn": ca.chn,
                "hidden_n": ca.hidden_n,
                "kernel_scale": ca.kernel_scale,
                "parameter_count": parameter_count(ca),
                "target_sha256": sha256_file(target_path),
                "state_layout": "RGBA in [0,1], no offset; zero state = dead",
            }
            (args.web_dir / f"nca_target_meta_{width}x{height}.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8")
            if (width, height) == primary_wh:
                shutil.copy2(args.web_dir / seed_image_name,
                             args.web_dir / "nca_target_seed.png")
                legacy = dict(meta, seed_image="nca_target_seed.png")
                (args.web_dir / "nca_target_meta.json").write_text(
                    json.dumps(legacy, indent=2), encoding="utf-8")
        weights_report = export_json_weights(
            ca, args.out_dir / f"nca_target_weights_{width}x{height}.json")
        report_path = args.out_dir / f"export_target_report_{width}x{height}.json"
        if (width, height) == primary_wh:
            shutil.copy2(args.out_dir / f"nca_target_weights_{width}x{height}.json",
                         args.out_dir / "nca_target_weights.json")
        report_path.write_text(json.dumps({
            "source_checkpoint": args.checkpoint.as_posix(),
            "source_sha256": sha256_file(args.checkpoint),
            "source_step": int(source_payload.get("step", 0)),
            "model_checkpoint": model_report,
            "onnx": [report],
            "weights": weights_report,
            "web_models": web_reports,
        }, indent=2), encoding="utf-8")
        if (width, height) == primary_wh:
            shutil.copy2(report_path, args.out_dir / "export_target_report.json")
        print(json.dumps(report, ensure_ascii=False), flush=True)
        print(json.dumps({"event": "export_complete", "kind": "target",
                          "report": report_path.as_posix()}, ensure_ascii=False))
        return

    model_report = save_delivery_checkpoint(ca, args.checkpoint, args.model_out, source_payload)
    print(json.dumps({"event": "model_checkpoint", **model_report}, ensure_ascii=False), flush=True)

    reports = []
    web_reports = []
    for size in args.sizes:
        if size not in DEFAULT_STEPS:
            raise ValueError(f"unsupported size {size}; expected one of {sorted(DEFAULT_STEPS)}")
        steps = DEFAULT_STEPS[size]
        out = args.out_dir / f"nca_starry_{size}.onnx"
        report = export_one(ca, size, steps, out, args.checkpoint)
        reports.append(report)
        if args.web_dir is not None:
            args.web_dir.mkdir(parents=True, exist_ok=True)
            web_path = args.web_dir / out.name
            shutil.copy2(out, web_path)
            web_reports.append({"path": web_path.as_posix(), "bytes": web_path.stat().st_size})
        print(json.dumps(report, ensure_ascii=False), flush=True)

    texture_meta = {
        "kind": "texture",
        "chn": ca.chn,
        "hidden_n": ca.hidden_n,
        "parameter_count": parameter_count(ca),
        "sizes": list(args.sizes),
        "steps_per_inference": {str(s): DEFAULT_STEPS[s] for s in args.sizes},
    }
    (args.out_dir / "nca_starry_meta.json").write_text(
        json.dumps(texture_meta, indent=2), encoding="utf-8")
    if args.web_dir is not None:
        (args.web_dir / "nca_starry_meta.json").write_text(
            json.dumps(texture_meta, indent=2), encoding="utf-8")
    weights_report = export_json_weights(ca, args.out_dir / "nca_starry_weights.json")
    report_path = args.out_dir / "export_report.json"
    report_path.write_text(
        json.dumps(
            {
                "source_checkpoint": args.checkpoint.as_posix(),
                "source_sha256": sha256_file(args.checkpoint),
                "source_step": int(source_payload.get("step", 0)),
                "model_checkpoint": model_report,
                "onnx": reports,
                "weights": weights_report,
                "web_models": web_reports,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"event": "export_complete", "report": report_path.as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
