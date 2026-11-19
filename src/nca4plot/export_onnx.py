"""Export fixed-size texture-NCA graphs for ONNX Runtime Web and JSON weights."""

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

from .model import CA, ExportCA, parameter_count


DEFAULT_STEPS = {128: 4, 256: 4, 512: 2}


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
    ca = CA()
    ca.load_state_dict(payload["model_state_dict"])
    ca.to(device).eval()
    if parameter_count(ca) != 5856:
        raise AssertionError(parameter_count(ca))
    return ca


def save_delivery_checkpoint(ca: CA, source: Path, out_path: Path, source_payload: dict) -> dict:
    """Save a small inference/retraining checkpoint without optimizer state."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    training = {
        "step": int(source_payload.get("step", 0)),
        "seed": source_payload.get("seed"),
        "exemplar": source_payload.get("exemplar"),
        "exemplar_sha256": source_payload.get("exemplar_sha256"),
        "training_config": source_payload.get("training_config"),
    }
    payload = {
        "format_version": 2,
        "architecture": {"chn": 12, "hidden_n": 96},
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
    state = torch.zeros(1, 12, size, size, dtype=torch.float32)
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
        "architecture": "texture NCA 12ch / 96 hidden / four fixed kernels",
        "parameter_count": "5856",
        "size": str(size),
        "steps_per_inference": str(steps),
        "state_layout": "NCHW float32, first 3 channels RGB offset by +0.5",
        "mask_layout": "K x 1 x H x W, values 0 or 1, applied per cell",
        "training_checkpoint": checkpoint.name,
        "training_exemplar": "128px Starry Night sky crop (public domain source)",
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


def export_json_weights(ca: CA, out_path: Path) -> dict:
    tensors = {}
    for name, tensor in ca.state_dict().items():
        arr = tensor.detach().cpu().contiguous().numpy()
        tensors[name] = {"shape": list(arr.shape), "data": arr.reshape(-1).astype(float).tolist()}
    payload = {
        "format": "nca4plot-weights-v1",
        "parameter_count": parameter_count(ca),
        "architecture": {"chn": 12, "hidden_n": 96},
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
    args = parser.parse_args()

    source_payload = load_payload(args.checkpoint)
    ca = load_model(args.checkpoint)
    args.out_dir.mkdir(parents=True, exist_ok=True)

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
