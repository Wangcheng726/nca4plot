"""Acceptance probes for the plant NCA (maps to the team's three damage tests).

  growth   : fixed seed -> 128 steps, contact sheet + MSE curve + mp4
  test 1   : no damage, continue to 2048 steps (stability / persistent behavior)
  test 2   : erase a small disc (leaf), watch recovery up to 256 steps
  test 3   : erase a large disc (stem + leaves), watch recovery up to 256 steps

All rollouts use a fixed RNG so results are reproducible. Failures are reported
honestly in the JSON report (the course brief requires truthful records).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .export_onnx import load_model
from .model import set_seed
from .train_plant import load_target_rgba, make_seed, rgba_uint8, save_sheet


def mse(x: torch.Tensor, t: torch.Tensor) -> float:
    return float(((x[:, :4] - t) ** 2).mean())


def erase_disc(x: torch.Tensor, cx: float, cy: float, r: float) -> torch.Tensor:
    h, w = x.shape[-2:]
    yy = torch.arange(h, device=x.device, dtype=torch.float32)[:, None]
    xx = torch.arange(w, device=x.device, dtype=torch.float32)[None, :]
    m = ((xx - cx) ** 2 + (yy - cy) ** 2 < r * r).float()
    return x * (1.0 - m)[None, None]


@torch.no_grad()
def run(ca, x, n, gen, device, capture=()):
    frames = {}
    for s in range(1, n + 1):
        mask = (torch.rand(1, 1, x.shape[-2], x.shape[-1], generator=gen, device=device) < 0.5).to(x.dtype)
        x = ca.step(x, mask)
        if s in capture:
            frames[s] = x.clone()
    return x, frames


def save_mp4(frames: list[torch.Tensor], path: Path, fps: int = 20) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as wr:
        for f in frames:
            wr.append_data(rgba_uint8(f))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--small-xy", default="45,50")
    ap.add_argument("--small-r", type=float, default=6.0)
    ap.add_argument("--big-xy", default="40,44")
    ap.add_argument("--big-r", type=float, default=16.0)
    ap.add_argument("--long-steps", type=int, default=2048)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(0)
    device = torch.device(args.device)
    ca = load_model(args.checkpoint, device)
    payload = torch.load(args.checkpoint, map_location="cpu")
    cfg = payload.get("training_config", {})
    target, w, h = load_target_rgba(Path(payload["target"]), cfg.get("target_max"), int(cfg.get("pad", 0)), device)
    sx, sy = payload.get("seed_xy", [w // 2, h // 2])
    seed = make_seed(ca.chn, h, w, sx, sy, device)
    out = args.out or args.checkpoint.parent / "probe"
    out.mkdir(parents=True, exist_ok=True)
    report = {"checkpoint": args.checkpoint.as_posix(), "step": payload.get("step"), "grid": [w, h]}

    # --- growth ---
    gen = torch.Generator(device=device).manual_seed(1234)
    marks = [8, 16, 32, 48, 64, 96, 128]
    x, frames = run(ca, seed.clone(), 128, gen, device, capture=marks)
    sheet = [("0", seed)] + [(str(s), frames[s]) for s in marks]
    save_sheet(sheet, out / "growth.png")
    report["growth_mse"] = {str(s): mse(frames[s], target) for s in marks}
    gif_frames = [seed]
    gen2 = torch.Generator(device=device).manual_seed(1234)
    xx = seed.clone()
    for s in range(1, 129):
        mask = (torch.rand(1, 1, h, w, generator=gen2, device=device) < 0.5).to(xx.dtype)
        xx = ca.step(xx, mask)
        if s % 2 == 0:
            gif_frames.append(xx.clone())
    save_mp4(gif_frames, out / "growth.mp4")

    # --- test 1: no damage, long run ---
    long_marks = [256, 512, 1024, args.long_steps]
    x2, lf = run(ca, x.clone(), args.long_steps - 128, gen, device, capture=[m - 128 for m in long_marks])
    report["long_mse"] = {str(m): mse(lf[m - 128], target) for m in long_marks}
    report["long_finite"] = bool(torch.isfinite(x2).all().item())
    save_sheet([(str(m), lf[m - 128]) for m in long_marks], out / "long.png")
    long_frames = [x.clone()]
    gen3 = torch.Generator(device=device).manual_seed(1234)
    xx = x.clone()
    for s in range(1, args.long_steps - 128 + 1):
        mask = (torch.rand(1, 1, h, w, generator=gen3, device=device) < 0.5).to(xx.dtype)
        xx = ca.step(xx, mask)
        if s % 8 == 0:
            long_frames.append(xx.clone())
    save_mp4(long_frames, out / "long.mp4", fps=30)

    # --- tests 2 & 3: damage after full growth ---
    for name, xy, r in (("small", args.small_xy, args.small_r), ("big", args.big_xy, args.big_r)):
        cx, cy = (float(v) for v in xy.split(","))
        g = torch.Generator(device=device).manual_seed(1234)
        xg, _ = run(ca, seed.clone(), 96, g, device)
        before = xg.clone()
        xd = erase_disc(xg, cx, cy, r)
        cap = [1, 16, 32, 64, 128, 256]
        xr, df = run(ca, xd, 256, g, device, capture=cap)
        curve = {str(s): mse(df[s], target) for s in cap}
        rec = next((s for s in cap if curve[str(s)] < 2e-3), None)
        report[f"damage_{name}"] = {"center": [cx, cy], "radius": r,
                                    "mse_before": mse(before, target), "mse_after_erase": mse(xd, target),
                                    "mse_curve": curve, "recovered_by_step": rec,
                                    "final_finite": bool(torch.isfinite(xr).all().item())}
        save_sheet([("grown", before), ("erased", xd)] + [(f"+{s}", df[s]) for s in cap], out / f"damage_{name}.png")
        dmg_frames = [before, xd]
        g2 = torch.Generator(device=device).manual_seed(1234)
        xx = erase_disc(run(ca, seed.clone(), 96, g2, device)[0], cx, cy, r)
        for s in range(1, 257):
            mask = (torch.rand(1, 1, h, w, generator=g2, device=device) < 0.5).to(xx.dtype)
            xx = ca.step(xx, mask)
            if s % 2 == 0:
                dmg_frames.append(xx.clone())
        save_mp4(dmg_frames, out / f"damage_{name}.mp4")

    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
