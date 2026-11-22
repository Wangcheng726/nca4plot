"""User-style regenerate probe (v11 acceptance harness).

Reproduces what the visitor actually does: several feathered tears including
the moon and the big swirl, then a LONG run (default 13000 CA steps, matching
the user's screenshot step counter). Reports per-wound:
  rgba_mse   - exact recall of the painting inside the wound core
  orient_deg - mean brush-orientation mismatch vs target (structure tensor)
  speckle    - fraction of high-frequency residual pixels (dark/colour noise)
  dead       - fraction of cells whose alpha collapsed (<0.1)
  hid_sat    - fraction of hidden channels outside the trained |h|<=2 range
Saves full frames + 4x wound zoom sheets (wounded | final | target).

Usage: .venv\Scripts\python.exe scripts\probe_user_regen.py --ckpt checkpoints\nca_target_v3.pt --tag v10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nca4plot.model import CA  # noqa: E402
from nca4plot.train_target import load_target, _boundary, rgba_to_uint8  # noqa: E402

# Wound layout in 512x288 grid coords, mapped from the user's red boxes
# (screenshot 2000x1024): top star, big swirl, MOON, below-moon patch,
# plus a drag stroke like a real pointer tear.
WOUNDS = [
    {"name": "star_top", "cx": 306, "cy": 34, "r": 16},
    {"name": "swirl_big", "cx": 350, "cy": 96, "r": 30},
    {"name": "moon", "cx": 426, "cy": 84, "r": 20},
    {"name": "below_moon", "cx": 392, "cy": 148, "r": 14},
]
STROKES = [((250, 60), (330, 120), 8)]


def smoothstep_keep(d, r, feather=1.6):
    t = ((d - (r - feather)) / (2 * feather)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def erase_web(state, cx, cy, r, feather=1.6):
    _, _, h, w = state.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=state.device),
                            torch.arange(w, device=state.device), indexing="ij")
    d = ((xx - cx) ** 2 + (yy - cy) ** 2).sqrt()
    keep = smoothstep_keep(d, r, feather)
    return state * keep[None, None]


def erase_stroke(state, p0, p1, r):
    n = max(1, int(np.hypot(p1[0] - p0[0], p1[1] - p0[1]) / (r * 0.42)))
    for i in range(n + 1):
        t = i / n
        state = erase_web(state, p0[0] + (p1[0] - p0[0]) * t,
                          p0[1] + (p1[1] - p0[1]) * t, r)
    return state


def box(a, k):
    pad = (k // 2,) * 4
    return F.avg_pool2d(F.pad(a, pad, mode="replicate"), k, stride=1)


def structure(rgba, k=9):
    lum = rgba[:, :3].mean(1, keepdim=True)
    g = F.pad(lum, (1, 1, 1, 1), mode="replicate")
    gx = g[:, :, 1:-1, 2:] - g[:, :, 1:-1, :-2]
    gy = g[:, :, 2:, 1:-1] - g[:, :, :-2, 1:-1]
    jxx, jyy, jxy = box(gx * gx, k), box(gy * gy, k), box(gx * gy, k)
    tr = jxx + jyy
    disc = ((jxx - jyy) ** 2 + 4 * jxy * jxy + 1e-8).sqrt()
    theta = 0.5 * torch.atan2(2 * jxy, jxx - jyy + 1e-8)
    return theta, tr, disc / (tr + 1e-4)


def wound_metrics(out, tgt, cx, cy, r):
    _, _, h, w = out.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=out.device),
                            torch.arange(w, device=out.device), indexing="ij")
    core = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (0.8 * r) ** 2
    core = core[None, None]
    n = core.sum().clamp(min=1)
    mse = float(((out[:, :4] - tgt[:, :4]) ** 2 * core).sum() / (n * 4))
    to, tr_o, coh_o = structure(out[:, :4])
    tt, tr_t, coh_t = structure(tgt[:, :4])
    dth = (to - tt).abs()
    dth = torch.minimum(dth, np.pi - dth)
    wgt = (tr_t * coh_t * core)
    orient = float((dth * wgt).sum() / wgt.sum().clamp(min=1e-6) * 180 / np.pi)
    res = out[:, :3] - tgt[:, :3]
    hp = (res - box(res, 5)).abs().amax(1, keepdim=True)
    speckle = float(((hp > 0.15) & core).sum() / n)
    hp_t = (tgt[:, :3] - box(tgt[:, :3], 5)).abs().amax(1, keepdim=True)
    speckle_ref = float(((hp_t > 0.15) & core).sum() / n)
    lum_o = out[:, :3].mean(1, keepdim=True)
    lum_t = tgt[:, :3].mean(1, keepdim=True)
    dark = float((((lum_t - lum_o) > 0.25) & (out[:, 3:4] > 0.3) & core).sum() / n)
    dead = float(((out[:, 3:4] < 0.1) & core).sum() / n)
    hid_sat = float(((out[:, 4:].abs() > 2) & core).sum() / (n * (out.shape[1] - 4)))
    return {"rgba_mse": round(mse, 5), "orient_deg": round(orient, 1),
            "speckle": round(speckle, 4),
            "speckle_excess": round(max(0.0, speckle - speckle_ref), 4),
            "dark": round(dark, 4),
            "dead": round(dead, 4), "hid_sat": round(hid_sat, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=ROOT / "checkpoints/nca_target_v3.pt")
    ap.add_argument("--tag", default="v10")
    ap.add_argument("--steps", type=int, default=13000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out or ROOT / "checkpoints" / f"user_probe_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = load_target(ROOT / "data/target_sky_512x288.png", device)
    ck = torch.load(args.ckpt, map_location=device)
    arch = ck.get("architecture", {"chn": 16, "hidden_n": 128})
    ca = CA(arch["chn"], arch["hidden_n"],
            kernel_scale=float(arch.get("kernel_scale", 1.0))).to(device)
    ca.life_gating = True
    ca.load_state_dict(ck["model_state_dict"])
    ca.eval()
    print(f"ckpt={args.ckpt.name} step={ck.get('step')} params={sum(p.numel() for p in ca.parameters())}", flush=True)

    x = torch.zeros(1, arch["chn"], 288, 512, device=device)
    x[:, :4] = target[:, :4]
    wounded = x.clone()
    for wd in WOUNDS:
        wounded = erase_web(wounded, wd["cx"], wd["cy"], wd["r"])
    for p0, p1, r in STROKES:
        wounded = erase_stroke(wounded, p0, p1, r)
    x = wounded.clone()
    Image.fromarray(rgba_to_uint8(x)).save(out_dir / "frame_00000_wounded.png")

    samples = [0, 300, 1000, 3000, 6000, args.steps]
    report = {"ckpt": str(args.ckpt), "ckpt_step": ck.get("step"), "steps": args.steps,
              "wounds": WOUNDS, "samples": {}}
    gen = torch.Generator(device=device).manual_seed(1234)
    t0 = time.perf_counter()
    with torch.no_grad():
        for step in range(args.steps + 1):
            if step in samples:
                rgb = rgba_to_uint8(x)
                Image.fromarray(rgb).save(out_dir / f"frame_{step:05d}.png")
                entry = {"global_mse": round(float(((x[:, :4] - target[:, :4]) ** 2).mean()), 5),
                         "alive": round(float((x[:, 3:4] > 0.1).float().mean()), 4),
                         "wounds": {}}
                for wd in WOUNDS:
                    entry["wounds"][wd["name"]] = wound_metrics(x, target, wd["cx"], wd["cy"], wd["r"])
                report["samples"][str(step)] = entry
                print(json.dumps({"step": step, **entry}, ensure_ascii=False), flush=True)
            if step == args.steps:
                break
            mask = (torch.rand(1, 1, 288, 512, generator=gen, device=device) < 0.5).to(x.dtype)
            x = _boundary(ca.step(x, mask), None)
    report["seconds"] = round(time.perf_counter() - t0, 1)

    # 4x zoom sheets: wounded | final | target per wound
    fin = np.asarray(Image.open(out_dir / f"frame_{args.steps:05d}.png"))
    wou = np.asarray(Image.open(out_dir / "frame_00000_wounded.png"))
    tgt = np.asarray(Image.open(ROOT / "data/target_sky_512x288.png").convert("RGB"))
    tiles = []
    for wd in WOUNDS:
        cx, cy, r = wd["cx"], wd["cy"], wd["r"]
        half = max(24, r + 8)
        y0, y1 = max(0, cy - half), min(288, cy + half)
        x0, x1 = max(0, cx - half), min(512, cx + half)
        row = []
        for img in (wou, fin, tgt):
            crop = Image.fromarray(img[y0:y1, x0:x1]).resize(((x1 - x0) * 4, (y1 - y0) * 4), Image.Resampling.NEAREST)
            row.append(np.asarray(crop))
        tiles.append((wd["name"], np.hstack(row)))
    hh = max(t.shape[0] for _, t in tiles)
    ww = max(t.shape[1] for _, t in tiles)
    sheet = np.zeros((hh * len(tiles) + 20 * len(tiles), ww, 3), np.uint8)
    from PIL import ImageDraw
    y = 0
    imgs = []
    for name, t in tiles:
        canvas = Image.new("RGB", (ww, hh + 20), (8, 12, 20))
        canvas.paste(Image.fromarray(t), (0, 0))
        ImageDraw.Draw(canvas).text((6, hh + 4), f"{name}: wounded | final@{args.steps} | target", fill=(226, 216, 182))
        imgs.append(canvas)
    sheet_img = Image.new("RGB", (ww, sum(i.size[1] for i in imgs)), (8, 12, 20))
    y = 0
    for i in imgs:
        sheet_img.paste(i, (0, y)); y += i.size[1]
    sheet_img.save(out_dir / "zoom_wounds.png")
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PROBE DONE", out_dir, flush=True)


if __name__ == "__main__":
    main()
