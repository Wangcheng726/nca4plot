"""Official-faithful Growing-CA trainer (Mordvintsev et al., Distill 2020).

Recipe from notebooks/growing_ca.ipynb ("Regenerating" experiment):
  * state: 16 channels (RGBA premultiplied + 12 hidden), life-gated
  * pool of 1024 states, batch 8; each iteration:
      - sort batch by current loss (desc), replace the worst with a fresh seed
      - damage the last --damage-n samples with random circle masks
        (official: center U(-0.5,0.5), radius U(0.1,0.4) in normalized coords)
      - roll out randint(64,96) steps, per-sample RGBA MSE vs padded target
      - per-tensor gradient normalisation g/(||g||+1e-8)
      - Adam 2e-3, x0.1 after 2000 iterations
      - commit rolled states back to the pool
  * --no-pool --damage-n 0 reproduces the official "Growing" experiment.

Checkpoint payload is compatible with export_onnx.load_model / export_target_one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .losses import normalize_gradients
from .model import CA, parameter_count, set_seed


def load_target_rgba(path: Path, max_size: int | None, pad: int, device) -> tuple[torch.Tensor, int, int]:
    img = Image.open(path).convert("RGBA")
    if max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr[..., :3] *= arr[..., 3:]  # official premultiply RGB by alpha
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    if pad:
        t = F.pad(t, (pad, pad, pad, pad))
    return t.to(device), int(t.shape[-1]), int(t.shape[-2])


def make_seed(chn: int, h: int, w: int, sx: int, sy: int, device) -> torch.Tensor:
    s = torch.zeros(1, chn, h, w, device=device)
    s[0, 3:, sy, sx] = 1.0
    return s


def circle_masks(n: int, h: int, w: int, device, r_lo: float = 0.1, r_hi: float = 0.4) -> torch.Tensor:
    """Official make_circle_masks: hard disc masks in normalized [-1,1] coords."""
    x = torch.linspace(-1.0, 1.0, w, device=device)[None, None, :]
    y = torch.linspace(-1.0, 1.0, h, device=device)[None, :, None]
    c = torch.rand(2, n, 1, 1, device=device) - 0.5
    r = torch.rand(n, 1, 1, device=device) * (r_hi - r_lo) + r_lo
    dx = (x - c[0]) / r
    dy = (y - c[1]) / r
    return (dx * dx + dy * dy < 1.0).float()[:, None]


def rgba_uint8(state: torch.Tensor) -> np.ndarray:
    """Premultiplied RGBA [0,1] -> uint8 RGB composited over white (official to_rgb)."""
    rgb = state[0, :3].clamp(0.0, 1.0)
    a = state[0, 3:4].clamp(0.0, 1.0)
    img = (1.0 - a + rgb).permute(1, 2, 0).detach().cpu().numpy()
    return (img * 255.0 + 0.5).astype(np.uint8)


def save_sheet(frames: list[tuple[str, torch.Tensor]], path: Path, zoom: int = 5) -> None:
    imgs = [(label, Image.fromarray(rgba_uint8(s))) for label, s in frames]
    w, h = imgs[0][1].size
    label_h, gap = 18, 6
    sheet = Image.new("RGB", (len(imgs) * (w * zoom + gap) + gap, h * zoom + label_h + gap), (250, 250, 248))
    d = ImageDraw.Draw(sheet)
    x = gap
    for label, im in imgs:
        sheet.paste(im.resize((w * zoom, h * zoom), Image.Resampling.NEAREST), (x, label_h + gap))
        d.text((x + 2, 3), label, fill=(30, 30, 30))
        x += w * zoom + gap
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


@torch.no_grad()
def growth_probe(ca: CA, seed: torch.Tensor, target: torch.Tensor, device,
                 marks=(8, 16, 32, 48, 64, 96, 128), probe_seed: int = 1234):
    gen = torch.Generator(device=device).manual_seed(probe_seed)
    x = seed.clone()
    frames = [("0", x.clone())]
    mse = {"0": float(((x[:, :4] - target) ** 2).mean())}
    for s in range(1, max(marks) + 1):
        mask = (torch.rand(1, 1, x.shape[-2], x.shape[-1], generator=gen, device=device) < 0.5).to(x.dtype)
        x = ca.step(x, mask)
        if s in marks:
            frames.append((str(s), x.clone()))
            mse[str(s)] = float(((x[:, :4] - target) ** 2).mean())
    return frames, mse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--target-max", type=int, default=None)
    ap.add_argument("--pad", type=int, default=0)
    ap.add_argument("--seed-xy", default=None, help="x,y; default: sidecar json seed_xy or grid center")
    ap.add_argument("--chn", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--kernel-scale", type=float, default=8.0)
    ap.add_argument("--kernels", type=int, default=3)
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--pool", type=int, default=1024)
    ap.add_argument("--no-pool", action="store_true")
    ap.add_argument("--damage-n", type=int, default=3)
    ap.add_argument("--damage-r-lo", type=float, default=0.1)
    ap.add_argument("--damage-r-hi", type=float, default=0.4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lr-drop", type=int, default=2000)
    ap.add_argument("--lr-factor", type=float, default=0.1)
    ap.add_argument("--steps-lo", type=int, default=64)
    ap.add_argument("--steps-hi", type=int, default=96)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    # Hidden-launch support (WMI-created processes have no console and do not
    # inherit the caller's env): tee all output into out/train.log, record pid.
    import os
    import sys

    class _Tee:
        def __init__(self, orig, fh):
            self.orig, self.fh = orig, fh

        def write(self, s):
            try:
                self.fh.write(s)
                self.fh.flush()
            except Exception:
                pass
            try:
                if self.orig is not None:
                    self.orig.write(s)
            except Exception:
                pass

        def flush(self):
            for f in (self.fh, self.orig):
                try:
                    if f is not None:
                        f.flush()
                except Exception:
                    pass

    _log_fh = (args.out / "train.log").open("a", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, _log_fh)
    sys.stderr = _Tee(sys.stderr, _log_fh)
    (args.out / "train.pid").write_text(str(os.getpid()), encoding="ascii")
    print(f"=== launch pid={os.getpid()} argv={sys.argv[1:]} ===", flush=True)

    target, w, h = load_target_rgba(args.target, args.target_max, args.pad, device)
    sx, sy = w // 2, h // 2
    sidecar = args.target.with_suffix(".json")
    if args.seed_xy:
        sx, sy = (int(v) for v in args.seed_xy.split(","))
    elif sidecar.exists() and not args.pad and not args.target_max:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if "seed_xy" in meta:
            sx, sy = (int(v) for v in meta["seed_xy"])
    seed = make_seed(args.chn, h, w, sx, sy, device)

    ca = CA(args.chn, args.hidden, args.kernel_scale, kernels=args.kernels, bias2=True)
    ca.life_gating = True
    ca.to(device)
    opt = torch.optim.Adam(ca.parameters(), lr=args.lr)
    start = 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu")
        ca.load_state_dict(payload["model_state_dict"])
        opt.load_state_dict(payload["optimizer_state_dict"])
        start = int(payload.get("step", 0))
        print(f"resumed from {args.resume} @ step {start}", flush=True)

    use_pool = not args.no_pool
    pool = seed.repeat(args.pool, 1, 1, 1).clone() if use_pool else None
    n_params = parameter_count(ca)
    print(f"target {w}x{h} seed=({sx},{sy}) params={n_params} pool={use_pool} damage_n={args.damage_n} device={device}", flush=True)

    sha = hashlib.sha256(args.target.read_bytes()).hexdigest()
    t0 = time.time()
    for it in range(start, args.iters):
        lr = args.lr if it < args.lr_drop else args.lr * args.lr_factor
        for g in opt.param_groups:
            g["lr"] = lr
        if use_pool:
            idx = torch.randperm(args.pool, device=device)[: args.batch]
            x0 = pool[idx].clone()
            with torch.no_grad():
                l0 = ((x0[:, :4] - target) ** 2).mean(dim=(1, 2, 3))
            order = l0.argsort(descending=True)
            x0, idx = x0[order], idx[order]
            x0[0] = seed[0]
            if args.damage_n:
                m = circle_masks(args.damage_n, h, w, device, args.damage_r_lo, args.damage_r_hi)
                x0[-args.damage_n:] *= 1.0 - m
        else:
            x0 = seed.repeat(args.batch, 1, 1, 1)
        n_steps = int(torch.randint(args.steps_lo, args.steps_hi, (1,)).item())
        x = x0
        for _ in range(n_steps):
            x = ca(x)
        per = ((x[:, :4] - target) ** 2).mean(dim=(1, 2, 3))
        loss = per.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        normalize_gradients(ca)
        opt.step()
        if use_pool:
            pool[idx] = x.detach()
        lv = float(loss.item())
        if (it + 1) % args.log_every == 0 or it == start:
            dt = time.time() - t0
            its = (it + 1 - start) / max(dt, 1e-9)
            eta = (args.iters - it - 1) / max(its, 1e-9) / 60.0
            print(f"it {it+1}/{args.iters} loss {lv:.6f} log10 {math.log10(max(lv,1e-12)):.2f} lr {lr:.1e} {its:.2f} it/s eta {eta:.1f}m", flush=True)
        if (it + 1) % args.ckpt_every == 0 or it + 1 == args.iters:
            frames, mse = growth_probe(ca, seed, target, device)
            save_sheet(frames, args.out / f"growth_{it+1:05d}.png")
            with (args.out / "growth_log.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": it + 1, "loss": lv, "mse": mse}) + "\n")
            payload = {
                "format_version": 2,
                "architecture": {"chn": ca.chn, "hidden_n": ca.hidden_n,
                                 "kernel_scale": ca.kernel_scale, "kernels": ca.kernels,
                                 "bias2": "w2.bias" in ca.state_dict()},
                "parameter_count": n_params,
                "model_state_dict": {k: v.detach().cpu() for k, v in ca.state_dict().items()},
                "optimizer_state_dict": opt.state_dict(),
                "step": it + 1,
                "life_gating": True,
                "target": args.target.as_posix(),
                "target_size": [w, h],
                "target_sha256": sha,
                "seed_xy": [sx, sy],
                "training_config": {k: (v.as_posix() if isinstance(v, Path) else v) for k, v in vars(args).items()},
            }
            torch.save(payload, args.out / "plant_latest.pt")
            if (it + 1) in (1000, 2000, 4000) or it + 1 == args.iters:
                torch.save(payload, args.out / f"plant_step{it+1:05d}.pt")
            print(f"  ckpt @ {it+1}: growth mse " + " ".join(f"{k}:{v:.5f}" for k, v in mse.items()), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
