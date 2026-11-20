"""Train the 5,856-parameter NCA to regrow the whole Starry Night sky.

Growing-CA style (Mordvintsev, Randazzo, Niklasson & Levin, Distill 2020):
the same 12-channel / 96-hidden architecture as the texture model, but the
first four channels carry RGBA directly in [0,1] (zero state = transparent =
dead) and the loss is a plain RGBA MSE against a FIXED target image.
Half the pool boots as the living painting itself (RGBA = target, hidden 0),
exactly like the web regenerate mode, and learns stability plus regrowth of
random circular damage; the other half grows from a single seeded cell so the
growing mode still watches one cell become a sky.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .losses import VGGStyleLoss, normalize_gradients, overflow_loss_ms
from .model import CA, parameter_count, set_seed
from .utils import save_loss_plot, state_buffer_summary


NIGHT = np.array([0.004, 0.008, 0.016], dtype=np.float32)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_target(path: Path, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGBA")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def make_seed(target: torch.Tensor, n: int = 1) -> torch.Tensor:
    """All-zero state except the centre cell, which carries the target RGBA."""
    _, _, h, w = target.shape
    state = torch.zeros(n, 12, h, w, device=target.device, dtype=target.dtype)
    cy, cx = h // 2, w // 2
    state[:, :4, cy, cx] = target[:, :4, cy, cx]
    return state


def make_full(target: torch.Tensor, n: int = 1) -> torch.Tensor:
    """The living painting itself: RGBA = target, hidden channels zero.

    This is exactly the state the web app boots in regenerate mode, so the
    pool must contain it: stability on the full picture is a trained skill,
    not an emergent hope.
    """
    state = torch.zeros(n, 12, target.shape[2], target.shape[3],
                        device=target.device, dtype=target.dtype)
    state[:, :4] = target[:, :4]
    return state


def make_remnant(target: torch.Tensor, rng: np.random.Generator,
                 grids: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep a small central disc of the living painting, erase everything else.

    v8: the lone-seed arc only gets gradient on 1/8 of steps and its blob is
    already mean-colour fog by step 40. A remnant boot puts the wound weight
    on the DEAD frontier, so every regrowth step must lay down remembered
    stroke content as the front advances -- structured growth gradient on
    another 1/8 of steps, at half the rollout cost.
    """
    _, _, h, w = target.shape
    yy, xx = grids
    min_dim = min(h, w)
    cx = w * (0.5 + rng.uniform(-0.08, 0.08))
    cy = h * (0.5 + rng.uniform(-0.08, 0.08))
    r = rng.uniform(0.06, 0.22) * min_dim
    keep = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    state = make_full(target, 1)
    state[:, :, ~keep] = 0.0
    wound = (~keep).to(target.dtype).unsqueeze(0).unsqueeze(0)
    return state, wound


def alive_mask(state: torch.Tensor) -> torch.Tensor:
    alpha = state[:, 3:4]
    return (F.max_pool2d(alpha, 3, stride=1, padding=1) > 0.1).to(state.dtype)


def target_loss(state: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Official growing-CA loss: plain RGBA MSE over every pixel.

    Dead cells are exactly zero and the life gate zeroes their gradient, so
    only living cells contribute learning signal (no alive-count renormalising,
    which previously amplified early-growth gradients and favoured a flat
    mean-colour attractor).
    """
    return (state[:, :4] - target).square().mean()


def _sobel_kernels(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype)
    ky = kx.t()
    return torch.stack([kx, ky]).unsqueeze(1).repeat(4, 1, 1, 1)


def gradient_loss_map(state: torch.Tensor, target: torch.Tensor,
                      kernels: torch.Tensor) -> torch.Tensor:
    """Per-pixel Sobel-edge MSE on RGBA [b,1,h,w]: a flat mean-colour fill has
    no edges, so this term pushes regrowth to reproduce stroke structure."""
    gs = F.conv2d(state[:, :4], kernels, padding=1, groups=4)
    gt = F.conv2d(target[:, :4], kernels, padding=1, groups=4)
    return (gs - gt).square().mean(dim=1, keepdim=True)


def composite_rgb(state: torch.Tensor) -> torch.Tensor:
    """RGBA over deep night, exactly what the viewer sees."""
    night = torch.tensor(NIGHT, device=state.device, dtype=state.dtype).view(1, 3, 1, 1)
    return (state[:, :3] * state[:, 3:4] + night * (1.0 - state[:, 3:4])).clamp(0, 1)


def perceptual_loss_map(vgg: VGGStyleLoss, state: torch.Tensor,
                        target_feats: list[torch.Tensor],
                        wound: torch.Tensor) -> torch.Tensor:
    """Wound-weighted per-pixel VGG feature MSE, normalised per layer.

    Sobel only sees 3x3 edges; VGG sees stroke texture at five scales. Unlike
    OT (distribution matching, right for growing from zero), feature MSE keeps
    strokes at their remembered places -- the regrowth perceptual anchor.
    """
    feats = vgg.features(composite_rgb(state))
    total = None
    for f, ft in zip(feats[1:], target_feats[1:]):
        scale = ft.square().mean() + 1e-6
        w = F.interpolate(wound, size=f.shape[-2:], mode="area")
        layer = ((f - ft).square().mean(dim=1, keepdim=True)
                 * (1.0 + 7.0 * w)).mean() / scale
        total = layer if total is None else total + layer
    return total / len(target_feats[1:])


def damage_state(
    state: torch.Tensor,
    rng: np.random.Generator,
    grids: tuple[torch.Tensor, torch.Tensor],
    *,
    prob: float,
    max_circles: int = 3,
    heavy_prob: float = 0.0,
) -> torch.Tensor:
    """Zero all 12 channels inside random discs on a fraction of the batch.

    Returns (state, wound) where wound is a float [b,1,h,w] mask of the erased
    discs so the loss can up-weight regrowth instead of letting the tiny wound
    area vanish inside a whole-image mean (the flat grey-scar attractor).
    """
    b, _, h, w = state.shape
    yy, xx = grids
    min_dim = min(h, w)
    wound = torch.zeros(b, 1, h, w, device=state.device, dtype=state.dtype)
    for i in range(b):
        if rng.random() >= prob:
            continue
        for _ in range(int(rng.integers(1, max_circles + 1))):
            cx = rng.uniform(0.12, 0.88) * w
            cy = rng.uniform(0.12, 0.88) * h
            if rng.random() < heavy_prob:
                r = rng.uniform(0.22, 0.42) * min_dim  # near-total erase
            else:
                r = rng.uniform(0.06, 0.17) * min_dim
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
            state[i, :, mask] = 0.0
            wound[i, :, mask] = 1.0
    return state, wound


def rgba_to_uint8(state: torch.Tensor) -> np.ndarray:
    """Composite RGBA over deep night so wounds read as darkness."""
    rgba = state[0, :4].detach().cpu().numpy().clip(0, 1)
    rgb = rgba[:3].transpose(1, 2, 0)
    alpha = rgba[3:4].transpose(1, 2, 0)
    out = rgb * alpha + NIGHT * (1 - alpha)
    return (out * 255).astype(np.uint8)


def save_sheet(frames: list[torch.Tensor], labels: list[str], path: Path, scale: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tiles = [Image.fromarray(rgba_to_uint8(f)) for f in frames]
    w, h = tiles[0].size
    tw, th = w * scale, h * scale
    label_h = 22
    sheet = Image.new("RGB", (tw * len(tiles), th + label_h), (8, 12, 20))
    from PIL import ImageDraw

    draw = ImageDraw.Draw(sheet)
    for i, tile in enumerate(tiles):
        sheet.paste(tile.resize((tw, th), Image.Resampling.NEAREST), (i * tw, 0))
        draw.text((i * tw + 6, th + 5), labels[i], fill=(226, 216, 182))
    sheet.save(path)


def rollout_train(ca: CA, x: torch.Tensor, step_n: int, chunk: int,
                  use_ckpt: bool, mid_at: tuple[int, ...] = ()):
    """Rollout with grad, optionally checkpointed, returning interior states.

    v7: a single loss at the end of a 400-step seed rollout vanishes inside
    the flat-fog attractor; interior anchors keep gradient on the still
    structured growth front (official growing-CA lives at 32-96 steps).
    """
    mids: dict[int, torch.Tensor] = {}
    bounds = sorted({0, step_n, *mid_at})
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = [ca] * (b - a)
        if use_ckpt:
            # torch floor-divides len(seg) by the segment count; a segment
            # shorter than `chunk` would yield chunk size 0 -> range() error.
            nseg = max(1, min(chunk, b - a))
            x = torch.utils.checkpoint.checkpoint_sequential(
                seg, nseg, x, use_reentrant=False)
        else:
            for _ in range(b - a):
                x = ca(x)
        if b in mid_at:
            mids[b] = x
    return x, mids


@torch.no_grad()
def rollout(ca: CA, state: torch.Tensor, steps: int, sample_at: tuple[int, ...], seed: int):
    gen = torch.Generator(device=state.device)
    gen.manual_seed(seed)
    wanted = sorted(set(sample_at))
    frames = []
    x = state.clone()
    for step in range(steps + 1):
        if step in wanted:
            frames.append(x.detach().cpu())
        if step == steps:
            break
        b, _, h, w = x.shape
        mask = (torch.rand(b, 1, h, w, generator=gen, device=x.device) < 0.5).to(x.dtype)
        x = ca.step(x, mask)
    return frames, x


def evaluate(ca: CA, target: torch.Tensor, seed: int, out_dir: Path, tag: str) -> dict:
    _, _, h, w = target.shape
    min_dim = min(h, w)
    grown_frames, grown = rollout(ca, make_seed(target), 400, (0, 40, 120, 240, 400), seed)
    save_sheet(grown_frames, [f"step {s}" for s in (0, 40, 120, 240, 400)],
               out_dir / f"{tag}_growth.png")
    growth_mse = float(target_loss(grown, target))
    growth_alive = float(alive_mask(grown).mean())

    yy, xx = torch.meshgrid(
        torch.arange(h, device=target.device),
        torch.arange(w, device=target.device), indexing="ij")
    # Regrow on the LIVING PAINTING (what web regenerate mode actually does),
    # not on the grown fog -- the old sheet measured regen on grey fog.
    mature = make_full(target, 1)
    cx, cy, r = w * 0.5, h * 0.45, min_dim * 0.18
    disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    mature[:, :, disc] = 0.0
    regen_frames, regenerated = rollout(ca, mature, 200, (0, 30, 80, 150, 200), seed + 7)
    save_sheet(regen_frames, [f"+{s}" for s in (0, 30, 80, 150, 200)],
               out_dir / f"{tag}_regen.png")
    regen_mse = float(target_loss(regenerated, target))
    regen_alive = float(alive_mask(regenerated).mean())

    stable_frames, stable = rollout(ca, make_full(target), 400, (0, 100, 400), seed + 13)
    save_sheet(stable_frames, ["full 0", "full 100", "full 400"],
               out_dir / f"{tag}_stable.png")
    stable_mse = float(target_loss(stable, target))
    stable_alive = float(alive_mask(stable).mean())

    keep = (xx - w * 0.5) ** 2 + (yy - h * 0.45) ** 2 <= (min_dim * 0.11) ** 2
    rem = make_full(target, 1)
    rem[:, :, ~keep] = 0.0
    rem_frames, rem_end = rollout(ca, rem, 240, (80, 240), seed + 29)
    remnant_mse = float(target_loss(rem_end, target))
    save_sheet(rem_frames, ["rem 80", "rem 240"], out_dir / f"{tag}_remnant.png")

    comp_frames = [target.cpu(), stable.cpu(), grown.cpu(), regenerated.cpu()]
    save_sheet(comp_frames, ["TARGET", "STABLE 400", "GROWN 400", "REGEN 200"],
               out_dir / f"{tag}_comparison.png", scale=2)
    return {
        "stable_mse": stable_mse,
        "stable_alive_fraction": stable_alive,
        "growth_mse": growth_mse,
        "growth_alive_fraction": growth_alive,
        "regen_mse": regen_mse,
        "regen_alive_fraction": regen_alive,
        "remnant_mse": remnant_mse,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=Path("data/target_sky_256x144.png"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/nca_target.pt"))
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--milestones", type=int, nargs="+", default=[2000, 4000, 6000])
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--damage-prob", type=float, default=0.5)
    parser.add_argument("--heavy-prob", type=float, default=0.25)
    parser.add_argument("--mid-weight", type=float, default=0.5)
    parser.add_argument("--reset-seed-pool", action="store_true")
    parser.add_argument("--fresh-scheduler", action="store_true")
    parser.add_argument("--lr-now", type=float, default=None)
    parser.add_argument("--official-regime", action="store_true")
    parser.add_argument("--gate-mse", type=float, default=0.01,
                        help="write-back gate: finals below this mse re-enter the pool")
    parser.add_argument("--seed-step-min", type=int, default=0)
    parser.add_argument("--seed-step-max", type=int, default=0)
    parser.add_argument("--step-min", type=int, default=64)
    parser.add_argument("--step-max", type=int, default=96)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-chunk", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume-state", type=Path, default=None)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        args.steps = min(args.steps, 3)
        args.pool_size = min(args.pool_size, 8)

    args.target = args.target.resolve()
    args.out = args.out.resolve()
    if not args.target.exists():
        raise FileNotFoundError(f"target not found: {args.target}")

    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()}", flush=True)

    target = load_target(args.target, device)
    _, _, th, tw = target.shape
    ca = CA().to(device)
    ca.life_gating = True
    pc = parameter_count(ca)
    if pc != 5856:
        raise AssertionError(f"expected 5856 parameters, got {pc}")

    optimizer = torch.optim.Adam(ca.parameters(), lr=args.lr, capturable=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    start_step = 0
    loss_history: list[float] = []
    n_full = args.pool_size // 2
    pool = torch.cat([make_full(target, n_full),
                      make_seed(target, args.pool_size - n_full)], dim=0)
    rng = np.random.default_rng(args.seed)
    if args.resume_state is not None:
        state = torch.load(args.resume_state, map_location=device)
        ca.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        pool = state["pool"].to(device)
        start_step = int(state["step"])
        rng.bit_generator.state = state["numpy_rng_state"]
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if device.type == "cuda" and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng_state"]])
        print(f"resumed from {args.resume_state} at step {start_step}", flush=True)

    if args.resume_state is not None and args.fresh_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.milestones, gamma=args.gamma)
    if args.lr_now is not None:
        for g in optimizer.param_groups:
            g["lr"] = args.lr_now
        print(json.dumps({"event": "lr_override", "lr": args.lr_now}), flush=True)

    yy, xx = torch.meshgrid(
        torch.arange(th, device=device), torch.arange(tw, device=device), indexing="ij")
    grids = (yy, xx)

    if args.reset_seed_pool or start_step == 0:
        # v7: the old seed half of the pool collapsed into flat fog and kept
        # re-teaching that attractor. Reboot it as fresh seeds alternating
        # with heavily-erased paintings: sparse living remnants carry the
        # composition memory that a lone seed cell cannot.
        with torch.no_grad():
            for i in range(n_full, args.pool_size):
                kind = (i - n_full) % 3
                if kind == 0:
                    pool[i] = make_seed(target, 1)[0]
                elif kind == 1:
                    st = make_full(target, 1)
                    st, _ = damage_state(st, rng, grids, prob=1.0,
                                         max_circles=2, heavy_prob=1.0)
                    pool[i] = st[0]
                else:
                    st, _ = make_remnant(target, rng, grids)
                    pool[i] = st[0]
        print(json.dumps({"event": "pool_surgery", "step": start_step,
                          "seed_slots": args.pool_size - n_full}), flush=True)

    stem = args.out.stem
    args.out.parent.mkdir(parents=True, exist_ok=True)
    internal_dir = args.out.parent / f"internal_{stem}"
    internal_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = args.out.parent / f"samples_{stem}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out.parent / f"{stem}_training_log.jsonl"

    def save_light(step: int) -> None:
        payload = {
            "format_version": 1,
            "model_state_dict": {k: v.detach().cpu() for k, v in ca.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "architecture": {"chn": 12, "hidden_n": 96},
            "parameter_count": pc,
            "life_gating": True,
            "step": step,
            "seed": args.seed,
            "target": args.target.as_posix(),
            "target_sha256": sha256_file(args.target),
            "target_size": [tw, th],
            "loss_history": loss_history,
            "training_config": {
                "pool_size": args.pool_size, "batch_size": args.batch_size,
                "learning_rate": args.lr, "milestones": args.milestones, "gamma": args.gamma,
                "damage_prob": args.damage_prob, "step_range": [args.step_min, args.step_max],
                "gradient_checkpointing": args.grad_checkpointing,
            },
            "device": str(device),
        }
        torch.save(payload, args.out)

    print(json.dumps({
        "event": "train_start", "kind": "target", "device": str(device),
        "parameter_count": pc, "steps": args.steps, "target": str(args.target),
        "target_size": [tw, th], "target_sha256": sha256_file(args.target),
        "damage_prob": args.damage_prob,
    }, ensure_ascii=False), flush=True)

    sobel_kernels = _sobel_kernels(device, torch.float32)
    vgg = VGGStyleLoss(device)
    target_feats = vgg.target_features(composite_rgb(target))
    started = time.perf_counter()
    for step in range(start_step, args.steps):
        with torch.no_grad():
            batch_idx = rng.choice(len(pool), args.batch_size, replace=False)
            x = pool[batch_idx].detach().clone()
            fresh_seed = step % 8 == 0
            fresh_full = step % 8 == 2
            front = (not args.official_regime) and step % 8 in (4, 6)
            if fresh_seed:
                x[:1] = make_seed(target, 1)
            if fresh_full:
                x[:1] = make_full(target, 1)
            remnant = remnant_w = None
            if front:
                remnant, remnant_w = make_remnant(target, rng, grids)
                x[:1] = remnant
            x, wound = damage_state(x, rng, grids, prob=args.damage_prob,
                                    heavy_prob=args.heavy_prob)
            if fresh_seed:
                x[:1] = make_seed(target, 1)  # keep the fresh seed undamaged
                wound[:1] = 0.0
            if front:
                x[:1] = remnant  # keep the frontier boot undamaged
                wound[:1] = remnant_w
            if fresh_full and step % 8 == 2:
                x[:1] = make_full(target, 1)  # undamaged: pure stability
                wound[:1] = 0.0
            mse_b = (x[:, :4] - target[:, :4]).square().mean(dim=(1, 2, 3))
            for i in range(args.batch_size):
                if float(mse_b[i]) < 0.02 and rng.random() < 0.5:
                    sigma = float(rng.uniform(0.005, 0.03))
                    noise = rng.normal(0, sigma, (4, th, tw)).astype(np.float32)
                    x[i, :4] = (x[i, :4] + torch.from_numpy(noise).to(device)).clamp(0, 1)
                    hid = rng.normal(0, 0.05, (8, th, tw)).astype(np.float32)
                    x[i, 4:] += torch.from_numpy(hid).to(device)

        step_n = int(rng.integers(args.step_min, args.step_max))
        mid_at: tuple[int, ...] = ()
        if args.official_regime:
            step_n = int(rng.integers(args.step_min, args.step_max))
            mid_at = ()
            if fresh_seed and args.seed_step_max > 0:
                # v12: fresh-seed rollouts run past coverage so the model can
                # learn "cover first, then refine fog into the painting".
                step_n = int(rng.integers(args.seed_step_min, args.seed_step_max))
        elif fresh_seed:
            if rng.random() < 0.25:
                step_n = int(rng.integers(256, 384))  # late-drift anchor
                mid_at = (64, 192)
            else:
                # official growing-CA regime: train the 0-60 step phase
                # where the blob must lay down remembered content.
                step_n = int(rng.integers(48, 128))
                mid_at = tuple(m for m in (32, 64) if m < step_n)
        elif front:
            # v10: short frontier rollouts keep gradient on the living dead
            # boundary; 2/8 of steps now train propagation into dead space.
            step_n = int(rng.integers(96, 192))
            mid_at = tuple(m for m in (48, 96) if m < step_n)
        x.requires_grad_(True)
        x, mids = rollout_train(ca, x, step_n, args.checkpoint_chunk,
                                args.grad_checkpointing, mid_at)
        x = x.clamp(-4.0, 4.0)  # safety net: never let a runaway rollout reach VGG

        diff_sq = (x[:, :4] - target[:, :4]).square()
        weight = 1.0 + 7.0 * wound
        weighted_mse = (diff_sq * weight).mean()
        gmap = gradient_loss_map(x, target, sobel_kernels)
        loss_sobel = (gmap * weight).mean()
        loss_perc = perceptual_loss_map(vgg, x, target_feats, wound)
        loss_mid = torch.zeros((), device=device)
        zero_wound = torch.zeros_like(wound)
        for m in mids.values():
            m = m.clamp(-4.0, 4.0)
            loss_mid = loss_mid + (
                (m[:, :4] - target[:, :4]).square().mean()
                + 0.5 * gradient_loss_map(m, target, sobel_kernels).mean()
                + 0.05 * perceptual_loss_map(vgg, m, target_feats, zero_wound))
        loss = (weighted_mse + 0.5 * loss_sobel + 0.05 * loss_perc
                + args.mid_weight * loss_mid + overflow_loss_ms(x))
        if not bool(torch.isfinite(loss)) or float(loss.detach()) > 10.0:
            # Bad batch (runaway rollout): skip the update, keep the clean pool.
            print(json.dumps({"event": "skip", "step": step + 1,
                              "loss": float(loss.detach())}), flush=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        normalize_gradients(ca)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            # v10: fog finals written back re-polluted the pool (seed AND
            # front slots) and re-taught the mean-colour attractor. Only
            # painting-like finals enter the curriculum pool.
            final_mse = (x[:, :4] - target[:, :4]).square().mean(dim=(1, 2, 3))
            for i in range(args.batch_size):
                if float(final_mse[i]) < args.gate_mse:
                    pool[batch_idx[i]] = x[i].detach()
            if fresh_seed:
                pool[batch_idx[0]] = make_seed(target, 1)[0]
            if front:
                pool[batch_idx[0]], _ = make_remnant(target, rng, grids)
            loss_value = float(loss.detach().cpu())
            loss_history.append(loss_value)

        if step % args.log_every == 0 or step == args.steps - 1:
            elapsed = time.perf_counter() - started
            speed = (step - start_step + 1) / max(elapsed, 1e-9)
            record = {
                "event": "step", "step": step + 1, "step_n": step_n, "loss": loss_value,
                "loss_mse": float(weighted_mse.detach()), "loss_sobel": float(loss_sobel.detach()),
                "loss_perc": float(loss_perc.detach()),
                "loss_mid": float(loss_mid.detach()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "alive_fraction": float(alive_mask(x).mean()),
                "iterations_per_second": speed, "elapsed_seconds": elapsed,
                "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / 1024**2 if device.type == "cuda" else 0,
            }
            print(json.dumps(record, ensure_ascii=False), flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if step % args.save_every == 0 or step == args.steps - 1:
            save_light(step + 1)
            torch.save({
                "model_state_dict": ca.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "pool": pool.cpu(), "step": step + 1,
                "numpy_rng_state": rng.bit_generator.state,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            }, internal_dir / "last_state.pt")
            torch.save({"model_state_dict": ca.state_dict(), "step": step + 1},
                       args.out.with_name(stem + "_latest.pt"))

        if step % args.sample_every == 0 or step == args.steps - 1:
            metrics = evaluate(ca, target, args.seed + step + 1, sample_dir, f"step_{step + 1:05d}")
            save_loss_plot(loss_history, sample_dir / f"step_{step + 1:05d}_loss.png")
            print(json.dumps({"event": "sample", "step": step + 1, **metrics}, ensure_ascii=False), flush=True)

    save_light(args.steps)
    save_loss_plot(loss_history, args.out.with_name(f"{stem}_loss_curve.png"))
    final_metrics = evaluate(ca, target, args.seed + 99991, args.out.parent, stem)
    (args.out.parent / f"{stem}_eval.json").write_text(
        json.dumps(final_metrics, indent=2), encoding="utf-8")
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(json.dumps({
        "event": "train_complete", "steps": args.steps,
        "seconds": time.perf_counter() - started, "checkpoint": str(args.out),
        "parameter_count": pc, **final_metrics,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()