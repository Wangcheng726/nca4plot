"""Render PyTorch NCA rollouts, including a genuine 12-channel erase and recovery test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .model import CA, parameter_count, set_seed
from .utils import save_gif, save_rollout_contact_sheet, save_state_png, state_buffer_summary


def erase_disc(state: torch.Tensor, cx: float, cy: float, radius: float, feather: float = 2.0) -> torch.Tensor:
    """Erase all CA channels, with a small soft rim in state space (no RGB fakery)."""
    _, _, h, w = state.shape
    yy = torch.arange(h, device=state.device, dtype=torch.float32)[:, None]
    xx = torch.arange(w, device=state.device, dtype=torch.float32)[None, :]
    dist = torch.sqrt((xx - cx).square() + (yy - cy).square())
    strength = ((dist - radius) / max(feather, 1e-3)).clamp(0.0, 1.0)
    strength = strength * strength * (3.0 - 2.0 * strength)
    return state * strength[None, None]


def load_ca(checkpoint: Path, device: torch.device) -> CA:
    payload = torch.load(checkpoint, map_location="cpu")
    ca = CA()
    ca.load_state_dict(payload["model_state_dict"])
    ca.to(device).eval()
    if parameter_count(ca) != 5856:
        raise AssertionError(parameter_count(ca))
    return ca


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/nca_starry.pt"))
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=420)
    parser.add_argument("--erase-at", type=int, default=180)
    parser.add_argument("--erase-ratio", type=float, default=0.12)
    parser.add_argument("--frame-every", type=int, default=4)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/rollouts"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    ca = load_ca(args.checkpoint, device)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    x = ca.seed(1, args.size, device=device)
    radius = args.size * args.erase_ratio
    frames = []
    pre_wound = None
    wound_stats = []
    with torch.no_grad():
        for step in range(args.steps + 1):
            if step == args.erase_at:
                save_state_png(x, args.out_dir / f"rollout_{args.size}_before_erase.png")
                pre_wound = x.detach().clone()
                x = erase_disc(x, args.size / 2, args.size / 2, radius)
                save_state_png(x, args.out_dir / f"rollout_{args.size}_erased.png")
            if step % args.frame_every == 0 or step in (args.erase_at, args.erase_at + 1):
                frames.append(x.detach().cpu())
            if step in (args.erase_at + 60, args.erase_at + 180, args.erase_at + 360):
                wound_stats.append({"step": step, "state": state_buffer_summary(x)})
            if step == args.steps:
                break
            b, c, h, w = x.shape
            mask = (torch.rand(b, 1, h, w, generator=gen, device=device) < 0.5).to(x.dtype)
            x = ca.step(x, mask)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_state_png(x, args.out_dir / f"rollout_{args.size}_final.png")
    picks = [
        frames[0],
        frames[min(len(frames) - 1, args.erase_at // args.frame_every)],
        frames[min(len(frames) - 1, args.erase_at // args.frame_every + 1)],
        frames[-1],
    ]
    save_rollout_contact_sheet(picks, args.out_dir / f"rollout_{args.size}_contact.png")
    save_gif(frames, args.out_dir / f"rollout_{args.size}.gif", fps=args.fps)

    report = {
        "size": args.size,
        "steps": args.steps,
        "erase_at": args.erase_at,
        "erase_radius": radius,
        "frames": len(frames),
        "final": state_buffer_summary(x),
        "window_stats": wound_stats,
    }
    (args.out_dir / f"rollout_{args.size}_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
