"""Train the 5,856-parameter texture NCA on a Starry Night sky crop."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .losses import VGGStyleLoss, normalize_gradients, overflow_loss
from .model import CA, parameter_count, set_seed, to_rgb
from .utils import (
    image_to_nchw,
    save_loss_plot,
    save_rollout_contact_sheet,
    state_buffer_summary,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_rollout(ca: CA, size: int, total_steps: int, sample_at: tuple[int, ...], seed: int) -> dict:
    gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    gen.manual_seed(seed)
    device = next(ca.parameters()).device
    x = ca.seed(1, size, device=device)
    wanted = sorted(set(sample_at))
    frames = []
    with torch.no_grad():
        for step in range(total_steps + 1):
            if step in wanted:
                frames.append(x.detach().cpu())
            if step == total_steps:
                break
            b, c, h, w = x.shape
            mask = (torch.rand(b, 1, h, w, generator=gen, device=device) < 0.5).to(x.dtype)
            x = ca.step(x, mask)
    return {"frames": frames, "indices": wanted, "summary": state_buffer_summary(x)}


def save_light_checkpoint(
    path: Path,
    ca: CA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    args: argparse.Namespace,
    loss_history: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_state_dict": {k: v.detach().cpu() for k, v in ca.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "architecture": {"chn": 12, "hidden_n": 96},
        "parameter_count": parameter_count(ca),
        "step": step,
        "seed": args.seed,
        "exemplar": args.exemplar.as_posix(),
        "exemplar_sha256": sha256_file(args.exemplar),
        "loss_history": loss_history,
        "training_config": {
            "pool_size": args.pool_size,
            "batch_size": args.batch_size,
            "train_size": args.train_size,
            "gradient_checkpointing": args.grad_checkpointing,
            "learning_rate": args.lr,
            "milestones": args.milestones,
            "gamma": args.gamma,
        },
        "device": str(next(ca.parameters()).device),
    }
    torch.save(payload, path)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exemplar", type=Path, default=Path("data/exemplar_128.png"))
    parser.add_argument("--exemplars", type=Path, nargs="+", default=None,
                        help="multiple sky crops; each batch sample matches a random one (larger training set)")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/nca_starry.pt"))
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--milestones", type=int, nargs="+", default=[1000, 2000])
    parser.add_argument("--gamma", type=float, default=0.3)
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

    args.exemplar = args.exemplar.resolve()
    args.out = args.out.resolve()
    if not args.exemplar.exists():
        raise FileNotFoundError(f"exemplar not found: {args.exemplar}")

    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()}")
        print(f"CUDA: {torch.version.cuda}; torch={torch.__version__}")

    ca = CA().to(device)
    pc = parameter_count(ca)
    if pc != 5856:
        raise AssertionError(f"unexpected parameter count: {pc}")
    optimizer = torch.optim.Adam(ca.parameters(), lr=args.lr, capturable=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.gamma
    )

    start_step = 0
    loss_history: list[float] = []
    pool = ca.seed(args.pool_size, args.train_size, device=device)
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
        print(f"resumed from {args.resume_state} at step {start_step}")

    exemplar_paths = list(args.exemplars) if args.exemplars else [args.exemplar]
    style_imgs = [image_to_nchw(p, device) for p in exemplar_paths]
    for p, im in zip(exemplar_paths, style_imgs):
        if tuple(im.shape[-2:]) != (args.train_size, args.train_size):
            raise ValueError(f"exemplar {p} must be {args.train_size}x{args.train_size}, got {im.shape[-2:]}")
    style_loss = VGGStyleLoss(device)
    target_features_list = [style_loss.target_features(im) for im in style_imgs]
    multi = len(style_imgs) > 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    (args.out.parent / "internal").mkdir(parents=True, exist_ok=True)
    sample_dir = args.out.parent / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out.parent / "training_log.jsonl"

    print(json.dumps({
        "event": "train_start", "device": str(device), "parameter_count": pc,
        "steps": args.steps, "resume_step": start_step, "exemplar": str(args.exemplar),
        "exemplars": [str(p) for p in exemplar_paths],
        "exemplar_sha256": sha256_file(args.exemplar),
        "gradient_checkpointing": args.grad_checkpointing,
    }, ensure_ascii=False), flush=True)

    started = time.perf_counter()
    for step in range(start_step, args.steps):
        with torch.no_grad():
            batch_idx = rng.choice(len(pool), args.batch_size, replace=False)
            x = pool[batch_idx].detach().clone()
            if step % 8 == 0:
                x[:1] = ca.seed(1, args.train_size, device=device)

        step_n = int(rng.integers(32, 96))
        if args.grad_checkpointing:
            x.requires_grad_(True)
            x = torch.utils.checkpoint.checkpoint_sequential(
                [ca] * step_n, args.checkpoint_chunk, x, use_reentrant=False
            )
        else:
            for _ in range(step_n):
                x = ca(x)

        rgb = to_rgb(x)
        if multi:
            assign = rng.integers(0, len(style_imgs), args.batch_size)
            style_term = style_loss.per_sample(rgb, target_features_list, assign)
        else:
            style_term = style_loss(rgb, target_features_list[0])
        loss = style_term + overflow_loss(x)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at step {step}: {loss.item()}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        normalize_gradients(ca)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            pool[batch_idx] = x.detach()
            loss_value = float(loss.detach().cpu())
            loss_history.append(loss_value)
            current_lr = float(optimizer.param_groups[0]["lr"])

        if step % args.log_every == 0 or step == args.steps - 1:
            elapsed = time.perf_counter() - started
            iters_done = step - start_step + 1
            speed = iters_done / max(elapsed, 1e-9)
            record = {
                "event": "step", "step": step + 1, "train_size": args.train_size,
                "step_n": step_n, "loss": loss_value, "lr": current_lr,
                "pool_rms": state_buffer_summary(pool[batch_idx]),
                "iterations_per_second": speed, "elapsed_seconds": elapsed,
                "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / 1024**2 if device.type == "cuda" else 0,
                "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / 1024**2 if device.type == "cuda" else 0,
            }
            print(json.dumps(record, ensure_ascii=False), flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if step % args.save_every == 0 or step == args.steps - 1:
            save_light_checkpoint(args.out, ca, optimizer, scheduler, step + 1, args, loss_history)
            torch.save({
                "model_state_dict": ca.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "pool": pool.cpu(),
                "step": step + 1,
                "numpy_rng_state": rng.bit_generator.state,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            }, args.out.parent / "internal" / "last_state.pt")
            torch.save({"model_state_dict": ca.state_dict(), "step": step + 1},
                       args.out.with_name(args.out.stem + "_latest.pt"))

        if step % args.sample_every == 0 or step == args.steps - 1:
            rollout = make_rollout(ca, args.train_size, 240, (0, 24, 80, 160, 240), args.seed + step + 1)
            save_rollout_contact_sheet(rollout["frames"], sample_dir / f"step_{step + 1:05d}_rollout.png")
            save_loss_plot(loss_history, sample_dir / f"step_{step + 1:05d}_loss.png")
            print(json.dumps({"event": "sample", "step": step + 1, "summary": rollout["summary"]}, ensure_ascii=False), flush=True)

    save_light_checkpoint(args.out, ca, optimizer, scheduler, args.steps, args, loss_history)
    save_loss_plot(loss_history, args.out.with_name("loss_curve.png"))
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(json.dumps({
        "event": "train_complete", "steps": args.steps,
        "seconds": time.perf_counter() - started,
        "checkpoint": str(args.out), "parameter_count": pc,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

