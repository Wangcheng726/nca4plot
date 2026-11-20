from __future__ import annotations

import io
from pathlib import Path

import matplotlib

# Windows default TkAgg backend crashes long trainings with Tcl_AsyncDelete
# ("main thread is not in main loop") when figures are created/destroyed.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def image_to_nchw(path: str | Path, device: torch.device | str = "cpu") -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    x = x.to(device=device, dtype=torch.float32)  # exemplar stays in [0,1] like official imread
    return x


def state_to_uint8(state: torch.Tensor) -> np.ndarray:
    x = (state[0, :3] + 0.5).detach().float().cpu().clamp(0.0, 1.0)
    return (x.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)


def save_state_png(state: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(state_to_uint8(state)).save(path)


def save_rollout_contact_sheet(states: list[torch.Tensor], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(state_to_uint8(s)) for s in states]
    w, h = imgs[0].size
    scale = max(1, 768 // w)
    imgs = [im.resize((w * scale, h * scale), Image.Resampling.NEAREST) for im in imgs]
    gap = 8
    sheet = Image.new("RGB", (sum(i.width for i in imgs) + gap * (len(imgs) - 1), imgs[0].height), (12, 14, 22))
    x = 0
    for im in imgs:
        sheet.paste(im, (x, 0))
        x += im.width + gap
    sheet.save(path, quality=95)


def save_gif(frames: list[torch.Tensor], path: str | Path, fps: int = 12) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(state_to_uint8(s)) for s in frames]
    duration = int(round(1000 / fps))
    imgs[0].save(
        path,
        save_all=True,
        append_images=imgs[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )


def save_loss_plot(history: list[float], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=160)
    ax.plot(history, color="#3155a5", linewidth=1.1)
    ax.set_yscale("log")
    ax.set_xlabel("training iteration")
    ax.set_ylabel("style + overflow loss")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def frames_to_mp4(frames: list[torch.Tensor], path: str | Path, fps: int = 20) -> None:
    """Write a small MP4 using imageio-ffmpeg if available; otherwise raise clearly."""
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("imageio is not installed; use GIF/contact sheet or install imageio[ffmpeg]") from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(state_to_uint8(frame))


def tensor_bytes_sha256(tensor: torch.Tensor) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def state_buffer_summary(state: torch.Tensor) -> dict[str, float]:
    x = state.detach().float()
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
        "rms": float(x.square().mean().sqrt()),
    }

