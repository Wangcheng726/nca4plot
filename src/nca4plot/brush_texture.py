"""Generate a seamless procedural brush/impasto material texture."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def fft_band_noise(rng: np.random.Generator, n: int, sx: float, sy: float) -> np.ndarray:
    """Periodic anisotropic noise; small sx/sy suppress frequencies along that axis."""
    white = rng.normal(size=(n, n))
    spec = np.fft.fft2(white)
    ky = np.fft.fftfreq(n)[:, None] * n
    kx = np.fft.fftfreq(n)[None, :] * n
    weight = np.exp(-0.5 * (kx / sx) ** 2 - 0.5 * (ky / sy) ** 2)
    out = np.fft.ifft2(spec * weight).real
    out -= out.min()
    return out / max(out.max(), 1e-8)


def normalize(x: np.ndarray) -> np.ndarray:
    x = x - x.min()
    return x / max(x.max(), 1e-8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("web/assets/textures/brush.png"))
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1889)
    args = parser.parse_args()

    n = args.size
    rng = np.random.default_rng(args.seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)

    long_streaks = fft_band_noise(rng, n, 1.4, 22.0)
    medium_streaks = fft_band_noise(rng, n, 3.4, 38.0)
    broad = fft_band_noise(rng, n, 14.0, 30.0)
    micro = fft_band_noise(rng, n, 100.0, 100.0)

    warp = normalize(broad) * 2.0 - 1.0
    phase = 2.0 * np.pi * (yy * 42.0 + warp * 2.3) / n
    bristle = 0.5 + 0.5 * np.sin(phase)
    bristle = normalize(0.58 * bristle + 0.42 * medium_streaks)

    ridge = normalize(0.50 * long_streaks + 0.30 * medium_streaks + 0.20 * broad)
    ridge = np.clip((ridge - 0.43) * 1.75 + 0.5, 0.0, 1.0)
    grain = normalize(0.70 * micro + 0.30 * broad)
    mottle = normalize(0.55 * broad + 0.45 * long_streaks)

    rgba = np.stack([
        (grain * 255.0 + 0.5).astype(np.uint8),
        (ridge * 255.0 + 0.5).astype(np.uint8),
        (bristle * 255.0 + 0.5).astype(np.uint8),
        (mottle * 255.0 + 0.5).astype(np.uint8),
    ], axis=-1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(args.out, optimize=True)
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes, {n}x{n})")


if __name__ == "__main__":
    main()
