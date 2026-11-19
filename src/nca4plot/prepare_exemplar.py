"""Prepare square Starry Night sky crops for texture-NCA training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


# Pixel boxes were selected against the 1280px Wikimedia public-domain scan.
# They avoid the moon, cypress, stars at the edges, hills, and village.
CANDIDATES = {
    "A-core-swirl": (430, 180, 830, 580),
    "B-upper-current": (480, 20, 880, 420),
    "C-east-current": (720, 150, 1120, 550),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_contact_sheet(source: Image.Image, out_path: Path) -> None:
    tile = 360
    label_h = 36
    sheet = Image.new("RGB", (tile * len(CANDIDATES), tile + label_h), (16, 18, 28))
    draw = ImageDraw.Draw(sheet)
    for i, (name, box) in enumerate(CANDIDATES.items()):
        crop = source.crop(box).resize((tile, tile), Image.Resampling.LANCZOS)
        sheet.paste(crop, (i * tile, 0))
        draw.text((i * tile + 12, tile + 10), name, fill=(236, 228, 203))
    sheet.save(out_path, quality=94)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/starry_night_source.jpg"))
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--media-dir", type=Path, default=Path("media"))
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), default="A-core-swirl")
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.media_dir.mkdir(parents=True, exist_ok=True)
    source = Image.open(args.source).convert("RGB")
    if source.size != (1280, 1014):
        raise ValueError(f"expected 1280x1014 source, got {source.size}")

    for name, box in CANDIDATES.items():
        crop = source.crop(box).resize((args.size, args.size), Image.Resampling.LANCZOS)
        crop.save(args.out_dir / f"exemplar_{name}.png")

    selected = source.crop(CANDIDATES[args.candidate]).resize(
        (args.size, args.size), Image.Resampling.LANCZOS
    )
    selected_path = args.out_dir / "exemplar_128.png"
    selected.save(selected_path)
    selected.resize((512, 512), Image.Resampling.NEAREST).save(
        args.media_dir / "exemplar_128_preview.png"
    )
    make_contact_sheet(source, args.media_dir / "exemplar_candidates.jpg")

    metadata = {
        "source": "Van Gogh, The Starry Night (1889), Google Art Project / Wikimedia Commons",
        "source_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
        "license": "Public domain (two-dimensional work)",
        "source_size": list(source.size),
        "source_sha256": sha256(args.source),
        "selected": args.candidate,
        "crop_box": CANDIDATES[args.candidate],
        "exemplar_size": args.size,
        "exemplar_sha256": sha256(selected_path),
        "candidates": {k: list(v) for k, v in CANDIDATES.items()},
    }
    (args.out_dir / "exemplar_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
