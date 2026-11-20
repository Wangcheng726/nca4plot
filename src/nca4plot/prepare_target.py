"""Prepare the full-sky rectangular target for the regeneration NCA.

Unlike the square texture exemplars (a small swirl crop), this target is the
WHOLE sky composition of The Starry Night: the cypress flame, the crescent
moon, both great swirls, every star and the hill horizon. The village and the
raw canvas edges of the scan are excluded. The 16:9 crop fills modern screens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


# left, top, right, bottom on the 1280x1014 public-domain scan.
CROP_BOX = (16, 8, 1264, 710)
SIZES = {"256x144": (256, 144), "384x216": (384, 216), "192x108": (192, 108),
         "128x72": (128, 72)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/starry_night_source.jpg"))
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--media-dir", type=Path, default=Path("media"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.media_dir.mkdir(parents=True, exist_ok=True)
    source = Image.open(args.source).convert("RGB")
    if source.size != (1280, 1014):
        raise ValueError(f"expected 1280x1014 source, got {source.size}")

    crop = source.crop(CROP_BOX)
    targets = {}
    for name, (w, h) in SIZES.items():
        img = crop.resize((w, h), Image.Resampling.LANCZOS).convert("RGBA")
        path = args.out_dir / f"target_sky_{name}.png"
        img.save(path)
        targets[name] = {"size": [w, h], "sha256": sha256(path), "path": path.as_posix()}

    crop.resize((1024, 576), Image.Resampling.LANCZOS).save(
        args.media_dir / "target_sky_preview.png"
    )

    metadata = {
        "source": "Van Gogh, The Starry Night (1889), Google Art Project / Wikimedia Commons",
        "source_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
        "license": "Public domain (two-dimensional work)",
        "source_size": list(source.size),
        "source_sha256": sha256(args.source),
        "crop_box": list(CROP_BOX),
        "crop_note": "whole sky: cypress, moon, both swirls, all stars, hill horizon; village excluded",
        "alpha": "opaque everywhere (the full rectangle is living matter)",
        "targets": targets,
    }
    (args.out_dir / "target_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()