"""Procedurally draw the team's original plant targets (flat colors, crisp shapes).

Candidates A/B/C on an 80x80 grid (+ a 64x64 fallback for A). RGBA with a
transparent background; drawn at 4x supersampling then LANCZOS-downscaled so
edges are cleanly antialiased like the official emoji targets. Geometry is
defined in 80-unit space and scaled for other grid sizes. The seed position
(stem base) is written to metadata for the trainer / web app.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

SS = 4  # supersampling factor

PALETTES = {
    "A": {"stem": "#2E7D32", "leaf": "#66BB6A", "leaf2": "#43A047",
          "petal": "#F48FB1", "center": "#FBC02D"},
    "B": {"stem": "#33691E", "leaf": "#8BC34A", "leaf2": "#689F38",
          "petal": "#FDD835", "center": "#F57F17"},
    "C": {"stem": "#2E7D32", "leaf": "#81C784", "leaf2": "#66BB6A",
          "petal": None, "center": "#FBC02D"},
}


def _hex(c: str) -> tuple[int, int, int, int]:
    c = c.lstrip("#")
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), 255)


def _bezier(p0, p1, p2, p3, t):
    u = 1.0 - t
    return (u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
            u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1])


def _leaf_polygon(ax, ay, angle_deg, length, width, curve=0.0):
    """Pointed-oval leaf from attach point (ax,ay) along angle (image coords, y down)."""
    a = math.radians(angle_deg)
    dx, dy = math.cos(a), math.sin(a)
    nx, ny = -dy, dx
    pts = []
    n = 20
    for i in range(n + 1):
        t = i / n
        bend = curve * math.sin(math.pi * t)
        px = ax + dx * length * t + nx * bend
        py = ay + dy * length * t + ny * bend
        w = (width * 0.5) * math.sin(math.pi * t) ** 0.85
        pts.append((px + nx * w, py + ny * w))
    for i in range(n, -1, -1):
        t = i / n
        bend = curve * math.sin(math.pi * t)
        px = ax + dx * length * t + nx * bend
        py = ay + dy * length * t + ny * bend
        w = (width * 0.5) * math.sin(math.pi * t) ** 0.85
        pts.append((px - nx * w, py - ny * w))
    return pts


def draw_plant(size: int, variant: str) -> tuple[Image.Image, dict]:
    """Return (RGBA image, metadata). Geometry authored in 80-unit space."""
    s = size / 80.0
    pal = PALETTES[variant]
    img = Image.new("RGBA", (size * SS, size * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    def S(p):
        return (p[0] * s * SS, p[1] * s * SS)

    # Stem: gentle S-curve from base (40,64) to top (40,26).
    p0, p1, p2, p3 = (40, 64), (37.5, 51), (42.5, 38), (40, 26)
    stem_w = 2.6 * s * SS
    pts = [_bezier(p0, p1, p2, p3, i / 40) for i in range(41)]
    d.line([S(p) for p in pts], fill=_hex(pal["stem"]), width=int(round(stem_w)), joint="curve")
    for p in (pts[0], pts[-1]):
        x, y = S(p)
        r = stem_w / 2
        d.ellipse([x - r, y - r, x + r, y + r], fill=_hex(pal["stem"]))

    # Leaves: alternating along the stem, pointing up-outward.
    leaf_ts = [0.16, 0.30, 0.44, 0.58, 0.72, 0.84]
    for k, t in enumerate(leaf_ts):
        if variant == "C" and k >= 4:
            break  # minimal sprout: 4 leaves
        ax, ay = _bezier(p0, p1, p2, p3, t)
        left = k % 2 == 0
        ang = 180 + 32 if left else -32  # up-left / up-right (y down)
        length = (12.5 - 1.2 * k) * s * SS
        width = (5.6 - 0.35 * k) * s * SS
        color = pal["leaf"] if k % 2 == 0 else pal["leaf2"]
        poly = _leaf_polygon(ax * s * SS, ay * s * SS, ang, length, width,
                             curve=(1.6 if left else -1.6) * s * SS)
        d.polygon(poly, fill=_hex(color))

    # Flower (A/B) or bud (C) at the stem top.
    fx, fy = 40, 21
    if pal["petal"]:
        pr, dist = 4.7 * s * SS, 4.9 * s * SS
        for k in range(5):
            a = math.radians(-90 + k * 72)
            cx = (fx + dist / (s * SS) * math.cos(a)) * s * SS
            cy = (fy + dist / (s * SS) * math.sin(a)) * s * SS
            d.ellipse([cx - pr, cy - pr, cx + pr, cy + pr], fill=_hex(pal["petal"]))
        cr = 3.3 * s * SS
        cx, cy = fx * s * SS, fy * s * SS
        d.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], fill=_hex(pal["center"]))
    else:
        cr = 3.6 * s * SS
        cx, cy = fx * s * SS, (fy + 1.5) * s * SS
        d.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], fill=_hex(pal["center"]))

    img = img.resize((size, size), Image.Resampling.LANCZOS)
    meta = {
        "variant": variant,
        "grid": size,
        "seed_xy": [round(40 * s), round(64 * s)],
        "palette": pal,
        "flower_center_xy": [round(fx * s), round(fy * s)],
        "note": "original procedural artwork; RGBA straight alpha; trainer premultiplies RGB",
    }
    return img, meta


def composite_white(img: Image.Image) -> Image.Image:
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, img).convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument("--media-dir", type=Path, default=Path("media"))
    ap.add_argument("--variants", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[80])
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.media_dir.mkdir(parents=True, exist_ok=True)

    metas = {}
    previews = []
    for v in args.variants:
        sizes = args.sizes + ([64] if v == "A" and 64 not in args.sizes else [])
        for sz in sizes:
            img, meta = draw_plant(sz, v)
            name = f"plant_{v}_{sz}"
            img.save(args.out_dir / f"{name}.png")
            (args.out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            metas[name] = meta
            if sz == 80:
                previews.append((name, composite_white(img)))
            print(f"{name}: seed={meta['seed_xy']} colors={[c for c in meta['palette'].values() if c]}")

    # Contact sheet: candidates zoomed x6 over white, with labels.
    zoom = 6
    gap = 16
    label_h = 28
    w = previews[0][1].width * zoom
    h = previews[0][1].height * zoom
    sheet = Image.new("RGB", (len(previews) * w + (len(previews) + 1) * gap,
                              h + label_h + 2 * gap), (250, 250, 248))
    sd = ImageDraw.Draw(sheet)
    for i, (name, im) in enumerate(previews):
        x = gap + i * (w + gap)
        sheet.paste(im.resize((w, h), Image.Resampling.NEAREST), (x, gap + label_h))
        sd.text((x + 4, gap + 4), f"Candidate {name}  ({name})", fill=(40, 40, 40))
    sheet.save(args.media_dir / "plant_candidates.png")
    (args.out_dir / "plant_metadata.json").write_text(json.dumps(metas, indent=2), encoding="utf-8")
    print("contact sheet:", args.media_dir / "plant_candidates.png")


if __name__ == "__main__":
    main()
