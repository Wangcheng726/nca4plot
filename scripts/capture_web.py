"""Capture the real interactive artwork as stills, MP4, GIF, and a process sheet."""

from __future__ import annotations

import argparse
import functools
import http.server
import io
import math
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
MEDIA_DIR = ROOT / "media"


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


@contextmanager
def local_web_server():
    handler = functools.partial(QuietHandler, directory=str(WEB_DIR))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        thread.join(timeout=3)


def find_chrome() -> str:
    candidates = [
        shutil.which("chrome"),
        shutil.which("google-chrome"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise RuntimeError("Chrome/Chromium not found; install Chrome or set it on PATH")


def wait_for_boot(page, min_steps: int = 240, timeout_ms: int = 120_000) -> None:
    page.wait_for_function(
        "(minimum) => window.nca4plot && window.nca4plot.entered && window.nca4plot.steps >= minimum",
        arg=min_steps,
        timeout=timeout_ms,
    )


def nearest_frame(frames: list[tuple[float, bytes]], target: float) -> bytes:
    times = np.asarray([item[0] for item in frames])
    index = int(np.argmin(np.abs(times - target)))
    return frames[index][1]


def decode_jpeg(payload: bytes, size: tuple[int, int] | None = None) -> Image.Image:
    image = Image.open(io.BytesIO(payload)).convert("RGB")
    if size and image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def record_sequence(
    page,
    duration_s: float,
    fps: int = 12,
) -> tuple[list[tuple[float, bytes]], dict[str, float]]:
    frames: list[tuple[float, bytes]] = []
    marks: dict[str, float] = {}
    recording_start = time.monotonic()

    def shot() -> None:
        now = time.monotonic()
        payload = page.screenshot(type="jpeg", quality=92, full_page=False)
        frames.append((now - recording_start, payload))

    def capture_for(seconds: float) -> None:
        end = time.monotonic() + seconds
        next_at = time.monotonic()
        while time.monotonic() < end:
            now = time.monotonic()
            if now >= next_at:
                shot()
                next_at = max(next_at + 1.0 / fps, time.monotonic() + 0.25 / fps)
            else:
                time.sleep(min(0.02, max(0.001, next_at - now)))

    # Zero seed -> growth.
    page.evaluate("document.querySelector('#resetButton').click()")
    capture_for(5.6)
    marks["tear_start"] = frames[-1][0]

    # A real pointer stroke enters the page event path and erases all 16 channels.
    points: list[tuple[float, float]] = []
    for i in range(29):
        t = i / 28
        x = 1320 - 720 * t + 92 * math.sin(t * math.pi * 3.2)
        y = 335 + 430 * t + 58 * math.cos(t * math.pi * 4.5)
        points.append((x, y))
    page.mouse.move(*points[0])
    page.mouse.down()
    tear_started = time.monotonic()
    next_shot = tear_started
    for index, point in enumerate(points[1:]):
        page.mouse.move(*point, steps=2)
        now = time.monotonic()
        if now >= next_shot:
            shot()
            next_shot = max(next_shot + 1.0 / fps, time.monotonic() + 0.25 / fps)
        time.sleep(0.04)
    page.mouse.up()
    page.wait_for_timeout(450)
    shot()
    marks["tear_end"] = frames[-1][0]

    # After the tear: target modes regrow the painting; persistent keeps the
    # wound damage-locked open while the rest of the sky drifts on.
    capture_for(max(0.2, duration_s - 6.05))
    marks["end"] = frames[-1][0]
    print(
        f"captured {len(frames)} frames across {marks['end'] - frames[0][0]:.2f}s",
        flush=True,
    )
    return frames, marks


def encode_mp4(
    frames: list[tuple[float, bytes]],
    path: Path,
    start_s: float,
    end_s: float,
    fps: int = 24,
) -> None:
    import imageio_ffmpeg

    times = np.asarray([item[0] for item in frames])
    sample_times = np.arange(start_s, end_s, 1.0 / fps)
    indices = np.searchsorted(times, sample_times).clip(0, len(frames) - 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        "1920x1080",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "22",
        "-profile:v",
        "high",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        path.as_posix(),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        for index in indices:
            image = decode_jpeg(frames[int(index)][1], (1920, 1080))
            process.stdin.write(np.asarray(image, dtype=np.uint8).tobytes())
    finally:
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {stderr}")
    print(f"wrote {path} ({path.stat().st_size / 1024 / 1024:.1f} MiB)", flush=True)


def save_gif(
    frames: list[tuple[float, bytes]],
    path: Path,
    start_s: float,
    duration_s: float,
    fps: int = 10,
) -> None:
    times = np.asarray([item[0] for item in frames])
    sample_times = np.arange(start_s, start_s + duration_s, 1.0 / fps)
    indices = np.searchsorted(times, sample_times).clip(0, len(frames) - 1)
    images: list[Image.Image] = []
    for index in indices:
        image = decode_jpeg(frames[int(index)][1], (720, 405))
        images.append(image.quantize(colors=64, method=Image.Quantize.MEDIANCUT))
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(round(1000 / fps)),
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"wrote {path} ({path.stat().st_size / 1024 / 1024:.1f} MiB)", flush=True)


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


SHEET_LABELS = {
    "regenerate": [
        "1  MEMORY: THE WHOLE SKY",
        "2  MEMORY REPLANTED",
        "3  THE SKY IS TORN",
        "4  THE WOUND REGROWS THE PAINTING",
        "5  MEMORY HOLDS, MATTER DRIFTS",
    ],
    "growing": [
        "1  MEMORY CONSTELLATION",
        "2  CONSTELLATION -> WHOLE SKY",
        "3  THE SKY IS TORN",
        "4  WOUND EDGE REWEAVING",
        "5  GROWN MEMORY",
    ],
    "persistent": [
        "1  MATURE DRIFT",
        "2  ZERO -> GROWTH",
        "3  THE SKY IS TORN",
        "4  THE WOUND STAYS OPEN",
        "5  NEVER SETTLING",
    ],
}


def save_contact_sheet(
    path: Path,
    mature_path: Path,
    frames: list[tuple[float, bytes]],
    marks: dict[str, float],
    mode: str = "persistent",
) -> None:
    start = frames[0][0]
    tear = marks["tear_start"]
    labels = SHEET_LABELS[mode]
    picks = [
        (Image.open(mature_path).convert("RGB"), labels[0]),
        (decode_jpeg(nearest_frame(frames, tear - 3.8)), labels[1]),
        (decode_jpeg(nearest_frame(frames, tear + 0.9)), labels[2]),
        (decode_jpeg(nearest_frame(frames, tear + 4.2)), labels[3]),
        (decode_jpeg(nearest_frame(frames, marks["end"] - 0.4)), labels[4]),
    ]
    if mode == "growing":
        # Reset replants the sparse memory constellation and the slow clock
        # coalesces it over ~2 s; the boot poster is already a merged sky, so
        # tiles 1-2 must come from the recorded regrowth arc, not the poster.
        picks[0] = (decode_jpeg(nearest_frame(frames, start + 0.35)), labels[0])
        picks[1] = (decode_jpeg(nearest_frame(frames, start + 1.1)), labels[1])
    tile_w, tile_h, gap, label_h = 640, 360, 18, 45
    columns = 3
    rows = 2
    sheet = Image.new(
        "RGB",
        (columns * tile_w + (columns + 1) * gap, rows * (tile_h + label_h) + (rows + 1) * gap),
        (5, 11, 17),
    )
    draw = ImageDraw.Draw(sheet)
    font = load_font(18)
    for index, (image, label) in enumerate(picks):
        row, column = divmod(index, columns)
        x = gap + column * (tile_w + gap)
        y = gap + row * (tile_h + label_h + gap)
        sheet.paste(image.resize((tile_w, tile_h), Image.Resampling.LANCZOS), (x, y))
        draw.text((x + 10, y + tile_h + 12), label, font=font, fill=(226, 216, 182))
    sheet.save(path, quality=92)
    print(f"wrote {path}", flush=True)


def capture_mobile(browser, url: str, path: Path) -> None:
    context = browser.new_context(
        viewport={"width": 390, "height": 844},
        device_scale_factor=2,
        is_mobile=True,
        has_touch=True,
    )
    page = context.new_page()
    page.goto(f"{url}?autostart=1&mode=regenerate", wait_until="domcontentloaded")
    wait_for_boot(page, 460)
    page.screenshot(path=path)
    context.close()
    print(f"wrote {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="regenerate",
                        choices=["regenerate", "growing", "persistent"])
    parser.add_argument("--duration", type=float, default=17.0)
    parser.add_argument("--out-dir", type=Path, default=MEDIA_DIR)
    parser.add_argument("--url", default=None, help="Use an existing server instead of starting one")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with local_web_server() as local_url, sync_playwright() as playwright:
        url = args.url or local_url
        browser = playwright.chromium.launch(
            headless=True,
            executable_path=find_chrome(),
            args=[
                "--enable-unsafe-webgpu",
                "--ignore-gpu-blocklist",
                "--disable-background-timer-throttling",
            ],
        )
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            device_scale_factor=1,
        )
        page = context.new_page()
        page.goto(
            f"{url}?autostart=1&mode={args.mode}",
            wait_until="domcontentloaded",
        )
        wait_for_boot(page, {"regenerate": 460, "growing": 60, "persistent": 560}[args.mode])
        page.wait_for_timeout(900)

        poster_site = args.out_dir / "poster_nca4plot.png"
        page.screenshot(path=poster_site)
        print(f"wrote {poster_site}", flush=True)

        page.add_style_tag(
            content=(
                ".masthead,.status-pill,.top-actions,.gesture-note,.controls,.mode-control,"
                ".colophon,.cursor,.toast"
                "{display:none!important}.edge-shade{opacity:.55}"
            )
        )
        poster_art = args.out_dir / "poster_sky_1920x1080.png"
        page.screenshot(path=poster_art)
        print(f"wrote {poster_art}", flush=True)

        frames, marks = record_sequence(page, args.duration, fps=12)
        if len(frames) < 20:
            raise RuntimeError(f"screencast produced too few frames: {len(frames)}")

        start = frames[0][0]
        end = marks["end"]
        encode_mp4(frames, args.out_dir / "nca4plot_demo.mp4", start, end, fps=24)
        gif_start = max(start, marks["tear_start"] - 2.1)
        gif_end = min(end, gif_start + 8.0)
        save_gif(
            frames,
            args.out_dir / "nca4plot_demo.gif",
            gif_start,
            gif_end - gif_start,
            fps=6,
        )
        save_contact_sheet(
            args.out_dir / "process_contact_sheet.jpg",
            poster_art,
            frames,
            marks,
            mode=args.mode,
        )
        context.close()

        capture_mobile(browser, url, args.out_dir / "mobile_390x844.png")
        browser.close()


if __name__ == "__main__":
    main()
