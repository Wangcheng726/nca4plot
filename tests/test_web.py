"""End-to-end browser regression for WebGPU/WASM boot, erasure, and recovery."""

from __future__ import annotations

import contextlib
import functools
import http.server
import json
import shutil
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture(scope="module")
def local_server():
    handler = functools.partial(QuietHandler, directory=str(WEB_DIR))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        thread.join(timeout=2)


def find_chrome() -> str | None:
    candidates = [
        shutil.which("chrome"),
        shutil.which("google-chrome"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    return next((path for path in candidates if path and Path(path).exists()), None)


def test_browser_boot_erase_recover_and_controls(local_server: str) -> None:
    playwright_module = pytest.importorskip("playwright.sync_api")
    chrome = find_chrome()
    if chrome is None:
        pytest.skip("Chrome/Chromium is not installed for the browser regression test")

    with playwright_module.sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            executable_path=chrome,
            args=[
                "--enable-unsafe-webgpu",
                "--ignore-gpu-blocklist",
                "--disable-background-timer-throttling",
            ],
        )
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        page_errors: list[str] = []
        failed_requests: list[str] = []
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.on(
            "requestfailed",
            lambda request: failed_requests.append(f"{request.url}: {request.failure}"),
        )

        page.goto(f"{local_server}?autostart=1&mode=regenerate", wait_until="domcontentloaded")
        page.wait_for_function(
            "() => window.nca4plot && window.nca4plot.entered && window.nca4plot.steps > 0",
            timeout=180_000,
        )
        start_steps = page.evaluate("window.nca4plot.steps")
        meta = json.loads((WEB_DIR / "models" / "nca_target_meta.json").read_text(encoding="utf-8"))
        assert page.evaluate("window.nca4plot.engine.kind") == "target"
        assert page.evaluate("window.nca4plot.engine.lockWounds") is False
        assert page.evaluate("window.nca4plot.engine.width") == int(meta["width"])
        assert page.evaluate("window.nca4plot.engine.height") == int(meta["height"])
        assert page.evaluate("window.nca4plot.backend") in {"webgpu", "wasm"}

        page.wait_for_function(
            "(start) => window.nca4plot.steps > start + 8", arg=start_steps, timeout=20_000
        )

        # Pause must stop progression and the pause button must reflect the state.
        page.click("#pauseButton")
        page.wait_for_timeout(250)
        paused_steps = page.evaluate("window.nca4plot.steps")
        page.wait_for_timeout(650)
        assert page.evaluate("window.nca4plot.steps") == paused_steps
        assert page.evaluate("window.nca4plot.paused") is True

        # Erase while paused, then inspect all CA channels in the central disc.
        page.evaluate("window.nca4plot.eraseNormalized(0.5, 0.5, 10)")
        empty = page.evaluate(
            """() => {
              const engine = window.nca4plot.engine;
              const w = engine.width, h = engine.height;
              const cx = w * 0.5;
              const cy = h * 0.5;
              const n2 = w * h;
              let maxAbs = 0;
              let samples = 0;
              for (let y = Math.floor(cy - 3); y <= Math.ceil(cy + 3); y += 1) {
                for (let x = Math.floor(cx - 3); x <= Math.ceil(cx + 3); x += 1) {
                  const pixel = y * w + x;
                  for (let channel = 0; channel < engine.chn; channel += 1) {
                    maxAbs = Math.max(maxAbs, Math.abs(engine.stateData[channel * n2 + pixel]));
                    samples += 1;
                  }
                }
              }
              return { maxAbs, samples };
            }"""
        )
        assert empty["samples"] > 0
        assert empty["maxAbs"] == 0, "erasure must clear every one of all CA channels"

        # Resume: regenerate semantics -- the wound core must start growing
        # back toward the painting while the rest of the sky keeps living.
        page.click("#pauseButton")
        page.wait_for_function(
            "(start) => window.nca4plot.steps > start + 120",
            arg=paused_steps,
            timeout=60_000,
        )
        retained = page.evaluate(
            """() => {
              const engine = window.nca4plot.engine;
              const w = engine.width, h = engine.height;
              const n2 = w * h;
              const cx = w * 0.5;
              const cy = h * 0.5;
              let coreMax = 0;
              let rimEnergy = 0;
              let rimCount = 0;
              for (let y = Math.floor(cy - 12); y <= Math.ceil(cy + 12); y += 1) {
                for (let x = Math.floor(cx - 12); x <= Math.ceil(cx + 12); x += 1) {
                  const pixel = y * w + x;
                  const dist = Math.hypot(x - cx, y - cy);
                  let cellMax = 0;
                  for (let channel = 0; channel < engine.chn; channel += 1) {
                    cellMax = Math.max(cellMax, Math.abs(engine.stateData[channel * n2 + pixel]));
                  }
                  if (dist <= 6) coreMax = Math.max(coreMax, cellMax);
                  if (dist >= 11) { rimEnergy += cellMax; rimCount += 1; }
                }
              }
              return {
                coreMax,
                rimMean: rimCount ? rimEnergy / rimCount : 0,
                locked: engine.lockedPixels.length,
                steps: engine.steps,
              };
            }"""
        )
        assert retained["locked"] == 0, "regenerate must not damage-lock cells"
        assert retained["coreMax"] > 0.2, "regenerate wound core did not start healing"
        assert retained["rimMean"] > 1e-3, "the sky around the wound stopped flowing"

        # Reset clears every damage lock and restores the living painting.
        page.click("#pauseButton")
        reset_state = page.evaluate(
            """() => {
              const e = window.nca4plot.engine;
              e.reset();
              const n2 = e.n2;
              let alphaMax = 0;
              for (let i = 0; i < n2; i += 97) alphaMax = Math.max(alphaMax, e.stateData[3 * n2 + i]);
              return { locks: e.lockedPixels.length, alphaMax };
            }"""
        )
        assert reset_state["locks"] == 0, "reset must clear all damage locks"
        assert reset_state["alphaMax"] > 0.9, "reset must restore the living painting"

        assert page_errors == []
        assert failed_requests == []
        context.close()
        browser.close()


PULL_RGBA = """() => {
  const e = window.nca4plot.engine;
  const w = e.width, h = e.height, n2 = w * h, d = e.stateData;
  const out = [];
  for (let y = 0; y < h; y += 2) for (let x = 0; x < w; x += 2) {
    const i = y * w + x;
    out.push(d[i], d[n2 + i], d[2 * n2 + i], d[3 * n2 + i]);
  }
  return { w, h, data: out };
}"""


def _rgba_mse(page, disc=None):
    import numpy as np
    from PIL import Image
    info = page.evaluate(PULL_RGBA)
    state = np.asarray(info["data"], dtype=np.float32).reshape(info["h"] // 2, info["w"] // 2, 4)
    target = np.asarray(
        Image.open(ROOT / "data" / f"target_sky_{info['w']}x{info['h']}.png").convert("RGBA"),
        dtype=np.float32)[::2, ::2] / 255.0
    if disc is not None:
        cx, cy, r = disc
        hh, ww = state.shape[:2]
        yy, xx = np.mgrid[0:hh, 0:ww]
        mask = (xx - cx / 2) ** 2 + (yy - cy / 2) ** 2 <= (r / 2) ** 2
        return float(((state[mask] - target[mask]) ** 2).mean())
    return float(((state - target) ** 2).mean())


def test_target_regenerate_only(local_server: str) -> None:
    playwright_module = pytest.importorskip("playwright.sync_api")
    chrome = find_chrome()
    if chrome is None:
        pytest.skip("Chrome/Chromium is not installed for the browser regression test")
    if not (WEB_DIR / "models" / "nca_target_meta.json").exists():
        pytest.skip("target model not exported yet")
    meta = json.loads((WEB_DIR / "models" / "nca_target_meta.json").read_text(encoding="utf-8"))
    tw, th = int(meta["width"]), int(meta["height"])

    with playwright_module.sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            executable_path=chrome,
            args=[
                "--enable-unsafe-webgpu",
                "--ignore-gpu-blocklist",
                "--disable-background-timer-throttling",
            ],
        )
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        page_errors: list[str] = []
        page.on("pageerror", lambda error: page_errors.append(str(error)))

        # Default mode is regenerate: the whole painting must already be there.
        page.goto(f"{local_server}?autostart=1", wait_until="domcontentloaded")
        page.wait_for_function(
            "() => window.nca4plot && window.nca4plot.entered && window.nca4plot.steps >= 440",
            timeout=180_000,
        )
        assert page.evaluate("window.nca4plot.engine.kind") == "target"
        assert page.evaluate("window.nca4plot.engine.width") == tw
        assert page.evaluate("window.nca4plot.engine.height") == th
        assert page.evaluate("window.nca4plot.mode") == "regenerate"
        assert page.locator('.mode-button[data-mode="regenerate"]').get_attribute("aria-pressed") == "true"
        assert page.evaluate("window.nca4plot.engine.lockWounds") is False
        assert page.evaluate("document.querySelector('#qualitySelect')") is None
        initial_mse = _rgba_mse(page)
        assert initial_mse < 0.012, f"initial state does not look like the painting: mse={initial_mse}"

        # Erase while paused: every CA channel must go to zero.
        page.click("#pauseButton")
        page.wait_for_timeout(200)
        paused_steps = page.evaluate("window.nca4plot.steps")
        page.evaluate("window.nca4plot.eraseNormalized(0.5, 0.5, 14)")
        empty = page.evaluate(
            """() => {
              const e = window.nca4plot.engine;
              const w = e.width, h = e.height, n2 = w * h;
              const cx = w * 0.5, cy = h * 0.5;
              let maxAbs = 0;
              for (let y = Math.floor(cy - 4); y <= Math.ceil(cy + 4); y += 1) {
                for (let x = Math.floor(cx - 4); x <= Math.ceil(cx + 4); x += 1) {
                  const i = y * w + x;
                  for (let c = 0; c < e.chn; c += 1) maxAbs = Math.max(maxAbs, Math.abs(e.stateData[c * n2 + i]));
                }
              }
              return maxAbs;
            }"""
        )
        assert empty == 0, "erasure must clear all CA channels"

        # Resume: the wound must regrow towards the painting, not just any texture.
        page.click("#pauseButton")
        page.wait_for_function(
            "(start) => window.nca4plot.steps > start + 200", arg=paused_steps, timeout=120_000)
        wound_mse = _rgba_mse(page, disc=(tw // 2, th // 2, max(16, th // 5)))
        assert wound_mse < 0.02, f"wound did not regrow the painting: mse={wound_mse}"

        # v11 scope: only regenerate is exposed. Growing / persistent must
        # not be reachable from the UI or the public debug surface.
        assert page.locator('.mode-button[data-mode="growing"]').count() == 0
        assert page.locator('.mode-button[data-mode="persistent"]').count() == 0
        page.evaluate("window.nca4plot.setMode('growing')")
        page.wait_for_timeout(300)
        assert page.evaluate("window.nca4plot.mode") == "regenerate"
        page.evaluate("window.nca4plot.setMode('persistent')")
        page.wait_for_timeout(300)
        assert page.evaluate("window.nca4plot.mode") == "regenerate"
        assert page.evaluate("window.nca4plot.engine.lockWounds") is False

        # A second tear must also heal back toward the painting.
        page.evaluate("window.nca4plot.eraseNormalized(0.62, 0.6, 12)")
        steps_before = page.evaluate("window.nca4plot.steps")
        page.wait_for_function(
            "(s) => window.nca4plot.steps > s + 400", arg=steps_before, timeout=120_000)
        healed_mse = _rgba_mse(page, disc=(int(tw * 0.62), int(th * 0.4), 24))
        assert healed_mse < 0.02, f"second wound did not regrow the painting: mse={healed_mse}"

        assert page_errors == []
        context.close()
        browser.close()


def test_mobile_layout_boots(local_server: str) -> None:
    playwright_module = pytest.importorskip("playwright.sync_api")
    chrome = find_chrome()
    if chrome is None:
        pytest.skip("Chrome/Chromium is not installed for the browser regression test")

    with playwright_module.sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            executable_path=chrome,
            args=["--enable-unsafe-webgpu", "--ignore-gpu-blocklist"],
        )
        context = browser.new_context(
            viewport={"width": 390, "height": 844},
            device_scale_factor=2,
            is_mobile=True,
            has_touch=True,
        )
        page = context.new_page()
        page.goto(f"{local_server}?autostart=1", wait_until="domcontentloaded")
        page.wait_for_function(
            "() => window.nca4plot && window.nca4plot.entered && window.nca4plot.steps >= 440",
            timeout=180_000,
        )
        target_meta = json.loads((WEB_DIR / "models" / "nca_target_meta.json").read_text(encoding="utf-8"))
        assert page.evaluate("window.nca4plot.engine.width") == int(target_meta["width"])
        assert page.locator(".mode-control").is_visible()
        assert page.locator("#pauseButton").is_visible()
        assert page.locator("#aboutButton").is_visible()
        body_width = page.evaluate("document.body.scrollWidth")
        viewport_width = page.evaluate("window.innerWidth")
        assert body_width <= viewport_width + 1
        context.close()
        browser.close()
