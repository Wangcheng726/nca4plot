"""End-to-end browser regression for WebGPU/WASM boot, erasure, and recovery."""

from __future__ import annotations

import contextlib
import functools
import http.server
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

        page.goto(f"{local_server}?autostart=1&quality=128", wait_until="domcontentloaded")
        page.wait_for_function(
            "() => window.nca4plot && window.nca4plot.entered && window.nca4plot.steps > 0",
            timeout=90_000,
        )
        start_steps = page.evaluate("window.nca4plot.steps")
        assert page.evaluate("window.nca4plot.engine.size") == 128
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

        # Erase while paused, then inspect all 12 channels in the central disc.
        page.evaluate("window.nca4plot.eraseNormalized(0.5, 0.5, 10)")
        empty = page.evaluate(
            """() => {
              const engine = window.nca4plot.engine;
              const n = engine.size;
              const cx = n * 0.5;
              const cy = n * 0.5;
              const n2 = n * n;
              let maxAbs = 0;
              let samples = 0;
              for (let y = Math.floor(cy - 3); y <= Math.ceil(cy + 3); y += 1) {
                for (let x = Math.floor(cx - 3); x <= Math.ceil(cx + 3); x += 1) {
                  const pixel = y * n + x;
                  for (let channel = 0; channel < 12; channel += 1) {
                    maxAbs = Math.max(maxAbs, Math.abs(engine.stateData[channel * n2 + pixel]));
                    samples += 1;
                  }
                }
              }
              return { maxAbs, samples };
            }"""
        )
        assert empty["samples"] > 0
        assert empty["maxAbs"] == 0, "erasure must clear every one of the 12 CA channels"

        # Resume and allow the local rule to repopulate the wound.
        page.click("#pauseButton")
        page.wait_for_function(
            "(start) => window.nca4plot.steps > start + 120",
            arg=paused_steps,
            timeout=60_000,
        )
        recovered = page.evaluate(
            """() => {
              const engine = window.nca4plot.engine;
              const n = engine.size;
              const n2 = n * n;
              const cx = n * 0.5;
              const cy = n * 0.5;
              let energy = 0;
              let count = 0;
              for (let y = Math.floor(cy - 9); y <= Math.ceil(cy + 9); y += 1) {
                for (let x = Math.floor(cx - 9); x <= Math.ceil(cx + 9); x += 1) {
                  const pixel = y * n + x;
                  for (let channel = 0; channel < 12; channel += 1) {
                    energy += Math.abs(engine.stateData[channel * n2 + pixel]);
                    count += 1;
                  }
                }
              }
              return { meanAbs: energy / count, steps: engine.steps };
            }"""
        )
        assert recovered["meanAbs"] > 1e-5, "the wound did not regrow"

        # Reset returns the state buffer to zero at least momentarily.
        page.click("#pauseButton")
        reset_state = page.evaluate(
            """() => {
              window.nca4plot.engine.reset();
              return Array.from(window.nca4plot.engine.stateData).every((value) => value === 0);
            }"""
        )
        assert reset_state is True

        # Quality switching loads a different fixed-shape graph and continues running.
        page.select_option("#qualitySelect", "256")
        page.wait_for_function(
            "() => window.nca4plot.engine.size === 256 && window.nca4plot.steps > 0",
            timeout=90_000,
        )
        assert page.evaluate("window.nca4plot.backend") in {"webgpu", "wasm"}
        page.select_option("#qualitySelect", "512")
        page.wait_for_function(
            "() => window.nca4plot.engine.size === 512 && window.nca4plot.steps > 0",
            timeout=120_000,
        )

        assert page_errors == []
        assert failed_requests == []
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
        page.goto(f"{local_server}?autostart=1&quality=128", wait_until="domcontentloaded")
        page.wait_for_function(
            "() => window.nca4plot && window.nca4plot.entered && window.nca4plot.steps > 0",
            timeout=90_000,
        )
        assert page.evaluate("window.nca4plot.engine.size") == 128
        assert page.locator("#pauseButton").is_visible()
        assert page.locator("#aboutButton").is_visible()
        body_width = page.evaluate("document.body.scrollWidth")
        viewport_width = page.evaluate("window.innerWidth")
        assert body_width <= viewport_width + 1
        context.close()
        browser.close()
