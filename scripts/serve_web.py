"""Static dev server for web/ with correct MIME types and no caching.

Usage: .venv\\Scripts\\python.exe scripts\\serve_web.py [port]
Serves the repo's web/ directory on 127.0.0.1 (default port 8765).
"""
import functools
import http.server
import mimetypes
import sys
from pathlib import Path

mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/octet-stream", ".onnx")

ROOT = Path(__file__).resolve().parents[1] / "web"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def log_message(self, fmt, *args):  # keep console quiet
        pass


handler = functools.partial(NoCacheHandler, directory=str(ROOT))
server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler)
print(f"serving {ROOT} at http://127.0.0.1:{PORT}/", flush=True)
server.serve_forever()
