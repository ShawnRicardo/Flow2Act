# ruff: noqa: N801, N802, N999
"""Serve a live, offline-capable loss dashboard for one Flow2Act run."""

import argparse
import contextlib
import gzip
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import subprocess
import threading
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    "loss",
    "action",
    "static",
    "motion2d",
    "motion3d",
    "lr",
    "grad_norm",
    "samples_per_second",
    "update_seconds",
    "elapsed_seconds",
    "eta_seconds",
)


class Flow2ActMetrics:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir).resolve()
        self.path = self.run_dir / "metrics_flow2act.jsonl"
        self.config = json.loads((self.run_dir / "config_flow2act.json").read_text())
        self.lock = threading.Lock()
        self.revision = None
        self.payload = None

    def snapshot(self):
        with self.lock:
            with self.path.open("rb") as stream:
                stat = os.fstat(stream.fileno())
                revision = f"{stat.st_mtime_ns}-{stat.st_size}"
                if revision == self.revision:
                    return self.payload
                content = stream.read(stat.st_size)
            records = {}
            invalid = duplicates = partial = 0
            for line in content.splitlines(keepends=True):
                if not line.endswith(b"\n"):
                    partial += 1
                    continue
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    step = record["update"]
                    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
                        raise ValueError("invalid update")
                    clean = {}
                    for key in FIELDS:
                        value = record.get(key)
                        clean[key] = value if isinstance(value, int | float) and math.isfinite(value) else None
                    duplicates += step in records
                    records[step] = clean
                except (ValueError, KeyError, TypeError):
                    invalid += 1
            steps = sorted(records)
            self.payload = {
                "revision": revision,
                "run_name": self.run_dir.name,
                "config": self.config,
                "mtime_ms": stat.st_mtime * 1000,
                "updates": steps,
                "columns": {key: [records[step][key] for step in steps] for key in FIELDS},
                "invalid_lines": invalid,
                "duplicate_updates": duplicates,
                "partial_lines": partial,
            }
            self.revision = revision
            return self.payload


def build_dashboard_flow2act(store, destination, plotly_path):
    """Write only the new dashboard subtree; training artifacts stay read-only."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite dashboard directory: {destination}")
    template = ROOT / "src/openpi/flow2act/loss_dashboard_flow2act.html"
    payload = json.dumps(store.snapshot(), ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    document = template.read_text().replace("__BOOTSTRAP_JSON__", payload)
    javascript = Path(plotly_path).read_bytes()
    destination.mkdir(parents=True)
    destination.chmod(0o777)
    for name, content in (("index.html", document.encode()), ("plotly.min.js", javascript)):
        path = destination / name
        path.write_bytes(content)
        path.chmod(0o777)


def handler_flow2act(store, destination):
    class Flow2ActHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlsplit(self.path).path
            if route == "/api/metrics":
                try:
                    payload = store.snapshot()
                    etag = '"' + payload["revision"] + '"'
                    if self.headers.get("If-None-Match") == etag:
                        self.send_response(304)
                        self.send_header("ETag", etag)
                        self.end_headers()
                        return
                    content = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
                except (OSError, ValueError) as error:
                    self.send_error(503, str(error))
                    return
                mime = "application/json; charset=utf-8"
            elif route in ("/", "/index.html", "/plotly.min.js"):
                filename = "plotly.min.js" if route == "/plotly.min.js" else "index.html"
                content = (destination / filename).read_bytes()
                mime = "application/javascript" if filename.endswith(".js") else "text/html; charset=utf-8"
                etag = None
            elif route == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            else:
                self.send_error(404)
                return
            compressed = "gzip" in self.headers.get("Accept-Encoding", "")
            if compressed:
                content = gzip.compress(content, compresslevel=3)
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Vary", "Accept-Encoding")
            if etag:
                self.send_header("ETag", etag)
            if compressed:
                self.send_header("Content-Encoding", "gzip")
            self.end_headers()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(content)

    return Flow2ActHandler


def main_flow2act():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--dashboard-dir", type=Path)
    parser.add_argument(
        "--serve-existing", action="store_true", help="Serve previously generated HTML without rewriting it"
    )
    parser.add_argument("--plotly-js", type=Path, default=ROOT / "assets/visualization_flow2act/plotly-2.35.2.min.js")
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    store = Flow2ActMetrics(args.run_dir)
    destination = (args.dashboard_dir or store.run_dir / "loss_dashboard_flow2act").resolve()
    # Reserve the port before publishing any dashboard files.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_flow2act(store, destination))
    try:
        if args.serve_existing:
            for name in ("index.html", "plotly.min.js"):
                if not (destination / name).is_file():
                    raise FileNotFoundError(destination / name)
        else:
            build_dashboard_flow2act(store, destination, args.plotly_js)
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"Flow2Act loss dashboard: {url}", flush=True)
        print(f"Source: {store.path}\nHTML: {destination / 'index.html'}", flush=True)
        if args.open_browser:
            browser = os.environ.get("BROWSER")
            if browser and Path(browser).is_file():
                # VS Code's browser helper resolves the remote URI through its tunnel service.
                subprocess.Popen([browser, url])
            else:
                print("No VS Code BROWSER helper; open the forwarded URL manually.", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main_flow2act()
