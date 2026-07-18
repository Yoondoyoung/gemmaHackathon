#!/usr/bin/env python3
"""Print Gemma prompt payloads from the iPhone to this Mac terminal.

Usage:
  python3 tools/prompt_log_server.py

The iOS app POSTs here on each question (same Wi-Fi).
"""
from __future__ import annotations

import json
import socket
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


PORT = 8765


def _lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            data = {"raw": raw.decode("utf-8", errors="replace")}

        now = datetime.now().strftime("%H:%M:%S")
        print("\n" + "=" * 60)
        print("[%s] Gemma prompt" % now)
        print("=" * 60)
        print("Q: %s" % data.get("question", ""))
        print("image: %s" % data.get("image", "none"))
        hints = data.get("detector_hints", "")
        if isinstance(hints, (dict, list)):
            print("detector_hints:")
            print(json.dumps(hints, ensure_ascii=False, indent=2))
        else:
            print("detector_hints: %s" % hints)
        print("=" * 60)
        try:
            import sys
            sys.stdout.flush()
        except Exception:
            pass

        self.send_response(204)
        self.end_headers()

    def log_message(self, fmt, *args):
        return


def main():
    ip = _lan_ip()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.allow_reuse_address = True
    print("Listening on http://%s:%s/log" % (ip, PORT), flush=True)
    print("Set GemmaChat.macLogHost to that IP, rebuild the app.", flush=True)
    print("Ctrl+C to stop.\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
