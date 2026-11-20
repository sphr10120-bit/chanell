"""Optional authenticated wake endpoint for a sleeping Railway worker.

The endpoint only wakes the process; it does not change configuration or send
messages. A platform can wake the service before it reaches this authentication
check, so exposing it does not guarantee protection from resource consumption.
"""
import hmac
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def start_wake_server(stop_event):
    port = os.getenv("PORT", "").strip()
    token = os.getenv("WAKE_TOKEN", "").strip()
    if not port or not token:
        return None
    if not port.isdigit() or not 1 <= int(port) <= 65535 or len(token) < 24:
        raise ValueError("PORT must be valid and WAKE_TOKEN must have at least 24 characters")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            provided = self.headers.get("Authorization", "")
            authorized = hmac.compare_digest(provided, "Bearer " + token)
            code = 200 if self.path == "/wake" and authorized else 404
            body = b"awake\n" if code == 200 else b"not found\n"
            self.send_response(code)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            # Do not log request headers, tokens, or client addresses.
            pass

    server = ThreadingHTTPServer(("0.0.0.0", int(port)), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="wake-http")
    thread.start()

    def stop_when_requested():
        stop_event.wait()
        server.shutdown()
        server.server_close()

    threading.Thread(target=stop_when_requested, daemon=True, name="wake-shutdown").start()
    return server
