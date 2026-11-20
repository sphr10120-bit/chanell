import os
import threading
import time
import unittest
from unittest.mock import patch

import requests

from wake import start_wake_server


class WakeTests(unittest.TestCase):
    def test_disabled_without_token(self):
        with patch.dict(os.environ, {"PORT": "8000"}, clear=True):
            self.assertIsNone(start_wake_server(threading.Event()))

    def test_invalid_token_is_rejected(self):
        with patch.dict(os.environ, {"PORT": "8000", "WAKE_TOKEN": "short"}, clear=True):
            with self.assertRaises(ValueError):
                start_wake_server(threading.Event())

    def test_authentication(self):
        import socket
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        stop = threading.Event()
        token = "secret-wake-token-at-least-24-characters"
        with patch.dict(os.environ, {"PORT": str(port), "WAKE_TOKEN": token}, clear=True):
            server = start_wake_server(stop)
            url = f"http://127.0.0.1:{port}/wake"
            self.assertEqual(requests.get(url, timeout=2).status_code, 404)
            self.assertEqual(requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=2).status_code, 200)
            self.assertEqual(requests.get(url + "?x=1", headers={"Authorization": f"Bearer {token}"}, timeout=2).status_code, 404)
            stop.set()
            time.sleep(.1)
            self.assertIsNotNone(server)


if __name__ == "__main__":
    unittest.main()
