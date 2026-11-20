"""fetch(): an expired TLS certificate falls back to an unverified fetch, and only the sha256 pin decides. Offline."""
import hashlib, io, os, ssl, subprocess, sys, tempfile, unittest, urllib.error
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
import tasks  # noqa: E402

GOOD = b"pinned bytes\n"


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def opener(payload):
    calls = []

    def urlopen(url, timeout=None, context=None):
        calls.append(context)
        if context is None:
            raise urllib.error.URLError(ssl.SSLCertVerificationError("certificate has expired"))
        return Resp(payload)
    return urlopen, calls


class Fetch(unittest.TestCase):
    def run_fetch(self, payload):
        src = {"x.data": ("https://example.invalid/x.data", hashlib.sha256(GOOD).hexdigest())}
        urlopen, calls = opener(payload)
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(tasks.SOURCES, src, clear=True), \
                mock.patch.object(tasks.urllib.request, "urlopen", urlopen):
            return tasks.fetch("x.data", d), calls

    def test_expired_cert_good_bytes_accepted(self):
        path, calls = self.run_fetch(GOOD)
        self.assertEqual(len(calls), 2)
        self.assertIsNotNone(calls[1])

    def test_expired_cert_wrong_bytes_refused(self):
        with self.assertRaises(SystemExit):
            self.run_fetch(b"tampered\n")

    def test_other_errors_still_raise(self):
        def urlopen(url, timeout=None, context=None):
            raise urllib.error.URLError("no route to host")
        src = {"x.data": ("https://example.invalid/x.data", hashlib.sha256(GOOD).hexdigest())}
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(tasks.SOURCES, src, clear=True), \
                mock.patch.object(tasks.urllib.request, "urlopen", urlopen), self.assertRaises(urllib.error.URLError):
            tasks.fetch("x.data", d)


class Decide(unittest.TestCase):
    def test_failure_exits_3(self):
        path = os.path.join(os.path.dirname(__file__), "decide.py")
        p = subprocess.run([sys.executable, path, "/no/such/results.json"], capture_output=True)
        self.assertEqual(p.returncode, 3)


if __name__ == "__main__":
    unittest.main()
