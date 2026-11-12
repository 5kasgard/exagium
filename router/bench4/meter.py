"""bench-4 meter: the only door between an agent container and the model. Standard library only.

usage: meter.py LISTEN_HOST PORT LOG_JSONL MODEL CAP
Env: OPENROUTER_API_KEY (stays on the host; containers get a dummy key).

Forwards POST {/v1,/api/v1}/chat/completions (and /responses, /messages) to https://openrouter.ai/api/v1/...
- Rewrites "model" to MODEL whatever the harness sent (every harness talks to the same worker) and
  adds usage accounting.
- Refuses with 429 once CAP model calls have been forwarded: the same budget for every harness,
  counted in model calls rather than in each harness's own idea of a "step".
- Logs one JSON line per call: t, path, status, stream, prompt/completion/cached/reasoning tokens,
  latency, and the generation id. Streaming responses pass through chunk by chunk; usage is read
  from the stream as it passes.
- GET /v1/models (and /api/v1/models) answers locally with MODEL, so harnesses that list models work.
- GET /meter/count returns {"calls": n} for the snapshot poller.
"""
import http.server, json, os, re, socketserver, sys, threading, time, urllib.error, urllib.request

HOST, PORT, LOG, MODEL, CAP = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5])
KEY = os.environ["OPENROUTER_API_KEY"]
UPSTREAM = os.environ.get("BENCH4_UPSTREAM", "https://openrouter.ai/api/v1")  # override only for local tests
lock = threading.Lock()
state = {"calls": 0, "refused": 0}
PATHS = re.compile(r"^(?:/api)?(?:/v1)?/(chat/completions|responses|messages|completions)$")


def log(rec):
    with lock, open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def usage_of(obj):
    u = (obj or {}).get("usage") or {}
    d = u.get("prompt_tokens_details") or u.get("input_tokens_details") or {}
    c = u.get("completion_tokens_details") or u.get("output_tokens_details") or {}
    return {"prompt": u.get("prompt_tokens", u.get("input_tokens")),
            "completion": u.get("completion_tokens", u.get("output_tokens")),
            "cached": d.get("cached_tokens"), "reasoning": c.get("reasoning_tokens"), "cost": u.get("cost")}


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/meter/count":
            with lock:
                return self._send(200, dict(state))
        if re.match(r"^(?:/api)?(?:/v1)?/models$", p):
            return self._send(200, {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "bench4",
                                                                  "context_length": 262144}]})
        self._send(404, {"error": {"message": "not served by the bench-4 meter"}})

    def do_POST(self):
        p = self.path.split("?")[0]
        m = PATHS.match(p)
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if not m:
            log({"t": time.time(), "path": p, "status": "refused_path"})
            return self._send(404, {"error": {"message": "path not served"}})
        with lock:
            if state["calls"] >= CAP:
                state["refused"] += 1
                refuse = True
            else:
                state["calls"] += 1
                n = state["calls"]
                refuse = False
        if refuse:
            log({"t": time.time(), "path": p, "status": "refused_cap"})
            return self._send(429, {"error": {"message": f"bench-4 call cap of {CAP} reached", "code": 429}})
        try:
            req = json.loads(body or b"{}")
        except ValueError:
            return self._send(400, {"error": {"message": "body is not JSON"}})
        asked = req.get("model")
        req["model"] = MODEL
        if m.group(1) == "chat/completions":
            req["usage"] = {"include": True}
            if req.get("stream"):
                req.setdefault("stream_options", {})["include_usage"] = True
        stream = bool(req.get("stream"))
        t0 = time.time()
        up = urllib.request.Request(f"{UPSTREAM}/{m.group(1)}", data=json.dumps(req).encode(), method="POST",
                                    headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
                                             "HTTP-Referer": "https://github.com/5kasgard/exagium", "X-Title": "exagium bench-4"})
        rec = {"t": t0, "n": n, "path": p, "asked_model": asked, "stream": stream}
        try:
            r = urllib.request.urlopen(up, timeout=600)
            status = r.status
        except urllib.error.HTTPError as e:
            r, status = e, e.code
        except Exception as e:  # network failure: the harness sees a 502
            rec.update(status=502, error=f"{type(e).__name__}: {e}"[:300], latency=round(time.time() - t0, 2))
            log(rec)
            return self._send(502, {"error": {"message": "upstream unreachable"}})
        ctype = r.headers.get("Content-Type", "application/json")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        if stream and status == 200:
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            last_usage, gen, buf = None, None, b""
            while True:
                chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
                if not chunk:
                    break
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.startswith(b"data:") and b"{" in line:
                        try:
                            o = json.loads(line[5:].strip())
                        except ValueError:
                            continue
                        gen = gen or o.get("id")
                        if o.get("usage"):
                            last_usage = o
            self.wfile.write(b"0\r\n\r\n")
            rec.update(status=status, usage=usage_of(last_usage), id=gen)
        else:
            data = r.read()
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            try:
                o = json.loads(data)
                rec.update(usage=usage_of(o), id=o.get("id"))
                if status != 200:
                    rec["error"] = json.dumps(o.get("error"))[:400]
            except ValueError:
                rec["error"] = data[:300].decode(errors="replace")
            rec["status"] = status
        rec["latency"] = round(time.time() - t0, 2)
        log(rec)


class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    print(f"meter on {HOST}:{PORT} model={MODEL} cap={CAP}", flush=True)
    S((HOST, PORT), H).serve_forever()
