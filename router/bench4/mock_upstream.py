"""Local test double for OpenRouter's chat API: answers every request without a model. Standard library only.

usage: mock_upstream.py PORT LOG_JSONL
Replies to POST .../chat/completions with one short assistant message ("DONE: nothing to do.") and a usage
block, in JSON or as an SSE stream, and logs each request's path, tool count and message count. Used only
to check that a harness reaches the meter, gets through it, and ends; it never decides a result.
"""
import http.server, json, socketserver, sys, time

PORT, LOG = int(sys.argv[1]), sys.argv[2]


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        with open(LOG, "a") as f:
            f.write(json.dumps({"t": time.time(), "path": self.path, "model": body.get("model"), "stream": body.get("stream"),
                                "tools": len(body.get("tools") or []), "messages": len(body.get("messages") or [])}) + "\n")
        text = "DONE: nothing to do."
        usage = {"prompt_tokens": 1000, "completion_tokens": 10, "total_tokens": 1010,
                 "prompt_tokens_details": {"cached_tokens": 0}}
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for o in ({"id": "gen-mock", "object": "chat.completion.chunk", "model": body.get("model"),
                       "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]},
                      {"id": "gen-mock", "object": "chat.completion.chunk", "model": body.get("model"),
                       "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                      {"id": "gen-mock", "object": "chat.completion.chunk", "model": body.get("model"), "choices": [], "usage": usage}):
                b = b"data: " + json.dumps(o).encode() + b"\n\n"
                self.wfile.write(b"%x\r\n%s\r\n" % (len(b), b))
            b = b"data: [DONE]\n\n"
            self.wfile.write(b"%x\r\n%s\r\n0\r\n\r\n" % (len(b), b))
            return
        o = {"id": "gen-mock", "object": "chat.completion", "model": body.get("model"),
             "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}], "usage": usage}
        b = json.dumps(o).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


S(("127.0.0.1", PORT), H).serve_forever()
