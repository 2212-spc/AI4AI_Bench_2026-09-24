"""Loopback API gateway: the only process in the harness that ever holds the upstream key.

Why this exists.  Both agents under test run with a shell tool, and one of them (the Claude Code agent)
runs an interactive CLI that insists on an API key in its own environment.  If that key were the real one,
a model under test could read its own credential out of `/proc/self/environ`, call the API directly, and -
more to the point for a benchmark - could in principle call *itself* outside the harness's accounting.  So
the sandboxed process is given a placeholder, and this gateway, which runs on the host side of the sandbox
boundary, swaps the placeholder for the real credential on the way out.

It is deliberately dumb: no parsing of bodies, no retries, no caching.  It forwards method, path, body and
the few headers that matter, and streams the response back byte for byte, so that server-sent events from
the Anthropic endpoint and long non-streaming `/v1/responses` calls both work unchanged.

Lifetime.  `start()` binds to loopback on a free port and serves from a daemon thread inside whatever
process called it, and it exports `GATEWAY_URL` into that process's environment so children inherit it.
It therefore dies with the driver and cannot outlive a tool call or collide with anything else on the box.
The key is read once at startup from the operator-only `.sec` directory and is never logged; the request
log records method, path, status and elapsed time only.
"""
import http.client
import http.server
import os
import socketserver
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

SEC = os.environ.get("SEC_DIR", os.path.expanduser("~/.bench_sec"))
PLACEHOLDER = "sk-sandbox-placeholder-not-a-credential"
# Hop-by-hop headers plus the ones we rewrite ourselves.
DROP = {"host", "authorization", "x-api-key", "connection", "keep-alive", "proxy-authorization",
        "te", "trailer", "transfer-encoding", "upgrade", "content-length", "accept-encoding"}
_log_lock = threading.Lock()


def _secret(name):
    with open(os.path.join(SEC, name)) as fh:
        return fh.read().strip()


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream = None          # (scheme, host, port) filled in by start()
    key = None
    logfile = None

    def log_message(self, *a):                       # silence the default stderr spew
        pass

    def _note(self, method, path, status, dt):
        if not self.logfile:
            return
        with _log_lock:
            self.logfile.write("%.0f %s %s %s %.1f\n" % (time.time(), method, path, status, dt))
            self.logfile.flush()

    def _forward(self, method):
        t0 = time.time()
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else None
        base = "%s://%s%s" % (self.upstream[0], self.upstream[1],
                              "" if self.upstream[2] in (80, 443) else ":%d" % self.upstream[2])
        out = {k: v for k, v in self.headers.items() if k.lower() not in DROP}
        # The sandbox never sees the real credential; whatever placeholder it sent is discarded here.
        out["Authorization"] = "Bearer " + self.key
        out["x-api-key"] = self.key
        out["Accept-Encoding"] = "identity"
        req = urllib.request.Request(base + self.path, data=body, headers=out, method=method)
        # urllib is used rather than http.client because this host reaches the internet only through the
        # environment's CONNECT proxy, and urllib's default opener is the thing that already knows how to
        # use it.  Egress policy stays where the platform put it instead of being re-implemented here.
        try:
            r = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e:
            r = e                                              # an HTTPError *is* the response
        except Exception as e:
            self.send_response(502)
            msg = ("upstream unreachable: %s" % type(e).__name__).encode()
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            self._note(method, self.path, "502", time.time() - t0)
            return
        status = r.status if hasattr(r, "status") else r.code
        self.send_response(status)
        for k, v in r.headers.items():
            if k.lower() in ("connection", "keep-alive", "transfer-encoding", "content-length",
                             "content-encoding"):
                continue
            self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        read1 = getattr(r, "read1", None) or r.read            # read1 returns as soon as bytes arrive,
        try:                                                   # which is what keeps SSE deltas prompt
            while True:
                chunk = read1(65536)
                if not chunk:
                    break
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except Exception:
            pass
        finally:
            r.close()
        self._note(method, self.path, str(status), time.time() - t0)

    def do_GET(self):
        self._forward("GET")

    def do_POST(self):
        self._forward("POST")

    def do_DELETE(self):
        self._forward("DELETE")


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start(log_path=None):
    """Bind loopback on a free port, serve in a daemon thread, export GATEWAY_URL.  Returns the URL."""
    if os.environ.get("GATEWAY_URL", "").startswith("http://127.0.0.1"):
        return os.environ["GATEWAY_URL"]                      # already running in this process
    u = urllib.parse.urlparse(_secret("url"))
    Handler.upstream = (u.scheme, u.hostname, u.port or (443 if u.scheme == "https" else 80))
    Handler.key = _secret("key")
    Handler.logfile = open(log_path, "a") if log_path else None
    srv = Server(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = "http://127.0.0.1:%d" % srv.server_address[1]
    os.environ["GATEWAY_URL"] = url
    return url


def _selftest():
    """Prove the swap works end to end: call an upstream read-only endpoint through the gateway with the
    placeholder credential the sandbox would use, and check the upstream accepted it."""
    url = start()
    host, port = url[len("http://"):].split(":")
    c = http.client.HTTPConnection(host, int(port), timeout=60)
    c.request("GET", "/v1/models", headers={"Authorization": "Bearer " + PLACEHOLDER,
                                            "x-api-key": PLACEHOLDER})
    r = c.getresponse()
    n = len(r.read())
    print("gateway", url, "upstream_status", r.status, "bytes", n)
    return 0 if r.status == 200 else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    url = start()
    print(url, flush=True)
    while True:
        time.sleep(3600)
