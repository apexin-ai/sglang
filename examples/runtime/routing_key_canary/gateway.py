"""Loopback-only routing-key guard/probe for an isolated SMG canary.

This is an opt-in front gateway, not a change to SMG's manual routing policy.
"""

import argparse
import asyncio
import hashlib
import ipaddress
import logging
import time
from collections import Counter, deque
from urllib.parse import urlsplit

from aiohttp import ClientError, ClientSession, ClientTimeout, web
from multidict import CIMultiDict

KEY_HEADER = b"x-smg-routing-key"
MAX_KEY_BYTES = 256
GENERATION_PATHS = frozenset(
    {
        "/generate",
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/responses",
        "/v1/messages",
    }
)
HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
LOG = logging.getLogger(__name__)


def generation_request(request):
    return request.method == "POST" and request.path.rstrip("/") in GENERATION_PATHS


def validate_routing_key(raw_headers):
    values = [value for name, value in raw_headers if name.lower() == KEY_HEADER]
    if not values:
        return "missing"
    if len(values) != 1:
        return "duplicate"
    if any(
        token.strip().lower() == KEY_HEADER
        for name, value in raw_headers
        if name.lower() == b"connection"
        for token in value.split(b",")
    ):
        return "invalid"
    key = values[0]
    if not key or not key.strip():
        return "empty"
    if len(key) > MAX_KEY_BYTES:
        return "too_long"
    # Commas may be a coalesced duplicate header; never silently pick one key.
    if any(byte < 0x21 or byte > 0x7E or byte == 0x2C for byte in key):
        return "invalid"
    return None


def forwarded_headers(headers, *, request=False):
    blocked = HOP_HEADERS | {
        token.strip().lower()
        for value in headers.getall("Connection", [])
        for token in value.split(",")
    }
    if request:
        blocked = blocked | {"host"}
    return CIMultiDict(
        (name, value) for name, value in headers.items() if name.lower() not in blocked
    )


def loopback_upstream(url):
    parsed = urlsplit(url)
    try:
        loopback = ipaddress.ip_address(parsed.hostname or "").is_loopback
        port = parsed.port
    except ValueError as exc:
        raise ValueError("upstream must have a literal loopback IP and port") from exc
    if (
        parsed.scheme != "http"
        or not loopback
        or not port
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("only a local canary HTTP upstream is permitted")
    return url.rstrip("/")


class CanaryProxy:
    def __init__(self, upstream, require_key, worker_id, instance_id):
        self.upstream = loopback_upstream(upstream)
        self.require_key = require_key
        self.worker_id = worker_id
        self.instance_id = instance_id
        self.counts = Counter()
        self.records = deque(maxlen=4096)
        self.session = None

    async def session_context(self, app):
        # The router owns the generation deadline; this proxy adds no SSE timer.
        async with ClientSession(
            auto_decompress=False,
            timeout=ClientTimeout(total=None, connect=5),
            trust_env=False,
        ) as session:
            self.session = session
            yield
        self.session = None

    async def state(self, request):
        return web.json_response(
            {
                "instance_id": self.instance_id,
                "worker_id": self.worker_id,
                "require_routing_key": self.require_key,
                "counts": dict(self.counts),
                "records": list(self.records),
            }
        )

    async def metrics(self, request):
        lines = [
            "# TYPE routing_key_guard_requests_total counter",
            *(
                f'routing_key_guard_requests_total{{outcome="{name}"}} {count}'
                for name, count in sorted(self.counts.items())
            ),
        ]
        return web.Response(text="\n".join(lines) + "\n", content_type="text/plain")

    async def proxy(self, request):
        generation = generation_request(request)
        if self.require_key and generation:
            reason = validate_routing_key(request.raw_headers)
            if reason:
                self.counts["rejected_" + reason] += 1
                return web.json_response(
                    {
                        "error": {
                            "message": "Provide one stable, non-empty X-SMG-Routing-Key "
                            "per conversation (1-256 visible ASCII bytes, no commas). "
                            "Reuse it across turns and retries.",
                            "type": "invalid_request_error",
                            "param": "X-SMG-Routing-Key",
                            "code": (
                                "missing_routing_key"
                                if reason in {"missing", "empty"}
                                else "invalid_routing_key"
                            ),
                        }
                    },
                    status=400,
                )

        self.counts["upstream_started"] += 1
        if generation:
            self.counts["generation_started"] += 1
            key_values = [
                value
                for name, value in request.raw_headers
                if name.lower() == KEY_HEADER
            ]
            self.records.append(
                {
                    "request_id": request.headers.get("X-Request-Id", "")[:128],
                    "worker_id": self.worker_id,
                    "key_sha256": (
                        hashlib.sha256(key_values[0]).hexdigest()
                        if len(key_values) == 1
                        else None
                    ),
                    "at_monotonic": time.monotonic(),
                    "path": request.path,
                }
            )

        downstream = None
        try:
            async with self.session.request(
                request.method,
                self.upstream + request.raw_path,
                headers=forwarded_headers(request.headers, request=True),
                data=(
                    request.content.iter_chunked(65536)
                    if request.can_read_body
                    else None
                ),
                allow_redirects=False,
            ) as upstream:
                headers = forwarded_headers(upstream.headers)
                if self.worker_id:
                    headers["X-Canary-Worker"] = self.worker_id
                downstream = web.StreamResponse(status=upstream.status, headers=headers)
                await downstream.prepare(request)
                async for chunk in upstream.content.iter_chunked(65536):
                    await downstream.write(chunk)
                await downstream.write_eof()
                self.counts["upstream_completed"] += 1
                return downstream
        except asyncio.CancelledError:
            self.counts["cancelled"] += 1
            raise
        except (ClientError, OSError, asyncio.TimeoutError) as exc:
            self.counts["upstream_failed"] += 1
            LOG.warning("canary upstream failed: %s", type(exc).__name__)
            if downstream is not None and downstream.prepared:
                # A partial HTTP 200 stream is not success. Do not invent [DONE].
                if request.transport:
                    request.transport.close()
                return downstream
            return web.json_response(
                {"error": {"type": "upstream_error", "code": "upstream_unavailable"}},
                status=502,
            )


def create_app(upstream, *, require_key=False, worker_id=None, instance_id="canary"):
    proxy = CanaryProxy(upstream, require_key, worker_id, instance_id)
    app = web.Application(client_max_size=16 * 1024 * 1024)
    app.cleanup_ctx.append(proxy.session_context)
    diagnostic = "/canary_worker" if worker_id else "/canary_guard"
    app.router.add_get(diagnostic + "/state", proxy.state)
    app.router.add_get(diagnostic + "/metrics", proxy.metrics)
    app.router.add_route("*", "/{tail:.*}", proxy.proxy)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--require-routing-key", action="store_true")
    parser.add_argument("--worker-id")
    parser.add_argument("--instance-id", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    web.run_app(
        create_app(
            args.upstream,
            require_key=args.require_routing_key,
            worker_id=args.worker_id,
            instance_id=args.instance_id,
        ),
        host="127.0.0.1",
        port=args.port,
        handler_cancellation=True,
        access_log=None,
        print=None,
    )


if __name__ == "__main__":
    main()
