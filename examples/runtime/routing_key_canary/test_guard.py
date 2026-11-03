"""Small CPU-only guard regressions; run on the allocated canary Pod."""

import unittest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from gateway import GENERATION_PATHS, create_app, validate_routing_key


class KeyValidationTests(unittest.TestCase):
    def test_key_shapes(self):
        cases = [
            ([], "missing"),
            ([(b"x-smg-routing-key", b"")], "empty"),
            ([(b"x-smg-routing-key", b" \t")], "empty"),
            ([(b"x-smg-routing-key", b"a b")], "invalid"),
            ([(b"x-smg-routing-key", b"a,b")], "invalid"),
            ([(b"x-smg-routing-key", b"\xff")], "invalid"),
            ([(b"x-smg-routing-key", b"a" * 257)], "too_long"),
            ([(b"x-smg-routing-key", b"a" * 256)], None),
            ([(b"X-SMG-Routing-Key", b"session:uuid-1")], None),
            (
                [(b"x-smg-routing-key", b"same"), (b"X-SMG-Routing-Key", b"same")],
                "duplicate",
            ),
            (
                [(b"x-smg-routing-key", b"a"), (b"connection", b"X-SMG-Routing-Key")],
                "invalid",
            ),
        ]
        for headers, reason in cases:
            with self.subTest(headers=headers):
                self.assertEqual(validate_routing_key(headers), reason)


class GuardProxyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.received = []

        async def upstream(request):
            self.received.append(
                (request.path, request.headers.getall("X-SMG-Routing-Key", []))
            )
            body = await request.read()
            return web.Response(
                body=body or b"healthy", headers={"Content-Type": "application/json"}
            )

        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", upstream)
        self.upstream = TestServer(app)
        await self.upstream.start_server()
        self.client = TestClient(
            TestServer(
                create_app(
                    str(self.upstream.make_url("/")),
                    require_key=True,
                    instance_id="unit-test",
                )
            )
        )
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        await self.upstream.close()

    async def test_all_generation_routes_reject_before_upstream(self):
        for path in sorted(GENERATION_PATHS):
            for headers in (
                [],
                [("X-SMG-Routing-Key", "")],
                [("X-SMG-Routing-Key", "a"), ("X-SMG-Routing-Key", "b")],
                [("X-SMG-Routing-Key", "a"), ("Connection", "X-SMG-Routing-Key")],
            ):
                with self.subTest(path=path, headers=headers):
                    async with self.client.post(
                        path, headers=headers, json={}
                    ) as response:
                        self.assertEqual(response.status, 400)
                        error = (await response.json())["error"]
                        self.assertEqual(error["type"], "invalid_request_error")
                        self.assertEqual(error["param"], "X-SMG-Routing-Key")
        self.assertEqual(self.received, [])

    async def test_valid_key_and_body_are_unchanged(self):
        payload = b'{"model":"tiny","stream":true}'
        async with self.client.post(
            "/v1/chat/completions",
            headers={"X-SMG-Routing-Key": "trajectory-123"},
            data=payload,
        ) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(await response.read(), payload)
        self.assertEqual(self.received, [("/v1/chat/completions", ["trajectory-123"])])

    async def test_health_and_models_do_not_need_key(self):
        for path in ("/health", "/readiness", "/v1/models", "/metrics"):
            async with self.client.get(path) as response:
                self.assertEqual(response.status, 200)
                await response.read()
        self.assertEqual(len(self.received), 4)

    async def test_permissive_default_is_backward_compatible(self):
        async with TestClient(
            TestServer(create_app(str(self.upstream.make_url("/"))))
        ) as client:
            async with client.post("/generate", json={}) as response:
                self.assertEqual(response.status, 200)
                await response.read()
        self.assertEqual(len(self.received), 1)


if __name__ == "__main__":
    unittest.main()
