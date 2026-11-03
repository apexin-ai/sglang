"""Deterministic SSE worker for deadline/cancellation contract tests only."""

import argparse
import asyncio
import time

from aiohttp import web


def create_app(instance_id):
    records = {}

    async def health(request):
        return web.json_response({"status": "ok"})

    async def server_info(request):
        return web.json_response(
            {
                "model_path": "canary-mock",
                "served_model_name": "canary-mock",
                "dp_size": 1,
                "max_total_num_tokens": 4096,
            }
        )

    async def model_info(request):
        return web.json_response(
            {"model_path": "canary-mock", "served_model_name": "canary-mock"}
        )

    async def models(request):
        return web.json_response(
            {"object": "list", "data": [{"id": "canary-mock", "object": "model"}]}
        )

    async def state(request):
        return web.json_response({"instance_id": instance_id, "records": records})

    async def chat(request):
        await request.json()
        request_id = request.headers.get("X-Request-Id", "")
        record = {"pulses": 0, "cancelled": False, "completed": False}
        records[request_id] = record
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        started = time.monotonic()
        try:
            # Every 200 ms there is body activity, yet total duration exceeds 2 s.
            for index in range(20):
                await response.write(
                    (
                        'data: {"id":"mock","object":"chat.completion.chunk",'
                        '"model":"canary-mock","choices":[{"index":0,"delta":'
                        '{"content":"tick"},"finish_reason":null}]}\n\n'
                    ).encode()
                )
                record["pulses"] += 1
                await asyncio.sleep(0.2)
            await response.write(
                b'data: {"choices":[{"index":0,"delta":{},'
                b'"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
            )
            await response.write_eof()
            record["completed"] = True
            return response
        except (asyncio.CancelledError, OSError):
            record["cancelled"] = True
            raise
        finally:
            record["elapsed_seconds"] = round(time.monotonic() - started, 6)

    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_get("/health_generate", health)
    app.router.add_get("/get_server_info", server_info)
    app.router.add_get("/server_info", server_info)
    app.router.add_get("/get_model_info", model_info)
    app.router.add_get("/model_info", model_info)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/canary_mock/state", state)
    app.router.add_post("/v1/chat/completions", chat)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--instance-id", required=True)
    args = parser.parse_args()
    web.run_app(
        create_app(args.instance_id),
        host="127.0.0.1",
        port=args.port,
        handler_cancellation=True,
        access_log=None,
        print=None,
    )


if __name__ == "__main__":
    main()
