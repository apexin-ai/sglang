"""Run isolated key-affinity/cache and scaled SSE contract tests on an ACK Pod."""

import argparse
import asyncio
import gc
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from aiohttp import ClientError, ClientSession, ClientTimeout

from gateway import GENERATION_PATHS

HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("x") as output:
        json.dump(value, output, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def workspace_mount():
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        fields = left.split()
        if fields[4] == "/workspace":
            filesystem, source, _ = right.split(" ", 2)
            require(
                "fuse" in filesystem or "efc" in filesystem or "bmcpfs" in filesystem,
                "workspace is not a real CPFS mount; no local-directory fallback",
            )
            return {
                "mountpoint": "/workspace",
                "filesystem": filesystem,
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "namespace": os.readlink("/proc/self/ns/mnt"),
            }
    raise AssertionError("/workspace has no distinct CPFS mount")


class Runtime:
    def __init__(self, root, attempt, allocation, node, source_commit):
        self.mount = workspace_mount()
        self.root = root.resolve()
        require(
            str(self.root).startswith("/workspace/jlw/"),
            "test root must be the authorized project directory on CPFS",
        )
        require(
            self.root.is_dir(), "create the project directory through owner cb first"
        )
        self.attempt = attempt
        self.logs = self.root / "logs" / attempt
        self.logs.mkdir(mode=0o700)
        self.result = self.root / "data" / (attempt + ".json")
        require(not self.result.exists(), "refuse to replay an existing attempt")
        self.process_state = self.root / "data" / (attempt + "-processes.json")
        require(
            not self.process_state.exists(), "attempt already has a process journal"
        )
        self.children = []
        self.reserved = {}
        self.report = {
            "schema": "sglang.routing-key-canary.v1",
            "attempt_id": attempt,
            "allocation_id": allocation,
            "node_id": node,
            "source_commit": source_commit,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "workspace": self.mount,
            "implementation": "opt-in loopback front gateway; unmodified SMG binary",
            "model_kind": "synthetic tiny Llama; not DeepSeek production weights",
            "production_mutations": [],
            "completed": False,
            "business_success": False,
            "stages": {},
            "processes": [],
        }

    def reserve_port(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        self.reserved[port] = sock
        return port

    def start(self, name, argv, ports):
        for port in ports:
            self.reserved.pop(port).close()
        log = (self.logs / (name + ".log")).open("xb")
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=HERE,
        )
        ticks = (
            Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
        )
        self.children.append((name, process, ticks, log))
        self.report["processes"].append(
            {"name": name, "pid": process.pid, "start_ticks": ticks, "ports": ports}
        )
        save_json(
            self.process_state,
            {"attempt_id": self.attempt, "processes": self.report["processes"]},
        )
        print(f"Started isolated {name} (pid={process.pid})", flush=True)
        return process

    def check_children(self):
        for name, process, _, _ in self.children:
            require(process.poll() is None, f"{name} exited; inspect its attempt log")

    def stop(self):
        outcomes = []
        for name, process, ticks, log in reversed(self.children):
            if process.poll() is None:
                stat = Path(f"/proc/{process.pid}/stat")
                require(
                    stat.read_text().rsplit(")", 1)[1].split()[19] == ticks,
                    "PID identity changed; refuse to signal a replacement",
                )
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=15)
            members = self.group_members(process.pid)
            if members:
                # A server can exit before its multiprocessing children. The
                # recorded session/group is ours; refuse a reused group leader.
                leader = Path(f"/proc/{process.pid}/stat")
                if leader.exists():
                    require(
                        leader.read_text().rsplit(")", 1)[1].split()[19] == ticks,
                        "group leader identity changed; refuse cleanup",
                    )
                os.killpg(process.pid, signal.SIGKILL)
                deadline = time.monotonic() + 15
                while self.group_members(process.pid) and time.monotonic() < deadline:
                    time.sleep(0.1)
                require(
                    not self.group_members(process.pid),
                    "own process group did not stop",
                )
            outcomes.append({"name": name, "returncode": process.returncode})
            log.close()
        for sock in self.reserved.values():
            sock.close()
        self.report["cleanup"] = {
            "own_child_processes_stopped": True,
            "outcomes": outcomes,
        }

    @staticmethod
    def group_members(group):
        members = []
        for stat in Path("/proc").glob("[0-9]*/stat"):
            try:
                fields = stat.read_text().rsplit(")", 1)[1].split()
                if fields[0] != "Z" and int(fields[2]) == group:
                    members.append(int(stat.parent.name))
            except (OSError, ValueError, IndexError):
                pass
        return members


async def ready(session, process, url, *, seconds=180, instance=None):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        require(
            process.poll() is None, f"startup exited for {url}; inspect attempt log"
        )
        try:
            async with session.get(url, timeout=ClientTimeout(total=5)) as response:
                if response.status == 200:
                    if instance is not None:
                        body = await response.json()
                        require(
                            body.get("instance_id") == instance,
                            "port is not our canary",
                        )
                    else:
                        await response.read()
                    return
        except (ClientError, asyncio.TimeoutError):
            pass
        await asyncio.sleep(1)
    raise AssertionError(f"startup did not become ready: {url}")


async def start_proxy(runtime, session, name, upstream, *, worker_id=None):
    port = runtime.reserve_port()
    argv = [
        sys.executable,
        str(HERE / "gateway.py"),
        "--upstream",
        upstream,
        "--port",
        str(port),
        "--instance-id",
        runtime.attempt,
    ]
    if worker_id:
        argv.extend(["--worker-id", worker_id])
    else:
        argv.append("--require-routing-key")
    process = runtime.start(name, argv, [port])
    url = f"http://127.0.0.1:{port}"
    diagnostic = "/canary_worker/state" if worker_id else "/canary_guard/state"
    await ready(session, process, url + diagnostic, instance=runtime.attempt)
    return url


async def start_router(runtime, session, name, workers, *, timeout, dp_aware=True):
    port, metric_port = runtime.reserve_port(), runtime.reserve_port()
    argv = [
        sys.executable,
        "-m",
        "sglang_router.launch_router",
        "--worker-urls",
        *workers,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--policy",
        "manual",
        "--assignment-mode",
        "min_group",
        "--request-timeout-secs",
        str(timeout),
        "--disable-retries",
        "--worker-startup-timeout-secs",
        "120",
        "--worker-startup-check-interval",
        "1",
        "--prometheus-host",
        "127.0.0.1",
        "--prometheus-port",
        str(metric_port),
        "--shutdown-grace-period-secs",
        "1",
        "--log-level",
        "warn",
    ]
    if dp_aware:
        argv.append("--dp-aware")
    process = runtime.start(name, argv, [port, metric_port])
    url = f"http://127.0.0.1:{port}"
    await ready(session, process, url + "/readiness")
    topology = await state(session, url + "/workers")
    observed = {worker["url"].split("@", 1)[0] for worker in topology["workers"]}
    require(
        observed == set(workers),
        "router port does not point exclusively to our canary workers",
    )
    require(process.poll() is None, "our router exited before topology verification")
    runtime.report.setdefault("routers", []).append(
        {"name": name, "url": url, "worker_urls": sorted(observed), "timeout": timeout}
    )
    return url, f"http://127.0.0.1:{metric_port}/metrics"


async def state(session, url):
    async with session.get(url) as response:
        require(response.status == 200, "canary diagnostic read failed")
        return await response.json()


async def worker_records(session, workers):
    result = {}
    total = 0
    for worker in workers:
        body = await state(session, worker + "/canary_worker/state")
        total += body["counts"].get("generation_started", 0)
        for record in body["records"]:
            request_id = record["request_id"]
            require(
                request_id not in result,
                "request reached multiple workers; no replay expected",
            )
            result[request_id] = record
    return total, result


async def guard_tests(runtime, session, guard, workers, metrics_url):
    await generate(
        session,
        guard,
        [123] * 32,
        runtime.attempt + ":guard-warmup",
        runtime.attempt + ":warmup",
    )
    before, _ = await worker_records(session, workers)
    async with session.get(metrics_url) as response:
        policy_before = await response.text()
    invalid = [
        [],
        [("X-SMG-Routing-Key", "")],
        [("X-SMG-Routing-Key", " ")],
        [("X-SMG-Routing-Key", "a"), ("X-SMG-Routing-Key", "b")],
        [("X-SMG-Routing-Key", "x" * 257)],
        [("X-SMG-Routing-Key", "a b")],
        [("X-SMG-Routing-Key", "a,b")],
        [("X-SMG-Routing-Key", "\u00e9")],
        [("X-SMG-Routing-Key", "a"), ("Connection", "X-SMG-Routing-Key")],
    ]
    rejected = 0
    for path in sorted(GENERATION_PATHS):
        for headers in invalid:
            payload = {
                "model": "routing-key-tiny",
                "messages": [{"role": "user", "content": "tok001"}],
                "prompt": "tok001",
                "input": "tok001",
                "input_ids": [150] * 32,
                "sampling_params": {"max_new_tokens": 1},
                "max_tokens": 1,
            }
            async with session.post(
                guard + path, headers=headers, json=payload
            ) as response:
                require(response.status == 400, f"key guard failed for {path}")
                error = (await response.json())["error"]
                require(
                    error["type"] == "invalid_request_error", "wrong error contract"
                )
                require(error["param"] == "X-SMG-Routing-Key", "wrong error parameter")
                rejected += 1
    after, _ = await worker_records(session, workers)
    require(before == after, "rejected generation reached a worker")
    async with session.get(metrics_url) as response:
        policy_after = await response.text()

    def policy(text):
        return sorted(
            line
            for line in text.splitlines()
            if line.startswith("smg_manual_policy_branch_total")
        )

    require(bool(policy(policy_before)), "router manual-policy metric is missing")
    require(
        policy(policy_before) == policy(policy_after), "rejection reached router policy"
    )
    health = {}
    for path in ("/health", "/readiness", "/v1/models"):
        async with session.get(guard + path) as response:
            health[path] = response.status
            await response.read()
            require(
                response.status == 200, f"keyless health/model discovery failed: {path}"
            )
    runtime.report["stages"]["strict_guard"] = {
        "rejected_requests": rejected,
        "http_status": 400,
        "generation_worker_count_delta": after - before,
        "router_policy_count_unchanged": True,
        "keyless_non_generation": health,
    }


def sse_objects(body):
    result, done = [], False
    body = body.replace(b"\r\n", b"\n")
    # A failed stream can end midway through a frame. Count complete frames only.
    for frame in body.split(b"\n\n")[:-1]:
        for line in frame.splitlines():
            if line.startswith(b"data:"):
                value = line[5:].strip()
                if value == b"[DONE]":
                    done = True
                elif value:
                    result.append(json.loads(value))
    return result, done


async def stream_request(session, url, payload, headers):
    started = time.monotonic()
    first = None
    chunks = []
    error = None
    status_code = None
    try:
        async with session.post(url, json=payload, headers=headers) as response:
            status_code = response.status
            async for chunk in response.content.iter_any():
                if first is None and chunk:
                    first = time.monotonic() - started
                chunks.append(chunk)
                require(
                    sum(map(len, chunks)) < 2 * 1024 * 1024,
                    "bounded smoke response exceeded",
                )
    except (ClientError, asyncio.TimeoutError) as exc:
        error = type(exc).__name__
    objects, done = sse_objects(b"".join(chunks))
    return {
        "status": status_code,
        "elapsed_seconds": time.monotonic() - started,
        "first_sse_seconds": first,
        "objects": objects,
        "done": done,
        "stream_error": error,
    }


async def generate(session, router, tokens, request_id, key):
    headers = {"X-Request-Id": request_id}
    if key is not None:
        headers["X-SMG-Routing-Key"] = key
    response = await stream_request(
        session,
        router + "/generate",
        {
            "input_ids": tokens,
            "sampling_params": {
                "max_new_tokens": 8,
                "temperature": 0,
                "ignore_eos": True,
            },
            "stream": True,
        },
        headers,
    )
    require(
        response["status"] == 200 and response["stream_error"] is None,
        "GPU generate failed",
    )
    require(bool(response["objects"]), "engine produced no SSE objects")
    meta = response["objects"][-1]["meta_info"]
    require(bool(meta.get("finish_reason")), "engine stream lacks finish reason")
    require("cached_tokens" in meta, "engine does not expose cache evidence")
    return {
        "request_id": request_id,
        "prompt_tokens": meta["prompt_tokens"],
        "cached_tokens": meta["cached_tokens"],
        "completion_tokens": meta["completion_tokens"],
        "first_sse_seconds": response["first_sse_seconds"],
    }


async def cache_tests(runtime, session, router, guard, probes, engines):
    cohorts = {}
    for mode, endpoint in (("no_key", router), ("stable_key", guard)):
        for engine in engines:
            async with session.post(engine + "/flush_cache") as response:
                require(response.status == 200, "isolated engine cache flush failed")
                await response.read()
        rows = []
        for index in range(24):
            # Unique first token prevents cache sharing between test trajectories.
            prefix = [16 + index] + [64 + index] * 511
            key = (
                runtime.attempt + f":trajectory:{index}"
                if mode == "stable_key"
                else None
            )
            pair = []
            for turn, tokens in enumerate((prefix, prefix + [200 + index] * 16)):
                request_id = runtime.attempt + f":{mode}:{index}:{turn}"
                pair.append(await generate(session, endpoint, tokens, request_id, key))
            rows.append({"trajectory": index, "turns": pair})
        _, records = await worker_records(session, probes)
        switches = 0
        for row in rows:
            for turn in row["turns"]:
                record = records.get(turn["request_id"])
                require(record is not None, "worker cannot attest the request ID")
                turn["worker"] = record["worker_id"]
                if mode == "stable_key":
                    expected = hashlib.sha256(
                        (runtime.attempt + f':trajectory:{row["trajectory"]}').encode()
                    ).hexdigest()
                    require(
                        record["key_sha256"] == expected,
                        "routing key changed in forwarding",
                    )
                else:
                    require(
                        record["key_sha256"] is None,
                        "baseline unexpectedly supplied a key",
                    )
            switches += row["turns"][0]["worker"] != row["turns"][1]["worker"]
        second_turns = [row["turns"][1] for row in rows]
        cohorts[mode] = {
            "trajectories": len(rows),
            "worker_switches_between_turns": switches,
            "same_worker_fraction": 1 - switches / len(rows),
            "second_turn_prompt_tokens": sum(
                turn["prompt_tokens"] for turn in second_turns
            ),
            "second_turn_cached_tokens": sum(
                turn["cached_tokens"] for turn in second_turns
            ),
            "second_turn_cache_fraction": sum(
                turn["cached_tokens"] for turn in second_turns
            )
            / sum(turn["prompt_tokens"] for turn in second_turns),
            "rows": rows,
        }
    require(
        cohorts["stable_key"]["worker_switches_between_turns"] == 0,
        "stable key lost affinity",
    )
    require(
        cohorts["stable_key"]["second_turn_cached_tokens"] > 0,
        "stable key has no engine cache hits",
    )
    runtime.report["stages"]["gpu_cache"] = cohorts

    chat = await stream_request(
        session,
        guard + "/v1/chat/completions",
        {
            "model": "routing-key-tiny",
            "messages": [{"role": "user", "content": "tok020 tok021 tok022"}],
            "max_tokens": 8,
            "stream": True,
        },
        {
            "X-SMG-Routing-Key": runtime.attempt + ":chat",
            "X-Request-Id": runtime.attempt + ":chat",
        },
    )
    finish = any(
        choice.get("finish_reason")
        for obj in chat["objects"]
        for choice in obj.get("choices", [])
    )
    require(
        chat["status"] == 200
        and chat["done"]
        and finish
        and chat["stream_error"] is None,
        "guarded GPU chat SSE did not finish cleanly",
    )
    runtime.report["stages"]["gpu_chat_sse"] = {
        "http_status": chat["status"],
        "finish_reason_seen": bool(finish),
        "done_seen": chat["done"],
        "first_sse_seconds": chat["first_sse_seconds"],
    }


def gpu_smoke_and_model(runtime):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    active = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    require(
        not active, "GPU compute processes already exist; do not take over or kill them"
    )
    require(
        torch.cuda.device_count() == 8, "the reservation is not an eight-GPU H20 Pod"
    )
    devices = []
    for index in range(8):
        name = torch.cuda.get_device_name(index)
        require("H20" in name, "reported hardware differs from the authorized H20 pool")
        with torch.cuda.device(index):
            tensor = torch.ones((64, 64), device=f"cuda:{index}")
            require((tensor @ tensor).sum().item() == 262144, "CUDA smoke failed")
            del tensor
            torch.cuda.empty_cache()
        devices.append({"index": index, "reported_name": name, "cuda_smoke": True})
    runtime.report["stages"]["gpu_readiness"] = {
        "devices": devices,
        "preexisting_compute_processes": 0,
        "multi_node_nccl_tested": False,
    }
    model_path = runtime.root / "checkpoints" / ("tiny-llama-" + runtime.attempt)
    model_path.mkdir(mode=0o700)
    torch.manual_seed(1234)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=512,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=4096,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    ).half()
    model.save_pretrained(model_path)
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    vocab.update({f"tok{index:03d}": index + 4 for index in range(508)})
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        model_max_length=4096,
        chat_template="{% for message in messages %}{{ message['role'] }} "
        "{{ message['content'] }} {% endfor %}{% if add_generation_prompt %}assistant{% endif %}",
    )
    fast.save_pretrained(model_path)
    del model, fast, tokenizer
    gc.collect()
    return model_path


async def mock_tests(runtime, session):
    port = runtime.reserve_port()
    process = runtime.start(
        "mock-worker",
        [
            sys.executable,
            str(HERE / "mock_worker.py"),
            "--port",
            str(port),
            "--instance-id",
            runtime.attempt,
        ],
        [port],
    )
    worker = f"http://127.0.0.1:{port}"
    await ready(
        session, process, worker + "/canary_mock/state", instance=runtime.attempt
    )
    routers = {}
    for timeout in (2, 8):
        router, _ = await start_router(
            runtime,
            session,
            f"mock-router-{timeout}",
            [worker],
            timeout=timeout,
            dp_aware=False,
        )
        routers[timeout] = await start_proxy(
            runtime, session, f"mock-guard-{timeout}", router
        )
    results = {}
    payload = {
        "model": "canary-mock",
        "messages": [{"role": "user", "content": "ping"}],
        "stream": True,
        "max_tokens": 32,
    }
    for timeout, guard in routers.items():
        response = await stream_request(
            session,
            guard + "/v1/chat/completions",
            payload,
            {
                "X-SMG-Routing-Key": runtime.attempt + ":mock",
                "X-Request-Id": runtime.attempt + f":timeout:{timeout}",
            },
        )
        results[str(timeout)] = {
            key: value for key, value in response.items() if key != "objects"
        }
        results[str(timeout)]["data_frames"] = len(response["objects"])
    require(
        results["2"]["status"] == 200 and not results["2"]["done"],
        "short total deadline did not reproduce a partial HTTP 200 stream",
    )
    require(
        1.5 < results["2"]["elapsed_seconds"] < 3.5
        and results["2"]["data_frames"] >= 5,
        "short deadline did not expire near 2 s despite continuous SSE activity",
    )
    require(
        results["8"]["status"] == 200
        and results["8"]["done"]
        and results["8"]["stream_error"] is None,
        "longer test-only deadline still truncated SSE",
    )
    runtime.report["stages"]["scaled_deadline_mock_only"] = {
        "worker_body_interval_seconds": 0.2,
        "worker_total_duration_seconds": 4,
        "router_total_deadlines": results,
        "production_request_causality_proven": False,
    }
    request_id = runtime.attempt + ":cancel"
    async with session.post(
        routers[8] + "/v1/chat/completions",
        json=payload,
        headers={
            "X-SMG-Routing-Key": runtime.attempt + ":cancel",
            "X-Request-Id": request_id,
        },
    ) as response:
        require(response.status == 200, "cancellation stream did not start")
        require(bool(await response.content.readany()), "cancellation got no SSE body")
        response.close()
    deadline = time.monotonic() + 3
    record = {}
    while time.monotonic() < deadline:
        body = await state(session, worker + "/canary_mock/state")
        record = body["records"].get(request_id, {})
        if record.get("cancelled"):
            break
        await asyncio.sleep(0.1)
    require(
        record.get("cancelled") and not record.get("completed"),
        "client cancellation did not propagate to the deterministic worker",
    )
    runtime.report["stages"]["mock_cancellation"] = record


async def run(runtime):
    import importlib.metadata

    runtime.report["packages"] = {
        name: importlib.metadata.version(name)
        for name in ("sglang", "sglang-router", "aiohttp", "torch")
    }
    from sglang_router.router_args import RouterArgs

    runtime.report["installed_router_default_timeout_seconds"] = (
        RouterArgs().request_timeout_secs
    )
    unit_log = runtime.logs / "guard-unit-tests.log"
    with unit_log.open("xb") as output:
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "-v", "test_guard"],
            cwd=HERE,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=60,
        )
    require(result.returncode == 0, "guard unit tests failed")
    runtime.report["stages"]["guard_unit_tests"] = {
        "passed": True,
        "returncode": result.returncode,
    }
    model = gpu_smoke_and_model(runtime)
    async with ClientSession(
        timeout=ClientTimeout(total=120), trust_env=False
    ) as session:
        engines, probes = [], []
        for index in range(2):
            port = runtime.reserve_port()
            process = runtime.start(
                f"gpu-worker-{index}",
                [
                    sys.executable,
                    "-m",
                    "sglang.launch_server",
                    "--model-path",
                    str(model),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--base-gpu-id",
                    str(index),
                    "--tp",
                    "1",
                    "--served-model-name",
                    "routing-key-tiny",
                    "--dtype",
                    "float16",
                    "--attention-backend",
                    "triton",
                    "--disable-cuda-graph",
                    "--max-total-tokens",
                    "8192",
                    "--context-length",
                    "4096",
                    "--mem-fraction-static",
                    "0.1",
                    "--enable-metrics",
                ],
                [port],
            )
            engine = f"http://127.0.0.1:{port}"
            await ready(session, process, engine + "/health", seconds=600)
            info = await state(session, engine + "/model_info")
            require(
                info.get("model_path") == str(model),
                "engine is not our pinned model",
            )
            engines.append(engine)
            probes.append(
                await start_proxy(
                    runtime,
                    session,
                    f"worker-probe-{index}",
                    engine,
                    worker_id=f"gpu-{index}",
                )
            )
        router, metrics_url = await start_router(
            runtime, session, "gpu-router", probes, timeout=120
        )
        guard = await start_proxy(runtime, session, "gpu-key-guard", router)
        await guard_tests(runtime, session, guard, probes, metrics_url)
        await cache_tests(runtime, session, router, guard, probes, engines)
        await mock_tests(runtime, session)
        async with session.get(metrics_url) as response:
            (runtime.logs / "router-metrics.txt").write_text(await response.text())
        runtime.check_children()
    runtime.report["completed"] = True
    runtime.report["business_success"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--allocation-id", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    require(
        args.attempt.isascii()
        and all(char.isalnum() or char == "-" for char in args.attempt),
        "attempt must be a unique ASCII filename component",
    )
    runtime = Runtime(
        args.root, args.attempt, args.allocation_id, args.node_id, args.source_commit
    )

    def terminate(signum, frame):
        raise KeyboardInterrupt("bounded canary stopped")

    signal.signal(signal.SIGTERM, terminate)
    try:
        asyncio.run(asyncio.wait_for(run(runtime), timeout=1500))
    except (Exception, KeyboardInterrupt) as exc:
        runtime.report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        try:
            runtime.stop()
        except Exception as exc:
            runtime.report["cleanup"] = {
                "own_child_processes_stopped": False,
                "error_type": type(exc).__name__,
            }
            runtime.report["business_success"] = False
        runtime.report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_json(runtime.result, runtime.report)
    print(
        json.dumps(
            {
                "result": str(runtime.result),
                "business_success": runtime.report["business_success"],
            }
        )
    )
    raise SystemExit(0 if runtime.report["business_success"] else 1)


if __name__ == "__main__":
    main()
