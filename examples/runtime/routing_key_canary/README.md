# Isolated Routing-Key Canary

This example tests an opt-in **front gateway** in front of an **unmodified**
SGLang Model Gateway with `manual` / `min_group` routing. It is not a new native
router flag, a production rollout, or a fix already deployed to DeepSeek.

## Contract

- With `--require-routing-key`, generation POSTs require exactly one
  `X-SMG-Routing-Key`: 1-256 visible ASCII bytes, no whitespace or comma.
- Missing/empty keys return HTTP 400 with `invalid_request_error` and
  `missing_routing_key`, before reading the request body or selecting a worker.
  Malformed/duplicate keys return 400 with `invalid_routing_key`.
- The five guarded paths are `/generate`, `/v1/chat/completions`,
  `/v1/completions`, `/v1/responses`, and `/v1/messages`. The last path is guarded
  defensively; this does not add Anthropic support to an upstream router that
  lacks it. Trailing slashes are also guarded, without making them valid routes.
- Health, readiness, metrics, model discovery, and other non-generation requests
  do not need a key. A `Connection` header cannot strip the required key in transit.
- Valid keys and bodies pass through without rewriting. SSE is streamed, not
  buffered until completion. Client disconnects close the upstream response;
  failed partial streams never get a fabricated finish reason or `[DONE]`.
- Strict validation is opt-in; the default is permissive. All listeners and
  upstreams are literal loopback addresses. No production URL is accepted.

Use one stable, non-secret ID per conversation/trajectory, including retries.
Do not generate a different key for every request or reuse one key globally.
Key presence does not authenticate a user, guarantee cache residency, survive
every failover, or share an assignment map across multiple router replicas.

## Resource And Safety Boundaries

Run only inside an ordinary ACK H20 Pod with a current-owner native allocation.
The runner requires a real CPFS `/workspace` mount and the authorized project
root under `/workspace/jlw/`. It refuses preexisting GPU compute processes,
reserves unique loopback ports, records child PIDs/start times, and stops only
its own process groups. It does not use host namespaces, production endpoints,
cloud expansion, existing worker weights, service restarts, or Pod deletion.

The runner makes a small, seeded random Llama checkpoint and local tokenizer
without model downloads. Two actual SGLang workers use GPUs 0 and 1; all eight
reserved H20 devices get a CUDA smoke check. The tiny-model cache result is a
functional experiment, **not** a DeepSeek KV-cache/TTFT benchmark or evidence of
a particular production client's header-loss rate.

## Run Through Native cb

Commit and push this source to your own fork before transferring its archive.
Create the project `workdir/`, `scripts/`, `logs/`, `data/`, and `checkpoints/`
through the current allocation after verifying the actual mount. Transfer
files using `cb file`, never a business `kubectl exec` or direct SSH.

```bash
cb alloc <allocation> run -- bash -lc '
  cd /workspace/jlw/<project>/workdir/routing_key_canary
  nohup python -u run_canary.py --root /workspace/jlw/<project> \
    --attempt <unique-ascii-attempt> --allocation-id <allocation> \
    --node-id <NodeId> --source-commit <pushed-commit> \
    > /workspace/jlw/<project>/logs/<unique-ascii-attempt>-runner.log 2>&1 < /dev/null &
  echo $!
'
```

The runner has a bounded 1500-second async deadline, independent attempt logs,
and a terminal JSON result. Check that exact attempt's real process and result;
do not replay an ambiguous launch. After it exits, retrieve the result using
`cb file get`, verify cleanup, then explicitly release your allocation if done.
Allocation release preserves the Pod and its Kubernetes GPU reservation;
deleting the Pod needs separate user authorization.

## Evidence

The result separates these checks:

1. CPU guard unit regressions and live 400 rejection before the SMG policy and
   worker generation counters change; keyless health/model discovery.
2. Real GPU generation: 24 two-turn trajectories without a key versus 24 with
   a stable key. Worker probes attest each request ID and the key's SHA256,
   not the raw key. Engine `cached_tokens` is distinct from manual-map hits.
3. A complete guarded GPU chat SSE with finish reason and `[DONE]`.
4. Deterministic **mock-only** SSE every 200 ms for four seconds: a two-second
   router total deadline truncates it; an eight-second test-only deadline lets
   it finish. A separate cancellation must reach the mock worker. This scaled
   test does not prove causality for any particular 1800-second production stream.

For production, prefer validation inside the authenticated ingress/native
generation route, before concurrency queues and policy selection. First audit
callers in observation mode, fix header forwarding and stable IDs, then enable
strict rejection in stages. Do not label 400 errors retryable, log raw keys, or
use a missing-key policy error as a misleading 503. This loopback canary is not
a full ingress implementation or a production-ready deployment package.
