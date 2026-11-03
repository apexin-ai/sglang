"""Qwen3.8 prompt-side special-token sanitization (Apex fork, --sanitize-control-tokens).

The default chat path renders the whole conversation to one string and encodes it,
so an added-token string inside a message body ("<|im_end|>" quoted by a user,
"</tool_call>" in a file being written) becomes the control id.  With sanitization
on, body fields are encoded as ordinary text instead, with the same rule the
training data uses (`qwen38_field_encode`, copied from the training repository).

- Fast path: a string scan of everything the template renders.  No added-token
  string (and no partial one at a string edge) -> the caller runs the unchanged
  code, so the ids are identical by construction.
- Slow path (hit only): render the mirror, require mirror text == jinja text,
  field-encode the jinja text, and require that the whole-string encoding of the
  raw tokenizer equals the served tokenizer's ids.  Any failed check -> the
  unchanged ids (counted as a fallback with its reason).

Only the Qwen3.8 tokenizer (added tokens exactly 248044..248076) with a known
Qwen3.8 chat template is supported; anything else leaves the server unchanged.
Not a security boundary: `input_ids`, `/generate` and `/v1/completions` bypass it.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from sglang.srt.parser import qwen38_field_encode as qfe

logger = logging.getLogger(__name__)

REQUEST_KWARG = "sanitize_control_tokens"

# sha256 of the chat templates the mirror covers.
TEMPLATE_V3_SHA256 = "60cf9fa2d00cc99d34fd045d3803c8d28c2515bc7c3f7ce3d9b6206850822b75"
TEMPLATE_BASE_SHA256 = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
KNOWN_TEMPLATES = {TEMPLATE_V3_SHA256: "v3", TEMPLATE_BASE_SHA256: "base"}

# The tokenizer.json the training data was encoded with (logged, not required).
TRAINING_TOKENIZER_SHA256 = (
    "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
)

# Startup probe: every added token plus text whose pre-tokenization differs between
# tokenizer classes (combining marks), so a served tokenizer that does not follow
# tokenizer.json is caught before any request.
_PROBE_TEXT = "Hello, world! हिन्दी café x́ 12345 \n\n  def f(a):\n\treturn a  "

_UNDEFINED = object()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_PROM_COUNTER = None


def _prom_counter():
    """One process-wide Prometheus counter (Chat and Responses servers share it)."""
    global _PROM_COUNTER
    if _PROM_COUNTER is None:
        from prometheus_client import Counter as PromCounter

        _PROM_COUNTER = PromCounter(
            name="sglang:prompt_sanitize_total",
            documentation="Qwen3.8 prompt sanitization events: hit, slow_path, "
            "sanitized, no_change, fallback (label reason)",
            labelnames=["event", "reason"],
        )
    return _PROM_COUNTER


class _Metrics:
    """In-process counters, mirrored to Prometheus when --enable-metrics is set."""

    def __init__(self, enable_prometheus: bool):
        self.counts: Counter = Counter()
        self._prom = None
        if enable_prometheus:
            try:
                self._prom = _prom_counter()
            except Exception as e:  # pragma: no cover - metrics are optional
                logger.warning("prompt sanitize: no Prometheus counter: %s", e)

    def inc(self, event: str, reason: str = "") -> None:
        self.counts[(event, reason)] += 1
        if self._prom is not None:
            self._prom.labels(event=event, reason=reason).inc()


class Qwen38PromptSanitizer:
    def __init__(self, encoder: qfe.FieldEncoder, enable_prometheus: bool = False):
        self.enc = encoder
        tokens = sorted(set(encoder.added.values()), key=len, reverse=True)
        self._token_re = re.compile("|".join(re.escape(t) for t in tokens))
        self._prefixes = {t[:k] for t in tokens for k in range(1, len(t))}
        self._suffixes = {t[k:] for t in tokens for k in range(1, len(t))}
        self._max_len = max(len(t) for t in tokens)
        self._template_kind: Dict[str, Optional[str]] = {}
        self.metrics = _Metrics(enable_prometheus)

    # ------------------------------------------------------------ construction

    @classmethod
    def build(
        cls,
        tokenizer: Any,
        tokenizer_dirs: List[Optional[str]],
        enable_prometheus: bool = False,
    ) -> Optional["Qwen38PromptSanitizer"]:
        """The sanitizer for this server, or None (with a log line) when the gate fails."""
        try:
            added = getattr(tokenizer, "added_tokens_decoder", None) or {}
            ids = {int(i) for i in added}
            if ids != set(range(qfe.LO, qfe.HI + 1)):
                logger.warning(
                    "sanitize-control-tokens=on but the tokenizer's added tokens are "
                    "not exactly %d..%d (got %d tokens); sanitization disabled",
                    qfe.LO,
                    qfe.HI,
                    len(ids),
                )
                return None
            path = None
            for d in tokenizer_dirs:
                if isinstance(d, str) and os.path.isfile(
                    os.path.join(d, "tokenizer.json")
                ):
                    path = os.path.join(d, "tokenizer.json")
                    break
            if path is None:
                logger.warning(
                    "sanitize-control-tokens=on but no local tokenizer.json in %s; "
                    "sanitization disabled",
                    tokenizer_dirs,
                )
                return None
            encoder = qfe.FieldEncoder(path)  # asserts added ids and BPE size LO
            served = {int(i): getattr(t, "content", str(t)) for i, t in added.items()}
            raw = dict(encoder.added)
            if served != raw:
                logger.warning(
                    "sanitize-control-tokens=on but tokenizer.json added tokens differ "
                    "from the served tokenizer; sanitization disabled"
                )
                return None
            probe = _PROBE_TEXT + "".join(raw[i] for i in sorted(raw))
            served_ids = list(tokenizer.encode(probe, add_special_tokens=False))
            if served_ids != list(encoder.encode_full(probe)[0]):
                logger.warning(
                    "sanitize-control-tokens=on but the served tokenizer does not "
                    "encode like %s (tokenizer class %s); sanitization disabled",
                    path,
                    type(tokenizer).__name__,
                )
                return None
        except Exception as e:
            logger.warning(
                "sanitize-control-tokens=on but the sanitizer could not be built "
                "(%s: %s); sanitization disabled",
                type(e).__name__,
                e,
            )
            return None
        logger.info(
            "sanitize-control-tokens=on: tokenizer %s sha256 %s (training tokenizer "
            "%s: %s)",
            path,
            encoder.sha256,
            TRAINING_TOKENIZER_SHA256[:8],
            "same" if encoder.sha256 == TRAINING_TOKENIZER_SHA256 else "DIFFERENT",
        )
        return cls(encoder, enable_prometheus)

    def template_kind(self, chat_template: Any) -> Optional[str]:
        """'v3' / 'base' for a known Qwen3.8 chat template, else None."""
        if not isinstance(chat_template, str):
            return None
        kind = self._template_kind.get(chat_template, _UNDEFINED)
        if kind is _UNDEFINED:
            sha = _sha256_text(chat_template)
            kind = KNOWN_TEMPLATES.get(sha)
            if kind is None:
                logger.warning(
                    "sanitize-control-tokens=on but chat template sha256 %s is not a "
                    "known Qwen3.8 template; requests are not sanitized",
                    sha,
                )
            else:
                logger.info("sanitize-control-tokens: chat template %s (%s)", kind, sha)
            self._template_kind = {chat_template: kind}
        return kind

    # ------------------------------------------------------------ request flag

    @staticmethod
    def pop_request_flag(template_kwargs: Dict[str, Any]) -> bool:
        """Remove the request kwarg (it never reaches jinja or the cache key) and
        return whether this request is sanitized (server mode on: default True)."""
        value = template_kwargs.pop(REQUEST_KWARG, True)
        if not isinstance(value, bool):
            raise ValueError(
                f"chat_template_kwargs.{REQUEST_KWARG} must be true or false"
            )
        return value

    # ------------------------------------------------------------ fast path scan

    def text_hit(self, s: str, edges: bool = True) -> bool:
        if "<" not in s and ">" not in s:
            return False
        if self._token_re.search(s):
            return True
        if not edges:
            return False
        for v in (s, s.strip()):
            n = min(len(v), self._max_len - 1)
            for k in range(1, n + 1):
                if v[-k:] in self._prefixes or v[:k] in self._suffixes:
                    return True
        return False

    def _walk_hit(self, x: Any) -> bool:
        if isinstance(x, str):
            return self.text_hit(x)
        if isinstance(x, dict):
            return any(
                (isinstance(k, str) and self.text_hit(k)) or self._walk_hit(v)
                for k, v in x.items()
            )
        if isinstance(x, (list, tuple)):
            return any(self._walk_hit(v) for v in x)
        return False

    def scan(self, messages: List[Dict[str, Any]], tools: Optional[List[Any]]) -> bool:
        """True when any string the template may render holds an added-token string,
        or a part of one at the start or end (it could join the template text)."""
        return self._walk_hit(messages) or self._walk_hit(tools)

    def count(self, event: str, reason: str = "") -> None:
        self.metrics.inc(event, reason)

    def fallback(self, reason: str, detail: str = "") -> None:
        self.metrics.inc("fallback", reason)
        logger.warning(
            "prompt sanitize fallback (%s)%s", reason, f": {detail}" if detail else ""
        )

    # ------------------------------------------------------------ slow path

    def _mirror(
        self,
        kind: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Any]],
        template_kwargs: Dict[str, Any],
        add_generation_prompt: bool,
    ):
        et = template_kwargs.get("enable_thinking", _UNDEFINED)
        instr_on = et is _UNDEFINED or et is True
        effort = "medium"
        if instr_on:
            eff = template_kwargs.get("reasoning_effort", _UNDEFINED)
            if kind == "v3":
                if eff is _UNDEFINED or eff is None:
                    eff = "xhigh"
                elif isinstance(eff, str) and eff in ("high", "max"):
                    eff = "xhigh"
                elif isinstance(eff, str) and eff == "minimal":
                    eff = "low"
            elif eff is _UNDEFINED:
                eff = "xhigh"
            if not isinstance(eff, str) or eff not in ("xhigh", "medium", "low"):
                raise qfe.RenderError(f"reasoning_effort {eff!r}")
            effort = eff
        pt = template_kwargs.get("preserve_thinking", _UNDEFINED)
        text, fields = qfe.render_spans(
            messages,
            tools,
            effort,
            enable_thinking=instr_on,
            preserve_thinking=pt is _UNDEFINED or pt is True,
        )
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
            if et is not _UNDEFINED and et is False:
                text += "<think>\n\n</think>\n\n"
            else:
                text += "<think>" if kind == "v3" else "<think>\n"
        return text, fields

    def encode(
        self,
        rendered: str,
        base_ids: List[int],
        *,
        chat_template: Any,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Any]],
        template_kwargs: Dict[str, Any],
        add_generation_prompt: bool,
        encode_kwargs: Dict[str, Any],
    ) -> List[int]:
        """Sanitized ids for the jinja-rendered `rendered` (whose normal ids are
        `base_ids`), or `base_ids` when a check fails or nothing is in a body field."""
        self.count("slow_path")
        try:
            kind = self.template_kind(chat_template)
            if kind is None:
                self.fallback("unknown_template")
                return base_ids
            if encode_kwargs.get("add_special_tokens"):
                self.fallback("add_special_tokens")
                return base_ids
            try:
                text, fields = self._mirror(
                    kind, messages, tools, template_kwargs, add_generation_prompt
                )
            except (qfe.RenderError, KeyError, TypeError, AttributeError) as e:
                self.fallback("mirror_error", f"{type(e).__name__}: {e}")
                return base_ids
            if text != rendered:
                at = next(
                    (i for i, (a, b) in enumerate(zip(text, rendered)) if a != b),
                    min(len(text), len(rendered)),
                )
                self.fallback(
                    "render_mismatch",
                    f"first difference at char {at} of {len(rendered)}",
                )
                return base_ids
            try:
                ids, _, body = self.enc.encode(
                    rendered, qfe.all_plain_ranges(fields)
                )
            except qfe.EncodeError as e:
                self.fallback("encode_error", str(e))
                return base_ids
            if not body:
                # The scan hit only a partial string or a non-rendered field.
                self.count("no_change")
                return base_ids
            if list(self.enc.encode_full(rendered)[0]) != list(base_ids):
                self.fallback("full_encode_mismatch")
                return base_ids
            self.count("sanitized")
            return ids
        except Exception as e:  # never fail the request because of the sanitizer
            self.fallback("error", f"{type(e).__name__}: {e}")
            return base_ids

    def encode_prefix(self, prefix: str) -> List[int]:
        """The continue_final_message prefix as ordinary text (encoded alone, as the
        unsanitized path does)."""
        self.count("sanitized", "prefix")
        return list(self.enc.plain.encode(prefix, add_special_tokens=False).ids)
