"""qwen3 reasoning + qwen3_coder tool calls over Qwen3.8 output token ids.

Qwen3.8 has `<think>`, `</think>`, `<tool_call>` and `</tool_call>` as single
added tokens (special=False). The model also writes the same strings with
ordinary tokens when it explains the format, quotes code, or writes a file.
Only the control tokens may act as markers.

The tokenizer is loaded from ``SGLANG_TEST_QWEN38_TOKENIZER`` (a local model
directory or a hub id), default ``Qwen/Qwen3.8-Flash-Next``.
"""

import functools
import json
import os
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

QWEN38_TOKENIZER = os.environ.get(
    "SGLANG_TEST_QWEN38_TOKENIZER", "Qwen/Qwen3.8-Flash-Next"
)
MARKERS = ("<think>", "</think>", "<tool_call>", "</tool_call>")


@functools.lru_cache(maxsize=None)
def _shared_tokenizer(path: str):
    from sglang.srt.utils.hf_transformers import get_tokenizer

    return get_tokenizer(path)


class _Spelled(str):
    """A marker written with ordinary tokens rather than its control token."""


def _fn(name, *params):
    return Tool(
        type="function",
        function=Function(
            name=name,
            parameters={
                "type": "object",
                "properties": {p: {"type": "string"} for p in params},
            },
        ),
    )


def _call_body(name, **params):
    body = f"\n<function={name}>\n"
    for key, value in params.items():
        body += f"<parameter={key}>\n{value}\n</parameter>\n"
    return body + "</function>\n"


class TestQwen38ControlTokenIds(unittest.TestCase):
    TOOLS = [_fn("bash", "command"), _fn("write", "path", "content")]
    BASH_LS = ["<tool_call>", _call_body("bash", command="ls"), "</tool_call>"]

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = _shared_tokenizer(QWEN38_TOKENIZER)

    def _encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _ids(self, parts):
        ids = []
        for part in parts:
            if isinstance(part, _Spelled):
                spelled = self._encode(part[0]) + self._encode(part[1:])
                self.assertNotIn(self.tokenizer.convert_tokens_to_ids(part), spelled)
                ids += spelled
            elif part in MARKERS:
                ids.append(self.tokenizer.convert_tokens_to_ids(part))
            else:
                encoded = self._encode(part)
                for marker in MARKERS:
                    marker_id = self.tokenizer.convert_tokens_to_ids(marker)
                    self.assertNotIn(marker_id, encoded)
                ids += encoded
        return ids

    def _chunks(self, ids, step):
        """(ids, text) per chunk, with text that ends in an incomplete
        character held back as the detokenizer does."""
        sent = ""
        for start in range(0, len(ids), step):
            end = min(start + step, len(ids))
            text = self.tokenizer.decode(ids[:end], skip_special_tokens=True)
            if end < len(ids):
                text = text.rstrip("�")
            yield ids[start:end], text[len(sent) :]
            sent = text

    def _parsers(self):
        reasoning_parser = ReasoningParser(
            "qwen3",
            stream_reasoning=self._stream_reasoning,
            force_reasoning=True,
            tokenizer=self.tokenizer,
        )
        tool_parser = FunctionCallParser(
            self.TOOLS, "qwen3_coder", tokenizer=self.tokenizer
        )
        return reasoning_parser, tool_parser

    def _parse(self, ids, *, step=None, stream_reasoning=True, with_ids=True):
        self._stream_reasoning = stream_reasoning
        reasoning_parser, tool_parser = self._parsers()
        if step is None:
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            reasoning, content = reasoning_parser.parse_non_stream(
                text, output_ids=ids if with_ids else None
            )
            content, calls = tool_parser.parse_non_stream(
                content, output_ids=reasoning_parser.take_content_output_ids()
            )
            return (
                reasoning,
                content.strip(),
                [(c.name, json.loads(c.parameters)) for c in calls],
            )

        reasoning, content, names, args = "", "", {}, {}
        chunks = list(self._chunks(ids, step))
        for i, (chunk_ids, text) in enumerate(chunks):
            if with_ids:
                reasoning_parser.observe_stream_output_ids(chunk_ids, cumulative=False)
            r, n = reasoning_parser.parse_stream_chunk(text)
            if i == len(chunks) - 1:
                end_r, end_n = reasoning_parser.parse_stream_end()
                r, n = r + end_r, n + end_n
            reasoning += r
            tool_parser.observe_output_ids(reasoning_parser.take_content_output_ids())
            n, calls = tool_parser.parse_stream_chunk(n)
            if i == len(chunks) - 1:
                end_n, end_calls = tool_parser.parse_stream_end()
                n, calls = n + end_n, calls + end_calls
            content += n
            for call in calls:
                names[call.tool_index] = names.get(call.tool_index, "") + (
                    call.name or ""
                )
                args[call.tool_index] = args.get(call.tool_index, "") + call.parameters
        calls = [(names[i], json.loads(args[i])) for i in sorted(names)]
        return reasoning, content.strip(), calls

    def _assert_parses(self, parts, expected):
        ids = self._ids(parts)
        self.assertEqual(self._parse(ids), expected)
        for step in (1, 2, 3, 7, len(ids)):
            for stream_reasoning in (True, False):
                with self.subTest(step=step, stream_reasoning=stream_reasoning):
                    got = self._parse(ids, step=step, stream_reasoning=stream_reasoning)
                    self.assertEqual(got, expected)

    def test_markers_resolve_to_single_tokens(self):
        """Without this, the parsers silently keep matching by text."""
        self._stream_reasoning = True
        reasoning_parser, tool_parser = self._parsers()
        self.assertIsNotNone(reasoning_parser.detector._control)
        self.assertIsNotNone(tool_parser.detector._control)
        for marker in MARKERS:
            self.assertEqual(len(self._encode(marker)), 1, marker)

    def test_real_markers_are_parsed(self):
        self._assert_parses(
            ["Let me list the files.", "</think>", "\n\n"] + self.BASH_LS,
            ("Let me list the files.", "", [("bash", {"command": "ls"})]),
        )
        self._assert_parses(
            ["Two calls.", "</think>", "\n\nRunning.\n"]
            + self.BASH_LS
            + ["\n", "<tool_call>", _call_body("bash", command="pwd"), "</tool_call>"],
            (
                "Two calls.",
                "Running.",
                [("bash", {"command": "ls"}), ("bash", {"command": "pwd"})],
            ),
        )

    def test_spelled_markers_are_text(self):
        """Spelled markers stay text in reasoning and in content, and a spelled
        call in content is not executed, for every chunking."""
        cases = [
            (
                [
                    "Split on ",
                    _Spelled("</think>"),
                    " or ",
                    _Spelled("<tool_call>"),
                    ".",
                    "</think>",
                    "\n\nWrite ",
                    _Spelled("<tool_call>"),
                    _call_body("bash", command="ls"),
                    _Spelled("</tool_call>"),
                    " to call.",
                ],
                (
                    "Split on </think> or <tool_call>.",
                    "Write <tool_call>" + _call_body("bash", command="ls")
                    # A spelled </tool_call> stays in the content.
                    + "</tool_call> to call.",
                    [],
                ),
            ),
            (
                ["龘", _Spelled("</think>"), "🦜", "</think>", "🦜", _Spelled("</think>")],
                ("龘</think>🦜", "🦜</think>", []),
            ),
        ]
        for parts, expected in cases:
            with self.subTest(expected=expected):
                self._assert_parses(parts, expected)

    def test_spelled_tool_call_tags_inside_an_argument(self):
        """Writing a file that contains tool-call markup: a spelled
        </tool_call> used to end the call in the middle of the value."""
        value = (
            "a",
            _Spelled("</tool_call>"),
            "b",
            _Spelled("<tool_call>"),
            "c ",
            _Spelled("</think>"),
            " d",
        )
        self._assert_parses(
            ["Writing.", "</think>", "\n\n", "<tool_call>"]
            + ["\n<function=write>\n<parameter=path>\nx.md\n</parameter>\n"]
            + ["<parameter=content>\n", *value, "\n</parameter>\n</function>\n"]
            + ["</tool_call>"],
            (
                "Writing.",
                "",
                [
                    (
                        "write",
                        {
                            "path": "x.md",
                            "content": "a</tool_call>b<tool_call>c </think> d",
                        },
                    )
                ],
            ),
        )

    def test_spelled_function_tag_before_a_real_call(self):
        """With ids, `<function=` only counts inside a `<tool_call>` token."""
        self._assert_parses(
            ["ok", "</think>", "\n\nFormat: ", _Spelled("<tool_call>")]
            + ["<function=x>", _Spelled("</tool_call>"), "\n"]
            + self.BASH_LS,
            (
                "ok",
                "Format: <tool_call><function=x></tool_call>",
                [("bash", {"command": "ls"})],
            ),
        )

    def test_tool_call_token_inside_reasoning(self):
        """A <tool_call> token inside reasoning stays reasoning when </think>
        follows; streaming used to end reasoning there. Without </think>
        it starts the call, as one-shot parsing decides."""
        self._assert_parses(
            ["See "] + self.BASH_LS + [" here.", "</think>", "\n\nDone."],
            (
                "See <tool_call>" + _call_body("bash", command="ls") + "</tool_call> here.",
                "Done.",
                [],
            ),
        )
        self._assert_parses(
            ["Look it up.\n"] + self.BASH_LS,
            ("Look it up.\n", "", [("bash", {"command": "ls"})]),
        )

    def test_without_output_ids_markers_match_as_text(self):
        ids = self._ids(["a ", _Spelled("</think>"), "b", "</think>", "c"])
        expected = ("a ", "b</think>c", [])
        self.assertEqual(self._parse(ids, with_ids=False), expected)
        self.assertEqual(self._parse(ids, step=1, with_ids=False), expected)

    def test_text_that_disagrees_with_the_ids_matches_as_text(self):
        self._stream_reasoning = True
        reasoning_parser, _ = self._parsers()
        ids = self._ids(["a", "</think>", "b"])
        self.assertEqual(
            reasoning_parser.parse_non_stream("x</think>y", output_ids=ids),
            ("x", "y"),
        )


if __name__ == "__main__":
    unittest.main()
