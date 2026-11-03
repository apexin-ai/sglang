"""Qwen3.8 field-aware prompt encoding, copied from the training repository.

Source: apexin-ai/qwen38-flash-training @ ab853ee41df6550fd0cb0350adb7f3a954e09c72
  - data/q38text.py: TOOLS_TAIL, EFFORT_TEXT, RenderError, tojson, IMAGE_BLOCK,
    is_image_part, _content_text, render_spans (verbatim)
  - data/field_encode.py: LO, HI, EncodeError, FieldEncoder (verbatim)
Added here (not in the source): all_plain_ranges.

`render_spans` is a Python mirror of the Qwen3.8 chat template (decide-at-think v3,
chat_template.q38-decide-v3.jinja sha256 60cf9fa2...; the base checkpoint's
chat_template.jinja c3cf9e34... renders the same text except the effort aliases and
the generation prompt, which the caller adds).  It returns the rendered text and the
character range of every message field.  `FieldEncoder.encode(text, plain)` cuts the
text only at the added tokens outside the `plain` ranges and encodes every piece
between two cuts with the BPE model alone (tokenizer.json without its added tokens),
so an added-token string inside a field becomes ordinary text.  A text whose plain
ranges hold no added-token string encodes to exactly the ids of the normal encoding.

Keep the copied parts byte-identical to the source so that training and serving
encode the same way; update them only together with the source SHA above.
"""

import bisect
import hashlib
import json

# ---------------------------------------------------------------- data/q38text.py

TOOLS_TAIL = ('\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n'
              '<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n'
              '<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\n'
              'multiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n'
              '- Function calls MUST follow the specified format: an inner <function=...></function> block must be '
              'nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n'
              '- You may provide optional reasoning for your function call in natural language BEFORE the function '
              'call, but NOT after\n- If there is no function call available, answer the question like normal with '
              'your current knowledge and do not tell the user about function calls\n</IMPORTANT>')
EFFORT_TEXT = {
    'xhigh': ('Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, '
              'consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final '
              'answer.'),
    'low': ('Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion '
            'without unnecessary elaboration.'),
    'medium': '',
}


class RenderError(ValueError):
    pass


def tojson(x):
    return json.dumps(x, ensure_ascii=False)


IMAGE_BLOCK = '<|vision_start|><|image_pad|><|vision_end|>'   # template v3 render_content for an image part


def is_image_part(item):
    return isinstance(item, dict) and ('image' in item or 'image_url' in item or item.get('type') == 'image')


def _content_text(content, where, images_ok=True):
    """A message content as template v3 render_content renders it: a string as is; a list of parts as text parts
    verbatim and an image part as IMAGE_BLOCK (TASK-VL F6: images in user / tool messages, e.g. CC Read results)."""
    if isinstance(content, str):
        return content
    if content is None:
        return ''
    if isinstance(content, list):
        out = []
        for item in content:
            if is_image_part(item):
                if not images_ok:
                    raise RenderError(f'{where}: an image part where the template does not allow one')
                out.append(IMAGE_BLOCK)
            elif isinstance(item, dict) and ('video' in item or item.get('type') == 'video'):
                raise RenderError(f'{where}: video parts are not supported')
            elif isinstance(item, dict) and 'text' in item:
                out.append(item['text'])
            else:
                raise RenderError(f'{where}: unexpected content item')
        return ''.join(out)
    raise RenderError(f'{where}: non-string content ({type(content).__name__}) is not supported by the mirror')


def render_spans(messages, tools=None, reasoning_effort='medium', enable_thinking=True, preserve_thinking=True):
    """(text, fields): the template-v3 text and [{msg, field, start, end, raw_off}] where text[start:end] is the
    field's rendered value; raw_off = characters the template trimmed from the start of the raw value (content /
    reasoning_content are trimmed; tool-call values, tool definitions are not)."""
    if not messages:
        raise RenderError('no messages')
    if reasoning_effort not in EFFORT_TEXT:
        raise RenderError(f'effort {reasoning_effort!r}')
    instr = EFFORT_TEXT[reasoning_effort] if enable_thinking else ''
    out, fields = [], []
    pos = [0]

    def put(s):
        out.append(s)
        pos[0] += len(s)

    def put_field(msg, field, raw, trim):
        v = raw.strip() if trim else raw
        off = (len(raw) - len(raw.lstrip())) if trim else 0
        fields.append({'msg': msg, 'field': field, 'start': pos[0], 'end': pos[0] + len(v), 'raw_off': off})
        put(v)

    m0 = messages[0]
    if tools:
        put('<|im_start|>system\n')
        if instr:
            put(instr + '\n\n')
        put('# Tools\n\nYou have access to the following functions:\n\n<tools>')
        for j, t in enumerate(tools):
            put('\n')
            put_field(None, f'tools[{j}]', tojson(t), False)
        put('\n</tools>')
        put(TOOLS_TAIL)
        if m0.get('role') == 'system':
            c = _content_text(m0.get('content'), 'msg 0', images_ok=False)
            if c.strip():
                put('\n\n')
                put_field(0, 'content', c, True)
        put('<|im_end|>\n')
    else:
        if m0.get('role') == 'system':
            c = _content_text(m0.get('content'), 'msg 0', images_ok=False)
            if c.strip():
                put('<|im_start|>system\n' + (instr + '\n\n' if instr else ''))
                put_field(0, 'content', c, True)
                put('<|im_end|>\n')
            elif instr:
                put('<|im_start|>system\n' + instr + '<|im_end|>\n')
        elif instr:
            put('<|im_start|>system\n' + instr + '<|im_end|>\n')
    last_query = None
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get('role') == 'user':
            c = _content_text(m.get('content'), f'msg {i}').strip()
            if not (c.startswith('<tool_response>') and c.endswith('</tool_response>')):
                last_query = i
                break
    if last_query is None:
        raise RenderError('No user query found in messages.')
    n = len(messages)
    for i, m in enumerate(messages):
        role = m.get('role')
        content = _content_text(m.get('content'), f'msg {i}', images_ok=role in ('user', 'tool'))
        if role == 'system':
            if i != 0:
                raise RenderError('System message must be at the beginning.')
        elif role == 'user':
            put('<|im_start|>user\n')
            put_field(i, 'content', content, True)
            put('<|im_end|>\n')
        elif role == 'assistant':
            rc = m.get('reasoning_content')
            rc = rc if isinstance(rc, str) else ''
            if preserve_thinking or i > last_query:
                put('<|im_start|>assistant\n<think>\n')
                put_field(i, 'reasoning_content', rc, True)
                put('\n</think>\n\n')
            else:
                put('<|im_start|>assistant\n')
            put_field(i, 'content', content, True)
            tcs = m.get('tool_calls')
            if tcs and isinstance(tcs, list):
                for k, tc in enumerate(tcs):
                    fn = tc['function'] if isinstance(tc, dict) and 'function' in tc else tc
                    name = fn['name']
                    if k == 0:
                        put('\n\n<tool_call>\n<function=' if content.strip() else '<tool_call>\n<function=')
                    else:
                        put('\n<tool_call>\n<function=')
                    put_field(i, f'tool_calls[{k}].name', name, False)
                    put('>\n')
                    args = fn.get('arguments')
                    if args is not None and args != '':
                        if not isinstance(args, dict):
                            raise RenderError(f'msg {i}: tool call arguments are {type(args).__name__}, not a mapping')
                        for an, av in args.items():
                            put('<parameter=' + an + '>\n')
                            put_field(i, f'tool_calls[{k}].arguments.{an}', av if isinstance(av, str) else tojson(av),
                                      False)
                            put('\n</parameter>\n')
                    put('</function>\n</tool_call>')
            put('<|im_end|>\n')
        elif role == 'tool':
            if i > 0 and messages[i - 1].get('role') != 'tool':
                put('<|im_start|>user')
            put('\n<tool_response>\n')
            put_field(i, 'content', content, True)
            put('\n</tool_response>')
            if i == n - 1 or messages[i + 1].get('role') != 'tool':
                put('<|im_end|>\n')
        else:
            raise RenderError(f'Unexpected message role {role!r}.')
    return ''.join(out), fields


# ---------------------------------------------------------------- data/field_encode.py

LO, HI = 248044, 248076


class EncodeError(ValueError):
    pass


class FieldEncoder:
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.path = str(path)
        raw = open(self.path, 'rb').read()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.full = Tokenizer.from_str(raw.decode())
        added = {i: t.content for i, t in self.full.get_added_tokens_decoder().items()}
        if set(added) != set(range(LO, HI + 1)):
            raise EncodeError(f'{self.path}: added tokens {min(added)}..{max(added)}, expected {LO}..{HI}')
        self.added = added
        j = json.loads(raw)
        j['added_tokens'] = []
        self.plain = Tokenizer.from_str(json.dumps(j))
        if self.plain.get_vocab_size(with_added_tokens=True) != LO:
            raise EncodeError('the BPE model alone must have exactly the ids below the added tokens')

    def encode_full(self, text):
        e = self.full.encode(text, add_special_tokens=False)
        return e.ids, e.offsets

    def encode(self, text, plain=()):
        """(ids, offsets, body): body = [(char start, end, added id)] of the added-token strings inside `plain` that
        are now ordinary text.  `plain` = sorted, non-overlapping (start, end) character ranges."""
        plain = sorted(plain)
        for (a0, b0), (a1, b1) in zip(plain, plain[1:]):
            if a1 < b0:
                raise EncodeError(f'plain ranges overlap: {a0}:{b0} and {a1}:{b1}')
        starts = [a for a, _ in plain]
        fids, foffs = self.encode_full(text)
        cuts, body = [], []
        for t, (a, b) in zip(fids, foffs):
            if not LO <= t <= HI:
                continue
            k = bisect.bisect_right(starts, a) - 1
            inside = k >= 0 and a < plain[k][1]
            if inside and b > plain[k][1]:
                raise EncodeError(f'added token {t} at {a}:{b} crosses the end of plain range {plain[k]}')
            if not inside and k + 1 < len(plain) and b > plain[k + 1][0]:
                raise EncodeError(f'added token {t} at {a}:{b} crosses the start of plain range {plain[k + 1]}')
            (body if inside else cuts).append((a, b, t))
        if not body:
            return list(fids), [tuple(o) for o in foffs], []
        ids, offs, pos = [], [], 0
        for a, b, t in cuts + [(len(text), len(text), None)]:
            if a > pos:
                e = self.plain.encode(text[pos:a], add_special_tokens=False)
                ids += e.ids
                offs += [(pos + x, pos + y) for x, y in e.offsets]
            if t is not None:
                ids.append(t)
                offs.append((a, b))
            pos = b
        # the tokenizer normalizes the text before BPE, so compare with what the whole-string ids decode to
        if self.full.decode(ids, skip_special_tokens=False) != self.full.decode(fids, skip_special_tokens=False):
            raise EncodeError('decode(field-encoded ids) != decode(whole-string ids)')
        for (a, b), t in zip(offs, ids):
            k = bisect.bisect_right(starts, a) - 1
            if LO <= t <= HI and k >= 0 and a < plain[k][1]:
                raise EncodeError(f'added id {t} left inside plain range {plain[k]}')
        return ids, offs, body


# ---------------------------------------------------------------- added for serving


def all_plain_ranges(fields):
    """Character ranges of every field of `render_spans` (all message fields and tools[j]): at serving time every
    field comes from the client, so all of them are body text."""
    return sorted((f['start'], f['end']) for f in fields if f['end'] > f['start'])
