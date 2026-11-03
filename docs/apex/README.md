# Apex 的 SGLang fork（apexin-ai/sglang）

本仓库是 [sgl-project/sglang](https://github.com/sgl-project/sglang) 的公开 fork。我们的改动放在 `apex/*` 分支。本文件只记 fork 的规则和我们加的功能。

## 分支

| 分支 | 用途 | 规则 |
|---|---|---|
| `main` | 上游 `main` 的镜像 | 不提交自己的改动。只用 GitHub merge-upstream 快进 |
| `apex/main` | 公司分支：上游 `main` + 测过的 `apex/pr/*` | 部署只用它。只合入，不 force-push |
| `apex/pr/<topic>` | 一个改动一个分支 | 从 `main` 切出；开发、测试后合入 `apex/main` |

- 不向上游开 PR、issue 或评论，除非用户另行决定。
- commit 身份：`Junwei Zhou <zjw330501@gmail.com>`。

### 快进 `main`

```bash
export GH_CONFIG_DIR=$HOME/.gh-config
gh api -X POST repos/apexin-ai/sglang/merge-upstream -f branch=main
```

- 只允许快进。如果 GitHub 报冲突，说明有人在 `main` 上提交了，先查清楚，不要强推。
- 快进 `main` 不会改 `apex/main`。要带上新的上游代码时，把 `main` 合进 `apex/main`，再跑测试。

## 功能：Qwen3.8 prompt 侧特殊 token 消毒（`--sanitize-control-tokens`）

分支 `apex/pr/qwen38-prompt-sanitize`。设计稿在 apex-workspace 仓库 `qwen38-flash-next-training/reports/2026-11-01-prompt-sanitization/README.md`。

### 作用

- 默认路径把整个对话渲染成一个字符串，再整段分词。消息正文里的 `<|im_end|>`、`<think>`、`<tool_call>` 等 33 个 added-token 字符串（id 248044–248076）会变成控制 id。
- 开启后，正文字段里的这些字符串按普通文本（纯 BPE）编码。模板结构仍是控制 id。规则与训练仓库 `data/field_encode.py` 相同。
- 正文字段：system / user / tool 的 content、assistant 的 content 和 `reasoning_content`、工具调用的 name 和参数值、`tools` 定义、`continue_final_message` 的前缀。

### 开关

- 服务端：`--sanitize-control-tokens {off,on}`。fork 默认 `off`，与上游行为相同。部署设 `on`。
- 请求：`chat_template_kwargs.sanitize_control_tokens`（`true` / `false`）。`on` 时请求可以关掉。值不是 bool 时返回 400。
- `--default-chat-template-kwargs` 也可以带这个键。合并之后读取，读后从 kwargs 删除，不传给模板，也不进模板缓存的键。
- `off` 时这个键不做处理，和上游一样原样传给模板。
- 没有 `force` 档。原因：这不是安全边界（见「限制」）。

### 生效条件（启动时检查）

1. 走默认 jinja 路径：没有 `--chat-template` 的内置模板名，没有 Python 模板，没有 `chat_encoding_spec`，渲染再分词不丢信息。
2. tokenizer 的 added token 正好是 248044–248076。
3. tokenizer 目录里有 `tokenizer.json`。去掉 added token 后的纯 BPE 词表大小正好是 248044。
4. 服务的 tokenizer 和 `tokenizer.json` 的编码相同（启动时用含组合字符的探针比对）。
5. 模板 sha256 是已知的 Qwen3.8 模板（每个请求检查）：v3 `60cf9fa2…`，或 base 检查点自带的 `c3cf9e34…`。

- 任何一条不满足：记一条 warning，服务行为与 `off` 相同。
- 启动日志写 `tokenizer.json` 的 sha256，并和训练用的 `0997f410…` 比对。

### 算法

1. 快路径：扫描模板要渲染的所有字符串（messages、tools、前缀），找 33 个字符串，以及字符串首尾的不完整片段。没有命中时，走原来的代码，ids 由构造保证相同，`parallel_prompt_encode` 和模板缓存不变。
2. 慢路径（只在命中时）：
   1. 照常用 jinja 渲染，整段分词。
   2. 用 mirror（`python/sglang/srt/parser/qwen38_field_encode.py`，从训练仓库原样复制）渲染，要求文本与 jinja 结果逐字相同。
   3. 用 `FieldEncoder` 按字段编码，所有字段都是正文。
   4. 要求原始 `tokenizer.json` 的整段编码等于服务 tokenizer 的 ids。
   5. 任何检查失败：用原来的 ids（回退），计数并记 warning。
3. `continue_final_message` 的前缀仍单独编码，命中时用纯 BPE。
4. 消毒后的请求直接送 ids（`MessageProcessingResult.sanitized_prompt_ids`），不让多模态 processor 再分词。
5. Anthropic `/v1/messages`：`on` 时 `supports_native_reasoning_history()` 返回 True。历史思考放进 `reasoning_content`，不再包成 `<think>…</think>` 文本放进 content。

### 指标

- 进程内计数器，开 `--enable-metrics` 时同时导出 Prometheus `sglang:prompt_sanitize_total{event,reason}`。
- `event`：`hit`（扫描命中）、`slow_path`、`sanitized`（ids 改了）、`no_change`（命中但正文里没有完整的 token）、`fallback`。
- `fallback` 的 `reason`：`media`、`unknown_template`、`add_special_tokens`、`mirror_error`、`render_mismatch`、`encode_error`（例如 token 跨越字段边界）、`full_encode_mismatch`、`error`。

### 限制

- 不是安全边界：`ChatCompletionRequest.input_ids`、`/generate`、`/v1/completions` 不经过它，由客户端自己负责。
- 只支持纯文本请求。带图片、视频、音频的请求在 `on` 时回退到原路径（计 `fallback/media`）。多模态的 ids 路径以后另开分支做。
- 覆盖的入口：Chat Completions、Responses、Anthropic `/v1/messages`、`/v1/tokenize`（都经过 `_process_messages`）。
- 不覆盖的入口：Ollama `/api/chat`、sgl-model-gateway 的 gRPC（Rust 渲染器）、`/v1/decisions`。
- 服务端带 `--chat-template` 或请求带 `chat_template` 覆盖时，模板 sha 不在已知列表，请求不消毒（计 `fallback/unknown_template`）。
- 工具参数名在模板里是结构文字，不消毒。

### 测试

```bash
SGLANG_TEST_QWEN38_TOKENIZER=/mnt/cpfs/models/Qwen3.8-Flash-Next \
SGLANG_TEST_QWEN38_TRAINING_REPO=<qwen38-flash-training checkout> \
python -m pytest -q test/registered/unit/entrypoints/openai/test_prompt_sanitize.py
```

- 没有 Qwen3.8 tokenizer 时整个文件跳过。没有训练仓库时，跳过和训练仓库逐 id 比对的部分和 v3 模板的用例。
- 更新 mirror 时，同时更新文件头里的训练仓库 SHA，并跑 `test_mirror_is_a_verbatim_copy_of_the_training_repo`。
