# tapedeck

<img src="docs/logo.svg" alt="tapedeck" width="420" align="center" />

[![ci](https://github.com/EAnge1o/tapedeck/actions/workflows/ci.yml/badge.svg)](https://github.com/EAnge1o/tapedeck/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tapedeck-llm)](https://pypi.org/project/tapedeck-llm/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org/)
[![Dependency](https://img.shields.io/badge/dependencies-0-informational.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**tapedeck** 是给 LLM Agent 用的**飞行记录仪**：录一次（REC ⏺），永久离线重放（PLAY ▶），精确 diff 出第一个漂移的步骤（⏩）。

> 每个做过 Agent 的人都遇到同一个夜晚：**昨天还好好的 agent，今天改了一行 prompt 就坏了。** 重跑一次要花钱，重跑十次才复现，日志里只有一坨 print。tapedeck 把整次运行（每次模型调用、每次工具调用、每份 usage）录成带版本号的 JSONL；之后重放零 API 成本、逐字节确定；两个 run 之间一条命令 diff，直接告诉你**第几步、哪个调用、从什么值变成了什么值**。

零依赖：纯 Python 标准库（连测试都是 `unittest`）。不上传任何数据——trace 就是本地文件，可以进 git、可以 cat、可以 rsync。

![tapedeck demo: record / replay / diff, rendered from real command output](docs/demo.svg)

*（截图由 `docs/make_demo_svg.py` 用真实命令输出渲染而成，不是摆拍。）*

English documentation: [README.md](README.md)

> **PyPI 包名说明**：`tapedeck` 这个名字在 PyPI 上自 2015 年起被一个无关的音乐播放器占用，所以本项目的发行包叫 **`tapedeck-llm`**——`pip install tapedeck-llm` 装出来的仍然是 `tapedeck` 包和 `tapedeck` 命令行工具。

## 它如何工作

```mermaid
flowchart LR
    A["你的 Agent<br/>tape.call(...)"] -->|"live 运行"| B["RecordingTape<br/>REC ⏺"]
    A -->|"改变代码后"| C["ReplayingTape<br/>PLAY ▶<br/>0 次 API 调用"]
    B --> D["trace.jsonl<br/>+ meta.json"]
    C -.->|"逐字回放"| D
    B --> E["tapedeck diff<br/>第一个漂移步骤"]
    F["第二次录制"] --> E
    E -->|"exit 0 / 1 / 2"| G["CI ✅ ❌"]
```

## 30 秒上手

```bash
pip install tapedeck-llm              # 或者直接把 tapedeck/ 放进你的项目
python examples/demo_agent.py --mode record --scenario stable
python examples/demo_agent.py --mode record --scenario drift   # 有人改了折扣策略
python examples/demo_agent.py --mode replay                    # 离线重放，$0.00
python -m tapedeck diff demo-stable demo-drift
```

真实输出（以上命令原样粘贴，非手写示例）：

```
=== replay ===
Your coffee is $3.50; with the 10% member discount you pay $6.30.

[replayed '20261026-151919-demo-stable': served 4 steps, 0 live calls, $0.00 spent]

=== diff demo-stable demo-drift ===
run A: demo-stable (4 steps)
run B: demo-drift (4 steps)
step 2 (tool:compute_total) output:
  A: 6.3
  B: 5.6
  context (steps 0..1):
    step 0 (model:model.plan) input: []
    step 1 (tool:get_price) input: []
run A: $0.0001
run B: $0.0001
```

diff 的退出码为 CI 而设计：`0` 完全一致，`1` 输出漂移，`2` 结构漂移（步骤增删）。

## 直接看管 SDK client（不改业务代码）

把每个调用改写成 `tape.call` 最诚实，但侵入也最大。如果你的 agent 直接调 provider SDK，把 client 交给 tape 就行——代码读起来还是普通的 SDK 调用：

```python
import tapedeck, openai

client = openai.OpenAI()                    # DeepSeek/vLLM/Ollama：同一个 client，配你的 base_url
with tapedeck.record("chat-v3") as tape:
    client = tape.watch(client)             # REC ⏺ —— live 运行，逐调用记档
    resp = client.chat.completions.create(model="deepseek-chat", messages=msgs)

with tapedeck.play("chat-v3") as tape:
    client = tape.watch(client)             # PLAY ▶ —— 从磁带回放，不走网络
    resp = client.chat.completions.create(model="deepseek-chat", messages=msgs)
```

拦截的方法路径：`chat.completions.create`（OpenAI SDK，经 `base_url` 同样覆盖所有 OpenAI 兼容端点——**DeepSeek、vLLM、Ollama**）、`responses.create`（OpenAI Responses API）、`messages.create`（Anthropic），均支持 `stream=True`（逐 chunk 录制、逐 chunk 回放）。按需安装：`pip install "tapedeck-llm[openai]"` / `"tapedeck-llm[anthropic]"`。SDK 只在这里被 import，核心保持零依赖。

重放时用 SDK 自己的类型从 trace 重建响应对象（`model_validate`），`resp.choices[0].message.content`、`resp.usage.prompt_tokens` 用起来和原来一模一样。如果装着的 SDK 版本已经和录制对不上，重放降级为属性 shim 而不是崩溃——内容访问仍然可用，严格的输入匹配也照常报告漂移。

## 接入 pytest

```python
def test_agent_replies(tape):                 # pip install "tapedeck-llm[pytest]"
    reply = my_agent(tape, "咖啡多少钱？")
    assert "3.50" in reply
```

每个请求 `tape` 的测试都会在 `.tapedeck/runs/` 下留一份 trace（测试失败也照样落盘——崩溃本身就是数据）。`tapedeck ls` 列出它们，`tapedeck diff` 比较其中任意两个。

## 接入 CI——回归测试

本地录一次，把 trace 进 git（或存成 artifact），然后把重放变成 PR 检查：

```yaml
# .github/workflows/agent-regression.yml
steps:
  - uses: actions/checkout@v4
  - uses: actions/setup-python@v5
    with: { python-version: "3.12" }
  - run: pip install tapedeck-llm
  - run: python my_agent.py --replay demo-stable     # 你的代码，内部 tape.play
  - run: |
      python -m tapedeck replay demo-stable
      python -m tapedeck diff demo-stable demo-drift --fail-on-drift  # 或断言退出码
```

`replay` 只回放录制值——这个 job 花费 $0.00、全程不碰网络。一旦 diff 出漂移，检查失败并直接打印第一个分歧步骤。

## 为什么不是 LangSmith / LangFuse

| | 云端可观测平台 | **tapedeck** |
|---|---|---|
| 定位 | 看运行（dashboard） | **复现运行（记录仪）** |
| 重放 | 重新调用 LLM（花钱、不确定） | **回放录制的输出（免费、确定）** |
| 回归测试 | 不适用 | `replay` + `diff` 进 CI |
| 数据 | 上传到第三方 | **只在本机，纯文件** |
| 集成 | SDK + 上报配置 | 两个函数调用，零依赖 |

两者是互补而非替代：平台上看趋势，tapedeck 里做取证。当你需要回答"这次事故的完整调用序列是什么、它和上周五那次差在哪"，本地文件比网页快照可靠得多。

## 三行接入

Agent 代码只面向一个接口：`tape.call(kind, name, fn, *args)`——live 录制时执行 `fn` 并记档，重放时**直接回放录制值、根本不调用 `fn`**（所以不花钱、不会失败）。

```python
import tapedeck

def run_agent(tape, message):                      # 接收 tape，而不是自己调 openai
    reply = tape.call("model", "deepseek.chat", chat, messages)   # fn = 你的真实调用
    return tape.call("tool", "postprocess", lambda: reply.strip())

with tapedeck.record("chat-v3", redact=tapedeck.Redactor()) as tape:   # REC：退出自动存盘
    run_agent(tape, "hello")

with tapedeck.play("chat-v3") as tape:                                 # PLAY：免费回归测试
    run_agent(tape, "hello")                                           # 退出时自动校验步数
```

接入原则：**model 调用和有副作用的工具调用经过 tape，纯函数不经过。** usage 跟着 model 调用一起记档（provider 返回什么就存什么），成本按你自己维护的价格表换算。

## CLI

```text
tapedeck ls                  # 所有 run：步数 / tokens / 成本（最新在前）
tapedeck report RUN          # 单个 run 的逐步明细表
tapedeck diff A B            # 第一个漂移点 + 上下文 + usage/成本对照
tapedeck prune --keep 20     # 只留最近 20 个 run
```

![tapedeck ls and report, rendered from real command output](docs/report.svg)

## 录制格式

每个 run 一个目录，两个纯文件（`DESIGN.md` 有完整 schema）：

```
.tapedeck/runs/20261026-151919-demo-stable/
    meta.json     # 名称、时间、主机、步数、是否脱敏
    trace.jsonl   # 每行一个步骤：kind/name/input/output/error/usage/wall_ms
```

```json
{"v": 1, "i": 0, "kind": "model", "name": "model.plan",
 "input": [], "output": {"tool": "get_price", "args": "coffee"},
 "usage": {"prompt_tokens": 118, "completion_tokens": 24, "model": "deepseek-chat"},
 "wall_ms": 0.142}
```

## 诚实边界

- **闭包里的变化检测不到。** replay 按调用点的输入做匹配；如果漂移藏在闭包捕获的变量里（不经过参数），输入匹配不会报警——这正是 `diff`（两次真实录制的对比）存在的意义，也是 README 建议"有副作用的调用把关键参数显式传参"的原因。
- **确定性重放只到 I/O 层。** 若 agent 内部有未播种的随机数或读时钟，重放结果可能不同于录制——这是所有 record/replay 方案（VCR.py、respx）的共同边界。
- **SDK 拦截有边界。** 请求参数必须 JSON 可序列化（messages、tools、采样参数都可以；pydantic 类和二进制负载不行——它们会记成 `repr`，重放时报输入漂移，这正是诚实的信号）。流式调用在流被消费完之后才落盘。装着的 SDK 若已无法校验某份录制，重放会改用属性 shim——形状访问可用，`isinstance` 判断不可用。
- **绝不背后 monkeypatch。** `tape.watch(client)` 是显式的、作用域受限的，返回代理对象——原 client 永远不被修改。

## Roadmap

- [x] OpenAI / DeepSeek / Anthropic 自动拦截（`tape.watch`，可选 extras）
- [x] pytest fixture（`tapedeck-llm[pytest]`）
- [ ] `tapedeck replay` 的 lenient 模式 CLI 化（从第一个漂移点切回 live 继续跑）
- [ ] trace 签名（防篡改哈希链）

## 设计取舍

见 [DESIGN.md](DESIGN.md)：为什么 trace 是 JSONL 而不是 SQLite、为什么按位置对齐而不是按内容对齐、为什么脱敏默认关闭而重放默认严格。

## 同系列

[nimbus-infer](https://github.com/EAnge1o/nimbus-infer) · [nimbus-infer-rs](https://github.com/EAnge1o/nimbus-infer-rs) · [nimbus-vm](https://github.com/EAnge1o/nimbus-vm) · [nimbus-kv](https://github.com/EAnge1o/nimbus-kv) —— tapedeck 是系列里第一个面向**产品与工作流**的工具：同样零依赖，但交付的不是引擎，是一条更顺的工作流。

姊妹项目 [mocktape](https://github.com/EAnge1o/mocktape)：把同样的"录制一次、离线重放"思想用在 **HTTP 层**——把真实 API 录成可读的 YAML cassette，供前端/集成测试离线回放，404 时给出最近磁带匹配的诊断。mocktape 管 HTTP 流量，tapedeck 管 agent 决策，互相不知道对方存在，但常常一起出现在同一个测试套件里。
