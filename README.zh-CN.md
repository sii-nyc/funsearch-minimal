# FunSearch 最小复现版中文说明

这个仓库是一个**可读性优先、教学优先**的 FunSearch 最小复现版。

它保留的核心思想只有几件事：

- 用一个固定程序骨架定义问题
- 只让模型进化其中一个目标函数
- 真的执行候选程序并打分
- 只把合法程序放进数据库
- 从历史程序构造 best-shot prompt
- 用简化版 island 机制维持搜索多样性

这个项目不是生产系统，也不追求工程完备性。它的目标是：让你可以比较轻松地把代码从头读到尾，看懂 FunSearch 的主循环到底在做什么。

如果你更习惯英文说明，可以看 [README.md](/Users/hariseldon/Desktop/codes/funsearch/README.md)。

## 项目结构

- [main.py](/Users/hariseldon/Desktop/codes/funsearch/main.py)：极薄的程序入口，只负责调用 CLI。
- [funsearch/cli.py](/Users/hariseldon/Desktop/codes/funsearch/funsearch/cli.py)：命令行参数解析、问题选择、LLM 选择。
- [funsearch/core.py](/Users/hariseldon/Desktop/codes/funsearch/funsearch/core.py)：`ProblemSpecification`、候选评估、搜索主循环。
- [funsearch/prompting.py](/Users/hariseldon/Desktop/codes/funsearch/funsearch/prompting.py)：构造 prompt，只抽取/替换目标函数。
- [funsearch/database.py](/Users/hariseldon/Desktop/codes/funsearch/funsearch/database.py)：程序数据库、island、按 signature 聚类、reset。
- [funsearch/llm.py](/Users/hariseldon/Desktop/codes/funsearch/funsearch/llm.py)：LLM 抽象、OpenAI 兼容实现、离线 `MockLLM`。
- [funsearch/tracing.py](/Users/hariseldon/Desktop/codes/funsearch/funsearch/tracing.py)：把 prompt、completion、事件和数据库快照写到本地。
- [funsearch/capset.py](/Users/hariseldon/Desktop/codes/funsearch/funsearch/capset.py)：cap set 示例问题。
- [funsearch/string_hash.py](/Users/hariseldon/Desktop/codes/funsearch/funsearch/string_hash.py)：字符串哈希示例问题。
- [tests/test_funsearch.py](/Users/hariseldon/Desktop/codes/funsearch/tests/test_funsearch.py)：单元测试和集成测试。

prompt 构造仍然把 skeleton 视为只读，但现在会把 seed program 里除目标函数外的固定源码一并放进上下文，让模型能看到目标函数在 `main` 和辅助函数里是怎么被调用的。

## 项目里有哪些问题

目前内置了两个 problem。

### 1. `capset`

这是更接近论文原始味道的示例。

- 固定程序骨架里有 `solve(n)`
- FunSearch 只进化 `priority(element, n)`
- evaluator 检查输出是不是合法 cap set
- 分数就是 cap set 的大小

如果你想看“论文风格”的最小复现，优先看它。

### 2. `string-hash`

这是更短、更好懂的教学示例。

- 固定 `main(problem)` evaluator
- FunSearch 只进化 `hash_string(s)`
- evaluator 把一组固定的混合真实字符串映射到 buckets
- 分数只看 bucket load variance，越均匀越好

如果你想更直接地看明白“只搜索一个小函数”是怎么形成闭环的，优先看它。

## 最常见运行方式

### 离线运行 cap set

```bash
uv run python main.py --llm mock --iterations 8 --islands 4 --reset-interval 4 --inputs 1,2,3,4
```

### 离线运行 string-hash

```bash
uv run python main.py --problem string-hash --llm mock --iterations 8 --islands 4 --reset-interval 4
```

### 自定义 string-hash 的桶数量和每组字符串数量

```bash
uv run python main.py \
  --problem string-hash \
  --llm mock \
  --iterations 8 \
  --islands 4 \
  --reset-interval 4 \
  --string-hash-buckets 23 \
  --string-hash-strings-per-case 12
```

### 调用 OpenAI 兼容接口

```bash
uv run python main.py \
  --llm openai-compatible \
  --base-url http://localhost:8000/v1 \
  --api-key dummy \
  --model your-model-name \
  --iterations 8 \
  --islands 4 \
  --reset-interval 4 \
  --inputs 1,2,3,4
```

### 保存搜索过程

```bash
uv run python main.py \
  --problem string-hash \
  --llm mock \
  --iterations 8 \
  --islands 4 \
  --reset-interval 4 \
  --trace-dir runs/hash-trace
```

生成 trace 后，目录里还会额外写出一个 `trace_report.txt`，把整次运行整理成单个纯文本文件，适合直接在编辑器里查看，不用担心终端窗口太窄或历史记录太长。

### 打开 trace 命令行可视化界面

先生成 trace，再启动终端 viewer：

```bash
uv run python -m funsearch.trace_viewer --trace-dir runs/hash-trace
```

这个界面会在终端里自动刷新，适合边跑边看。它会集中展示：

- 当前运行状态
- 已完成到第几轮
- 每轮选中了哪个 island
- 该 island 当时有哪些程序
- prompt 选了哪些历史程序
- prompt 内容
- 模型 completion
- 重建出的 candidate program
- 候选是否通过评估
- 分数或拒绝原因
- 本轮是否触发 reset

常用按键：

- `←` / `→` 或 `h` / `l`：上一轮 / 下一轮
- `[` / `]` 或 `tab`：切换详情 section
- `j` / `k`：滚动当前 section
- `f`：切换是否自动跟随最新轮次
- `r`：立即刷新
- `q`：退出

### 分析某一个 hash 程序的桶分布

如果你已经拿到某个完整程序，想看它在当前固定输入上到底把字符串分到哪些桶里，可以直接运行：

```bash
uv run python -m funsearch.hash_analysis \
  --program runs/hash-trace/programs/program_000001.py \
  --string-hash-buckets 101 \
  --string-hash-strings-per-case 100
```

这个分析入口会输出：

- 当前固定输入上的 score 和 variance
- 使用了多少个桶、空桶多少、冲突多少、最大桶负载多少
- bucket load histogram
- 每个桶的负载
- 每个桶里具体有哪些字符串

## 所有命令行参数说明

命令行参数定义在 [funsearch/cli.py](/Users/hariseldon/Desktop/codes/funsearch/funsearch/cli.py)。

### `--problem`

作用：选择要运行的内置问题。

可选值：

- `capset`
- `string-hash`

默认值：

- `capset`

说明：

- 如果你不写这个参数，默认跑 `capset`
- 想跑字符串哈希示例时，需要显式写 `--problem string-hash`

### `--llm`

作用：选择使用哪种 LLM 后端。

可选值：

- `mock`
- `openai-compatible`

默认值：

- `openai-compatible`

说明：

- `mock`：完全离线，不依赖任何在线模型，适合快速演示
- `openai-compatible`：通过官方 OpenAI Python SDK 调用兼容接口

### `--base-url`

作用：指定 OpenAI 兼容接口的基地址。

示例：

```bash
--base-url http://localhost:8000/v1
```

默认来源：

- 环境变量 `FUNSEARCH_BASE_URL`

说明：

- 只有在 `--llm openai-compatible` 时才需要
- 如果使用 `mock`，这个参数不会被用到

### `--api-key`

作用：指定访问模型接口用的 API key。

默认来源：

- 环境变量 `FUNSEARCH_API_KEY`

默认值：

- 空字符串

说明：

- 只有在 `--llm openai-compatible` 时才有意义
- 某些本地兼容服务可能允许传一个占位值，例如 `dummy`

### `--model`

作用：指定要调用的模型名。

示例：

```bash
--model gpt-4.1-mini
```

默认来源：

- 环境变量 `FUNSEARCH_MODEL`

说明：

- 只有在 `--llm openai-compatible` 时必需
- 不同服务支持的模型名不同

### `--iterations`

作用：控制搜索主循环跑多少轮。

默认值：

- `20`

说明：

- 每一轮都会：
  - 从数据库采样历史程序
  - 构造 prompt
  - 调用 LLM 生成新版本目标函数
  - 评估候选程序
  - 尝试把合法程序加入数据库
- 值越大，搜索越久，通常也越容易找到更好的程序
- 对教学演示来说，`4` 到 `10` 往往就足够看流程

### `--islands`

作用：控制数据库里 island 的数量。

默认值：

- `4`

说明：

- 每个 island 可以理解成一个子种群
- 不同 island 有助于维持多样性，避免所有搜索都挤到同一个局部模式
- 值太小，多样性可能不足
- 值太大，在很短的 demo 里未必有明显收益

### `--reset-interval`

作用：控制多久执行一次 island reset。

默认值：

- `10`

说明：

- 这里的 reset 不是按时间触发，而是按“已评估候选数”触发
- 当评估次数达到这个间隔时，最差的一半 island 会被较强 island 的最佳程序重新播种
- 值越小，reset 越频繁
- 值设为 `0` 或负数时，相当于不触发 reset

### `--inputs`

作用：为 `capset` 问题指定评测输入。

默认值：

```text
1,2,3,4
```

说明：

- 这个参数只在 `--problem capset` 时生效
- 它表示要评估哪些维度的 cap set，例如：

```bash
--inputs 1,2,3
```

表示会分别在 `n=1`、`n=2`、`n=3` 上执行 `main(n)`，得到一个 3 维 signature。

补充：

- `string-hash` 问题不会用这个参数
- `string-hash` 使用内置固定测试集

### `--temperature`

作用：控制调用在线 LLM 时使用的采样温度。

默认值：

- `0.8`

说明：

- 只在 `--llm openai-compatible` 时使用
- 温度越高，输出通常越发散
- 温度越低，输出通常越保守
- 这个参数不会影响数据库采样温度；数据库内部的采样温度在当前版本里写在代码默认配置里

### `--seed`

作用：控制搜索过程中的随机种子。

默认值：

- `0`

说明：

- 它会影响数据库采样和 reset 过程中的随机选择
- 使用相同种子，通常更容易复现相同的搜索轨迹
- `string-hash` 默认测试集本身也是固定的，便于复现

### `--trace-dir`

作用：指定一个空目录，把搜索过程写进去。

示例：

```bash
--trace-dir runs/mock-trace
```

说明：

- 目录必须是空的，否则程序会报错
- 如果不提供这个参数，就不会保存 trace 文件
- 这是理解 FunSearch 过程非常有用的一个参数

## 运行后会输出什么

CLI 最终会打印：

- `Best aggregate score`
  含义：最佳程序在所有输入上的总分
- `Best signature`
  含义：最佳程序在每个输入上的逐项得分
- `Trace directory`
  含义：如果你开了 `--trace-dir`，这里会打印 trace 路径
- `Best program`
  含义：当前数据库中最佳程序的完整源码

在运行过程中，CLI 现在还会默认把每一轮的记录直接输出到终端，包括：

- 当前选中的 island 以及它里面的程序
- 用来构造 prompt 的历史程序
- prompt 全文
- 模型 completion 原文
- 重建出的 candidate program
- 候选接受 / 拒绝结果
- reset 动作
- 本轮结束后的数据库摘要

如果你只想看最终结果，不想看中间每轮记录，可以加：

```bash
--no-live-report
```

## `Best signature` 是什么

这是理解 FunSearch 的一个关键概念。

每个候选程序都不会只在一个输入上打分，而是会在 `ProblemSpecification.inputs` 里的所有输入上各跑一次，得到一组分数。例如：

```python
(-2.0, -1.3, -3.8, -0.9)
```

这组 per-input 分数就叫 `signature`。

然后再通过 `aggregate_scores(...)` 把它们聚合成一个总分。

当前两个问题的聚合方式都很简单：

- 直接取平均值

## Trace 目录里有什么

如果使用了 `--trace-dir`，目录里会有这些内容：

- `run.json`
  含义：本次运行的参数、问题信息、LLM 类型
- `events.jsonl`
  含义：逐条事件日志，例如 prompt 采样、candidate 接受/拒绝、reset 等
- `prompts/`
  含义：每一轮发给模型的 prompt
- `completions/`
  含义：模型原始输出
- `candidates/`
  含义：从 completion 中提取并重建出的完整候选程序
- `programs/`
  含义：所有被数据库引用过的已接受程序
- `snapshots/`
  含义：数据库在初始化、每轮之后、最终结束时的完整快照

如果你想真正看懂搜索过程，建议优先看：

1. `prompts/`
2. `completions/`
3. `events.jsonl`
4. `snapshots/`

不过现在默认更推荐直接看主命令运行时输出的内嵌记录；如果你想在运行结束后再回看 trace，仍然可以使用上面的终端版 `trace_viewer`。

## 两个问题的 evaluator 分别在做什么

### `capset` evaluator

定义在 seed program 中，逻辑很直接：

- 调用 `solve(n)` 生成候选集合
- 检查它是不是合法 cap set
- 如果合法，返回集合大小
- 如果不合法，返回 `None`

而 `core.py` 会把 `None` 视为非法得分并拒绝程序。

### `string-hash` evaluator

定义也在 seed program 中。这里的 seed `hash_string` 故意很弱，只做非常基础的滚动累加；evaluator 的逻辑是：

- 取一组固定的混合真实字符串
- 用固定的 `hash_string(s)` 逐个求 hash
- 映射到 `num_buckets` 个桶
- 统计 bucket load variance
- 返回 `-variance`

因为要最大化分数，所以负的 variance 越接近 0 越好，表示桶越均匀。

## 超时保护是怎么工作的

当前版本会给候选程序执行加一个小的超时保护，定义在 [funsearch/core.py](/Users/hariseldon/Desktop/codes/funsearch/funsearch/core.py)。

目的很简单：

- 如果 LLM 生成了死循环，不要让整个搜索直接卡死

保护范围包括：

- 顶层 `exec(program_source, namespace)`
- 每次 `entrypoint(problem_input)` 调用

当前默认超时：

- `1.0` 秒

这个值来自 `ProblemSpecification.evaluation_timeout`。

## 最推荐的阅读顺序

如果你想看懂整个项目，我建议按这个顺序读：

1. [main.py](/Users/hariseldon/Desktop/codes/funsearch/main.py)
2. [funsearch/cli.py](/Users/hariseldon/Desktop/codes/funsearch/funsearch/cli.py)
3. [funsearch/core.py](/Users/hariseldon/Desktop/codes/funsearch/funsearch/core.py)
4. [funsearch/prompting.py](/Users/hariseldon/Desktop/codes/funsearch/funsearch/prompting.py)
5. [funsearch/database.py](/Users/hariseldon/Desktop/codes/funsearch/funsearch/database.py)
6. [funsearch/string_hash.py](/Users/hariseldon/Desktop/codes/funsearch/funsearch/string_hash.py)
7. [funsearch/capset.py](/Users/hariseldon/Desktop/codes/funsearch/funsearch/capset.py)
8. [tests/test_funsearch.py](/Users/hariseldon/Desktop/codes/funsearch/tests/test_funsearch.py)

如果你只想先看一个最短闭环，建议先读：

1. [funsearch/string_hash.py](/Users/hariseldon/Desktop/codes/funsearch/funsearch/string_hash.py)
2. [funsearch/core.py](/Users/hariseldon/Desktop/codes/funsearch/funsearch/core.py)
3. [funsearch/prompting.py](/Users/hariseldon/Desktop/codes/funsearch/funsearch/prompting.py)

## 测试命令

```bash
uv run python -m unittest -v
```

## 一句话总结

这个仓库的重点不是“搜得多强”，而是“你能清楚看懂 FunSearch 是怎么工作的”。
