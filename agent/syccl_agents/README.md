# SyCCL Two-Agent 搜索流程

本文档说明 `syccl-two-agent` 的工程入口、每轮搜索的数据流、输出文件和常见排错点。这里的 two-agent 指一个 proposal agent 负责生成 SyCCL compact SketchDSL，一个 record agent 负责把评估结果压缩成下一轮可用的历史记录和方向提示。

当前主入口是 [`../scripts/run_syccl_two_agent.py`](../scripts/run_syccl_two_agent.py)。它复用 SimpleTES 的 LLM 后端、checkpoint 和日志系统，但把 SimpleTES 默认的多链搜索替换成 SyCCL 专用的顺序 proposal/evaluate/record 循环，核心实现位于 [`../simpletes/engine/syccl_two_agent.py`](../simpletes/engine/syccl_two_agent.py)。

## 快速运行

先准备 Python 环境和 `flow-sim-rs` binary：

```bash
cd /home/antl/wzd/llm-ccl/agent
uv sync

cd /home/antl/wzd/llm-ccl/Flow-Simulator/flow-sim-rs
cargo build
```

在 `agent/env.toml` 中配置 LLM：

```toml
model = "deepseek/deepseek-v4-flash"
api_base = "https://api.deepseek.com"
api_key = "..."
```

从 `agent/` 目录启动一次 1 轮搜索：

```bash
cd /home/antl/wzd/llm-ccl/agent

uv run python scripts/run_syccl_two_agent.py \
  --topo examples/topologies/clos_2host.py \
  --output-dir test_result/syccl_two_agent_demo \
  --rounds 1 \
  --flow-sim-bin ../Flow-Simulator/flow-sim-rs/target/debug/flow-sim-rs \
  --save-llm-io
```

脚本结束时会打印 `instance_id`、`checkpoint_dir`、SimpleTES `best_score`，以及从 SyCCL `summary.json` 读取的 `syccl_best_time_us`、`syccl_best_source`、`syccl_best_source_round` 和 `syccl_summary_json`。SyCCL two-agent 的主要产物在：

```text
<output-dir>/checkpoints/<date>/instance-<id>/syccl_two_agent/
```

## 输入

### TopoDSL

`--topo` 指向一个 TopoDSL 文件，由 [`topodsl.py`](topodsl.py) 读取。支持三种形式：

- Python 文件定义 `topology()`，返回参数字典。
- Python 文件实例化 `BaseTopology` 子类，例如 [`../examples/topologies/clos_2host.py`](../examples/topologies/clos_2host.py)。
- 文本文件使用 `key = value` 行。

最小字典形式示例：

```python
def topology():
    return {
        "family": "clos",
        "hosts": 2,
        "gpus_per_host": 2,
        "nics_per_host": 1,
        "leaf_switches": 1,
        "spine_switches": 1,
        "message_size": 4096,
        "collective": "allgather",
    }
```

当前支持的 topology family 是 `clos` 和 `multirail`。当前 smoke workflow 只支持 `allgather`；如果传入其他 collective，会在 TopoDSL 归一化阶段失败。

### 配置覆盖

`run_syccl_two_agent.py` 的常用参数：

| 参数 | 作用 |
| --- | --- |
| `--topo` | 输入 TopoDSL 文件。 |
| `--output-dir` | 本次运行的根输出目录。 |
| `--rounds` | proposal 轮数，映射到 SimpleTES `max_generations`。 |
| `--collective` | 可选覆盖/校验 TopoDSL 中的 collective。 |
| `--message-size` | 可选覆盖/校验 TopoDSL 中的 message size。 |
| `--flow-sim-bin` | `flow-sim-rs` 可执行文件路径，默认假设在 `PATH` 中。 |
| `--env-toml` | LLM 配置文件，默认 `agent/env.toml`。 |
| `--temperature` | proposal/record LLM 温度。 |
| `--max-tokens` | 单次 LLM 输出 token 上限。 |
| `--timeout` | LLM 调用超时。flow-sim 使用 SimpleTES `eval_timeout` 默认值，当前 CLI 未单独暴露该参数。 |
| `--save-llm-io` | 保存完整 prompt、completion 和 raw output。 |

## 搜索主流程

启动脚本会先在 `<output-dir>/simpletes_task/` 写入临时的 `init_program.py`、`evaluator.py` 和 `instruction.txt`。这些文件保留了 SimpleTES 的任务文件复制语义；`init_program.py` 会被 SyCCL runtime 当作初始候选真实执行和 flow-sim 评分，`evaluator.py` 只是兼容 SimpleTES 配置接口的占位文件。

初始化阶段：

1. SimpleTES engine 建立 checkpoint 目录。
2. runtime 读取 TopoDSL，得到规范化的 `TopologyParams` 和稳定 `topodsl_config_id`。
3. [`config_render.py`](config_render.py) 把 TopoDSL 参数渲染成 `flow-sim-config.json`。
4. runtime 执行复制出的 `init_program.py`，生成 `initial-candidate.py`、`initial-sketch.json` 和 `initial-flow-sim.json`。
5. 初始候选被写成 SimpleTES root node，`combined_score = -time_us`，并作为第一轮 proposal prompt 的 root-node inspiration。
6. runtime 记录 `syccl-sketch-search` 当前被跳过：该 Rust crate 目前输出 SyCCL native JSON，而 proposal prompt 需要 LLM 友好的 compact DSL。

每一轮 `round_id = 1..rounds` 执行：

1. 应用已经完成的 record-agent 结果，更新 rolling record summary 和 direction hint。
2. 构造 proposal prompt。prompt 由 [`../prompt/syccl_two_agent/proposal_base.txt`](../prompt/syccl_two_agent/proposal_base.txt) 和 [`../prompt/syccl_two_agent/proposal_dynamic_context.txt`](../prompt/syccl_two_agent/proposal_dynamic_context.txt) 拼接而成，包含：
   - 原始 TopoDSL；
   - topology summary；
   - layer/group 语义；
   - compact SketchDSL seed；
   - sketch enumeration skipped note；
   - 初始程序代码和真实 flow-sim 指标；
   - rolling record summary；
   - direction hint；
   - SimpleTES evolve-block scaffold，包括 `EXACT_PREFIX` 和 `EXACT_SUFFIX`。
3. proposal agent 返回一个 Python code block，必须包含 `# EVOLVE-BLOCK-START` 和 `# EVOLVE-BLOCK-END`，并实现 `construct_sketches()`。
4. runtime 抽取 evolve block，保存为 `rounds/round-XXXX-candidate.py`，执行 `construct_sketches()` 或 `run_code()`。
5. [`sketch_dsl.py`](sketch_dsl.py) 把候选规范化为 compact transmission 列表，并写入 `rounds/round-XXXX-sketch.json`。
6. [`flow_sim.py`](flow_sim.py) 调用：

```text
flow-sim-rs simulate-sketch --config flow-sim-config.json --sketch round-XXXX-sketch.json --output round-XXXX-flow-sim.json
```

7. runtime 校验 flow-sim 输出必须包含：
   - `time_us`：本轮 full allgather schedule 的完成时间；
   - `flow_count`：展开后的 flow 数量；
   - `bottleneck_profile`：以 sketch transmission 为单位的瓶颈诊断。
8. 生成本轮指标：
   - `combined_score = -time_us`；
   - `completion_time = time_us`；
   - `algorithm_bandwidth = message_size / time_us`；
   - `validity_status`；
   - `best_so_far`。
9. 若本轮比当前 best 更快，写入 `best_sketch.json`，并在 summary 中把 `best_source` 标记为 `proposal`。
10. 构造 record prompt，由 [`../prompt/syccl_two_agent/record_agent.txt`](../prompt/syccl_two_agent/record_agent.txt) 约束 record agent 输出 JSON。
11. record agent 异步运行，把本轮 proposal、candidate sketch、metrics 和 bottleneck profile 压缩成历史记录。proposal 搜索不会等待每个 record 立即完成，但最多允许 5 个 record backlog。已完成的 record 结果按 round 顺序应用；只有成功解析并应用的 LLM record 输出会进入后续 proposal prompt。
12. runtime 写入 `rounds.jsonl`、`llm_calls.jsonl`，并把本轮完整 Python candidate 作为 SimpleTES node 提交到 checkpoint。所有 proposal node 都挂在初始 root node 下，避免把搜索历史伪装成线性代码演化链。

结束阶段会 drain 已完成的 record tasks，取消仍未完成的 record tasks，写入 `summary.json`，然后让 SimpleTES 写最终 checkpoint。

## Compact SketchDSL

proposal agent 只生成 root GPU 0 的单根 broadcast tree。评估器会把这个 root-0 tree 按 GPU id 旋转到所有 root，展开为完整 allgather schedule 后再仿真。

推荐返回形式：

```python
# EVOLVE-BLOCK-START
def construct_sketches():
    return [
        (0, 1, 0, 0, [1]),
        (1, 3, 0, 0, [2]),
        (2, 1, 1, 2, [3]),
    ]
# EVOLVE-BLOCK-END
```

每个 transmission 字段为：

```text
(step, layer, group, srcs, dsts)
```

也可以使用 dict 形式：

```python
{"step": 1, "layer": 3, "group": 0, "srcs": [0], "dsts": [2]}
```

关键约束：

- GPU 0 初始拥有 root chunk。
- 每个非 root GPU 必须正好收到一次；GPU 0 不能出现在 `dsts`。
- `srcs` 中的 GPU 必须在更早 step 已经拥有 chunk。
- 同一 step 不能用刚收到 chunk 的 GPU 继续转发。
- `layer/group` 必须和 TopoDSL 派生出的 layer/group membership 匹配。
- Clos 主要使用 layer `1`、`3`、`4`；multirail 主要使用 layer `1`、`3`。

## 输出文件

典型目录结构：

```text
test_result/syccl_two_agent_demo/
  simpletes_task/
    init_program.py
    evaluator.py
    instruction.txt
  checkpoints/
    <date>/
      instance-<id>/
        run.log
        db_state_<timestamp>/
        syccl_two_agent/
          flow-sim-config.json
          rounds.jsonl
          llm_calls.jsonl
          summary.json
          best_sketch.json
          record_errors.jsonl
          rounds/
            round-0001-candidate.py
            round-0001-sketch.json
            round-0001-flow-sim.json
          llm_io/
            round-0001-proposal-prompt.txt
            round-0001-proposal-completion.txt
            round-0001-record-attempt-01-prompt.txt
```

`rounds.jsonl` 每行对应一轮 proposal/evaluate 结果，核心字段由 [`records.py`](records.py) 校验：

| 字段 | 含义 |
| --- | --- |
| `topodsl_config_id` | TopoDSL 规范化参数的稳定 hash。 |
| `collective` | 当前 collective，目前应为 `allgather`。 |
| `message_size` | collective 消息大小，单位 byte。 |
| `round_id` | 搜索轮次，从 1 开始。 |
| `candidate_sketch` | 规范化后的 compact SketchDSL。 |
| `validity_status` | `ok` 或 `invalid: <reason>`。 |
| `combined_score` | `-completion_time`，越大越好。 |
| `completion_time` | flow-sim `time_us`。 |
| `algorithm_bandwidth` | `message_size / completion_time`。 |
| `bottleneck_profile` | sketch-level 瓶颈诊断。 |
| `best_so_far` | 本轮是否刷新 best sketch。 |
| `candidate_code_path` | 本轮候选 Python 文件相对路径。 |

`llm_calls.jsonl` 记录 proposal 和 record agent 调用：

- `agent_type`：`proposal` 或 `record`；
- `round_id`；
- `model_name`；
- `input_tokens` / `output_tokens`；
- `cached_input_tokens` / `reasoning_tokens`；
- `wall_clock_time_ms`；
- `prompt_hash` / `completion_hash`；
- `attempt`：record agent 重试时出现。

`summary.json` 汇总本次运行：

- `initial_time_us`、`initial_combined_score`、`initial_validity_status`；
- `rounds_completed`；
- `best_time_us`；
- `best_source` 和 `best_source_round`；
- `best_sketch_stages`；
- `enumeration_status` 和 `enumeration_note`；
- record task started/applied/cancelled/failed/retried；
- proposal 等待 record backlog 的次数和耗时。

## 当前限制

- 当前 workflow 只支持 `allgather`。
- `syccl-sketch-search` 暂未接入 proposal prompt；初始化阶段会显式记录 `enumeration_status = "skipped"`。
- `flow-sim-rs simulate-sketch` 必须输出 `bottleneck_profile`。如果只输出 legacy `links`，runtime 会失败并提示需要更新 flow-sim。
- proposal 产物无效时，本轮会被记录为 `invalid`，score 使用大负数；但 flow-sim 输出结构错误、binary 不存在或超时会终止运行。
- record agent 输出依赖 `json-repair` 做容错 JSON 解析。单轮最多重试 2 次；失败后会写入 `record_errors.jsonl`，该轮 record summary 会被跳过，但 proposal 搜索继续。
- direct [`runner.py`](runner.py) 提供测试/工具用的同步 runner；主 CLI 使用 SimpleTES runtime，以保留 checkpoint、日志和 LLM I/O 记录。

## 调试建议

- 看完整 prompt：运行时加 `--save-llm-io`，检查 `syccl_two_agent/llm_io/`。
- 看候选是否被正确抽取：检查 `rounds/round-XXXX-candidate.py`。
- 看 DSL 规范化结果：检查 `rounds/round-XXXX-sketch.json`。
- 看仿真输出和瓶颈：检查 `rounds/round-XXXX-flow-sim.json` 中的 `bottleneck_profile`。
- 看 record agent 是否可解析：检查 `llm_calls.jsonl` 中 `agent_type=record` 的 completion，以及 `record_errors.jsonl`。
- 看最终最佳结果：检查 `best_sketch.json` 和 `summary.json` 的 `best_time_us`、`best_source`。

## 测试

单元测试：

```bash
cd /home/antl/wzd/llm-ccl/agent
uv run pytest tests/test_syccl_two_agent.py -q
```

集成测试默认跳过，需要显式打开：

```bash
cd /home/antl/wzd/llm-ccl/agent
RUN_SYCCL_INTEGRATION=1 uv run pytest tests/integration/test_syccl_two_agent_runtime.py -q
```
