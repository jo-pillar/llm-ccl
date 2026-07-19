# LLM-CCL 实验流水线

这套新代码独立放在 `agent/scripts/llm-ccl/`，旧的四个实验准备脚本保持不变。

主流程固定为：

```text
prepare
  -> 所有 case 使用 llm_elite + all 搜索
  -> 每个 case 选择最新成功 attempt 中 time_us 最小的有效候选
  -> 等全部搜索和筛选结束后，只 resim 每个 case 的 best 候选
  -> report
```

搜索失败的 case 不做 resim，但会出现在最终报告中。重复执行 `search` 时，成功 case 会跳过，失败 case 会创建新的 `attempt-NNNN`；成功 resim 也不会重复运行。

## 使用方法

在 `agent/` 下运行：

```bash
.venv/bin/python scripts/llm-ccl/run.py projects
```

完整运行一个项目：

```bash
.venv/bin/python scripts/llm-ccl/run.py run v100_dgx2_clos \
  --model hosted_vllm/deepseek-ai/DeepSeek-V4-Flash \
  --api-base http://127.0.0.1:8000/v1 \
  --synthesize-bin /path/to/syccl/build/synthesize \
  --jobs 4
```

也可以分阶段执行，适合中断后继续：

```bash
.venv/bin/python scripts/llm-ccl/run.py prepare v100_dgx2_clos \
--launch-id my-run

.venv/bin/python scripts/llm-ccl/run.py search \
  ../experiments/llm-ccl/v100_dgx2_clos/my-run \
  --model hosted_vllm/deepseek-ai/DeepSeek-V4-Flash \
  --api-base http://127.0.0.1:8000/v1 \
  --jobs 4

.venv/bin/python scripts/llm-ccl/run.py select \
  ../experiments/llm-ccl/v100_dgx2_clos/my-run

.venv/bin/python scripts/llm-ccl/run.py resim \
  ../experiments/llm-ccl/v100_dgx2_clos/my-run \
  --synthesize-bin /path/to/syccl/build/synthesize

.venv/bin/python scripts/llm-ccl/run.py report \
  ../experiments/llm-ccl/v100_dgx2_clos/my-run
```

最终报告位于 bundle 的 `reports/report.json` 和 `reports/report.csv`。

## 添加新的实验项目

不需要修改中央 registry。只需要：

1. 准备 TopoDSL 模板、instruction 模板和带有 `EVOLVE-BLOCK` 的初始程序。
2. 在 `llm_ccl/projects/` 下新增一个 Python 文件。
3. 在该文件中导出名为 `PROJECT` 的 `ExperimentSpec`。

最小示例：

```python
from pathlib import Path

from llm_ccl.models import ExperimentSpec, LayerShape, ScaleSpec, case_matrix


TEMPLATE_ROOT = Path("/absolute/path/to/templates")

SCALE = ScaleSpec(
    "8hosts-64gpu",
    64,
    (
        LayerShape(1, group_num=8, node_num=8),
        LayerShape(2, group_num=64, node_num=1),
    ),
)

PROJECT = ExperimentSpec(
    name="my_new_project",
    topology_template=TEMPLATE_ROOT / "topology.py",
    instruction_template=TEMPLATE_ROOT / "prompt.txt",
    initial_program=TEMPLATE_ROOT / "initial_program.py",
    cases=case_matrix(
        (SCALE,),
        ("allgather",),
        (65536, 262144, 1048576),
    ),
)
```

注意：

- `total_message_size` 是整个 collective 的消息量；框架只在准备阶段除以 GPU 数一次，生成 `config.json` 的 `coll.byte`。
- `LayerShape` 只修改不同规模下的 `group_num/node_num`。带宽和延迟继续以 TopoDSL 模板中的 `LinkSpec` 为统一来源。
- 项目模块会被自动发现；可用 `run.py projects` 检查是否加载成功。
- 当前搜索策略固定为 `llm_elite + all`，没有跨策略比较。
