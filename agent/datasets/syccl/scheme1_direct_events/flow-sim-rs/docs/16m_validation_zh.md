# 16M allgather 验证记录

## 数据与路径

- candidate 包：`/home/antl/mntdisk/syccl-llm-scheme1-direct-events/simpletes/multirail-512gpu/multirail-512gpu-allgather-16m-candidates-with-config.tar.gz`
- 解压目录：`/home/antl/mntdisk/new-simulator/inputs/16m-package`
- 使用 config：`/home/antl/mntdisk/syccl-llm-scheme1-direct-events/configs/multirail-512gpu/allgather/multirail-512gpu-ag-16m.json`
- Rust 输出：`/home/antl/mntdisk/new-simulator/16m/rust-runs`
- Rust manifest：`/home/antl/mntdisk/new-simulator/16m/benchmarks.json`
- Rust summary：`/home/antl/mntdisk/new-simulator/16m/reports/rust-summary.csv`
- SimAI cache：`/home/antl/mntdisk/new-simulator/16m/simai-cache`

## Rust 全量结果

已解压并运行 166 个 `candidate-translated.json`。Rust 批处理命令：

```bash
/home/antl/mntdisk/new-simulator/cargo-target/release/flow-sim-rs batch \
  --config /home/antl/mntdisk/syccl-llm-scheme1-direct-events/configs/multirail-512gpu/allgather/multirail-512gpu-ag-16m.json \
  --input-dir /home/antl/mntdisk/new-simulator/inputs/16m-package/16m/full/eval_artifacts/scheme1_direct_events \
  --output-dir /home/antl/mntdisk/new-simulator/16m/rust-runs \
  --manifest /home/antl/mntdisk/new-simulator/16m/benchmarks.json \
  --summary /home/antl/mntdisk/new-simulator/16m/reports/rust-summary.csv
```

结果统计：

- case 数：166
- min：51142.163 us
- p25：53145.409 us
- median：54562.659 us
- p75：56572.226 us
- max：200928.553 us
- mean：58646.287934 us

## SimAI 16M baseline 状态

修复了 SimAI baseline 输入生成中的硬编码：`syccl-flow-size` 和 workload 中的 `ALLGATHER` 大小现在优先从 translated schedule 的 `chunk_size_byte` 读取，缺失时再使用 config 的 `coll.byte`。对 16M dry-run 的检查结果：

```text
syccl-flow-size: 16777216
eval_20260513-214112_937056_pet1t0qs -1 1 ALLGATHER 16777216 1 NONE 0 1 NONE 0 1
```

随后启动了代表集中的第一个 case：

```bash
python3 scripts/run_simai_baselines.py \
  --root /home/antl/mntdisk/new-simulator/16m \
  --config /home/antl/mntdisk/syccl-llm-scheme1-direct-events/configs/multirail-512gpu/allgather/multirail-512gpu-ag-16m.json \
  --selection representative \
  --jobs 1 \
  --force
```

SimAI 日志确认读取到 `syccl-flow-size: 16777216`。首个 case 为 Rust 排序中最快的一类：

```text
eval_20260513-214112_937056_pet1t0qs
rust_time_us=51142.163
flow_count=261632
channel_count=512
```

首次试跑时，该 SimAI 进程运行约 16 分钟后仍未产生完整 `EndToEnd.csv`，只写出 3584 行 FCT，且进程仍占用约 21 GB RSS 和一个满载 CPU。后续按“资源充足，不用在乎资源占用”的策略继续运行，不再因为单 case 耗时长而提前停止。

后续验证了 `ns3.36.1-AstraSimNetwork-optimized` 不可作为 16M baseline：两轮 24 case
并行启动都会在写出有效 FCT/EndToEnd 之前 SIGSEGV。当前可信 baseline 仍使用 debug
binary。为了避免 MockNcclLog 写满根分区，脚本默认设置：

```text
AS_LOG_LEVEL=3
AS_LOG_DIR=/home/antl/mntdisk/new-simulator/mocknccl-logs
SIMAI_LOG_PATH=/home/antl/mntdisk/new-simulator/mocknccl-logs
```

其中 `SIMAI_LOG_PATH` 是当前 SimAI 代码实际读取的日志路径变量。

当前长作业命令：

```bash
TMPDIR=/home/antl/mntdisk/new-simulator/tmp \
python3 scripts/run_simai_baselines.py \
  --root /home/antl/mntdisk/new-simulator/16m \
  --config /home/antl/mntdisk/syccl-llm-scheme1-direct-events/configs/multirail-512gpu/allgather/multirail-512gpu-ag-16m.json \
  --selection all \
  --jobs 24 \
  --truncate-legacy-mocknccl-logs
```

该命令先启动 Rust 排序最前的 24 个 case。为覆盖慢端样本，另建了一个独立的 tail
采样目录，避免与主目录重复写同一批 case：

```bash
TMPDIR=/home/antl/mntdisk/new-simulator/tmp \
python3 scripts/run_simai_baselines.py \
  --root /home/antl/mntdisk/new-simulator/16m-tail \
  --config /home/antl/mntdisk/syccl-llm-scheme1-direct-events/configs/multirail-512gpu/allgather/multirail-512gpu-ag-16m.json \
  --selection tail \
  --limit 8 \
  --jobs 8 \
  --truncate-legacy-mocknccl-logs
```

截至 2026-05-18 21:04 CST，主批次 24 个 debug 进程和 tail 批次 8 个 debug 进程均在运行，
尚未产生有效的非空 `*EndToEnd.csv`，因此 16M 暂时还不能生成与 SimAI 对齐的 compare
报告。4K 已完成样本的完整 `fct.txt` 行数为 261632；当前主批次最大 39936 行、总进度约
6.48%，tail 批次 8 个 case 均在 3584 行、总进度约 1.37%。这说明 SimAI/ns-3 baseline
仍在正常推进，但 16M 的包级仿真 wall time 明显长于 4K。

另启动了一个单 case 的 debug `-t 8` 试验：

```bash
TMPDIR=/home/antl/mntdisk/new-simulator/tmp \
python3 scripts/run_simai_baselines.py \
  --root /home/antl/mntdisk/new-simulator/16m-debug-t8 \
  --config /home/antl/mntdisk/syccl-llm-scheme1-direct-events/configs/multirail-512gpu/allgather/multirail-512gpu-ag-16m.json \
  --selection head \
  --limit 1 \
  --jobs 1 \
  --simai-threads 8 \
  --truncate-legacy-mocknccl-logs
```

该进程未复现 optimized binary 的 SIGSEGV，但在高并发内存压力下并没有快速写出有效
`EndToEnd.csv`。当前仍以 debug `-t 1` 批次作为可信 baseline，`-t 8` 只作为独立测速样本。

脚本会跳过已有有效 `EndToEnd.csv` 的 case，并持续更新：

```text
/home/antl/mntdisk/new-simulator/16m/simai-cache/baseline-manifest.json
```

每个 case 目录下的 `simai.lock` 用于避免重复启动仍在运行的 case。新版本 lock 记录 JSON
元数据，包括真实 AstraSimNetwork 子进程 `pid`、runner `owner_pid`、case 名和命令；旧版纯
PID lock 仍可兼容读取。若重复启动脚本时看到 `locked` 状态，表示该 case 已有 SimAI 进程在跑，
等待后 resume 即可。
早期启动的主批次可能没有保留当前 lock 文件，因此判断作业是否仍在运行时应优先看监控脚本的
`running_simai_processes`，不要只看 `active_locks`。

每个 case 的 SimAI stdout/stderr 会追加到：

```text
/home/antl/mntdisk/new-simulator/16m/simai-cache/<case>/simai-run.log
```

在 `baseline-manifest.json` 中有有效 SimAI case 后，再运行 compare：

```bash
/home/antl/mntdisk/new-simulator/cargo-target/release/flow-sim-rs compare \
  --manifest /home/antl/mntdisk/new-simulator/16m/simai-cache/baseline-manifest.json \
  --output /home/antl/mntdisk/new-simulator/16m/reports/baseline-compare.json
```

若需要合并主批次和 tail 采样结果，先合并 manifest：

```bash
python3 scripts/merge_baseline_manifests.py \
  --output /home/antl/mntdisk/new-simulator/16m/reports/merged-baseline-manifest.json \
  /home/antl/mntdisk/new-simulator/16m/simai-cache/baseline-manifest.json \
  /home/antl/mntdisk/new-simulator/16m-tail/simai-cache/baseline-manifest.json

/home/antl/mntdisk/new-simulator/cargo-target/release/flow-sim-rs compare \
  --manifest /home/antl/mntdisk/new-simulator/16m/reports/merged-baseline-manifest.json \
  --output /home/antl/mntdisk/new-simulator/16m/reports/merged-baseline-compare.json
```

compare 报告中的 `passes_alignment_requirement` 会同时检查：

- 有效 SimAI case 中至少 95% 的绝对倍率误差不超过 5 倍；
- 去重 schedule 后、忽略 SimAI 两方案时间差不超过较小值 5% 的近似并列 pair，
  pairwise 趋势一致率不低于 95%。

也可以用监控脚本一次性汇总两个目录的状态，并在已有有效 baseline 时自动生成 compare：

```bash
python3 scripts/monitor_simai_baselines.py \
  --root /home/antl/mntdisk/new-simulator/16m \
  --root /home/antl/mntdisk/new-simulator/16m-tail \
  --output-manifest /home/antl/mntdisk/new-simulator/16m/reports/merged-baseline-manifest.json \
  --compare-output /home/antl/mntdisk/new-simulator/16m/reports/merged-baseline-compare.json
```

## 当前 Rust 模型验证结果

在 32 个已有有效 SimAI baseline case 上，当前 Rust 模型输出目录为：

```text
/home/antl/mntdisk/new-simulator/16m-rust-roundtrip-crossdominant-valid
```

运行命令：

```bash
target/release/flow-sim-rs batch \
  --config /home/antl/mntdisk/syccl-llm-scheme1-direct-events/configs/multirail-512gpu/allgather/multirail-512gpu-ag-16m.json \
  --input-dir /home/antl/mntdisk/new-simulator/16m-rust-hop0-rr-ackfix-valid/inputs \
  --output-dir /home/antl/mntdisk/new-simulator/16m-rust-roundtrip-crossdominant-valid/rust-runs \
  --manifest /home/antl/mntdisk/new-simulator/16m-rust-roundtrip-crossdominant-valid/benchmarks.json \
  --summary /home/antl/mntdisk/new-simulator/16m-rust-roundtrip-crossdominant-valid/reports/rust-summary.csv
```

结果：

```text
runtime: 119.48 s
abs_ratio_passed: 32 / 32
tolerant_unique_pairwise_consistency_rate: 152 / 154 = 0.987012987012987
passes_alignment_requirement: true
```

对应 4K 全量 107 case 的当前模型输出目录为：

```text
/home/antl/mntdisk/new-simulator/4k-rust-roundtrip-crossdominant
```

结果：

```text
runtime: 382.29 s
abs_ratio_passed: 107 / 107
tolerant_unique_pairwise_consistency_rate: 515 / 523 = 0.9847036328871893
passes_alignment_requirement: true
```

如果 `merged_valid_cases == 0`，脚本只写合并 manifest 和进度摘要，不运行 compare。
监控输出中的关键字段：

- `running_simai_processes`：按 `root/simai-cache/` 边界匹配到的真实 AstraSimNetwork 进程数；
- `active_locks`：当前仍存在的 `simai.lock` 数，主要用于 resume 防重；
- `fct_line_total` / `fct_line_target` / `fct_progress_rate`：按 261632 行完整 FCT 估算的当前 FCT
  写出进度；
- `nonempty_end_to_end_files` 和 `merged_valid_cases`：只有大于 0 后才有真实 SimAI baseline 可比。
