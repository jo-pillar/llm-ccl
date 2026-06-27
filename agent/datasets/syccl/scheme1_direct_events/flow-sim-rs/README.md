# SYCCL Rust Flow Simulator

这是一个面向 SYCCL `candidate-translated.json` 的 Rust 流级仿真器。它的目标不是替代
SimAI/ns-3 的包级网络仿真，而是在保留流级仿真速度优势的同时，复刻 SimAI 对
SYCCL schedule 的关键执行语义，使结果能够用于快速筛选候选方案。

当前已验证版本：

```text
commit: ca332f0
tag: checkpoint-roundtrip-crossdominant-4k-16m-pass-20260519
```

## 支持范围

- collective：`allgather`
- 拓扑：SYCCL multirail
- 输入 config：SYCCL config JSON
- 输入 schedule：默认遍历 `algorithms[*].final_schedule.Schedule.Events[*].sends[*]`
- 约束：`host_nic_num == host_gpu_num == switch_num`

当前主要测试集是 512 GPU multirail AllGather：

- 4K config：`/home/antl/mntdisk/syccl-llm-scheme1-direct-events/configs/multirail-512gpu/allgather/multirail-512gpu-ag-4k.json`
- 16M config：`/home/antl/mntdisk/syccl-llm-scheme1-direct-events/configs/multirail-512gpu/allgather/multirail-512gpu-ag-16m.json`

## 构建

```bash
cd /home/antl/wzd/Flow-Simulator/flow-sim-rs
cargo build --release
```

生成的可执行文件：

```text
target/release/flow-sim-rs
```

## 单个 case 仿真

```bash
target/release/flow-sim-rs simulate \
  --config /path/to/syccl-config.json \
  --translated /path/to/candidate-translated.json \
  --output /home/antl/mntdisk/new-simulator/one-case.json
```

默认会按 SYCCL JSON 中 `algorithms` 的原始顺序仿真所有 solution。只跑一条时可加：

```bash
target/release/flow-sim-rs simulate \
  --config /path/to/syccl-config.json \
  --translated /path/to/candidate-translated.json \
  --output /home/antl/mntdisk/new-simulator/one-case.json \
  --algorithm-index 0
```

输出 JSON 顶层为 `solutions` 数组，每项包含：

- `solution_index`
- `syccl_time_us`
- `rust_time_us`
- `flow_count` / `channel_count`
- `result`：完整 `SimulationResult`，包含 `finish_time_ns`、queue wait、epoch 诊断和链路统计

## 批量仿真

`batch` 会在 `input-dir` 的一层或两层子目录中查找 `candidate-translated.json`。

```bash
OUT=/home/antl/mntdisk/new-simulator/4k-rust-roundtrip-crossdominant
mkdir -p "$OUT/reports" "$OUT/rust-runs"

/usr/bin/time -f 'elapsed=%e' target/release/flow-sim-rs batch \
  --config /home/antl/mntdisk/syccl-llm-scheme1-direct-events/configs/multirail-512gpu/allgather/multirail-512gpu-ag-4k.json \
  --input-dir /home/antl/mntdisk/syccl-llm-scheme1-direct-events/simpletes/multirail-512gpu/allgather/4k/full/eval_artifacts/scheme1_direct_events \
  --output-dir "$OUT/rust-runs" \
  --manifest "$OUT/benchmarks.json" \
  --summary "$OUT/reports/rust-summary.csv"
```

## 与 SimAI baseline 对比

`compare` 需要 manifest 中每个 case 带有 `simai_time_us`，或者带有可读取的
`simai_end_to_end_csv`。

```bash
target/release/flow-sim-rs compare \
  --manifest /home/antl/mntdisk/new-simulator/4k-rust-roundtrip-crossdominant/benchmarks-with-simai.json \
  --output /home/antl/mntdisk/new-simulator/4k-rust-roundtrip-crossdominant/reports/baseline-compare.json
```

默认判定标准：

- 绝对误差：`max(rust_time, simai_time) / min(rust_time, simai_time) <= 5`
- 趋势一致：去重 schedule 后做 pairwise 比较；SimAI 两方案时间差不超过较小值 5% 的 pair 不参与趋势判断
- 通过要求：绝对误差通过率至少 95%，趋势一致率至少 95%

## SimAI baseline 辅助脚本

仓库提供了可续跑的 SimAI baseline 辅助脚本：

```bash
python3 scripts/run_simai_baselines.py \
  --root /home/antl/mntdisk/new-simulator/16m \
  --config /path/to/syccl-config.json \
  --selection representative \
  --jobs 1
```
eval_20260513-223547_939383_9b0zc34v
eval_20260514-010234_947733_h1vcy059
也可以监控多个 baseline 目录并合并 manifest：

```bash
python3 scripts/monitor_simai_baselines.py \
  --roots /home/antl/mntdisk/new-simulator/16m /home/antl/mntdisk/new-simulator/16m-tail \
  --output-manifest /home/antl/mntdisk/new-simulator/16m/reports/merged-baseline-manifest.json \
  --compare-output /home/antl/mntdisk/new-simulator/16m/reports/merged-baseline-compare.json
```

## 文档

- [仿真器设计文档](docs/simulator_design_zh.md)
- [速度与 SimAI 对齐统计](docs/performance_alignment_zh.md)
- [16M 验证记录](docs/16m_validation_zh.md)
