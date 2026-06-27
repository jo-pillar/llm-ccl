# Rust 流级仿真器设计与使用说明

## 目标

本工具用于快速评估 SYCCL 展开后的 `candidate-translated.json`。首版只支持
AllGather + multirail 拓扑，输入和 SimAI 的 `sycclFlowModel` 保持一致：

- 配置文件使用 SYCCL config JSON。
- 调度文件默认按输入顺序遍历 `algorithms[*].final_schedule.Schedule.Events[*].sends[*]`。
- 输出结果、manifest、报告建议放在 `/home/antl/mntdisk/new-simulator`。

## 建模机制

仿真流程分三步：

1. 解析 translated schedule，按 `(epoch, 原始顺序)` 排序 send。
2. 复刻 SimAI `SycclFlowModel` 的依赖构建方式：按 `(src_chunk, chunk_index)` 建 channel，
   按 chunk 持有关系和 epoch 建 parent/child flow。
3. 将 parent 依赖拆成两类事件：
   - chunk 持有依赖使用接收端 `QpComplete` 时刻释放；
   - 同一 `(src_chunk, chunk_index, src_gpu)` 的后续 epoch 发送依赖使用发送端
     `PacketSentFinished` 时刻释放。
4. 根据 config 构建 multirail 路由，并在每条物理链路上维护队列。

数据链路传输时间为：

```text
latency_ns + size_bits / bandwidth_bps
```

其中带宽和延迟直接来自 config：

```text
bw_bps = bw_mbpus * 8 * 1024 * 1024 * 1_000_000
latency_ns = lat_us * 1000
```

当前实现没有修改 config 中的带宽、延迟等物理参数，也没有隐藏的拟合倍率。额外加入的
机制性开销均由调度结构和 config 推导：

```text
source_cross_epoch_pressure_ns =
  max_source_cross_epoch_fanout * flow_serialization_time_on_cross_bottleneck
  仅在 same-host staging flow 数量不少于 cross-host flow 数量时启用

epoch_barrier_ns = 0
```

其中 `max_source_cross_epoch_fanout` 统计同一 `(epoch, src_gpu)` 内跨 host 发送的最大数量，
跨 host 瓶颈带宽取 `min(link_nic.bw_mbpus, netlink.bw_mbpus)`。这个项用于刻画 host 内
staging 仍占主要流量时，同一源端 rail/NIC 在一个 epoch 内对多个跨 host flow 的串行注入压力。
如果一个 schedule 的跨 host flow 数量已经超过 host 内 flow 数量，则 NIC/rail 队列压力已经由
显式链路事件建模，此时保留 `max_source_cross_epoch_fanout` 统计但不再额外叠加
`source_cross_epoch_pressure_ns`。`epoch_barrier_ns` 保留为输出字段，但当前不再由 raw `epoch`
标签推导额外延迟；SYCCL final schedule 的 `epoch` 用作发送排序/优先级标签，依赖传播已经由
flow graph 和链路事件建模。

接收端完成事件使用数据正向路径的到达时间；ACK 会在反向路径上独立序列化，用于链路占用和
ACK 优先级建模，但不再延迟接收完成。发送侧后续依赖仍在首跳链路序列化完成后释放，对应
SimAI 的 `SendComplete` / `PacketSentFinished`。

同 host 的大消息数据在 host RR 首跳上按 packet head 释放下游链路，避免把整个 flow 的尾包
完成时间误当成下游链路可开始时间。跨 host GPU->NIC 和网络路径仍按尾包完成传播，避免对
网络端过度 cut-through。

## 路由范围

首版支持的拓扑约束：

- `coll.name == "allgather"`
- `switch_topo == "multirail"`
- `host_nic_num == host_gpu_num == switch_num`
- 当前测试配置为 64 host、每 host 8 GPU、8 NIC、8 rail switch。

同 host 通信使用：

```text
src_gpu -> host_nvswitch -> dst_gpu
```

跨 host 通信使用同 local GPU index 对应的 rail：

```text
src_gpu -> src_pseudo_nic -> rail_switch -> dst_pseudo_nic -> dst_gpu
```

## 常用命令

构建与测试：

```bash
cd /home/antl/wzd/Flow-Simulator/flow-sim-rs
cargo test
cargo build --release
```

单个候选：

```bash
cargo run --release -- simulate \
  --config /home/antl/mntdisk/syccl-llm-scheme1-direct-events/configs/multirail-512gpu/allgather/multirail-512gpu-ag-4k.json \
  --translated /path/to/candidate-translated.json \
  --output /home/antl/mntdisk/new-simulator/rust-runs/<case>.json
```

生成 manifest：

```bash
cargo run --release -- manifest \
  --config /home/antl/mntdisk/syccl-llm-scheme1-direct-events/configs/multirail-512gpu/allgather/multirail-512gpu-ag-4k.json \
  --input-dir /home/antl/mntdisk/syccl-llm-scheme1-direct-events/simpletes/multirail-512gpu/allgather/4k/full/eval_artifacts/scheme1_direct_events \
  --output-dir /home/antl/mntdisk/new-simulator/rust-runs \
  --manifest /home/antl/mntdisk/new-simulator/benchmarks.json
```

批量运行 107 个候选：

```bash
cargo run --release -- batch \
  --config /home/antl/mntdisk/syccl-llm-scheme1-direct-events/configs/multirail-512gpu/allgather/multirail-512gpu-ag-4k.json \
  --input-dir /home/antl/mntdisk/syccl-llm-scheme1-direct-events/simpletes/multirail-512gpu/allgather/4k/full/eval_artifacts/scheme1_direct_events \
  --output-dir /home/antl/mntdisk/new-simulator/rust-runs \
  --manifest /home/antl/mntdisk/new-simulator/benchmarks.json \
  --summary /home/antl/mntdisk/new-simulator/reports/rust-summary.csv
```

如果 manifest 中补充了 SimAI baseline，可以运行对比：

```bash
cargo run --release -- compare \
  --manifest /home/antl/mntdisk/new-simulator/benchmarks.json \
  --output /home/antl/mntdisk/new-simulator/reports/compare.json
```

## 输出说明

单个结果 JSON 包含：

- `time_us`：总完成时间。
- `flow_count`：flow 数量。
- `channel_count`：channel 数量。
- `total_queue_wait_ns`：所有链路排队等待总和。
- `max_source_cross_epoch_fanout`：同一 `(epoch, src_gpu)` 内跨 host 发送 fanout 的最大值。
- `source_cross_epoch_pressure_ns`：由上述 fanout 推导出的源端串行压力。
- `max_epoch`：调度中的最大 epoch 编号。
- `epoch_barrier_ns`：保留字段；当前为 0，不再由 raw epoch 标签推导额外延迟。
- `critical_flow_id`：完成最晚的 flow。
- `links`：有流量的链路统计，包括 transmission 数量、busy time 和 queue wait。

汇总 CSV 按 `time_us` 升序排列，可用于快速观察候选趋势。

## SimAI baseline

`cache-simai` 命令会为 manifest 中的候选准备 SimAI 输入，并可选调用 NS3。它会把命令记录在
`/home/antl/mntdisk/new-simulator/simai-cache/commands.log`，便于后台复跑或人工接管。

建议流程是先运行 Rust batch 得到快速排序，再选择少量候选用 SimAI 完整验证并把结果缓存下来。

当前工程提供了可续跑的 SimAI baseline 脚本。代表集验证：

```bash
python3 scripts/run_simai_baselines.py --selection representative
```

全量验证可以并行运行，已完成 case 会从
`/home/antl/mntdisk/new-simulator/simai-cache` 读取缓存：

```bash
TMPDIR=/home/antl/mntdisk/new-simulator/tmp \
python3 scripts/run_simai_baselines.py \
  --selection all \
  --jobs 24 \
  --truncate-legacy-mocknccl-logs
```

当前脚本默认选择稳定的 `ns3.36.1-AstraSimNetwork-debug`。16M 采样中验证过
`ns3.36.1-AstraSimNetwork-optimized` 会在进入有效 FCT 输出前 SIGSEGV，因此不能作为
SimAI baseline。脚本默认设置：

```text
AS_SEND_LAT=3
AS_NVLS_ENABLE=1
AS_LOG_LEVEL=3
AS_LOG_DIR=/home/antl/mntdisk/new-simulator/mocknccl-logs
SIMAI_LOG_PATH=/home/antl/mntdisk/new-simulator/mocknccl-logs
```

其中 `SIMAI_LOG_PATH` 是当前 SimAI `MockNcclLog.h` 实际读取的日志目录变量；
`AS_LOG_LEVEL=3` 只保留 ERROR 级日志，避免 `/etc/astra-sim` 或根分区被大量日志写满。
每个 case 目录下会写 `simai.lock`，用于避免重复启动正在运行的 case。遇到 `locked`
状态时可以等待已有进程结束后再 resume。

脚本会持续更新：

```text
/home/antl/mntdisk/new-simulator/simai-cache/baseline-manifest.json
```

随后运行：

```bash
cargo run --release -- compare \
  --manifest /home/antl/mntdisk/new-simulator/simai-cache/baseline-manifest.json \
  --output /home/antl/mntdisk/new-simulator/simai-cache/baseline-compare.json
```

对比报告会给出每个 case 的绝对倍率误差，以及所有有 SimAI baseline 的 pairwise 趋势一致性。
验收字段含义如下：

- `abs_ratio_case_total`：有 SimAI baseline 的 case 数。
- `abs_ratio_passed` / `abs_ratio_failed`：满足或不满足
  `max(rust_time, simai_time) / min(rust_time, simai_time) <= abs_ratio_limit`
  的 case 数，默认 `abs_ratio_limit=5.0`。
- `abs_ratio_requirement`：绝对倍率误差通过率门槛，当前为 `0.95`。
- `passes_abs_ratio_requirement`：当至少有一个有效 baseline 且
  `abs_ratio_pass_rate >= abs_ratio_requirement` 时为 `true`。
- `tolerant_unique_pairwise_consistency_rate`：对去重后的 schedule 做 pairwise 趋势比较，
  忽略 SimAI 两方案时间差不超过较小值 5% 的近似并列 pair。
- `passes_trend_requirement`：当上述趋势一致率不低于 `trend_consistency_requirement=0.95`
  时为 `true`。
- `passes_alignment_requirement`：绝对误差和趋势两个要求都可判定且都通过时为 `true`。

## 当前验证结果

4K 全量 107 个有效 case，当前模型输出目录：

```text
/home/antl/mntdisk/new-simulator/4k-rust-roundtrip-crossdominant
```

对比结果：

```text
abs_ratio_passed: 107 / 107
tolerant_unique_pairwise_consistency_rate: 515 / 523 = 0.9847036328871893
passes_alignment_requirement: true
runtime: 382.29 s
```

16M 有 SimAI baseline 的 32 个有效 case，当前模型输出目录：

```text
/home/antl/mntdisk/new-simulator/16m-rust-roundtrip-crossdominant-valid
```

对比结果：

```text
abs_ratio_passed: 32 / 32
tolerant_unique_pairwise_consistency_rate: 152 / 154 = 0.987012987012987
passes_alignment_requirement: true
runtime: 119.48 s
```
