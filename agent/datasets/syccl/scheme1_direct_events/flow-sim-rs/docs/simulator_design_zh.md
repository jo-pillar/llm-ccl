# Rust 流级仿真器设计文档

## 设计目标

本仿真器用于快速评估 SYCCL 展开后的 `candidate-translated.json`，并尽量对齐
SimAI/ns-3 包级仿真的排序结果。它的定位是“SimAI 语义对齐的流级事件仿真器”：

- 保留流级仿真速度，不逐包运行完整 ns-3。
- 输入与 SimAI `sycclFlowModel` 一致，便于同一组 candidate 同时跑 Rust 和 SimAI。
- 不修改 config 中的物理带宽、延迟等参数。
- 通过更贴近 SimAI 执行语义的机制建模提升趋势一致性，而不是靠调参拟合。

## 与传统流级仿真器的区别

传统流级仿真器通常把一次 send 抽象成一个连续流，只按路径瓶颈、理想共享或简单 FIFO
估算完成时间。这种做法速度快，但对 SYCCL/SimAI 的 collective schedule 会丢掉几个关键因素：

- 不区分 `SendComplete` 和 `QpComplete`，容易把发送侧可继续调度时间和接收侧持有 chunk 的时间混在一起。
- 不理解 SYCCL schedule 中的 chunk 持有依赖、source epoch 依赖和 channel 结构。
- 往往只做链路级流量叠加，不建模 RDMA ACK、QP 调度、ACK 优先级和 endpoint 事件延迟。
- 对 multirail 拓扑容易简化成一个全局瓶颈，不能表达每个 GPU 对应 pseudo-NIC 和 rail 的路径差异。
- 对小消息和大量 fanout 的场景，容易低估事件轮次、依赖传播和队列细节对排序的影响。

本仿真器保留“以 flow 为主”的速度优势，但在会影响 SimAI 排序的地方加入事件语义：

- 按 SimAI `SycclFlowModel` 的方式构建 flow dependency graph。
- 把 parent 依赖拆成接收侧完成依赖和发送侧 source-epoch 依赖。
- 每条物理链路维护队列和 busy time，而不是只计算全局瓶颈。
- host/NIC 链路使用 QP round-robin 数据调度，并支持 ACK 优先。
- same-host 大消息使用 packet-head cut-through，避免把尾包完成时间误当成下游链路启动时间。
- cross-host 路径仍按尾包传播，避免网络端过度 cut-through。
- 保留 schedule 中的 epoch 作为发送排序键和 source-epoch 依赖键，但不把 epoch 标签再折算成额外全局延迟。

因此它不是传统意义上的“单流公式估算器”，而是一个按 SimAI schedule 语义驱动的轻量事件仿真器。

## 输入模型

### Config

读取 SYCCL config JSON 中的：

- `coll.name` / `coll.byte`
- `hosts.host_num`
- `hosts.host_gpu_num`
- `hosts.host_nic_num`
- multirail switch layer 的 `switch_num`
- `link_spec` 中的 `memcpy`、host link、NIC link、net link 带宽和延迟

带宽、延迟转换：

```text
bw_bps = bw_mbpus * 8 * 1024 * 1024 * 1_000_000
latency_ns = lat_us * 1000
```

### Translated Schedule

读取：

```text
algorithms[*].final_schedule.Schedule.Events[*].sends[*]
```

每个 send 至少需要：

- `src_gpu`
- `dst_gpu`
- `epoch`
- 所在 event 的 `src_chunk`

`simulate` 默认按输入顺序仿真所有 `algorithms`；指定 `--algorithm-index` 时只仿真对应下标。
`chunk_size_byte` 优先作为 flow size；缺失时使用 config 中的 `coll.byte`。

## 依赖建模

仿真器会先按 `(epoch, 原始顺序)` 对 send 排序，然后构建 flow graph。

每个 `(src_chunk, chunk_index)` 对应一个 channel。每个 send 生成一个 flow，并根据两类规则建立 parent：

- chunk 持有依赖：某 GPU 只有收到 chunk 后才能继续发送该 chunk。
- source epoch 依赖：同一 `(src_chunk, chunk_index, src_gpu)` 的后续 epoch 发送，需要等待上一 epoch 的发送侧完成。

这与 SimAI `SycclFlowModel` 的依赖构建方式一致，但 Rust 仿真器在执行时进一步区分 parent 的释放时间：

- chunk 持有依赖在接收端数据到达后释放，对应接收侧 `QpComplete`。
- source epoch 依赖在首跳发送序列化完成后释放，对应发送侧 `PacketSentFinished`。

## 拓扑与路由

当前支持 multirail：

```text
same-host:
src_gpu -> host_nvswitch -> dst_gpu

cross-host:
src_gpu -> src_pseudo_nic -> rail_switch -> dst_pseudo_nic -> dst_gpu
```

约束：

```text
host_nic_num == host_gpu_num == switch_num
```

这与 SimAI 的 SYCCL multirail topofile 生成方式一致：每个 local GPU id 绑定一个 pseudo-NIC 和一条 rail。

## 链路队列与事件语义

每条链路维护：

- `busy_until_ns`
- data queue
- ACK queue
- transmissions
- busy time
- queue wait time

数据包大小使用 flow size 加 RDMA data header；ACK 使用固定 wire size。ACK 在链路队列中优先于普通数据，用于近似 RDMA completion 对后续依赖的影响。

host/NIC 链路采用 QP round-robin 数据服务。same-host 大消息在 host 链路上可以按 packet head 释放下游 hop，使下游链路尽早开始，而不是等待整个 flow 尾包。

## 机制性修正项

所有修正项都来自 schedule 结构和 config，不改物理参数。

### Source Cross-Epoch Fanout Pressure

定义：

```text
max_source_cross_epoch_fanout =
  max count(cross-host sends grouped by (epoch, src_gpu))
```

当 same-host staging flow 数量不少于 cross-host flow 数量时，额外加入：

```text
source_cross_epoch_pressure_ns =
  max_source_cross_epoch_fanout *
  serialization_time(flow_size + RDMA header, min(nic_bw, net_bw))
```

这个项用于刻画 host 内 staging 仍占主导时，同一源端在一个 epoch 内向网络注入多个跨 host flow
产生的串行压力。

当一个 schedule 中 cross-host flow 数量多于 same-host flow 数量时，这个全局压力不再额外叠加。
原因是这类 schedule 的主要压力已经在显式 GPU->NIC、NIC->rail、rail->NIC 链路队列中体现；
继续按源端 fanout 叠加会重复计算。

### Epoch Barrier

定义：

```text
epoch_barrier_ns = 0
```

SYCCL final schedule 中的 `epoch` 字段在 perf model 里用于发送优先级和队列排序，不是需要额外乘
网络往返延迟的仿真轮次数。依赖传播已经由 flow graph 的 recv/source parent 和链路事件处理，
因此不再根据 `max_epoch` 叠加全局 barrier；`max_epoch` 仍保留在输出中作为诊断信息。

## 输出指标

单 case 输出包含：

- `time_us`：最终仿真完成时间。
- `finish_time_ns`：最终完成时间，单位 ns。
- `flow_count`：flow 数量。
- `channel_count`：channel 数量。
- `total_queue_wait_ns`：所有链路排队等待总和。
- `max_source_cross_epoch_fanout`：同一 `(epoch, src_gpu)` 的最大跨 host fanout。
- `source_cross_epoch_pressure_ns`：源端 fanout 压力。
- `max_epoch`：最大 epoch 编号。
- `epoch_barrier_ns`：保留字段；当前为 0，不再由 raw epoch 标签推导额外延迟。
- `links`：每条活跃链路的 transmissions、busy time、queue wait 和最后完成时间。

## 对齐判定

对已有 SimAI baseline 的 case，compare 报告使用：

```text
absolute_ratio = max(rust_time, simai_time) / min(rust_time, simai_time)
```

默认要求：

- 有效 case 中至少 95% 满足 `absolute_ratio <= 5`。
- 去重 schedule 后做 pairwise 趋势比较；SimAI 两方案时间差不超过较小值 5% 的 pair 不参与判断。
- 剩余 pair 的趋势一致率至少 95%。

## 当前限制

- 只支持 AllGather。
- 只支持 multirail，且要求一 GPU 对一 NIC/rail。
- 不模拟完整 ns-3 packet stack、拥塞控制、随机丢包或交换机内部 buffer。
- 当前模型针对 SYCCL translated schedule 和 SimAI `sycclFlowModel` 语义设计，不保证直接适用于其他 collective runtime。
