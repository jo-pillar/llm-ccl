# 运行速度与 SimAI 对齐统计

本文记录当前 Rust 流级仿真器版本在已有 4K/16M SimAI baseline 上的速度和误差统计。

当前版本：

```text
commit: ca332f0
tag: checkpoint-roundtrip-crossdominant-4k-16m-pass-20260519
```

## 统计口径

- Rust 运行速度：`/usr/bin/time -f 'elapsed=%e'` 统计 batch wall time。
- 绝对误差倍率：`max(rust_time, simai_time) / min(rust_time, simai_time)`。
- 相对误差：`abs(rust_time - simai_time) / simai_time`。
- 趋势一致：去重 schedule 后做 pairwise 比较。
- 趋势过滤：如果 SimAI 两方案时间差不超过较小 SimAI 时间的 5%，该 pair 不参与趋势判断。
- 16M 只统计已有有效 SimAI baseline 的 32 个 case。

## 运行速度

| 数据集 | Rust 输出目录 | case 数 | batch wall time | 平均每 case |
|---|---|---:|---:|---:|
| 4K AllGather | `/home/antl/mntdisk/new-simulator/4k-rust-roundtrip-crossdominant` | 107 | 382.29 s | 3.57 s |
| 16M AllGather | `/home/antl/mntdisk/new-simulator/16m-rust-roundtrip-crossdominant-valid` | 32 | 119.48 s | 3.73 s |

说明：这里统计的是 Rust 仿真器实际运行 wall time，不是仿真输出中的 collective 完成时间。

## 与 SimAI 的绝对误差

| 数据集 | 有效 SimAI case | 通过 `<=5x` | 通过率 | abs ratio min | p50 | p90 | p95 | max | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K | 107 | 107 | 100.00% | 1.025 | 1.254 | 1.400 | 1.400 | 1.406 | 1.257 |
| 16M | 32 | 32 | 100.00% | 1.055 | 1.300 | 1.643 | 1.848 | 2.095 | 1.346 |

两组有效 baseline 都满足绝对误差 5 倍以内要求。

## 相对误差

| 数据集 | min | p50 | p90 | p95 | max | mean |
|---|---:|---:|---:|---:|---:|---:|
| 4K | 2.55% | 25.41% | 40.02% | 40.02% | 40.56% | 25.68% |
| 16M | 5.46% | 30.02% | 64.32% | 84.79% | 109.50% | 34.61% |

16M 的最大相对误差仍远小于 5x 绝对倍率限制；当前主要优化目标是趋势一致，而不是逐 case 做线性拟合。

## 趋势一致性

| 数据集 | 参与判断 pair | 一致 pair | 不一致 pair | 一致率 | 是否通过 |
|---|---:|---:|---:|---:|---|
| 4K | 523 | 515 | 8 | 98.47% | 是 |
| 16M | 154 | 152 | 2 | 98.70% | 是 |

当前两组数据都超过 95% 趋势一致率要求。

## Rust 与 SimAI 输出时间分布

| 数据集 | Rust time min | Rust p50 | Rust p90 | Rust max | SimAI min | SimAI p50 | SimAI p90 | SimAI max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K | 85.704 us | 106.730 us | 134.762 us | 387.929 us | 69.867 us | 84.015 us | 111.323 us | 378.295 us |
| 16M | 52,970.056 us | 80,244.750 us | 115,111.811 us | 375,743.019 us | 49,474.379 us | 61,866.478 us | 70,052.531 us | 179,348.605 us |

## 报告路径

4K：

```text
/home/antl/mntdisk/new-simulator/4k-rust-roundtrip-crossdominant/reports/baseline-compare.json
/home/antl/mntdisk/new-simulator/4k-rust-roundtrip-crossdominant/reports/rust-summary.csv
```

16M：

```text
/home/antl/mntdisk/new-simulator/16m-rust-roundtrip-crossdominant-valid/reports/baseline-compare.json
/home/antl/mntdisk/new-simulator/16m-rust-roundtrip-crossdominant-valid/reports/rust-summary.csv
```

## 复现实验命令

4K Rust batch：

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

16M Rust batch：

```bash
OUT=/home/antl/mntdisk/new-simulator/16m-rust-roundtrip-crossdominant-valid
mkdir -p "$OUT/reports" "$OUT/rust-runs"

/usr/bin/time -f 'elapsed=%e' target/release/flow-sim-rs batch \
  --config /home/antl/mntdisk/syccl-llm-scheme1-direct-events/configs/multirail-512gpu/allgather/multirail-512gpu-ag-16m.json \
  --input-dir /home/antl/mntdisk/new-simulator/16m-rust-hop0-rr-ackfix-valid/inputs \
  --output-dir "$OUT/rust-runs" \
  --manifest "$OUT/benchmarks.json" \
  --summary "$OUT/reports/rust-summary.csv"
```

