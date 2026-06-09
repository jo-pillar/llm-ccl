# SyCCL Sketch Search

This crate extracts the SyCCL sketch exploration stage into a standalone Rust
component. It generates SyCCL-compatible single-root sketch JSON arrays, matching
the shape written by SyCCL's `save_sketch` path and accepted by `use_sketch_input`
or `resim --sketch`.

It intentionally stops before SyCCL's later stages:

- expanding a single-root sketch into full allgather/alltoall sketches
- combining multiple expanded sketches
- running MIP/performance-model solve and filtering

Those stages still depend on SyCCL's C++ `PerfModel` and solver stack.

## Build And Test

```bash
cargo test
```

## CLI

Generate at most 8 sketches from a SyCCL config:

```bash
cargo run -- \
  --config ../syccl/config/single-host-8gpu.json \
  --output /tmp/syccl-sketches.json \
  --limit 8 \
  --pretty
```

The output is a JSON array:

```json
[
  {
    "ngpus": 4,
    "src_gpu": 0,
    "nodes": [
      {
        "id": 0,
        "step": 0,
        "layer": 1,
        "group": 0,
        "src_dest_pair": {
          "srcs": [0],
          "dsts": [1, 2, 3]
        },
        "deps": [],
        "next": []
      }
    ]
  }
]
```

Use the generated file from SyCCL by setting:

```json
"sketch": {
  "customize_sketch": false,
  "use_sketch_input": true,
  "save_sketch": false,
  "sketch_path": "/tmp/syccl-sketches.json"
}
```

## Supported Config Subset

The Rust component reads the SyCCL config sections needed by sketch search:

- `coll`: `allgather` and `alltoall` are searched as root-GPU-0 single-root
  sketches, matching SyCCL's isomorphic sub-collective path.
- `hosts`: `nvswitch` and `nvlink` host links.
- `topo`: `local`, `host`, `nic`, and `switch` layers with `pod` or
  first-level `multirail` switch grouping.
- `prune`: the layer, group-size, source/destination, and step pruning flags
  used by the extracted search.

Unsupported topology families fail early with an explicit error instead of
silently producing incompatible sketches.
