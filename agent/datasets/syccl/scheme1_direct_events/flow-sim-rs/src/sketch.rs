use std::collections::{BTreeSet, HashMap};
use std::io::{BufReader, Read};

use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;

use crate::config::SimConfig;
use crate::schedule::{SendEvent, SketchTransmissionSource, TranslatedSchedule};
const ROOT_GPU: usize = 0;
const DEFAULT_MAX_STEP: u64 = 31;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SketchGraph {
    pub ngpus: usize,
    pub src_gpu: usize,
    pub nodes: Vec<SketchNode>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SketchNode {
    pub id: usize,
    pub step: u64,
    pub layer: i64,
    pub group: usize,
    pub srcs: Vec<usize>,
    pub dsts: Vec<usize>,
    pub deps: Vec<usize>,
    pub next: Vec<usize>,
    pub transmission_index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Transmission {
    index: usize,
    step: u64,
    layer: i64,
    group: usize,
    srcs: Vec<usize>,
    dsts: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum RawSketchInput {
    Native(RawNativeSketch),
    List(Vec<serde_json::Value>),
}

#[derive(Debug, Clone, Deserialize)]
struct RawNativeSketch {
    ngpus: usize,
    src_gpu: usize,
    nodes: Vec<RawNativeNode>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawNativeNode {
    step: serde_json::Value,
    layer: serde_json::Value,
    group: serde_json::Value,
    src_dest_pair: RawSrcDestPair,
}

#[derive(Debug, Clone, Deserialize)]
struct RawSrcDestPair {
    srcs: serde_json::Value,
    dsts: serde_json::Value,
}

pub fn parse_compact_sketches<R: Read>(reader: R, config: &SimConfig) -> Result<Vec<SketchGraph>> {
    let raw: RawSketchInput = serde_json::from_reader(BufReader::new(reader))
        .context("failed to parse compact sketch JSON")?;
    normalize_sketches(raw, config)
}

pub fn parse_compact_sketch_candidates<R: Read>(
    reader: R,
    config: &SimConfig,
) -> Result<Vec<SketchGraph>> {
    let raw: RawSketchInput = serde_json::from_reader(BufReader::new(reader))
        .context("failed to parse compact sketch JSON")?;
    normalize_sketch_candidates(raw, config)
}

pub fn sketches_to_translated_schedule(
    sketches: &[SketchGraph],
    config: &SimConfig,
) -> Result<TranslatedSchedule> {
    if sketches.len() != 1 {
        bail!("flow-sim sketch evaluator expects exactly one sketch");
    }
    let graph = &sketches[0];
    let mut sends = Vec::new();
    match config.coll_name.as_str() {
        "allgather" => {
            for root in 0..graph.ngpus {
                rotate_graph_sends(graph, root, config, &mut sends);
            }
        }
        "alltoall" => {
            for src in 0..graph.ngpus {
                for dst in 0..graph.ngpus {
                    let epoch = alltoall_chunk_epoch(src, dst, config);
                    rotate_path_sends(graph, src, dst, epoch, config, &mut sends)?;
                }
            }
        }
        other => bail!("flow-sim sketch translator does not support collective {other}"),
    }
    if sends.is_empty() {
        bail!("translated schedule contains no sends");
    }
    Ok(TranslatedSchedule {
        coll_name: Some(config.coll_name.clone()),
        ngpus: graph.ngpus,
        chunk_size_byte: Some(config.coll_bytes),
        sends,
    })
}

fn normalize_sketches(raw: RawSketchInput, config: &SimConfig) -> Result<Vec<SketchGraph>> {
    let candidates = normalize_sketch_candidates(raw, config)?;
    if candidates.len() != 1 {
        bail!(
            "flow-sim sketch evaluator expects exactly one sketch, got {}",
            candidates.len()
        );
    }
    Ok(candidates)
}

fn normalize_sketch_candidates(
    raw: RawSketchInput,
    config: &SimConfig,
) -> Result<Vec<SketchGraph>> {
    let candidate_values = match raw {
        RawSketchInput::Native(native) => {
            return Ok(vec![build_graph(
                native_to_transmissions(native, config)?,
                total_gpus(config),
            )?])
        }
        RawSketchInput::List(values) => values,
    };

    if candidate_values.is_empty() {
        bail!("no sketches returned");
    }
    let candidates = if looks_like_transmission_list(&candidate_values) {
        vec![serde_json::Value::Array(candidate_values)]
    } else {
        candidate_values
    };

    candidates
        .into_iter()
        .map(|candidate| normalize_candidate(candidate, config))
        .map(|result| result.and_then(|txs| build_graph(txs, total_gpus(config))))
        .collect()
}

fn native_to_transmissions(
    native: RawNativeSketch,
    config: &SimConfig,
) -> Result<Vec<Transmission>> {
    let ngpus = total_gpus(config);
    if native.ngpus != ngpus {
        bail!("native sketch ngpus must be {ngpus}");
    }
    if native.src_gpu != ROOT_GPU {
        bail!("native sketch src_gpu must be {ROOT_GPU}");
    }
    native
        .nodes
        .into_iter()
        .enumerate()
        .map(|(index, node)| {
            normalize_transmission_fields(
                node.step,
                node.layer,
                node.group,
                node.src_dest_pair.srcs,
                node.src_dest_pair.dsts,
                index,
                config,
            )
        })
        .collect()
}

fn normalize_candidate(
    candidate: serde_json::Value,
    config: &SimConfig,
) -> Result<Vec<Transmission>> {
    if let Ok(native) = serde_json::from_value::<RawNativeSketch>(candidate.clone()) {
        return native_to_transmissions(native, config);
    }
    let raw_txs = candidate
        .as_array()
        .ok_or_else(|| anyhow!("each sketch must be a non-empty list of transmissions"))?;
    if raw_txs.is_empty() {
        bail!("each sketch must be a non-empty list of transmissions");
    }
    let mut transmissions = Vec::with_capacity(raw_txs.len());
    for (index, raw) in raw_txs.iter().enumerate() {
        transmissions.push(normalize_transmission(raw.clone(), index, config)?);
    }
    transmissions.sort_by(|a, b| {
        (a.step, a.layer, a.group, &a.srcs, &a.dsts)
            .cmp(&(b.step, b.layer, b.group, &b.srcs, &b.dsts))
    });
    Ok(transmissions)
}

fn normalize_transmission(
    raw: serde_json::Value,
    index: usize,
    config: &SimConfig,
) -> Result<Transmission> {
    match raw {
        serde_json::Value::Array(values) if values.len() == 5 => normalize_transmission_fields(
            values[0].clone(),
            values[1].clone(),
            values[2].clone(),
            values[3].clone(),
            values[4].clone(),
            index,
            config,
        ),
        serde_json::Value::Object(map) => {
            let step = map.get("step").cloned().unwrap_or(serde_json::Value::Null);
            let layer = map.get("layer").cloned().unwrap_or(serde_json::Value::Null);
            let group = map.get("group").cloned().unwrap_or(serde_json::Value::Null);
            let srcs = map
                .get("srcs")
                .or_else(|| map.get("src"))
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            let dsts = map
                .get("dsts")
                .or_else(|| map.get("dst"))
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            normalize_transmission_fields(step, layer, group, srcs, dsts, index, config)
        }
        _ => bail!("transmission {index} must be a dict or (step, layer, group, srcs, dsts)"),
    }
}

fn normalize_transmission_fields(
    step_raw: serde_json::Value,
    layer_raw: serde_json::Value,
    group_raw: serde_json::Value,
    srcs_raw: serde_json::Value,
    dsts_raw: serde_json::Value,
    index: usize,
    config: &SimConfig,
) -> Result<Transmission> {
    let step = as_u64(&step_raw, &format!("transmission {index}.step"))?;
    // if step > DEFAULT_MAX_STEP {
    //     bail!("transmission {index}.step must be in [0, {DEFAULT_MAX_STEP}]");
    // }
    let layer = as_i64(&layer_raw, &format!("transmission {index}.layer"))?;
    let group = as_usize(&group_raw, &format!("transmission {index}.group"))?;
    let srcs = as_gpu_vec(
        &srcs_raw,
        &format!("transmission {index}.srcs"),
        total_gpus(config),
    )?;
    let dsts = as_gpu_vec(
        &dsts_raw,
        &format!("transmission {index}.dsts"),
        total_gpus(config),
    )?;
    if srcs.iter().any(|src| dsts.contains(src)) {
        bail!("transmission {index} has overlapping srcs/dsts");
    }
    check_layer_group(config, layer, group, &srcs, &dsts)?;
    Ok(Transmission {
        index,
        step,
        layer,
        group,
        srcs,
        dsts,
    })
}

fn build_graph(transmissions: Vec<Transmission>, ngpus: usize) -> Result<SketchGraph> {
    let mut transmissions_by_source = Vec::new();
    for tx in transmissions {
        transmissions_by_source.extend(split_transmission_by_source(tx)?);
    }

    let mut reached_by_node: HashMap<usize, isize> = HashMap::new();
    let mut reached_step: HashMap<usize, u64> = HashMap::new();
    let mut first_delivery_step: HashMap<usize, u64> = HashMap::new();
    reached_by_node.insert(ROOT_GPU, -1);
    reached_step.insert(ROOT_GPU, 0);
    for tx in &transmissions_by_source {
        for dst in &tx.dsts {
            first_delivery_step
                .entry(*dst)
                .and_modify(|step| *step = (*step).min(tx.step))
                .or_insert(tx.step);
        }
    }

    let mut nodes = Vec::new();
    for tx in transmissions_by_source {
        let node_id = nodes.len();
        let mut deps = BTreeSet::new();
        for src in &tx.srcs {
            let dep = match reached_by_node.get(src).copied() {
                Some(dep) => dep,
                None => {
                    if let Some(delivery_step) = first_delivery_step.get(src) {
                        bail!(
                            "source GPU {src} has not been reached before step {}; GPU {src} is first reached at step {delivery_step}, so it can only be used as a source in a strictly later step",
                            tx.step
                        );
                    }
                    bail!(
                        "source GPU {src} has not been reached before step {}",
                        tx.step
                    );
                }
            };
            if dep >= 0 {
                let src_reached_step = reached_step[src];
                if src_reached_step >= tx.step {
                    bail!(
                        "source GPU {src} is reached at step {src_reached_step} but reused at step {}; dependent steps must be strictly later",
                        tx.step
                    );
                }
                deps.insert(dep as usize);
            }
        }
        for dst in &tx.dsts {
            if reached_by_node.contains_key(dst) {
                bail!("destination GPU {dst} is reached more than once");
            }
        }
        nodes.push(SketchNode {
            id: node_id,
            step: tx.step,
            layer: tx.layer,
            group: tx.group,
            srcs: tx.srcs.clone(),
            dsts: tx.dsts.clone(),
            deps: deps.into_iter().collect(),
            next: Vec::new(),
            transmission_index: tx.index,
        });
        for dst in &tx.dsts {
            reached_by_node.insert(*dst, node_id as isize);
            reached_step.insert(*dst, tx.step);
        }
    }

    let missing: Vec<_> = (0..ngpus)
        .filter(|gpu| !reached_by_node.contains_key(gpu))
        .collect();
    if !missing.is_empty() {
        bail!("sketch does not cover all GPUs; missing {missing:?}");
    }

    for node_id in 0..nodes.len() {
        let deps = nodes[node_id].deps.clone();
        for dep in deps {
            nodes[dep].next.push(node_id);
        }
    }
    for node in &mut nodes {
        node.next.sort_unstable();
    }

    Ok(SketchGraph {
        ngpus,
        src_gpu: ROOT_GPU,
        nodes,
    })
}

fn split_transmission_by_source(tx: Transmission) -> Result<Vec<Transmission>> {
    if tx.srcs.is_empty() {
        bail!("transmission {} must have at least one source", tx.index);
    }
    if tx.dsts.len() % tx.srcs.len() != 0 {
        bail!(
            "transmission {} dsts length {} must be divisible by srcs length {}",
            tx.index,
            tx.dsts.len(),
            tx.srcs.len()
        );
    }
    if tx.srcs.len() == 1 {
        return Ok(vec![tx]);
    }

    let dsts_per_src = tx.dsts.len() / tx.srcs.len();
    let Transmission {
        index,
        step,
        layer,
        group,
        srcs,
        dsts,
    } = tx;
    let mut split = Vec::with_capacity(srcs.len());
    for (src_index, src) in srcs.into_iter().enumerate() {
        let dst_start = src_index * dsts_per_src;
        let dst_end = dst_start + dsts_per_src;
        split.push(Transmission {
            index,
            step,
            layer,
            group,
            srcs: vec![src],
            dsts: dsts[dst_start..dst_end].to_vec(),
        });
    }
    Ok(split)
}

fn rotate_graph_sends(
    graph: &SketchGraph,
    root: usize,
    config: &SimConfig,
    sends: &mut Vec<SendEvent>,
) {
    let mut nodes = graph.nodes.clone();
    nodes.sort_by_key(|node| (node.step, node.id));
    for node in nodes {
        for src in &node.srcs {
            for dst in &node.dsts {
                let order = sends.len();
                sends.push(SendEvent {
                    src_chunk: root,
                    chunk_index: 0,
                    src_gpu: rotate_gpu(*src, root, config),
                    dst_gpu: rotate_gpu(*dst, root, config),
                    epoch: node.step,
                    layer_used: Some(node.layer),
                    order,
                    sketch_source: Some(node_source(&node)),
                });
            }
        }
    }
}

fn rotate_path_sends(
    graph: &SketchGraph,
    src_root: usize,
    dst_gpu: usize,
    epoch: u64,
    config: &SimConfig,
    sends: &mut Vec<SendEvent>,
) -> Result<()> {
    if src_root == dst_gpu {
        return Ok(());
    }
    let target = unrotate_gpu(dst_gpu, src_root, config);
    let mut parent_by_dst = HashMap::new();
    let mut node_by_dst = HashMap::new();
    for node in &graph.nodes {
        if let Some(parent) = node.srcs.first() {
            for dst in &node.dsts {
                parent_by_dst.insert(*dst, *parent);
                node_by_dst.insert(*dst, node);
            }
        }
    }

    let mut path_nodes = Vec::new();
    let mut current = target;
    let mut visited = BTreeSet::new();
    while current != ROOT_GPU {
        if !visited.insert(current) {
            bail!("cycle while tracing alltoall path to GPU {dst_gpu}");
        }
        let parent = parent_by_dst
            .get(&current)
            .copied()
            .ok_or_else(|| anyhow!("sketch has no path from root 0 to GPU {target}"))?;
        let node = node_by_dst
            .get(&current)
            .copied()
            .ok_or_else(|| anyhow!("sketch has no node for GPU {target}"))?;
        path_nodes.push((parent, current, node));
        current = parent;
    }
    path_nodes.reverse();

    for (parent, child, node) in path_nodes {
        let order = sends.len();
        sends.push(SendEvent {
            src_chunk: src_root,
            chunk_index: dst_gpu,
            src_gpu: rotate_gpu(parent, src_root, config),
            dst_gpu: rotate_gpu(child, src_root, config),
            epoch,
            layer_used: Some(node.layer),
            order,
            sketch_source: Some(node_source(node)),
        });
    }
    Ok(())
}

fn alltoall_chunk_epoch(src_root: usize, dst_gpu: usize, config: &SimConfig) -> u64 {
    let host_count = config.hosts.host_num;
    let gpus_per_host = config.hosts.gpus_per_host;
    let src_host = src_root / gpus_per_host;
    let src_local = src_root % gpus_per_host;
    let dst_host = dst_gpu / gpus_per_host;
    let dst_local = dst_gpu % gpus_per_host;
    let remote_rounds = gpus_per_host * host_count.saturating_sub(1);

    if src_host == dst_host {
        return (remote_rounds + dst_local) as u64;
    }

    let host_offset = (dst_host + host_count - src_host) % host_count;
    (src_local * (host_count - 1) + host_offset - 1) as u64
}

fn node_source(node: &SketchNode) -> SketchTransmissionSource {
    SketchTransmissionSource {
        transmission_index: node.transmission_index,
        step: node.step,
        layer: node.layer,
        group: node.group,
        srcs: node.srcs.clone(),
        dsts: node.dsts.clone(),
    }
}

fn check_layer_group(
    config: &SimConfig,
    layer: i64,
    group: usize,
    srcs: &[usize],
    dsts: &[usize],
) -> Result<()> {
    // 检查如果是clos topo或者multirail的话dsl中不允许出现layer1
    // if config.topology == TopologyKind::Clos && layer == 1 {
    //     bail!("layer 1 is not allowed in clos topology");
    // }
    // if config.topology == TopologyKind::Multirail && layer == 1 {
    //     bail!("layer 1 is not allowed in multirail topology");
    // }
    let layer_id = usize::try_from(layer).map_err(|_| anyhow!("unsupported layer {layer}"))?;
    let groups = config.layer_groups.get(&layer_id).ok_or_else(|| {
        let mut allowed: Vec<_> = config.layer_groups.keys().copied().collect();
        allowed.sort_unstable();
        anyhow!("unsupported layer {layer}; allowed layers are {allowed:?}")
    })?;
    let members = groups.get(&group).ok_or_else(|| {
        let mut allowed: Vec<_> = groups.keys().copied().collect();
        allowed.sort_unstable();
        anyhow!("unsupported group {group} for layer {layer}; allowed groups are {allowed:?}")
    })?;
    let outside: Vec<_> = srcs
        .iter()
        .chain(dsts.iter())
        .copied()
        .filter(|gpu| !members.contains(gpu))
        .collect();
    if !outside.is_empty() {
        bail!("layer {layer} group {group} cannot connect GPUs {outside:?}");
    }
    Ok(())
}

fn as_u64(value: &serde_json::Value, field: &str) -> Result<u64> {
    value
        .as_u64()
        .ok_or_else(|| anyhow!("{field} must be a non-negative integer"))
}

fn as_i64(value: &serde_json::Value, field: &str) -> Result<i64> {
    value
        .as_i64()
        .ok_or_else(|| anyhow!("{field} must be an integer"))
}

fn as_usize(value: &serde_json::Value, field: &str) -> Result<usize> {
    let raw = as_u64(value, field)?;
    usize::try_from(raw).map_err(|_| anyhow!("{field} is too large"))
}

fn as_gpu_vec(value: &serde_json::Value, field: &str, ngpus: usize) -> Result<Vec<usize>> {
    let raw_values = match value {
        serde_json::Value::Number(_) => vec![value.clone()],
        serde_json::Value::Array(values) => values.clone(),
        _ => bail!("{field} must be a GPU id or a list of GPU ids"),
    };
    if raw_values.is_empty() {
        bail!("{field} must not be empty");
    }
    let mut seen = BTreeSet::new();
    let mut gpus = Vec::with_capacity(raw_values.len());
    for raw in raw_values {
        if raw.is_array() {
            bail!("{field} must be a GPU id or a flat list of GPU ids");
        }
        let gpu = as_usize(&raw, field)?;
        if gpu >= ngpus {
            bail!("{field} contains out-of-range GPU {gpu}");
        }
        if !seen.insert(gpu) {
            bail!("{field} contains duplicate GPU {gpu}");
        }
        gpus.push(gpu);
    }
    Ok(gpus)
}

fn looks_like_transmission_list(values: &[serde_json::Value]) -> bool {
    values.iter().all(looks_like_transmission)
}

fn looks_like_transmission(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Array(items) if items.len() == 5 => {
            items[0].is_u64() && items[1].is_i64() && items[2].is_u64()
        }
        serde_json::Value::Object(map) => {
            map.contains_key("step")
                && map.contains_key("layer")
                && map.contains_key("group")
                && ((map.contains_key("srcs") && map.contains_key("dsts"))
                    || (map.contains_key("src") && map.contains_key("dst")))
        }
        _ => false,
    }
}

fn rotate_gpu(gpu: usize, root: usize, config: &SimConfig) -> usize {
    let host_gpu_num = config.hosts.gpus_per_host;
    let src_host = gpu / host_gpu_num;
    let src_local = gpu % host_gpu_num;
    let root_host = root / host_gpu_num;
    let root_local = root % host_gpu_num;
    let host = (src_host + root_host) % config.hosts.host_num;
    let local = (src_local + root_local) % host_gpu_num;
    host * host_gpu_num + local
}

fn unrotate_gpu(gpu: usize, root: usize, config: &SimConfig) -> usize {
    let host_gpu_num = config.hosts.gpus_per_host;
    let host = gpu / host_gpu_num;
    let local = gpu % host_gpu_num;
    let root_host = root / host_gpu_num;
    let root_local = root % host_gpu_num;
    let src_host = (host + config.hosts.host_num - root_host) % config.hosts.host_num;
    let src_local = (local + host_gpu_num - root_local) % host_gpu_num;
    src_host * host_gpu_num + src_local
}

fn total_gpus(config: &SimConfig) -> usize {
    config.hosts.host_num * config.hosts.gpus_per_host
}
