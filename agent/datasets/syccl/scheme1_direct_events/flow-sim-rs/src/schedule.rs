use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

use crate::config::SimConfig;

#[derive(Debug, Clone, Deserialize)]
struct RawTranslated {
    coll_name: Option<String>,
    ngpus: usize,
    chunk_size_byte: Option<u64>,
    algorithms: Vec<RawAlgorithm>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawAlgorithm {
    final_schedule: RawFinalSchedule,
}

#[derive(Debug, Clone, Deserialize)]
struct RawFinalSchedule {
    #[serde(rename = "Time")]
    time: Option<f64>,
    #[serde(rename = "Schedule")]
    schedule: RawSchedule,
}

#[derive(Debug, Clone, Deserialize)]
struct RawSchedule {
    #[serde(rename = "Events")]
    events: Vec<RawEvent>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawEvent {
    src_chunk: String,
    sends: Vec<RawSend>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawSend {
    src_gpu: usize,
    dst_gpu: usize,
    epoch: u64,
    layer_used: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SendEvent {
    pub src_chunk: usize,
    pub chunk_index: usize,
    pub src_gpu: usize,
    pub dst_gpu: usize,
    pub epoch: u64,
    pub layer_used: Option<i64>,
    pub order: usize,
    pub sketch_source: Option<SketchTransmissionSource>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SketchTransmissionSource {
    pub transmission_index: usize,
    pub step: u64,
    pub layer: i64,
    pub group: usize,
    pub srcs: Vec<usize>,
    pub dsts: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct TranslatedSchedule {
    pub coll_name: Option<String>,
    pub ngpus: usize,
    pub chunk_size_byte: Option<u64>,
    pub sends: Vec<SendEvent>,
}

#[derive(Debug, Clone)]
pub struct AlgorithmSchedule {
    pub algorithm_index: usize,
    pub syccl_time_us: Option<f64>,
    pub schedule: TranslatedSchedule,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Flow {
    pub id: usize,
    pub channel_id: usize,
    pub src_chunk: usize,
    pub chunk_index: usize,
    pub src: usize,
    pub dst: usize,
    pub epoch: u64,
    pub order: usize,
    pub size_bytes: u64,
    pub recv_parents: Vec<usize>,
    pub send_parents: Vec<usize>,
    pub children: Vec<usize>,
    pub sketch_source: Option<SketchTransmissionSource>,
}

#[derive(Debug, Clone)]
pub struct FlowGraph {
    pub flows: Vec<Flow>,
    pub channel_count: usize,
}

pub fn to_syccl_resim_input(
    schedule: &TranslatedSchedule,
    config: &SimConfig,
) -> Result<serde_json::Value> {
    let chunk_size_byte = schedule.chunk_size_byte.unwrap_or(config.coll_bytes);
    let total_gpus = config.hosts.host_num * config.hosts.gpus_per_host;
    if schedule.ngpus != total_gpus {
        return Err(anyhow!(
            "translated schedule ngpus {} does not match config topology {}",
            schedule.ngpus,
            total_gpus
        ));
    }
    if schedule
        .coll_name
        .as_deref()
        .is_some_and(|name| name != config.coll_name)
    {
        return Err(anyhow!(
            "translated schedule collective {:?} does not match config collective {}",
            schedule.coll_name,
            config.coll_name
        ));
    }

    let chunk_keys = syccl_resim_chunk_keys(schedule, config)?;
    let mut sends_by_chunk: BTreeMap<(usize, usize), Vec<&SendEvent>> = BTreeMap::new();
    for send in &schedule.sends {
        sends_by_chunk
            .entry((send.src_chunk, send.chunk_index))
            .or_default()
            .push(send);
    }

    let init_devs: Vec<_> = chunk_keys
        .iter()
        .copied()
        .map(|(src_chunk, chunk_index)| {
            let dev = syccl_gpu_device_id(src_chunk, config)?;
            Ok(serde_json::json!([[src_chunk, chunk_index], [dev]]))
        })
        .collect::<Result<Vec<_>>>()?;
    let chunk_sizes: Vec<_> = chunk_keys
        .iter()
        .copied()
        .map(|(src_chunk, chunk_index)| {
            serde_json::json!([[src_chunk, chunk_index], chunk_size_byte])
        })
        .collect();
    let events: Vec<_> = sends_by_chunk
        .into_iter()
        .map(|((src_chunk, chunk_index), mut sends)| {
            sends.sort_by_key(|send| (send.epoch, send.order));
            let sends_json: Vec<_> = sends
                .into_iter()
                .map(|send| {
                    serde_json::json!({
                        "src_gpu": send.src_gpu,
                        "dst_gpu": send.dst_gpu,
                        "epoch": send.epoch,
                        "copy": true,
                        "reduce": false,
                        "layer_used": send.layer_used.unwrap_or(-1),
                    })
                })
                .collect();
            serde_json::json!({
                "src_chunk": format!("({}, {})", src_chunk, chunk_index),
                "sends": sends_json,
            })
        })
        .collect();

    Ok(serde_json::json!({
        "coll_name": config.coll_name,
        "ngpus": schedule.ngpus,
        "chunk_size_byte": chunk_size_byte,
        "algorithms": [{
            "chunk_size_B": chunk_size_byte,
            "final_schedule": {
                "Schedule": {
                    "init_devs": init_devs,
                    "chunk_sizes": chunk_sizes,
                    "Events": events,
                }
            }
        }]
    }))
}

#[derive(Serialize)]
struct SycclSendOutput {
    src_gpu: usize,
    dst_gpu: usize,
    epoch: u64,
    copy: bool,
    reduce: bool,
    layer_used: i64,
}

pub fn write_syccl_resim_input<W: Write>(
    schedule: &TranslatedSchedule,
    config: &SimConfig,
    mut writer: W,
) -> Result<()> {
    let chunk_size_byte = schedule.chunk_size_byte.unwrap_or(config.coll_bytes);
    let total_gpus = config.hosts.host_num * config.hosts.gpus_per_host;
    if schedule.ngpus != total_gpus {
        return Err(anyhow!(
            "translated schedule ngpus {} does not match config topology {}",
            schedule.ngpus,
            total_gpus
        ));
    }
    if schedule
        .coll_name
        .as_deref()
        .is_some_and(|name| name != config.coll_name)
    {
        return Err(anyhow!(
            "translated schedule collective {:?} does not match config collective {}",
            schedule.coll_name,
            config.coll_name
        ));
    }

    let chunk_keys = syccl_resim_chunk_keys(schedule, config)?;
    let mut sends_by_chunk: BTreeMap<(usize, usize), Vec<&SendEvent>> = BTreeMap::new();
    for send in &schedule.sends {
        sends_by_chunk
            .entry((send.src_chunk, send.chunk_index))
            .or_default()
            .push(send);
    }
    for sends in sends_by_chunk.values_mut() {
        sends.sort_by_key(|send| (send.epoch, send.order));
    }

    writer.write_all(b"{\"coll_name\":")?;
    serde_json::to_writer(&mut writer, &config.coll_name)?;
    writer.write_all(b",\"ngpus\":")?;
    serde_json::to_writer(&mut writer, &schedule.ngpus)?;
    writer.write_all(b",\"chunk_size_byte\":")?;
    serde_json::to_writer(&mut writer, &chunk_size_byte)?;
    writer.write_all(b",\"algorithms\":[{\"chunk_size_B\":")?;
    serde_json::to_writer(&mut writer, &chunk_size_byte)?;
    writer.write_all(b",\"final_schedule\":{\"Schedule\":{\"init_devs\":[")?;

    for (index, (src_chunk, chunk_index)) in chunk_keys.iter().copied().enumerate() {
        if index > 0 {
            writer.write_all(b",")?;
        }
        let chunk_key = [src_chunk, chunk_index];
        let devices = [syccl_gpu_device_id(src_chunk, config)?];
        serde_json::to_writer(&mut writer, &(&chunk_key, &devices))?;
    }

    writer.write_all(b"],\"chunk_sizes\":[")?;
    for (index, (src_chunk, chunk_index)) in chunk_keys.iter().copied().enumerate() {
        if index > 0 {
            writer.write_all(b",")?;
        }
        let chunk_key = [src_chunk, chunk_index];
        serde_json::to_writer(&mut writer, &(&chunk_key, chunk_size_byte))?;
    }

    writer.write_all(b"],\"Events\":[")?;
    for (event_index, ((src_chunk, chunk_index), sends)) in sends_by_chunk.into_iter().enumerate() {
        if event_index > 0 {
            writer.write_all(b",")?;
        }
        writer.write_all(b"{\"src_chunk\":")?;
        serde_json::to_writer(&mut writer, &format!("({}, {})", src_chunk, chunk_index))?;
        writer.write_all(b",\"sends\":[")?;
        for (send_index, send) in sends.into_iter().enumerate() {
            if send_index > 0 {
                writer.write_all(b",")?;
            }
            serde_json::to_writer(
                &mut writer,
                &SycclSendOutput {
                    src_gpu: send.src_gpu,
                    dst_gpu: send.dst_gpu,
                    epoch: send.epoch,
                    copy: true,
                    reduce: false,
                    layer_used: send.layer_used.unwrap_or(-1),
                },
            )?;
        }
        writer.write_all(b"]}")?;
    }
    writer.write_all(b"]}}}]}")?;
    Ok(())
}

static ATOMIC_WRITE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub fn write_syccl_resim_input_atomic(
    schedule: &TranslatedSchedule,
    config: &SimConfig,
    output: &Path,
) -> Result<()> {
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).with_context(|| format!("failed to create {}", parent.display()))?;
    let temporary = atomic_temporary_path(output)?;

    let result = (|| {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .with_context(|| format!("failed to create {}", temporary.display()))?;
        let mut writer = BufWriter::new(file);
        write_syccl_resim_input(schedule, config, &mut writer)?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        drop(writer);
        fs::rename(&temporary, output).with_context(|| {
            format!(
                "failed to replace {} with {}",
                output.display(),
                temporary.display()
            )
        })?;
        Ok(())
    })();

    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn atomic_temporary_path(output: &Path) -> Result<PathBuf> {
    let file_name = output
        .file_name()
        .ok_or_else(|| anyhow!("output path has no file name: {}", output.display()))?;
    let sequence = ATOMIC_WRITE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let mut temporary_name = OsString::from(".");
    temporary_name.push(file_name);
    temporary_name.push(format!(".{}.{}.tmp", std::process::id(), sequence));
    Ok(output.with_file_name(temporary_name))
}

fn syccl_resim_chunk_keys(
    schedule: &TranslatedSchedule,
    config: &SimConfig,
) -> Result<Vec<(usize, usize)>> {
    match config.coll_name.as_str() {
        "allgather" => Ok((0..schedule.ngpus).map(|gpu| (gpu, 0)).collect()),
        "alltoall" => Ok((0..schedule.ngpus)
            .flat_map(|src| (0..schedule.ngpus).map(move |dst| (src, dst)))
            .collect()),
        other => Err(anyhow!(
            "unsupported collective {other}, expected allgather or alltoall"
        )),
    }
}

fn syccl_gpu_device_id(gpu: usize, config: &SimConfig) -> Result<usize> {
    let total_gpus = config.hosts.host_num * config.hosts.gpus_per_host;
    if gpu >= total_gpus {
        return Err(anyhow!(
            "GPU {} is out of range for topology with {} GPUs",
            gpu,
            total_gpus
        ));
    }
    let switch_count = match config.hosts.host_links.as_str() {
        "nvswitch" => 1,
        "nvlink" => 0,
        other => {
            return Err(anyhow!(
                "unsupported host_links {other:?}, expected nvswitch or nvlink"
            ))
        }
    };
    let host = gpu / config.hosts.gpus_per_host;
    let local_gpu = gpu % config.hosts.gpus_per_host;
    let devices_per_host = config.hosts.gpus_per_host + config.hosts.nics_per_host + switch_count;
    Ok(host * devices_per_host + local_gpu)
}

pub fn parse_translated_schedule<R: Read>(reader: R) -> Result<TranslatedSchedule> {
    let raw = parse_raw_translated(reader)?;
    parse_algorithm_schedule(&raw, 0).map(|algorithm| algorithm.schedule)
}

pub fn parse_all_translated_schedules<R: Read>(reader: R) -> Result<Vec<AlgorithmSchedule>> {
    let raw = parse_raw_translated(reader)?;
    raw.algorithms
        .iter()
        .enumerate()
        .map(|(algorithm_index, _)| parse_algorithm_schedule(&raw, algorithm_index))
        .collect()
}

fn parse_raw_translated<R: Read>(reader: R) -> Result<RawTranslated> {
    let raw: RawTranslated = serde_json::from_reader(BufReader::new(reader))
        .context("failed to parse translated schedule JSON")?;
    if raw.algorithms.is_empty() {
        return Err(anyhow!("translated schedule has no algorithms"));
    }
    if raw
        .coll_name
        .as_deref()
        .is_some_and(|name| name != "allgather" && name != "alltoall")
    {
        return Err(anyhow!(
            "unsupported translated collective {:?}",
            raw.coll_name
        ));
    }

    Ok(raw)
}

fn parse_algorithm_schedule(
    raw: &RawTranslated,
    algorithm_index: usize,
) -> Result<AlgorithmSchedule> {
    let algorithm = raw
        .algorithms
        .get(algorithm_index)
        .ok_or_else(|| anyhow!("algorithm_index {} is out of range", algorithm_index))?;
    let sends = parse_algorithm_sends(&algorithm.final_schedule.schedule.events)
        .with_context(|| format!("failed to parse algorithm {}", algorithm_index))?;
    if sends.is_empty() {
        return Err(anyhow!(
            "translated schedule algorithm {} contains no sends",
            algorithm_index
        ));
    }
    Ok(AlgorithmSchedule {
        algorithm_index,
        syccl_time_us: algorithm.final_schedule.time,
        schedule: TranslatedSchedule {
            coll_name: raw.coll_name.clone(),
            ngpus: raw.ngpus,
            chunk_size_byte: raw.chunk_size_byte,
            sends,
        },
    })
}

fn parse_algorithm_sends(events: &[RawEvent]) -> Result<Vec<SendEvent>> {
    let mut sends = Vec::new();
    for event in events {
        let (src_chunk, chunk_index) = parse_src_chunk(&event.src_chunk)?;
        for raw_send in &event.sends {
            sends.push(SendEvent {
                src_chunk,
                chunk_index,
                src_gpu: raw_send.src_gpu,
                dst_gpu: raw_send.dst_gpu,
                epoch: raw_send.epoch,
                layer_used: raw_send.layer_used,
                order: sends.len(),
                sketch_source: None,
            });
        }
    }
    Ok(sends)
}

fn parse_src_chunk(value: &str) -> Result<(usize, usize)> {
    let trimmed = value.trim();
    let inner = trimmed
        .strip_prefix('(')
        .and_then(|v| v.strip_suffix(')'))
        .ok_or_else(|| anyhow!("invalid src_chunk {}", value))?;
    let mut parts = inner.split(',').map(str::trim);
    let root = parts
        .next()
        .ok_or_else(|| anyhow!("invalid src_chunk {}", value))?
        .parse::<usize>()
        .with_context(|| format!("invalid src_chunk root {}", value))?;
    let chunk = parts
        .next()
        .ok_or_else(|| anyhow!("invalid src_chunk {}", value))?
        .parse::<usize>()
        .with_context(|| format!("invalid src_chunk index {}", value))?;
    if parts.next().is_some() {
        return Err(anyhow!("invalid src_chunk {}", value));
    }
    Ok((root, chunk))
}

pub fn build_flows(schedule: &TranslatedSchedule, flow_size: u64) -> Result<FlowGraph> {
    let mut sends = schedule.sends.clone();
    sends.sort_by_key(|send| (send.epoch, send.order));

    let mut chunk_ids = BTreeMap::new();
    for send in &sends {
        let key = (send.src_chunk, send.chunk_index);
        let next_id = chunk_ids.len();
        chunk_ids.entry(key).or_insert(next_id);
    }

    let mut holder_to_flows: HashMap<(usize, usize, usize), Vec<usize>> = HashMap::new();
    let mut has_chunk: HashSet<(usize, usize, usize)> = HashSet::new();
    let mut flows = Vec::with_capacity(sends.len());

    for send in &sends {
        let src_holder = (send.src_chunk, send.chunk_index, send.src_gpu);
        if send.src_gpu == send.src_chunk {
            has_chunk.insert(src_holder);
        }
        if !has_chunk.contains(&src_holder) {
            return Err(anyhow!(
                "send order invalid: GPU {} sends chunk {} before holding it",
                send.src_gpu,
                send.src_chunk
            ));
        }

        // recv_parents: flows that delivered the chunk to this sender GPU.
        let mut recv_parents = holder_to_flows
            .get(&src_holder)
            .cloned()
            .unwrap_or_default();
        recv_parents.sort_unstable();
        recv_parents.dedup();

        let flow_id = flows.len();
        flows.push(Flow {
            id: flow_id,
            channel_id: *chunk_ids
                .get(&(send.src_chunk, send.chunk_index))
                .expect("chunk id allocated"),
            src_chunk: send.src_chunk,
            chunk_index: send.chunk_index,
            src: send.src_gpu,
            dst: send.dst_gpu,
            epoch: send.epoch,
            order: send.order,
            size_bytes: flow_size,
            recv_parents,
            send_parents: Vec::new(),
            children: Vec::new(),
            sketch_source: send.sketch_source.clone(),
        });

        let dst_holder = (send.src_chunk, send.chunk_index, send.dst_gpu);
        has_chunk.insert(dst_holder);
        holder_to_flows.insert(dst_holder, vec![flow_id]);
    }

    // Build child edges from the parent lists for dependency release.
    for id in 0..flows.len() {
        let mut parents = flows[id].recv_parents.clone();
        parents.extend(flows[id].send_parents.iter().copied());
        parents.sort_unstable();
        parents.dedup();
        for parent in parents {
            if parent >= flows.len() {
                return Err(anyhow!("flow {} has invalid parent {}", id, parent));
            }
            flows[parent].children.push(id);
        }
    }

    Ok(FlowGraph {
        flows,
        channel_count: chunk_ids.len(),
    })
}
