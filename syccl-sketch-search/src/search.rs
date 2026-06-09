use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::Instant,
};

use crate::{
    config::{PruneConfig, SearchConfig, SycclConfig},
    graph::{SketchGraph, SketchNode, SrcDestPair},
    topology::{LayerRelation, Topology},
};

#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub limit: Option<usize>,
    pub parallelism: usize,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            limit: None,
            parallelism: 1,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SearchStats {
    pub layer_combinations: usize,
    pub group_configurations: usize,
    pub schedules: usize,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct SearchTimings {
    pub layer_ms: f64,
    pub group_ms: f64,
    pub step_ms: f64,
    pub graph_ms: f64,
    pub total_ms: f64,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub sketches: Vec<SketchGraph>,
    pub stats: SearchStats,
    pub timings: SearchTimings,
}

#[derive(Debug, Clone)]
struct Schedule {
    id: usize,
    sender: usize,
    gpu_num: usize,
    per_layer_size: BTreeMap<usize, LayerConfig>,
    tmp_acc_dest: usize,
    topo: TopoSubset,
    comm_by_gpus: BTreeMap<usize, SourceRecord>,
    comm_by_layer: BTreeMap<usize, LayerComm>,
    comm_by_step: BTreeMap<usize, StepComm>,
    layer_prunes: BTreeMap<usize, LayerPrune>,
    cur_step_sources: BTreeSet<usize>,
    nxt_step_sources: BTreeSet<usize>,
}

#[derive(Debug, Clone, Default)]
struct LayerConfig {
    config: Vec<LayerSizeConfig>,
}

#[derive(Debug, Clone)]
struct LayerSizeConfig {
    group_id: Option<usize>,
    used_group_num: usize,
    comm_source_per_group: usize,
    non_comm_source_per_group: usize,
    dest_per_group: usize,
}

#[derive(Debug, Clone, Default)]
struct TopoSubset {
    layer_group_src_dest: BTreeMap<usize, BTreeMap<usize, BTreeMap<usize, BTreeSet<usize>>>>,
    gpu_layer_group: BTreeMap<usize, BTreeMap<usize, usize>>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SourceRecord {
    step: isize,
    layer: isize,
    group: isize,
    sources: BTreeSet<usize>,
}

#[derive(Debug, Clone, Default)]
struct LayerComm {
    comm_groups: BTreeMap<usize, GroupComm>,
    config_id_to_groups: BTreeMap<usize, BTreeSet<usize>>,
    group_to_config_id: BTreeMap<usize, usize>,
}

#[derive(Debug, Clone, Default)]
struct StepComm {
    layer_group_comm: BTreeMap<usize, BTreeMap<usize, GroupComm>>,
}

#[derive(Debug, Clone, Default)]
struct GroupComm {
    src_dest_pairs: Vec<(BTreeSet<usize>, BTreeSet<usize>)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum DestPrune {
    Undefined,
    Balance,
    Imbalance,
}

impl Default for DestPrune {
    fn default() -> Self {
        Self::Undefined
    }
}

#[derive(Debug, Clone, Default)]
struct LayerPrune {
    step_split: isize,
    num_group_per_step: Vec<usize>,
    num_source_per_group_per_step: Vec<usize>,
    dst_prune: DestPrune,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct PartitionKey {
    comm_paths: Vec<Vec<usize>>,
    layer_gpus: BTreeMap<usize, usize>,
}

#[derive(Debug, Clone)]
struct LayerCand {
    layer: usize,
    group_sources: BTreeMap<usize, BTreeSet<usize>>,
    group_dests: BTreeMap<usize, BTreeSet<usize>>,
}

#[derive(Debug, Clone, Copy)]
struct LayerSize {
    layer: usize,
    source_per_group: usize,
    group_num: usize,
}

#[derive(Debug, Clone, Default)]
struct DestCombPrune {
    layer_group_finished_gpu_num: BTreeMap<usize, Vec<usize>>,
    balance_ratio: f64,
}

static COMPARE_STEP_PRUNE_WARNING_EMITTED: AtomicBool = AtomicBool::new(false);

pub fn search_sketches(
    config: &SycclConfig,
    options: SearchOptions,
) -> anyhow::Result<(Vec<SketchGraph>, SearchStats)> {
    let result = search_sketches_profiled(config, options)?;
    Ok((result.sketches, result.stats))
}

pub fn search_sketches_profiled(
    config: &SycclConfig,
    options: SearchOptions,
) -> anyhow::Result<SearchResult> {
    let total_start = Instant::now();
    let search_config = SearchConfig::try_from(config)?;
    let topology = Topology::from_syccl_config(config)?;
    if search_config.ngpus != topology.num_gpus() {
        anyhow::bail!(
            "config GPU count {} does not match topology GPU count {}",
            search_config.ngpus,
            topology.num_gpus()
        );
    }

    let initial = initialize_schedule(&search_config);
    let layer_start = Instant::now();
    let layer_combinations = search_sched_layers(&search_config.prune, &topology, initial)?;
    let layer_ms = elapsed_ms(layer_start);
    let layer_combination_count = layer_combinations.len();
    let group_start = Instant::now();
    let group_configurations = search_sched_groups(&search_config.prune, layer_combinations)?;
    let group_ms = elapsed_ms(group_start);
    let group_configuration_count = group_configurations.len();
    let step_start = Instant::now();
    let schedules = search_sched_steps(
        &search_config.prune,
        group_configurations,
        options.limit,
        options.parallelism,
    );
    let step_ms = elapsed_ms(step_start);
    let schedule_count = schedules.len();
    let graph_start = Instant::now();
    let sketches = schedules
        .iter()
        .map(schedule_to_graph)
        .collect::<anyhow::Result<Vec<_>>>()?;
    let graph_ms = elapsed_ms(graph_start);

    let stats = SearchStats {
        layer_combinations: layer_combination_count,
        group_configurations: group_configuration_count,
        schedules: schedule_count,
    };

    let timings = SearchTimings {
        layer_ms,
        group_ms,
        step_ms,
        graph_ms,
        total_ms: elapsed_ms(total_start),
    };

    Ok(SearchResult {
        sketches,
        stats,
        timings,
    })
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn initialize_schedule(config: &SearchConfig) -> Schedule {
    let mut cur_step_sources = BTreeSet::new();
    cur_step_sources.insert(config.sender);
    let mut comm_by_gpus = BTreeMap::new();
    comm_by_gpus.insert(
        config.sender,
        SourceRecord {
            step: -1,
            layer: -1,
            group: -1,
            sources: BTreeSet::new(),
        },
    );
    Schedule {
        id: 0,
        sender: config.sender,
        gpu_num: config.ngpus,
        per_layer_size: BTreeMap::new(),
        tmp_acc_dest: 1,
        topo: TopoSubset::default(),
        comm_by_gpus,
        comm_by_layer: BTreeMap::new(),
        comm_by_step: BTreeMap::new(),
        layer_prunes: BTreeMap::new(),
        cur_step_sources,
        nxt_step_sources: BTreeSet::new(),
    }
}

fn search_sched_layers(
    prune: &PruneConfig,
    topo: &Topology,
    input: Schedule,
) -> anyhow::Result<Vec<Schedule>> {
    let mut outputs = Vec::new();
    let mut queue = VecDeque::from([input]);
    for layer in 0..topo.layers().len() {
        let n = queue.len();
        for _ in 0..n {
            let schedule = queue.pop_front().expect("queue length");
            queue.push_back(schedule.clone());
            let mut with_layer = schedule;
            with_layer
                .per_layer_size
                .insert(layer, LayerConfig::default());
            if output_sched_layers(prune, &with_layer, topo) {
                finish_sched_layer(&mut with_layer, topo)?;
                outputs.push(with_layer);
            } else if !prune_sched_layers(prune, &with_layer) {
                queue.push_back(with_layer);
            }
        }
    }
    Ok(outputs)
}

fn output_sched_layers(prune: &PruneConfig, schedule: &Schedule, topo: &Topology) -> bool {
    if prune.max_comb_layer_num != -1
        && schedule.per_layer_size.len() > prune.max_comb_layer_num as usize
    {
        return false;
    }
    if !prune
        .must_contain_layers
        .iter()
        .all(|layer| schedule.per_layer_size.contains_key(layer))
    {
        return false;
    }

    for &layer in schedule.per_layer_size.keys() {
        if topo.layer(layer).groups.len() == 1 {
            return true;
        }
        for &other_layer in schedule.per_layer_size.keys() {
            if other_layer != layer
                && topo.layer(layer).relation_to(other_layer) == LayerRelation::Cross
            {
                return true;
            }
        }
    }
    false
}

fn prune_sched_layers(prune: &PruneConfig, schedule: &Schedule) -> bool {
    if prune
        .ignored_layers
        .iter()
        .any(|layer| schedule.per_layer_size.contains_key(layer))
    {
        return true;
    }
    prune.max_comb_layer_num != -1
        && schedule.per_layer_size.len() == prune.max_comb_layer_num as usize
}

fn finish_sched_layer(schedule: &mut Schedule, topo: &Topology) -> anyhow::Result<()> {
    for &layer in schedule.per_layer_size.keys() {
        let layer_info = topo.layer(layer);
        for group in &layer_info.groups {
            for &src in &group.connected_gpus {
                schedule
                    .topo
                    .gpu_layer_group
                    .entry(src)
                    .or_default()
                    .insert(layer, group.id);
                schedule
                    .topo
                    .layer_group_src_dest
                    .entry(layer)
                    .or_default()
                    .entry(group.id)
                    .or_default()
                    .insert(src, group.connected_gpus.clone());
            }
        }
    }

    let layers = schedule.per_layer_size.keys().copied().collect::<Vec<_>>();
    for &layer in &layers {
        for &lower_layer in &layers {
            if lower_layer >= layer {
                continue;
            }
            let lower_groups = schedule
                .topo
                .layer_group_src_dest
                .get(&lower_layer)
                .cloned()
                .unwrap_or_default();
            for (_lower_group, srcs_to_dsts) in lower_groups {
                for (src, lower_group_gpus) in srcs_to_dsts {
                    let Some(group) = schedule
                        .topo
                        .gpu_layer_group
                        .get(&src)
                        .and_then(|layers| layers.get(&layer))
                        .copied()
                    else {
                        anyhow::bail!("GPU {src} is missing layer {layer} group mapping");
                    };
                    if let Some(dsts) = schedule
                        .topo
                        .layer_group_src_dest
                        .get_mut(&layer)
                        .and_then(|groups| groups.get_mut(&group))
                        .and_then(|sources| sources.get_mut(&src))
                    {
                        for dst in lower_group_gpus {
                            dsts.remove(&dst);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn search_sched_groups(
    policy: &PruneConfig,
    inputs: Vec<Schedule>,
) -> anyhow::Result<Vec<Schedule>> {
    let mut outputs = Vec::new();
    for input in inputs {
        search_sched_groups_func(policy, input, &mut outputs)?;
    }
    Ok(outputs)
}

fn search_sched_groups_func(
    policy: &PruneConfig,
    input: Schedule,
    outputs: &mut Vec<Schedule>,
) -> anyhow::Result<()> {
    let mut queue = VecDeque::from([input.clone()]);
    let (mut remained_dest_max, mut remained_dest_min) = get_remained_dest(&input);
    for &layer in input.per_layer_size.keys() {
        let group_num = layer_num_groups(&input.topo, layer);
        let group_gpu_num = layer_num_gpus_per_group(&input.topo, layer);
        remained_dest_max -= group_num * (group_gpu_num - 1);
        remained_dest_min -= 1;
        let n = queue.len();
        for _ in 0..n {
            let schedule = queue.pop_front().expect("queue length");
            let cur_layer_min_dest = usize_saturating_max(
                1,
                schedule
                    .gpu_num
                    .saturating_sub(remained_dest_max + schedule.tmp_acc_dest),
            );
            let cur_layer_max_dest = std::cmp::min(
                group_num * (group_gpu_num - 1),
                schedule
                    .gpu_num
                    .saturating_sub(remained_dest_min + schedule.tmp_acc_dest),
            );
            for used_group in 1..=std::cmp::min(group_num, cur_layer_max_dest) {
                if policy.all_groups_used && used_group != group_num {
                    continue;
                }
                let dest_min = std::cmp::max(cur_layer_min_dest, used_group);
                let dest_max = std::cmp::min(cur_layer_max_dest, used_group * (group_gpu_num - 1));
                for dest_size in dest_min..=dest_max {
                    let source_size = group_gpu_num * used_group - dest_size;
                    let mut candidates =
                        update_size(policy, &schedule, layer, used_group, source_size, dest_size)?;
                    if policy.allow_source_group_special {
                        update_source_size(
                            &schedule,
                            layer,
                            used_group,
                            source_size,
                            dest_size,
                            &mut candidates,
                        );
                    }
                    for candidate in candidates {
                        if sched_finish(&candidate) {
                            if output_sched_size(policy, &candidate) {
                                outputs.push(candidate);
                            }
                        } else {
                            queue.push_back(candidate);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn update_size(
    policy: &PruneConfig,
    input: &Schedule,
    layer: usize,
    group_num: usize,
    source_num: usize,
    dest_num: usize,
) -> anyhow::Result<Vec<Schedule>> {
    if group_num == 0 || source_num == 0 || (source_num + dest_num) % group_num != 0 {
        return Ok(Vec::new());
    }
    let num_gpu_per_group = (source_num + dest_num) / group_num;
    let mut result = Vec::new();
    let start = std::cmp::max(1, dest_num / source_num);
    let end = std::cmp::min(dest_num, layer_num_links_per_gpu(&input.topo, layer));
    for d_per_s in start..=end {
        if dest_num % d_per_s != 0 {
            continue;
        }
        let total_comm_s_num = dest_num / d_per_s;
        if total_comm_s_num % group_num != 0 && !policy.allow_unequal_source_per_group {
            continue;
        }
        if source_num % group_num != 0 && (!policy.allow_unequal_source_per_group || d_per_s != 1) {
            continue;
        }
        if d_per_s * source_num != dest_num
            && (!policy.allow_unequal_dest_per_source || (d_per_s != 1 && d_per_s != dest_num))
        {
            continue;
        }

        let mut comm_source_per_group = total_comm_s_num / group_num;
        let mut tmp_num_groups = if total_comm_s_num % group_num == 0 {
            group_num
        } else {
            group_num * (comm_source_per_group + 1) - total_comm_s_num
        };
        let mut acc_num_groups = 0;
        let mut schedule = input.clone();
        while acc_num_groups < group_num {
            let tmp_dest_per_group = comm_source_per_group * d_per_s;
            let tmp_non_comm_source_per_group = num_gpu_per_group as isize
                - comm_source_per_group as isize
                - tmp_dest_per_group as isize;
            if comm_source_per_group != 0 && tmp_num_groups != 0 {
                schedule
                    .per_layer_size
                    .get_mut(&layer)
                    .ok_or_else(|| anyhow::anyhow!("layer {layer} missing"))?
                    .config
                    .push(LayerSizeConfig {
                        group_id: None,
                        used_group_num: tmp_num_groups,
                        comm_source_per_group,
                        non_comm_source_per_group: tmp_non_comm_source_per_group.max(0) as usize,
                        dest_per_group: tmp_dest_per_group,
                    });
                schedule.tmp_acc_dest += tmp_dest_per_group * tmp_num_groups;
            }
            if tmp_non_comm_source_per_group < 0 {
                schedule
                    .per_layer_size
                    .get_mut(&layer)
                    .expect("layer exists")
                    .config
                    .clear();
                break;
            }
            comm_source_per_group += 1;
            acc_num_groups += tmp_num_groups;
            tmp_num_groups = group_num.saturating_sub(tmp_num_groups);
        }
        if schedule
            .per_layer_size
            .get(&layer)
            .is_some_and(|config| !config.config.is_empty())
        {
            result.push(schedule);
        }
    }
    Ok(result)
}

fn update_source_size(
    input: &Schedule,
    layer: usize,
    group_num: usize,
    source_num: usize,
    dest_num: usize,
    result: &mut Vec<Schedule>,
) {
    if group_num != input.topo.layer_group_src_dest[&layer].len() || group_num == 1 {
        return;
    }
    let num_gpu_per_group = (source_num + dest_num) / group_num;
    let Some(source_group) = gpu_group_in_layer(&input.topo, input.sender, layer) else {
        return;
    };
    for g0_source_num in 1..std::cmp::min(source_num, num_gpu_per_group) {
        let g0_dest_num = num_gpu_per_group - g0_source_num;
        let g0_non_comm_source = num_gpu_per_group - g0_source_num - g0_dest_num;
        if g0_dest_num % g0_source_num != 0 {
            continue;
        }
        let other_group_num = group_num - 1;
        let Some(remaining_sources) = source_num.checked_sub(g0_source_num) else {
            continue;
        };
        let Some(remaining_dests) = dest_num.checked_sub(g0_dest_num) else {
            continue;
        };
        if remaining_sources % other_group_num != 0 || remaining_dests % other_group_num != 0 {
            continue;
        }
        let other_source_num = remaining_sources / other_group_num;
        let other_dest_num = remaining_dests / other_group_num;
        if other_source_num == 0 || other_dest_num == 0 || other_dest_num % other_source_num != 0 {
            continue;
        }
        let other_non_comm_source =
            num_gpu_per_group as isize - other_source_num as isize - other_dest_num as isize;
        if other_non_comm_source < 0 {
            continue;
        }
        if other_dest_num / other_source_num == g0_dest_num / g0_source_num
            && other_source_num == g0_source_num
        {
            continue;
        }
        let mut schedule = input.clone();
        let config = schedule
            .per_layer_size
            .get_mut(&layer)
            .expect("layer exists");
        config.config.push(LayerSizeConfig {
            group_id: Some(source_group),
            used_group_num: 1,
            comm_source_per_group: g0_source_num,
            non_comm_source_per_group: g0_non_comm_source,
            dest_per_group: g0_dest_num,
        });
        config.config.push(LayerSizeConfig {
            group_id: None,
            used_group_num: other_group_num,
            comm_source_per_group: other_source_num,
            non_comm_source_per_group: other_non_comm_source as usize,
            dest_per_group: other_dest_num,
        });
        schedule.tmp_acc_dest += g0_dest_num + other_dest_num * other_group_num;
        result.push(schedule);
    }
}

fn output_sched_size(policy: &PruneConfig, schedule: &Schedule) -> bool {
    let mut has_layer_without_zero_dest_source = !policy.at_least_one_layer_no_source_with_0_dest;
    for config in schedule.per_layer_size.values() {
        for group_config in &config.config {
            if group_config.non_comm_source_per_group == 0 {
                has_layer_without_zero_dest_source = true;
            }
        }
    }
    if !has_layer_without_zero_dest_source {
        return false;
    }

    for &layer in schedule.per_layer_size.keys() {
        if let Some(uplimit) = policy.layer_comm_uplimit.get(&layer) {
            if *uplimit < layer_comm(&schedule.per_layer_size[&layer]) {
                return false;
            }
        }
        if let Some(less_layers) = policy.layer_comm_ordering.get(&layer) {
            for other_layer in less_layers {
                if schedule.per_layer_size.contains_key(other_layer)
                    && layer_comm(&schedule.per_layer_size[&layer])
                        < layer_comm(&schedule.per_layer_size[other_layer])
                {
                    return false;
                }
            }
        }
    }

    for (&layer, config) in &schedule.per_layer_size {
        let group_num = layer_num_groups(&schedule.topo, layer);
        let is_all_group_used = group_num == total_num_groups_for_layer(schedule, layer);
        let one_group_config_used = config.config.len() == 1;
        let source_group = gpu_group_in_layer(&schedule.topo, schedule.sender, layer);
        let source_group_special = config
            .config
            .iter()
            .any(|c| source_group.is_some_and(|g| c.group_id == Some(g)));
        if is_all_group_used && (one_group_config_used || source_group_special) {
            return true;
        }
    }

    for (&layer, config) in &schedule.per_layer_size {
        let group_num = layer_num_groups(&schedule.topo, layer);
        let is_all_group_used = group_num == total_num_groups_for_layer(schedule, layer);
        if is_all_group_used && config.config.len() == 1 {
            return true;
        }
    }
    false
}

fn sched_finish(schedule: &Schedule) -> bool {
    schedule
        .per_layer_size
        .values()
        .all(|config| !config.config.is_empty())
}

fn search_sched_steps(
    policy: &PruneConfig,
    inputs: Vec<Schedule>,
    limit: Option<usize>,
    parallelism: usize,
) -> Vec<Schedule> {
    if limit.is_none() && parallelism > 1 && inputs.len() > 1 {
        return search_sched_steps_parallel(policy, inputs, parallelism);
    }

    let mut outputs = Vec::new();
    for input in inputs {
        search_sched_steps_func(policy, input, &mut outputs, limit);
        if limit.is_some_and(|limit| outputs.len() >= limit) {
            outputs.truncate(limit.unwrap());
            break;
        }
    }
    outputs
}

fn search_sched_steps_parallel(
    policy: &PruneConfig,
    inputs: Vec<Schedule>,
    parallelism: usize,
) -> Vec<Schedule> {
    let worker_count = parallelism.min(inputs.len());
    let mut buckets = (0..worker_count).map(|_| Vec::new()).collect::<Vec<_>>();
    for (idx, input) in inputs.into_iter().enumerate() {
        buckets[idx % worker_count].push(input);
    }

    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for bucket in buckets {
            handles.push(scope.spawn(move || {
                let mut local_outputs = Vec::new();
                for input in bucket {
                    search_sched_steps_func(policy, input, &mut local_outputs, None);
                }
                local_outputs
            }));
        }

        let mut outputs = Vec::new();
        for handle in handles {
            outputs.extend(handle.join().expect("sketch search worker panicked"));
        }
        outputs
    })
}

fn search_sched_steps_func(
    policy: &PruneConfig,
    input: Schedule,
    outputs: &mut Vec<Schedule>,
    limit: Option<usize>,
) {
    let mut queue = VecDeque::from([input.clone()]);
    let total_group = total_num_groups(&input);
    let max_step = if policy.limit_num_steps != -1 {
        policy.limit_num_steps as usize
    } else if policy.limit_num_steps_ratio != -1.0 {
        (total_group as f64 * policy.limit_num_steps_ratio).floor() as usize
    } else {
        input.gpu_num - 1
    };
    for step in 0..=max_step {
        if queue.is_empty() {
            break;
        }
        let size_to_search = queue.len();
        for _ in 0..size_to_search {
            let schedule = queue.pop_front().expect("queue length");
            if output_sched(&schedule) {
                outputs.push(schedule);
                if limit.is_some_and(|limit| outputs.len() >= limit) {
                    return;
                }
            } else if step < max_step {
                search_step(policy, schedule, &mut queue, step);
            }
        }
    }
}

fn search_step(
    policy: &PruneConfig,
    input: Schedule,
    next_step_queue: &mut VecDeque<Schedule>,
    step: usize,
) {
    let mut queue = VecDeque::from([input]);
    while let Some(mut schedule) = queue.pop_front() {
        let mut start_src_dests = Vec::new();
        if search_start_sources(policy, &schedule, step, &mut start_src_dests) {
            for src_dests in start_src_dests {
                if !src_dests.group_sources.is_empty() {
                    search_layer(
                        policy,
                        src_dests.layer,
                        src_dests.group_sources,
                        src_dests.group_dests,
                        schedule.clone(),
                        step,
                        &mut queue,
                    );
                }
            }
        }

        if step_has_comm(&schedule, step) {
            schedule.cur_step_sources = schedule.nxt_step_sources.clone();
            schedule.nxt_step_sources.clear();
            if !schedule.cur_step_sources.is_empty() || output_sched(&schedule) {
                next_step_queue.push_back(schedule);
            }
        }
    }
}

fn search_start_sources(
    policy: &PruneConfig,
    schedule: &Schedule,
    step: usize,
    start_sources: &mut Vec<LayerCand>,
) -> bool {
    for &layer in schedule.per_layer_size.keys() {
        let layer_already_searched = step_has_comm(schedule, step)
            && schedule.comm_by_step[&step]
                .layer_group_comm
                .keys()
                .any(|other_layer| *other_layer >= layer);
        if layer_already_searched {
            continue;
        }
        if policy.same_step_diff {
            if let Some(layer_prune) = schedule.layer_prunes.get(&layer) {
                if layer_prune.step_split != -1
                    && layer_prune.num_group_per_step.len() + layer_prune.step_split as usize
                        != step + 1
                {
                    continue;
                }
            }
        }

        let mut candidate = LayerCand {
            layer,
            group_sources: BTreeMap::new(),
            group_dests: BTreeMap::new(),
        };
        for &src in &schedule.cur_step_sources {
            let Some(group) = gpu_group_in_layer(&schedule.topo, src, layer) else {
                continue;
            };
            if policy.start_source_diff_layer
                && schedule
                    .comm_by_gpus
                    .get(&src)
                    .is_some_and(|record| record.layer == layer as isize)
            {
                continue;
            }
            if group_has_comm(schedule, layer, group) {
                let group_not_reached_gpus = group_num_not_reached_gpus(schedule, layer, group);
                let group_cur_dest_gpus = group_num_dest(schedule, layer, group);
                let group_conf_id = schedule.comm_by_layer[&layer].group_to_config_id[&group];
                let group_obj_dest =
                    schedule.per_layer_size[&layer].config[group_conf_id].dest_per_group;
                if group_not_reached_gpus < group_obj_dest - group_cur_dest_gpus {
                    return false;
                }
                if group_obj_dest == group_cur_dest_gpus {
                    continue;
                }
            } else if layer_has_comm(schedule, layer)
                && schedule.comm_by_layer[&layer].comm_groups.len()
                    == total_num_groups_for_layer(schedule, layer)
            {
                continue;
            }

            if let Some(dsts) = schedule
                .topo
                .layer_group_src_dest
                .get(&layer)
                .and_then(|groups| groups.get(&group))
                .and_then(|sources| sources.get(&src))
            {
                for &gpu in dsts {
                    if !gpu_has_comm(schedule, gpu) {
                        candidate.group_dests.entry(group).or_default().insert(gpu);
                    }
                }
            }
            if candidate.group_dests.contains_key(&group) {
                candidate
                    .group_sources
                    .entry(group)
                    .or_default()
                    .insert(src);
            }
        }
        if !candidate.group_sources.is_empty() {
            start_sources.push(candidate);
        }
    }
    true
}

fn search_layer(
    policy: &PruneConfig,
    layer: usize,
    sources: BTreeMap<usize, BTreeSet<usize>>,
    dests: BTreeMap<usize, BTreeSet<usize>>,
    schedule: Schedule,
    step: usize,
    result: &mut VecDeque<Schedule>,
) {
    let group_partitions = partition_groups(&schedule, layer, &sources, &dests);
    let layer_sizes = search_layer_sizes(&schedule, layer, &group_partitions);
    let pruned_layer_sizes = prune_layer_sizes(policy, &schedule, &layer_sizes, step);
    for (src_prune, layer_size) in pruned_layer_sizes {
        let layer_groups = search_layer_groups(&group_partitions, layer_size);
        for groups in layer_groups {
            if is_valid_groups(&schedule, layer, &groups) {
                let params = LayerFuncParams {
                    layer,
                    groups,
                    group_sources: &sources,
                    step,
                    prune_info: src_prune.clone(),
                    source_per_group: layer_size.source_per_group,
                };
                search_layer_func(policy, schedule.clone(), params, result);
            }
        }
    }
}

struct LayerFuncParams<'a> {
    layer: usize,
    groups: BTreeSet<usize>,
    group_sources: &'a BTreeMap<usize, BTreeSet<usize>>,
    step: usize,
    prune_info: LayerPrune,
    source_per_group: usize,
}

fn search_layer_func(
    policy: &PruneConfig,
    input: Schedule,
    params: LayerFuncParams<'_>,
    result: &mut VecDeque<Schedule>,
) {
    let mut queue = VecDeque::from([input]);
    for group in params.groups {
        let n = queue.len();
        for _ in 0..n {
            let schedule = queue.pop_front().expect("queue length");
            let source_combs = search_sources(
                policy,
                &schedule,
                params.group_sources,
                group,
                params.source_per_group,
            );
            for source_comb in source_combs {
                if prune_sources(&schedule, params.step, params.layer, group, &source_comb) {
                    continue;
                }
                let dest_params = DestSearchParams {
                    source_comb: &source_comb,
                    step: params.step,
                    layer: params.layer,
                    group,
                    prune_info: params.prune_info.clone(),
                };
                search_dests(policy, &schedule, dest_params, &mut queue);
                if policy.max_dest_combs != -1 && n > policy.max_dest_combs as usize {
                    break;
                }
            }
        }
    }
    while let Some(mut schedule) = queue.pop_front() {
        schedule.id += 1;
        result.push_back(schedule);
    }
}

struct DestSearchParams<'a> {
    source_comb: &'a BTreeSet<usize>,
    step: usize,
    layer: usize,
    group: usize,
    prune_info: LayerPrune,
}

fn search_dests(
    policy: &PruneConfig,
    schedule: &Schedule,
    params: DestSearchParams<'_>,
    result: &mut VecDeque<Schedule>,
) {
    let mut dsts = BTreeSet::new();
    search_destinations(
        schedule,
        params.layer,
        params.group,
        params.source_comb,
        &mut dsts,
    );
    let dest_combs = if policy.partition_dests {
        let dest_partitions = partition_gpus(schedule, &dsts);
        let dest_num =
            layer_d_per_s(schedule, params.layer, params.group) * params.source_comb.len();
        gpu_combinations(dest_num, &dsts, &dest_partitions)
    } else {
        vec![dsts]
    };
    let pruned_dest_combs = prune_dest_combs(
        policy,
        schedule,
        params.step,
        params.layer,
        params.group,
        params.source_comb,
        dest_combs,
    );
    for (dest_prune, combs) in pruned_dest_combs {
        for dest_comb in combs {
            let mut new_schedule = copy_scheduling(
                schedule,
                params.source_comb,
                &dest_comb,
                params.step,
                params.layer,
                params.group,
                params.prune_info.clone(),
            );
            new_schedule
                .layer_prunes
                .entry(params.layer)
                .or_default()
                .dst_prune = dest_prune;
            result.push_back(new_schedule);
        }
    }
}

fn search_destinations(
    schedule: &Schedule,
    layer: usize,
    group: usize,
    source_comb: &BTreeSet<usize>,
    dsts: &mut BTreeSet<usize>,
) {
    for src in source_comb {
        if let Some(src_dsts) = schedule
            .topo
            .layer_group_src_dest
            .get(&layer)
            .and_then(|groups| groups.get(&group))
            .and_then(|sources| sources.get(src))
        {
            for &dst in src_dsts {
                if !gpu_has_comm(schedule, dst) {
                    dsts.insert(dst);
                }
            }
        }
    }
}

fn copy_scheduling(
    input: &Schedule,
    sources: &BTreeSet<usize>,
    dests: &BTreeSet<usize>,
    step: usize,
    layer: usize,
    group: usize,
    prune: LayerPrune,
) -> Schedule {
    let mut schedule = input.clone();
    update_scheduling(&mut schedule, sources, dests, step, layer, group, prune);
    schedule
}

fn update_scheduling(
    schedule: &mut Schedule,
    sources: &BTreeSet<usize>,
    dests: &BTreeSet<usize>,
    step: usize,
    layer: usize,
    group: usize,
    prune: LayerPrune,
) {
    for &dest in dests {
        schedule.comm_by_gpus.insert(
            dest,
            SourceRecord {
                step: step as isize,
                layer: layer as isize,
                group: group as isize,
                sources: sources.clone(),
            },
        );
        schedule.nxt_step_sources.insert(dest);
    }
    schedule.layer_prunes.insert(layer, prune);
    if !group_has_comm(schedule, layer, group) {
        if let Some(config_id) = new_group_config_id(schedule, layer, group) {
            schedule
                .comm_by_layer
                .entry(layer)
                .or_default()
                .config_id_to_groups
                .entry(config_id)
                .or_default()
                .insert(group);
            schedule
                .comm_by_layer
                .entry(layer)
                .or_default()
                .group_to_config_id
                .insert(group, config_id);
        }
    }
    schedule
        .comm_by_layer
        .entry(layer)
        .or_default()
        .comm_groups
        .entry(group)
        .or_default()
        .src_dest_pairs
        .push((sources.clone(), dests.clone()));
    schedule
        .comm_by_step
        .entry(step)
        .or_default()
        .layer_group_comm
        .entry(layer)
        .or_default()
        .entry(group)
        .or_default()
        .src_dest_pairs
        .push((sources.clone(), dests.clone()));
}

fn schedule_to_graph(schedule: &Schedule) -> anyhow::Result<SketchGraph> {
    let mut nodes: Vec<SketchNode> = Vec::new();
    let mut received_node = vec![None; schedule.gpu_num];
    for (&step, step_comm) in &schedule.comm_by_step {
        for (&layer, layer_comm) in &step_comm.layer_group_comm {
            for (&group, group_comm) in layer_comm {
                if group_comm.src_dest_pairs.len() != 1 {
                    anyhow::bail!(
                        "SyCCL sketch graph expects one src/dst pair per step/layer/group node"
                    );
                }
                let (srcs, dsts) = &group_comm.src_dest_pairs[0];
                let id = nodes.len();
                for &dst in dsts {
                    if dst >= received_node.len() {
                        anyhow::bail!("destination GPU {dst} is out of range");
                    }
                    if received_node[dst].is_some() {
                        anyhow::bail!("destination GPU {dst} receives data more than once");
                    }
                    received_node[dst] = Some(id);
                }
                let mut deps = BTreeSet::new();
                for &src in srcs {
                    if src != schedule.sender {
                        let dep = received_node.get(src).and_then(|dep| *dep).ok_or_else(|| {
                            anyhow::anyhow!("source GPU {src} has not received data yet")
                        })?;
                        deps.insert(dep);
                    }
                }
                for dep in &deps {
                    nodes[*dep].next.push(id);
                }
                nodes.push(SketchNode {
                    id,
                    step,
                    layer,
                    group,
                    src_dest_pair: SrcDestPair {
                        srcs: srcs.clone(),
                        dsts: dsts.clone(),
                    },
                    deps: deps.into_iter().collect(),
                    next: Vec::new(),
                });
            }
        }
    }
    Ok(SketchGraph {
        ngpus: schedule.gpu_num,
        src_gpu: schedule.sender,
        nodes,
    })
}

fn partition_groups(
    schedule: &Schedule,
    layer: usize,
    sources: &BTreeMap<usize, BTreeSet<usize>>,
    _dests: &BTreeMap<usize, BTreeSet<usize>>,
) -> BTreeMap<usize, BTreeSet<usize>> {
    let mut result: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    for (&group, src) in sources {
        let max_source_num = src.len();
        let group_cur_dest_gpus = group_num_dest(schedule, layer, group);
        let group_obj_dest = group_num_obj_dest(schedule, layer, group);
        let d_per_s = layer_d_per_s(schedule, layer, group);
        if d_per_s == 0 {
            continue;
        }
        let max_allow_num = std::cmp::min(
            max_source_num,
            (group_obj_dest - group_cur_dest_gpus) / d_per_s,
        );
        result.entry(max_allow_num).or_default().insert(group);
    }
    result
}

fn search_layer_sizes(
    schedule: &Schedule,
    layer: usize,
    group_partitions: &BTreeMap<usize, BTreeSet<usize>>,
) -> Vec<LayerSize> {
    if group_partitions.is_empty() {
        return Vec::new();
    }
    let mut result = Vec::new();
    let mut max_source_per_group = schedule.per_layer_size[&layer]
        .config
        .iter()
        .map(|conf| conf.comm_source_per_group)
        .max()
        .unwrap_or(0);
    if let Some(max_partition) = group_partitions.keys().next_back() {
        max_source_per_group = std::cmp::min(max_source_per_group, *max_partition);
    }
    let mut acc_group_num = group_partitions.values().map(BTreeSet::len).sum::<usize>();
    let mut group_iter = group_partitions.iter().peekable();
    for source_per_group in 1..=max_source_per_group {
        while group_iter
            .peek()
            .is_some_and(|(max_sources, _)| **max_sources < source_per_group)
        {
            if let Some((_, groups)) = group_iter.next() {
                acc_group_num = acc_group_num.saturating_sub(groups.len());
            }
        }
        if group_iter.peek().is_none() {
            break;
        }
        let max_group_num =
            std::cmp::min(acc_group_num, total_num_groups_for_layer(schedule, layer));
        for group_num in 1..=max_group_num {
            result.push(LayerSize {
                layer,
                source_per_group,
                group_num,
            });
        }
    }
    result
}

fn prune_layer_sizes(
    policy: &PruneConfig,
    schedule: &Schedule,
    orig_sizes: &[LayerSize],
    step: usize,
) -> Vec<(LayerPrune, LayerSize)> {
    let max_num_group = orig_sizes.iter().map(|s| s.group_num).max().unwrap_or(0);
    let max_num_source = orig_sizes
        .iter()
        .map(|s| s.source_per_group)
        .max()
        .unwrap_or(0);
    let mut result = Vec::new();
    for &size in orig_sizes {
        let mut prune_info = schedule
            .layer_prunes
            .get(&size.layer)
            .cloned()
            .unwrap_or_default();
        update_src_prune(&mut prune_info, step, size.group_num, size.source_per_group);
        if prune_source_arithmetic_or_geometric(policy, &prune_info) {
            continue;
        }
        if policy.drop_small_sources
            && max_num_source != size.source_per_group
            && (!policy.keep_large_groups || max_num_group != size.group_num)
        {
            continue;
        }
        if policy.drop_small_groups
            && max_num_group != size.group_num
            && (!policy.keep_large_sources || max_num_source != size.source_per_group)
        {
            continue;
        }
        result.push((prune_info, size));
    }
    result
}

fn update_src_prune(
    prune: &mut LayerPrune,
    step: usize,
    group_num: usize,
    source_per_group: usize,
) {
    if prune.step_split == 0 {
        prune.step_split = -1;
    }
    if prune.num_group_per_step.len() < step {
        if prune.step_split != -1 {
            debug_assert_eq!(
                prune.num_group_per_step.len() + prune.step_split as usize,
                step + 1
            );
        }
        if !prune.num_group_per_step.is_empty() {
            prune.step_split = (step + 1 - prune.num_group_per_step.len()) as isize;
        }
        for _ in prune.num_group_per_step.len()..step {
            prune.num_group_per_step.push(0);
            prune.num_source_per_group_per_step.push(0);
        }
    }
    prune.num_group_per_step.push(group_num);
    prune.num_source_per_group_per_step.push(source_per_group);
}

fn prune_source_arithmetic_or_geometric(policy: &PruneConfig, prune: &LayerPrune) -> bool {
    if compare_step_arithmetic_geometric_violation(policy, prune)
        && !COMPARE_STEP_PRUNE_WARNING_EMITTED.swap(true, Ordering::Relaxed)
    {
        eprintln!(
            "warning: prune_sources_by_compare_steps matched a non-arithmetic/non-geometric \
             cross-step pattern, but this check is warning-only and no branch was pruned"
        );
    }
    false
}

fn compare_step_arithmetic_geometric_violation(policy: &PruneConfig, prune: &LayerPrune) -> bool {
    if !policy.prune_sources_by_compare_steps {
        return false;
    }
    let mut is_arithmetic = policy.allow_source_per_group_arithmetic_per_step;
    let mut is_geometric = policy.allow_source_per_group_geometric_per_step;
    check_arithmetic_or_geometric(
        &prune.num_source_per_group_per_step,
        &mut is_arithmetic,
        &mut is_geometric,
        prune.step_split,
    );
    if !is_arithmetic && !is_geometric {
        return true;
    }

    is_arithmetic = policy.allow_num_groups_arithmetic_per_step;
    is_geometric = policy.allow_num_groups_geometric_per_step;
    check_arithmetic_or_geometric(
        &prune.num_group_per_step,
        &mut is_arithmetic,
        &mut is_geometric,
        prune.step_split,
    );
    !is_arithmetic || !is_geometric
}

fn check_arithmetic_or_geometric(
    numbers: &[usize],
    is_arithmetic: &mut bool,
    is_geometric: &mut bool,
    step_split: isize,
) {
    let split = if step_split == -1 {
        1
    } else {
        step_split as usize
    };
    for s in 0..numbers.len() {
        if *is_arithmetic
            && s >= 2 * split
            && numbers[s] + numbers[s - 2 * split] != numbers[s - split] + numbers[s - split]
        {
            *is_arithmetic = false;
        }
        if *is_geometric
            && s >= 2 * split
            && numbers[s] * numbers[s - 2 * split] != numbers[s - split] * numbers[s - split]
        {
            *is_geometric = false;
        }
    }
}

fn search_layer_groups(
    group_partitions: &BTreeMap<usize, BTreeSet<usize>>,
    layer_size: LayerSize,
) -> Vec<BTreeSet<usize>> {
    let mut queue = VecDeque::from([BTreeSet::new()]);
    let mut total_group_num = group_partitions.values().map(BTreeSet::len).sum::<usize>();
    for (&max_sources, groups_for_partition) in group_partitions {
        total_group_num -= groups_for_partition.len();
        if max_sources < layer_size.source_per_group {
            continue;
        }
        let n = queue.len();
        for _ in 0..n {
            let groups = queue.pop_front().expect("queue length");
            let group_num_to_add = layer_size.group_num.saturating_sub(groups.len());
            let min_group_num = group_num_to_add.saturating_sub(total_group_num);
            let max_group_num = if group_num_to_add > total_group_num {
                std::cmp::min(group_num_to_add, groups_for_partition.len())
            } else {
                0
            };
            for count in min_group_num..=max_group_num {
                let mut new_groups = groups.clone();
                for group in groups_for_partition.iter().take(count) {
                    new_groups.insert(*group);
                }
                queue.push_back(new_groups);
            }
        }
    }
    queue.into_iter().collect()
}

fn search_sources(
    policy: &PruneConfig,
    schedule: &Schedule,
    group_sources: &BTreeMap<usize, BTreeSet<usize>>,
    group: usize,
    source_per_group: usize,
) -> Vec<BTreeSet<usize>> {
    let Some(sources) = group_sources.get(&group) else {
        return Vec::new();
    };
    if policy.partition_sources {
        let source_partitions = partition_gpus(schedule, sources);
        gpu_combinations(source_per_group, sources, &source_partitions)
    } else {
        vec![sources.clone()]
    }
}

fn prune_sources(
    schedule: &Schedule,
    step: usize,
    layer: usize,
    group: usize,
    sources: &BTreeSet<usize>,
) -> bool {
    if !step_has_comm(schedule, step)
        || !schedule.comm_by_step[&step]
            .layer_group_comm
            .contains_key(&layer)
    {
        return false;
    }
    if gpu_group_in_layer(&schedule.topo, schedule.sender, layer) == Some(group) {
        return false;
    }
    let group_map = &schedule.topo.layer_group_src_dest[&layer][&group];
    let Some(g1) = group_map.keys().next().copied() else {
        return false;
    };
    let Some(g2) = group_map.keys().next_back().copied() else {
        return false;
    };
    if g1 == g2 {
        return false;
    }

    let sender_group = gpu_group_in_layer(&schedule.topo, schedule.sender, layer);
    for (&group_id, comm) in &schedule.comm_by_step[&step].layer_group_comm[&layer] {
        if group_id < group && sender_group != Some(group_id) {
            let other_group_map = &schedule.topo.layer_group_src_dest[&layer][&group_id];
            let Some(other_g1) = other_group_map.keys().next().copied() else {
                continue;
            };
            let Some(other_g2) = other_group_map.keys().next_back().copied() else {
                continue;
            };
            if other_g1 == other_g2 {
                continue;
            }
            let source_num = comm.src_dest_pairs.len();
            if sources.len() > source_num {
                return false;
            }
            if source_num == sources.len() {
                for (src, other_pair) in sources.iter().zip(comm.src_dest_pairs.iter()) {
                    let Some(other_first_src) = other_pair.0.iter().next() else {
                        continue;
                    };
                    let i1 = (*src - g1) / (g2 - g1);
                    let i2 = (*other_first_src - other_g1) / (other_g2 - other_g1);
                    if i1 > i2 {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn prune_dest_combs(
    policy: &PruneConfig,
    schedule: &Schedule,
    step: usize,
    layer: usize,
    group: usize,
    source_comb: &BTreeSet<usize>,
    dest_combs: Vec<BTreeSet<usize>>,
) -> BTreeMap<DestPrune, Vec<BTreeSet<usize>>> {
    if !policy.prune_dests {
        return BTreeMap::from([(DestPrune::Undefined, dest_combs)]);
    }

    let mut best_ratio = DestCombPrune {
        balance_ratio: -1.0,
        ..Default::default()
    };
    let mut indexes: BTreeMap<DestPrune, BTreeSet<usize>> = BTreeMap::new();
    for (i, dest_comb) in dest_combs.iter().enumerate() {
        let attr = get_ratio(schedule, step, layer, group, source_comb, dest_comb);
        let dst_prune = schedule
            .layer_prunes
            .get(&layer)
            .map(|p| p.dst_prune)
            .unwrap_or(DestPrune::Undefined);

        if dst_prune == DestPrune::Undefined
            || (policy.allow_dest_balanced && dst_prune == DestPrune::Balance)
        {
            if best_ratio.balance_ratio < 0.0 || attr.balance_ratio < best_ratio.balance_ratio {
                best_ratio.balance_ratio = attr.balance_ratio;
                indexes.entry(DestPrune::Balance).or_default().clear();
                indexes.entry(DestPrune::Balance).or_default().insert(i);
            } else if (attr.balance_ratio - best_ratio.balance_ratio).abs() < f64::EPSILON {
                indexes.entry(DestPrune::Balance).or_default().insert(i);
            }
        }

        if dst_prune == DestPrune::Undefined
            || (policy.allow_dest_centralized && dst_prune == DestPrune::Imbalance)
        {
            if best_ratio.layer_group_finished_gpu_num.is_empty()
                || attr_is_larger_than(&attr, &best_ratio)
            {
                best_ratio.layer_group_finished_gpu_num = attr.layer_group_finished_gpu_num.clone();
                indexes.entry(DestPrune::Imbalance).or_default().clear();
                indexes.entry(DestPrune::Imbalance).or_default().insert(i);
            } else if attr_is_larger_than(&attr, &best_ratio) {
                indexes.entry(DestPrune::Imbalance).or_default().insert(i);
            }
        }
    }
    merge_prunes(indexes, dest_combs)
}

fn merge_prunes(
    mut indexes: BTreeMap<DestPrune, BTreeSet<usize>>,
    dest_combs: Vec<BTreeSet<usize>>,
) -> BTreeMap<DestPrune, Vec<BTreeSet<usize>>> {
    let mut result: BTreeMap<DestPrune, Vec<BTreeSet<usize>>> = BTreeMap::new();
    let balance = indexes.remove(&DestPrune::Balance).unwrap_or_default();
    let mut imbalance = indexes.remove(&DestPrune::Imbalance).unwrap_or_default();
    for index in balance {
        if imbalance.remove(&index) {
            result
                .entry(DestPrune::Undefined)
                .or_default()
                .push(dest_combs[index].clone());
        } else {
            result
                .entry(DestPrune::Balance)
                .or_default()
                .push(dest_combs[index].clone());
        }
    }
    for index in imbalance {
        result
            .entry(DestPrune::Imbalance)
            .or_default()
            .push(dest_combs[index].clone());
    }
    result
}

fn get_ratio(
    schedule: &Schedule,
    step: usize,
    layer: usize,
    group: usize,
    source_comb: &BTreeSet<usize>,
    dest_comb: &BTreeSet<usize>,
) -> DestCombPrune {
    let simulated = copy_scheduling(
        schedule,
        source_comb,
        dest_comb,
        step,
        layer,
        group,
        schedule
            .layer_prunes
            .get(&layer)
            .cloned()
            .unwrap_or_default(),
    );
    let mut result = DestCombPrune {
        balance_ratio: 0.0,
        ..Default::default()
    };
    for &other_layer in simulated.per_layer_size.keys() {
        if layer == other_layer {
            continue;
        }
        for &other_group in simulated.topo.layer_group_src_dest[&other_layer].keys() {
            let finished_gpu_num = group_num_reached_gpus(&simulated, other_layer, other_group);
            result
                .layer_group_finished_gpu_num
                .entry(other_layer)
                .or_default()
                .push(finished_gpu_num);
        }
        if let Some(values) = result.layer_group_finished_gpu_num.get_mut(&other_layer) {
            values.sort();
            result.balance_ratio += mean_dev(values);
        }
    }
    result
}

fn attr_is_larger_than(left: &DestCombPrune, right: &DestCombPrune) -> bool {
    for (&layer, left_values) in &left.layer_group_finished_gpu_num {
        let Some(right_values) = right.layer_group_finished_gpu_num.get(&layer) else {
            return true;
        };
        for (left_v, right_v) in left_values.iter().rev().zip(right_values.iter().rev()) {
            if left_v > right_v {
                return true;
            }
            if left_v < right_v {
                return false;
            }
        }
    }
    false
}

fn mean_dev(values: &[usize]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let avg = values.iter().sum::<usize>() as f64 / values.len() as f64;
    values
        .iter()
        .map(|v| {
            let diff = *v as f64 - avg;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64
}

fn partition_gpus(
    schedule: &Schedule,
    gpus: &BTreeSet<usize>,
) -> BTreeMap<PartitionKey, BTreeSet<usize>> {
    let mut result: BTreeMap<PartitionKey, BTreeSet<usize>> = BTreeMap::new();
    for &gpu in gpus {
        result
            .entry(get_gpu_key(schedule, gpu))
            .or_default()
            .insert(gpu);
    }
    result
}

fn get_gpu_key(schedule: &Schedule, gpu: usize) -> PartitionKey {
    let mut layer_gpus = BTreeMap::new();
    for &layer in schedule.per_layer_size.keys() {
        if let Some(group) = gpu_group_in_layer(&schedule.topo, gpu, layer) {
            layer_gpus.insert(layer, group_num_reached_gpus(schedule, layer, group));
        }
    }
    PartitionKey {
        comm_paths: if gpu_has_comm(schedule, gpu) {
            comm_path_layers(schedule, gpu)
        } else {
            Vec::new()
        },
        layer_gpus,
    }
}

fn gpu_combinations(
    num: usize,
    gpus: &BTreeSet<usize>,
    partitions: &BTreeMap<PartitionKey, BTreeSet<usize>>,
) -> Vec<BTreeSet<usize>> {
    if num == 0 {
        return vec![BTreeSet::new()];
    }
    if num > gpus.len() {
        return Vec::new();
    }
    let mut gpu_partition = BTreeMap::new();
    for (partition, partition_gpus) in partitions {
        for gpu in partition_gpus {
            gpu_partition.insert(*gpu, partition);
        }
    }

    let mut result = Vec::new();
    let mut queue = VecDeque::from([BTreeSet::new()]);
    let mut remained_gpu_num = gpus.len();
    for &gpu in gpus {
        let n = queue.len();
        remained_gpu_num -= 1;
        for _ in 0..n {
            let selected = queue.pop_front().expect("queue length");
            if remained_gpu_num + selected.len() >= num {
                queue.push_back(selected.clone());
            }
            let mut is_valid = true;
            if let Some(partition) = gpu_partition.get(&gpu) {
                if let Some(partition_gpus) = partitions.get(*partition) {
                    for gpu1 in partition_gpus {
                        if *gpu1 < gpu && !selected.contains(gpu1) {
                            is_valid = false;
                            break;
                        }
                    }
                }
            }
            if is_valid {
                let mut with_gpu = selected;
                with_gpu.insert(gpu);
                if with_gpu.len() == num {
                    result.push(with_gpu);
                } else {
                    queue.push_back(with_gpu);
                }
            }
        }
    }
    result
}

fn comm_path_layers(schedule: &Schedule, gpu: usize) -> Vec<Vec<usize>> {
    let mut result = vec![Vec::new()];
    let Some(record) = schedule.comm_by_gpus.get(&gpu) else {
        return result;
    };
    if record.step == -1 {
        return result;
    }
    let mut individual_sources: Vec<SourceRecord> = Vec::new();
    for &src in &record.sources {
        let Some(src_record) = schedule.comm_by_gpus.get(&src).cloned() else {
            continue;
        };
        if individual_sources.contains(&src_record) {
            continue;
        }
        individual_sources.push(src_record);
        for mut path in comm_path_layers(schedule, src) {
            path.push(record.layer as usize);
            result.push(path);
        }
    }
    result
}

fn is_valid_groups(schedule: &Schedule, layer: usize, groups: &BTreeSet<usize>) -> bool {
    let mut group_num = if !layer_has_comm(schedule, layer) {
        groups.len()
    } else {
        schedule.comm_by_layer[&layer].comm_groups.len()
    };
    if layer_has_comm(schedule, layer) {
        for group in groups {
            if !group_has_comm(schedule, layer, *group) {
                group_num += 1;
            }
        }
    }
    group_num <= total_num_groups_for_layer(schedule, layer)
}

fn layer_num_groups(topo: &TopoSubset, layer: usize) -> usize {
    topo.layer_group_src_dest[&layer].len()
}

fn layer_num_gpus_per_group(topo: &TopoSubset, layer: usize) -> usize {
    topo.layer_group_src_dest[&layer]
        .values()
        .next()
        .map(BTreeMap::len)
        .unwrap_or(0)
}

fn layer_num_links_per_gpu(topo: &TopoSubset, layer: usize) -> usize {
    topo.layer_group_src_dest[&layer]
        .values()
        .next()
        .and_then(|sources| sources.values().next())
        .map(BTreeSet::len)
        .unwrap_or(0)
}

fn layer_comm(config: &LayerConfig) -> usize {
    config
        .config
        .iter()
        .map(|conf| conf.dest_per_group * conf.used_group_num)
        .sum()
}

fn get_remained_dest(schedule: &Schedule) -> (usize, usize) {
    let mut max = 0;
    let mut min = 0;
    for &layer in schedule.per_layer_size.keys() {
        max += layer_num_groups(&schedule.topo, layer)
            * (layer_num_gpus_per_group(&schedule.topo, layer) - 1);
        min += 1;
    }
    (max, min)
}

fn total_num_groups(schedule: &Schedule) -> usize {
    schedule
        .per_layer_size
        .keys()
        .map(|&layer| total_num_groups_for_layer(schedule, layer))
        .sum()
}

fn total_num_groups_for_layer(schedule: &Schedule, layer: usize) -> usize {
    schedule
        .per_layer_size
        .get(&layer)
        .map(|config| config.config.iter().map(|conf| conf.used_group_num).sum())
        .unwrap_or(0)
}

fn step_has_comm(schedule: &Schedule, step: usize) -> bool {
    schedule.comm_by_step.contains_key(&step)
}

fn layer_has_comm(schedule: &Schedule, layer: usize) -> bool {
    schedule.comm_by_layer.contains_key(&layer)
}

fn group_has_comm(schedule: &Schedule, layer: usize, group: usize) -> bool {
    schedule
        .comm_by_layer
        .get(&layer)
        .is_some_and(|layer_comm| layer_comm.comm_groups.contains_key(&group))
}

fn gpu_has_comm(schedule: &Schedule, gpu: usize) -> bool {
    schedule.comm_by_gpus.contains_key(&gpu)
}

fn group_num_dest(schedule: &Schedule, layer: usize, group: usize) -> usize {
    schedule
        .comm_by_layer
        .get(&layer)
        .and_then(|layer_comm| layer_comm.comm_groups.get(&group))
        .map(|group_comm| {
            group_comm
                .src_dest_pairs
                .iter()
                .map(|(_, dsts)| dsts.len())
                .sum()
        })
        .unwrap_or(0)
}

fn group_num_not_reached_gpus(schedule: &Schedule, layer: usize, group: usize) -> usize {
    schedule.topo.layer_group_src_dest[&layer][&group]
        .keys()
        .filter(|gpu| !gpu_has_comm(schedule, **gpu))
        .count()
}

fn group_num_reached_gpus(schedule: &Schedule, layer: usize, group: usize) -> usize {
    schedule.topo.layer_group_src_dest[&layer][&group]
        .keys()
        .filter(|gpu| gpu_has_comm(schedule, **gpu))
        .count()
}

fn new_group_config_id(schedule: &Schedule, layer: usize, group: usize) -> Option<usize> {
    if !layer_has_comm(schedule, layer) {
        return Some(0);
    }
    for (config_id, config) in schedule.per_layer_size[&layer].config.iter().enumerate() {
        if config.group_id.is_some_and(|group_id| group_id != group) {
            continue;
        }
        let used = schedule.comm_by_layer[&layer]
            .config_id_to_groups
            .get(&config_id)
            .map(BTreeSet::len)
            .unwrap_or(0);
        if used < config.used_group_num {
            return Some(config_id);
        }
    }
    None
}

fn group_num_obj_dest(schedule: &Schedule, layer: usize, group: usize) -> usize {
    let config_id = if group_has_comm(schedule, layer, group) {
        schedule.comm_by_layer[&layer].group_to_config_id[&group]
    } else {
        new_group_config_id(schedule, layer, group).unwrap_or(0)
    };
    schedule.per_layer_size[&layer].config[config_id].dest_per_group
}

fn layer_d_per_s(schedule: &Schedule, layer: usize, group: usize) -> usize {
    let config_id = if group_has_comm(schedule, layer, group) {
        schedule.comm_by_layer[&layer].group_to_config_id[&group]
    } else {
        new_group_config_id(schedule, layer, group).unwrap_or(0)
    };
    let config = &schedule.per_layer_size[&layer].config[config_id];
    if config.comm_source_per_group == 0 {
        0
    } else {
        config.dest_per_group / config.comm_source_per_group
    }
}

fn gpu_group_in_layer(topo: &TopoSubset, gpu: usize, layer: usize) -> Option<usize> {
    topo.gpu_layer_group
        .get(&gpu)
        .and_then(|layers| layers.get(&layer))
        .copied()
}

fn output_sched(schedule: &Schedule) -> bool {
    schedule.comm_by_gpus.len() == schedule.gpu_num
}

fn usize_saturating_max(left: usize, right: usize) -> usize {
    std::cmp::max(left, right)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_step_arithmetic_geometric_check_warns_without_pruning() {
        let mut policy = PruneConfig::default();
        policy.prune_sources_by_compare_steps = true;
        policy.allow_source_per_group_arithmetic_per_step = false;
        policy.allow_source_per_group_geometric_per_step = false;
        policy.allow_num_groups_arithmetic_per_step = false;
        policy.allow_num_groups_geometric_per_step = false;

        let prune = LayerPrune {
            step_split: -1,
            num_group_per_step: vec![1, 3, 2],
            num_source_per_group_per_step: vec![1, 4, 2],
            ..Default::default()
        };

        assert!(compare_step_arithmetic_geometric_violation(&policy, &prune));
        COMPARE_STEP_PRUNE_WARNING_EMITTED.store(false, Ordering::Relaxed);
        assert!(!prune_source_arithmetic_or_geometric(&policy, &prune));
        assert!(COMPARE_STEP_PRUNE_WARNING_EMITTED.load(Ordering::Relaxed));
    }
}
