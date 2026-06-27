use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet, VecDeque};

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use crate::config::{spec_to_bps, SimConfig};
use crate::schedule::{build_flows, SketchTransmissionSource, TranslatedSchedule};
use crate::topology::Topology;

const SIMAI_ENDPOINT_EVENT_DELAY_NS: u64 = 10;
const SIMAI_RDMA_APP_START_DELAY_NS: u64 = 3_000;
const RDMA_PACKET_PAYLOAD_BYTES: u64 = 9_000;
const RDMA_DATA_HEADER_BYTES_ON_WIRE: u64 = 52;
const RDMA_ACK_BYTES_ON_WIRE: u64 = 60;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LinkKey {
    pub src: usize,
    pub dst: usize,
}

impl LinkKey {
    pub fn new(src: usize, dst: usize) -> Self {
        Self { src, dst }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RouteKind {
    SameHost,
    CrossHost,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    pub time_us: f64,
    #[serde(default)]
    pub bottleneck_profile: BottleneckProfile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckProfile {
    pub critical_flow_chain: CriticalFlowChain,
}

impl Default for BottleneckProfile {
    fn default() -> Self {
        Self {
            critical_flow_chain: CriticalFlowChain::default(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CriticalFlowChain {
    pub collective_start_ns: u64,
    pub collective_finish_ns: u64,
    pub collective_duration_ns: u64,
    pub latest_flow_id: Option<usize>,
    pub chain: Vec<CriticalFlowChainEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalFlowChainEntry {
    pub flow_id: usize,
    pub chunk: CriticalFlowChunk,
    pub previous_bottleneck_flow_id: Option<usize>,
    pub previous_bottleneck_flow_end_time: Option<u64>,
    pub current_flow_start_time: u64,
    pub current_flow_duration_ns: u64,
    pub sketch_transmission: Option<SketchTransmissionSource>,
    pub propagation_delay_ns: u64,
    pub transmission_time_ns: CriticalFlowTransmissionTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalFlowChunk {
    pub src_chunk: usize,
    pub chunk_index: usize,
    pub src_gpu: usize,
    pub dst_gpu: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalFlowTransmissionTime {
    pub link_busy_ns: u64,
    pub link_wait_ns: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct LinkRuntime {
    bandwidth_bps: u64,
    latency_ns: u64,
    busy_until_ns: u64,
    active: bool,
    data_queue_mode: DataQueueMode,
    path_kind: LinkPathKind,
    data_fifo: VecDeque<QueuedPacket>,
    data_queues: BTreeMap<usize, VecDeque<QueuedPacket>>,
    rr_active_data: BTreeMap<usize, ActiveData>,
    rr_data_last_update_ns: u64,
    rr_generation: u64,
    ack_queue: VecDeque<QueuedPacket>,
    rr_last_flow_id: Option<usize>,
    transmissions: usize,
    busy_ns: u64,
    queue_wait_ns: u64,
    last_finish_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DataQueueMode {
    Fifo,
    RoundRobinQp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LinkPathKind {
    Host,
    Nic,
    Net,
}

#[derive(Debug, Clone)]
struct FlowRoutes {
    forward: Vec<usize>,
    ack: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PacketKind {
    Data,
    Ack,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Packet {
    flow_id: usize,
    kind: PacketKind,
    hop_index: usize,
}

#[derive(Debug, Clone, Copy)]
struct QueuedPacket {
    packet: Packet,
    arrival_ns: u64,
}

#[derive(Debug, Clone, Copy)]
struct ActiveData {
    packet: Packet,
    arrival_ns: u64,
    total_bytes: u64,
    head_remaining_bytes: Option<f64>,
    head_released: bool,
    remaining_bytes: f64,
}

#[derive(Debug, Clone, Default)]
struct FlowTiming {
    start_ns: Option<u64>,
    send_complete_ns: Option<u64>,
    recv_complete_ns: Option<u64>,
    queue_wait_ns: u64,
    route_busy_ns: u64,
    route_latency_ns: u64,
    parent_release_ns: u64,
}

#[derive(Debug, Clone, Copy)]
enum EventKind {
    StartFlow { flow_id: usize },
    LinkArrive { link_id: usize, packet: Packet },
    LinkReady { link_id: usize },
    RrDataComplete { link_id: usize, generation: u64 },
    FlowSendComplete { flow_id: usize },
    FlowRecvComplete { flow_id: usize },
}

#[derive(Debug, Clone, Copy)]
struct Event {
    time_ns: u64,
    seq: u64,
    kind: EventKind,
}

impl EventKind {
    fn priority(self) -> u8 {
        // Tie-breaker for same timestamp: lower value is processed first.
        match self {
            EventKind::LinkReady { .. } => 1,
            EventKind::RrDataComplete { .. } => 1,
            _ => 0,
        }
    }
}

impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .time_ns
            .cmp(&self.time_ns)
            .then_with(|| other.kind.priority().cmp(&self.kind.priority()))
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.time_ns == other.time_ns && self.seq == other.seq
    }
}

impl Eq for Event {}

pub fn simulate_case(
    config: &SimConfig,
    schedule: &TranslatedSchedule,
) -> Result<SimulationResult> {
    // Event-driven flow-level simulation of the translated schedule.
    if schedule.ngpus != config.hosts.host_num * config.hosts.gpus_per_host {
        return Err(anyhow!(
            "translated ngpus {} does not match config GPUs {}",
            schedule.ngpus,
            config.hosts.host_num * config.hosts.gpus_per_host
        ));
    }
    let flow_size = schedule.chunk_size_byte.unwrap_or(config.coll_bytes);
    let graph = build_flows(schedule, flow_size)?;
    let (_, source_cross_epoch_pressure_ns) =
        source_cross_epoch_pressure(config, &graph.flows, flow_size);
    let epoch_barrier_ns = 0;
    let topo = Topology::from_config(config)?;
    let link_pairs = topo.link_runtimes();
    let mut links = Vec::with_capacity(link_pairs.len());
    for (_, runtime) in link_pairs {
        links.push(runtime);
    }
    // Precompute forward and ACK link sequences for each flow.
    let mut route_cache = Vec::with_capacity(graph.flows.len());
    for flow in &graph.flows {
        route_cache.push(FlowRoutes {
            forward: topo.route_link_ids(flow.src, flow.dst)?,
            ack: topo.route_link_ids(flow.dst, flow.src)?,
        });
    }

    let mut remaining_parents: Vec<usize> = graph
        .flows
        .iter()
        .map(|flow| flow.recv_parents.len() + flow.send_parents.len())
        .collect();
    let mut parent_max_finish = vec![0_u64; graph.flows.len()];
    let mut finish_times = vec![0_u64; graph.flows.len()];
    let mut flow_timings = vec![FlowTiming::default(); graph.flows.len()];
    let mut events = BinaryHeap::new();
    let mut next_event_seq = 0_u64;
    for flow in &graph.flows {
        if remaining_parents[flow.id] == 0 {
            push_event(
                &mut events,
                &mut next_event_seq,
                SIMAI_ENDPOINT_EVENT_DELAY_NS,
                EventKind::StartFlow { flow_id: flow.id },
            );
        }
    }

    // BinaryHeap uses reversed ordering so the earliest event is popped first.
    let mut completed = 0_usize;
    while let Some(event) = events.pop() {
        match event.kind {
            EventKind::StartFlow { flow_id } => {
                flow_timings[flow_id].start_ns = Some(event.time_ns);
                let route = &route_cache[flow_id].forward;
                if route.is_empty() {
                    // Local transfer: no network hops, complete immediately.
                    push_event(
                        &mut events,
                        &mut next_event_seq,
                        event.time_ns,
                        EventKind::FlowSendComplete { flow_id },
                    );
                    push_event(
                        &mut events,
                        &mut next_event_seq,
                        event.time_ns,
                        EventKind::FlowRecvComplete { flow_id },
                    );
                    continue;
                }
                push_event(
                    &mut events,
                    &mut next_event_seq,
                    event.time_ns + SIMAI_RDMA_APP_START_DELAY_NS,
                    EventKind::LinkArrive {
                        link_id: route[0],
                        packet: Packet {
                            flow_id,
                            kind: PacketKind::Data,
                            hop_index: 0,
                        },
                    },
                );
            }
            EventKind::LinkArrive { link_id, packet } => {
                let link = &mut links[link_id];
                let queued = QueuedPacket {
                    packet,
                    arrival_ns: event.time_ns,
                };
                match packet.kind {
                    PacketKind::Data => match link.data_queue_mode {
                        DataQueueMode::Fifo => link.data_fifo.push_back(queued),
                        DataQueueMode::RoundRobinQp if packet.hop_index == 0 => {
                            // First hop on RR links enters the active sharing set.
                            enqueue_rr_data(link, &graph.flows, queued, event.time_ns);
                            schedule_rr_data_completion(
                                link_id,
                                event.time_ns,
                                &mut links,
                                &mut events,
                                &mut next_event_seq,
                            );
                            continue;
                        }
                        DataQueueMode::RoundRobinQp => link
                            .data_queues
                            .entry(packet.flow_id)
                            .or_default()
                            .push_back(queued),
                    },
                    // ACKs use a separate queue and are prioritized on RR links.
                    PacketKind::Ack => link.ack_queue.push_back(queued),
                }
                if !link.active && link.busy_until_ns <= event.time_ns {
                    push_event(
                        &mut events,
                        &mut next_event_seq,
                        event.time_ns,
                        EventKind::LinkReady { link_id },
                    );
                }
            }
            EventKind::LinkReady { link_id } => match links[link_id].data_queue_mode {
                DataQueueMode::Fifo => {
                    try_start_link(
                        link_id,
                        event.time_ns,
                        &mut links,
                        &graph.flows,
                        &route_cache,
                        &mut flow_timings,
                        &mut events,
                        &mut next_event_seq,
                    );
                }
                DataQueueMode::RoundRobinQp => {
                    // Serve ACKs first; otherwise send queued data or advance RR service.
                    if !links[link_id].ack_queue.is_empty() {
                        try_start_rr_ack(
                            link_id,
                            event.time_ns,
                            &mut links,
                            &route_cache,
                            &mut flow_timings,
                            &mut events,
                            &mut next_event_seq,
                        );
                    } else if !links[link_id].data_queues.is_empty() {
                        try_start_link(
                            link_id,
                            event.time_ns,
                            &mut links,
                            &graph.flows,
                            &route_cache,
                            &mut flow_timings,
                            &mut events,
                            &mut next_event_seq,
                        );
                    } else {
                        schedule_rr_data_completion(
                            link_id,
                            event.time_ns,
                            &mut links,
                            &mut events,
                            &mut next_event_seq,
                        );
                    }
                }
            },
            EventKind::RrDataComplete {
                link_id,
                generation,
            } => {
                complete_rr_data(
                    link_id,
                    generation,
                    event.time_ns,
                    &mut links,
                    &route_cache,
                    &mut flow_timings,
                    &mut events,
                    &mut next_event_seq,
                );
            }
            EventKind::FlowSendComplete { flow_id } => {
                flow_timings[flow_id].send_complete_ns = Some(
                    flow_timings[flow_id]
                        .send_complete_ns
                        .unwrap_or(event.time_ns),
                );
                // Release children waiting on send-side dependencies.
                for child in &graph.flows[flow_id].children {
                    let child_flow = &graph.flows[*child];
                    if child_flow.send_parents.binary_search(&flow_id).is_ok() {
                        release_parent(
                            *child,
                            event.time_ns,
                            &mut parent_max_finish,
                            &mut remaining_parents,
                            &mut flow_timings,
                            &mut events,
                            &mut next_event_seq,
                        );
                    }
                }
            }
            EventKind::FlowRecvComplete { flow_id } => {
                if finish_times[flow_id] != 0 {
                    continue;
                }
                finish_times[flow_id] = event.time_ns;
                flow_timings[flow_id].recv_complete_ns = Some(event.time_ns);
                completed += 1;
                // Release children waiting on receive-side dependencies.
                for child in &graph.flows[flow_id].children {
                    let child_flow = &graph.flows[*child];
                    if child_flow.recv_parents.binary_search(&flow_id).is_ok() {
                        release_parent(
                            *child,
                            event.time_ns,
                            &mut parent_max_finish,
                            &mut remaining_parents,
                            &mut flow_timings,
                            &mut events,
                            &mut next_event_seq,
                        );
                    }
                }
            }
        }
    }

    if completed != graph.flows.len() {
        return Err(anyhow!(
            "flow dependency graph did not complete: {}/{} completed",
            completed,
            graph.flows.len()
        ));
    }

    let event_finish_time_ns = finish_times.iter().copied().max().unwrap_or(0);
    let finish_time_ns = event_finish_time_ns
        .saturating_add(source_cross_epoch_pressure_ns)
        .saturating_add(epoch_barrier_ns);
    let critical_flow_id = finish_times
        .iter()
        .enumerate()
        .max_by_key(|(_, finish)| **finish)
        .map(|(id, _)| id);
    let bottleneck_profile = build_bottleneck_profile(
        &graph.flows,
        &flow_timings,
        critical_flow_id,
        finish_time_ns,
    );

    Ok(SimulationResult {
        time_us: finish_time_ns as f64 / 1000.0,
        bottleneck_profile,
    })
}

fn build_bottleneck_profile(
    flows: &[crate::schedule::Flow],
    timings: &[FlowTiming],
    critical_flow_id: Option<usize>,
    collective_finish_ns: u64,
) -> BottleneckProfile {
    let collective_start_ns = 0;
    BottleneckProfile {
        critical_flow_chain: CriticalFlowChain {
            collective_start_ns,
            collective_finish_ns,
            collective_duration_ns: collective_finish_ns.saturating_sub(collective_start_ns),
            latest_flow_id: critical_flow_id,
            chain: critical_flow_id
                .map(|flow_id| build_critical_flow_chain(flows, timings, flow_id))
                .unwrap_or_default(),
        },
    }
}

fn build_critical_flow_chain(
    flows: &[crate::schedule::Flow],
    timings: &[FlowTiming],
    critical_flow_id: usize,
) -> Vec<CriticalFlowChainEntry> {
    let mut flow_ids = Vec::new();
    let mut seen = HashSet::new();
    let mut current = Some(critical_flow_id);
    while let Some(flow_id) = current {
        if !seen.insert(flow_id) {
            break;
        }
        flow_ids.push(flow_id);
        current = previous_bottleneck_flow(flows, timings, flow_id).map(|parent| parent.flow_id);
    }
    flow_ids.reverse();

    flow_ids
        .into_iter()
        .map(|flow_id| critical_flow_chain_entry(flows, timings, flow_id))
        .collect()
}

#[derive(Debug, Clone, Copy)]
struct PreviousBottleneckFlow {
    flow_id: usize,
    release_ns: u64,
}

fn critical_flow_chain_entry(
    flows: &[crate::schedule::Flow],
    timings: &[FlowTiming],
    flow_id: usize,
) -> CriticalFlowChainEntry {
    let flow = &flows[flow_id];
    let previous = previous_bottleneck_flow(flows, timings, flow_id);
    let current_flow_start_time = previous.map(|parent| parent.release_ns).unwrap_or(0);
    let timing = &timings[flow_id];
    let current_flow_duration_ns = timing
        .recv_complete_ns
        .unwrap_or(current_flow_start_time)
        .saturating_sub(current_flow_start_time);
    CriticalFlowChainEntry {
        flow_id,
        chunk: CriticalFlowChunk {
            src_chunk: flow.src_chunk,
            chunk_index: flow.chunk_index,
            src_gpu: flow.src,
            dst_gpu: flow.dst,
        },
        previous_bottleneck_flow_id: previous.map(|parent| parent.flow_id),
        previous_bottleneck_flow_end_time: previous.map(|parent| parent.release_ns),
        current_flow_start_time,
        current_flow_duration_ns,
        sketch_transmission: flow.sketch_source.clone(),
        propagation_delay_ns: timing
            .route_latency_ns
            .saturating_add(fixed_scheduling_delay_ns()),
        transmission_time_ns: CriticalFlowTransmissionTime {
            link_busy_ns: timing.route_busy_ns,
            link_wait_ns: timing.queue_wait_ns,
        },
    }
}

fn previous_bottleneck_flow(
    flows: &[crate::schedule::Flow],
    timings: &[FlowTiming],
    flow_id: usize,
) -> Option<PreviousBottleneckFlow> {
    let flow = flows.get(flow_id)?;
    flow.recv_parents
        .iter()
        .copied()
        .filter_map(|parent_id| {
            timings
                .get(parent_id)
                .and_then(|timing| timing.recv_complete_ns)
                .map(|release_ns| PreviousBottleneckFlow {
                    flow_id: parent_id,
                    release_ns,
                })
        })
        .chain(flow.send_parents.iter().copied().filter_map(|parent_id| {
            timings
                .get(parent_id)
                .and_then(|timing| timing.send_complete_ns)
                .map(|release_ns| PreviousBottleneckFlow {
                    flow_id: parent_id,
                    release_ns,
                })
        }))
        .max_by_key(|parent| parent.release_ns)
}

fn fixed_scheduling_delay_ns() -> u64 {
    SIMAI_ENDPOINT_EVENT_DELAY_NS.saturating_add(SIMAI_RDMA_APP_START_DELAY_NS)
}

fn source_cross_epoch_pressure(
    config: &SimConfig,
    flows: &[crate::schedule::Flow],
    flow_size: u64,
) -> (usize, u64) {
    // Estimate extra delay when cross-host fanout crosses epoch boundaries.
    let mut cross_fanout_by_source_epoch: HashMap<(u64, usize), usize> = HashMap::new();
    let mut cross_host_flows = 0_usize;
    let mut same_host_flows = 0_usize;
    for flow in flows {
        if flow.src / config.hosts.gpus_per_host == flow.dst / config.hosts.gpus_per_host {
            same_host_flows += 1;
            continue;
        }
        cross_host_flows += 1;
        *cross_fanout_by_source_epoch
            .entry((flow.epoch, flow.src))
            .or_default() += 1;
    }

    let max_fanout = cross_fanout_by_source_epoch
        .values()
        .copied()
        .max()
        .unwrap_or(0);
    if max_fanout == 0 {
        return (0, 0);
    }
    if cross_host_flows > same_host_flows {
        return (max_fanout, 0);
    }

    let cross_bottleneck_bps =
        spec_to_bps(config.nic.bw_mbpus).min(spec_to_bps(config.net.bw_mbpus));
    let flow_bytes = flow_size.saturating_add(RDMA_DATA_HEADER_BYTES_ON_WIRE);
    let per_flow_ns = serialization_duration_ns(flow_bytes, cross_bottleneck_bps);
    (max_fanout, per_flow_ns.saturating_mul(max_fanout as u64))
}

fn release_parent(
    child: usize,
    parent_finish_ns: u64,
    parent_max_finish: &mut [u64],
    remaining_parents: &mut [usize],
    flow_timings: &mut [FlowTiming],
    events: &mut BinaryHeap<Event>,
    next_event_seq: &mut u64,
) {
    parent_max_finish[child] = parent_max_finish[child].max(parent_finish_ns);
    flow_timings[child].parent_release_ns =
        flow_timings[child].parent_release_ns.max(parent_finish_ns);
    remaining_parents[child] -= 1;
    if remaining_parents[child] == 0 {
        push_event(
            events,
            next_event_seq,
            parent_max_finish[child] + SIMAI_ENDPOINT_EVENT_DELAY_NS,
            EventKind::StartFlow { flow_id: child },
        );
    }
}

fn push_event(
    events: &mut BinaryHeap<Event>,
    next_event_seq: &mut u64,
    time_ns: u64,
    kind: EventKind,
) {
    let seq = *next_event_seq;
    *next_event_seq += 1;
    events.push(Event { time_ns, seq, kind });
}

fn try_start_link(
    link_id: usize,
    now_ns: u64,
    links: &mut [LinkRuntime],
    flows: &[crate::schedule::Flow],
    routes: &[FlowRoutes],
    flow_timings: &mut [FlowTiming],
    events: &mut BinaryHeap<Event>,
    next_event_seq: &mut u64,
) {
    let link = &mut links[link_id];
    if link.active || link.busy_until_ns > now_ns {
        return;
    }

    let Some(queued) = pop_next_packet(link) else {
        link.active = false;
        return;
    };

    let bytes = match queued.packet.kind {
        PacketKind::Data => flows[queued.packet.flow_id]
            .size_bytes
            .saturating_add(RDMA_DATA_HEADER_BYTES_ON_WIRE),
        PacketKind::Ack => RDMA_ACK_BYTES_ON_WIRE,
    };
    let start_ns = now_ns.max(link.busy_until_ns);
    let wait = start_ns - queued.arrival_ns;
    let serialization_ns = serialization_duration_ns(bytes, link.bandwidth_bps);
    let tx_finish_ns = start_ns + serialization_ns;
    let hop_finish_ns = tx_finish_ns + link.latency_ns;

    link.active = true;
    link.busy_until_ns = tx_finish_ns;
    link.transmissions += 1;
    link.busy_ns += serialization_ns;
    link.queue_wait_ns += wait;
    link.last_finish_ns = hop_finish_ns;
    if queued.packet.kind == PacketKind::Data {
        let timing = &mut flow_timings[queued.packet.flow_id];
        timing.queue_wait_ns = timing.queue_wait_ns.saturating_add(wait);
        timing.route_busy_ns = timing.route_busy_ns.saturating_add(serialization_ns);
        timing.route_latency_ns = timing.route_latency_ns.saturating_add(link.latency_ns);
    }

    push_event(
        events,
        next_event_seq,
        tx_finish_ns,
        EventKind::LinkReady { link_id },
    );

    complete_packet_transmission(
        queued.packet,
        tx_finish_ns,
        hop_finish_ns,
        routes,
        events,
        next_event_seq,
    );

    link.active = false;
    if !link.ack_queue.is_empty() || has_pending_data(link) {
        push_event(
            events,
            next_event_seq,
            tx_finish_ns,
            EventKind::LinkReady { link_id },
        );
    }
}

fn complete_packet_transmission(
    packet: Packet,
    tx_finish_ns: u64,
    hop_finish_ns: u64,
    routes: &[FlowRoutes],
    events: &mut BinaryHeap<Event>,
    next_event_seq: &mut u64,
) {
    match packet.kind {
        PacketKind::Data => {
            // Send completion is recorded at the end of the first hop.
            if packet.hop_index == 0 {
                push_event(
                    events,
                    next_event_seq,
                    tx_finish_ns,
                    EventKind::FlowSendComplete {
                        flow_id: packet.flow_id,
                    },
                );
            }
            let route = &routes[packet.flow_id].forward;
            let next_hop = packet.hop_index + 1;
            if next_hop < route.len() {
                push_event(
                    events,
                    next_event_seq,
                    hop_finish_ns,
                    EventKind::LinkArrive {
                        link_id: route[next_hop],
                        packet: Packet {
                            hop_index: next_hop,
                            ..packet
                        },
                    },
                );
            } else {
                // Destination reached: mark receive complete and enqueue ACK on reverse route.
                push_event(
                    events,
                    next_event_seq,
                    hop_finish_ns,
                    EventKind::FlowRecvComplete {
                        flow_id: packet.flow_id,
                    },
                );
                let ack_route = &routes[packet.flow_id].ack;
                if let Some(first_ack_link) = ack_route.first() {
                    push_event(
                        events,
                        next_event_seq,
                        hop_finish_ns,
                        EventKind::LinkArrive {
                            link_id: *first_ack_link,
                            packet: Packet {
                                flow_id: packet.flow_id,
                                kind: PacketKind::Ack,
                                hop_index: 0,
                            },
                        },
                    );
                }
            }
        }
        PacketKind::Ack => {
            let route = &routes[packet.flow_id].ack;
            let next_hop = packet.hop_index + 1;
            if next_hop < route.len() {
                push_event(
                    events,
                    next_event_seq,
                    hop_finish_ns,
                    EventKind::LinkArrive {
                        link_id: route[next_hop],
                        packet: Packet {
                            hop_index: next_hop,
                            ..packet
                        },
                    },
                );
            }
        }
    }
}

fn release_packet_head(
    packet: Packet,
    head_finish_ns: u64,
    routes: &[FlowRoutes],
    events: &mut BinaryHeap<Event>,
    next_event_seq: &mut u64,
) {
    // Head release allows the next hop to start before full tail completion.
    if packet.kind != PacketKind::Data {
        return;
    }
    let route = &routes[packet.flow_id].forward;
    let next_hop = packet.hop_index + 1;
    if next_hop < route.len() {
        push_event(
            events,
            next_event_seq,
            head_finish_ns,
            EventKind::LinkArrive {
                link_id: route[next_hop],
                packet: Packet {
                    hop_index: next_hop,
                    ..packet
                },
            },
        );
    }
}

fn complete_packet_tail(
    packet: Packet,
    tx_finish_ns: u64,
    tail_finish_ns: u64,
    head_already_released: bool,
    routes: &[FlowRoutes],
    events: &mut BinaryHeap<Event>,
    next_event_seq: &mut u64,
) {
    match packet.kind {
        PacketKind::Data => {
            // Tail completion gates recv completion and ACK generation.
            if packet.hop_index == 0 {
                push_event(
                    events,
                    next_event_seq,
                    tx_finish_ns,
                    EventKind::FlowSendComplete {
                        flow_id: packet.flow_id,
                    },
                );
            }
            let route = &routes[packet.flow_id].forward;
            let next_hop = packet.hop_index + 1;
            if next_hop < route.len() {
                if !head_already_released {
                    release_packet_head(packet, tail_finish_ns, routes, events, next_event_seq);
                }
            } else {
                push_event(
                    events,
                    next_event_seq,
                    tail_finish_ns,
                    EventKind::FlowRecvComplete {
                        flow_id: packet.flow_id,
                    },
                );
                let ack_route = &routes[packet.flow_id].ack;
                if let Some(first_ack_link) = ack_route.first() {
                    push_event(
                        events,
                        next_event_seq,
                        tail_finish_ns,
                        EventKind::LinkArrive {
                            link_id: *first_ack_link,
                            packet: Packet {
                                flow_id: packet.flow_id,
                                kind: PacketKind::Ack,
                                hop_index: 0,
                            },
                        },
                    );
                }
            }
        }
        PacketKind::Ack => complete_packet_transmission(
            packet,
            tx_finish_ns,
            tail_finish_ns,
            routes,
            events,
            next_event_seq,
        ),
    }
}

fn enqueue_rr_data(
    link: &mut LinkRuntime,
    flows: &[crate::schedule::Flow],
    queued: QueuedPacket,
    now_ns: u64,
) {
    // RR model: active flows share bandwidth; host path releases a packet head early.
    advance_rr_data(link, now_ns);
    let bytes = flows[queued.packet.flow_id]
        .size_bytes
        .saturating_add(RDMA_DATA_HEADER_BYTES_ON_WIRE);
    let head_remaining_bytes = if link.path_kind == LinkPathKind::Host {
        Some(
            flows[queued.packet.flow_id]
                .size_bytes
                .min(RDMA_PACKET_PAYLOAD_BYTES)
                .saturating_add(RDMA_DATA_HEADER_BYTES_ON_WIRE) as f64,
        )
    } else {
        None
    };
    link.rr_active_data.insert(
        queued.packet.flow_id,
        ActiveData {
            packet: queued.packet,
            arrival_ns: queued.arrival_ns,
            total_bytes: bytes,
            head_remaining_bytes,
            head_released: false,
            remaining_bytes: bytes as f64,
        },
    );
}

fn complete_rr_data(
    link_id: usize,
    generation: u64,
    now_ns: u64,
    links: &mut [LinkRuntime],
    routes: &[FlowRoutes],
    flow_timings: &mut [FlowTiming],
    events: &mut BinaryHeap<Event>,
    next_event_seq: &mut u64,
) {
    let link = &mut links[link_id];
    if link.data_queue_mode != DataQueueMode::RoundRobinQp || generation != link.rr_generation {
        return;
    }
    // Advance RR service, then release any heads and complete any tails that finished.
    advance_rr_data(link, now_ns);
    let mut head_ready_packets = Vec::new();
    for active in link.rr_active_data.values_mut() {
        if !active.head_released
            && active
                .head_remaining_bytes
                .is_some_and(|remaining| remaining <= 1e-6)
        {
            active.head_released = true;
            head_ready_packets.push(active.packet);
        }
    }
    let completed_ids: Vec<_> = link
        .rr_active_data
        .iter()
        .filter_map(|(flow_id, active)| {
            if active.remaining_bytes <= 1e-6 {
                Some(*flow_id)
            } else {
                None
            }
        })
        .collect();
    if head_ready_packets.is_empty() && completed_ids.is_empty() {
        schedule_rr_data_completion(link_id, now_ns, links, events, next_event_seq);
        return;
    }

    let latency_ns = link.latency_ns;
    let bandwidth_bps = link.bandwidth_bps;
    for packet in head_ready_packets {
        release_packet_head(packet, now_ns + latency_ns, routes, events, next_event_seq);
    }
    for flow_id in completed_ids {
        let active = link
            .rr_active_data
            .remove(&flow_id)
            .expect("completed RR data flow must still be active");
        let serialized_alone_ns = serialization_duration_ns(active.total_bytes, bandwidth_bps);
        let queue_wait_ns = now_ns
            .saturating_sub(active.arrival_ns)
            .saturating_sub(serialized_alone_ns);
        let hop_finish_ns = now_ns + latency_ns;
        link.transmissions += 1;
        link.queue_wait_ns = link.queue_wait_ns.saturating_add(queue_wait_ns);
        link.last_finish_ns = link.last_finish_ns.max(hop_finish_ns);
        let timing = &mut flow_timings[active.packet.flow_id];
        timing.queue_wait_ns = timing.queue_wait_ns.saturating_add(queue_wait_ns);
        timing.route_busy_ns = timing.route_busy_ns.saturating_add(serialized_alone_ns);
        timing.route_latency_ns = timing.route_latency_ns.saturating_add(latency_ns);
        complete_packet_tail(
            active.packet,
            now_ns,
            hop_finish_ns,
            active.head_released,
            routes,
            events,
            next_event_seq,
        );
    }
    schedule_rr_data_completion(link_id, now_ns, links, events, next_event_seq);
}

fn try_start_rr_ack(
    link_id: usize,
    now_ns: u64,
    links: &mut [LinkRuntime],
    routes: &[FlowRoutes],
    _flow_timings: &mut [FlowTiming],
    events: &mut BinaryHeap<Event>,
    next_event_seq: &mut u64,
) {
    // ACKs consume link time and update RR accounting on the same link.
    let link = &mut links[link_id];
    if link.active || link.busy_until_ns > now_ns {
        return;
    }
    let Some(queued) = link.ack_queue.pop_front() else {
        schedule_rr_data_completion(link_id, now_ns, links, events, next_event_seq);
        return;
    };

    advance_rr_data(link, now_ns);
    let serialization_ns = serialization_duration_ns(RDMA_ACK_BYTES_ON_WIRE, link.bandwidth_bps);
    let tx_finish_ns = now_ns + serialization_ns;
    let hop_finish_ns = tx_finish_ns + link.latency_ns;
    let wait = now_ns - queued.arrival_ns;

    link.active = true;
    link.busy_until_ns = tx_finish_ns;
    link.rr_data_last_update_ns = tx_finish_ns;
    link.transmissions += 1;
    link.busy_ns += serialization_ns;
    link.queue_wait_ns += wait;
    link.last_finish_ns = link.last_finish_ns.max(hop_finish_ns);

    complete_packet_transmission(
        queued.packet,
        tx_finish_ns,
        hop_finish_ns,
        routes,
        events,
        next_event_seq,
    );

    link.active = false;
    if !link.ack_queue.is_empty() || !link.data_queues.is_empty() {
        push_event(
            events,
            next_event_seq,
            tx_finish_ns,
            EventKind::LinkReady { link_id },
        );
    }
    schedule_rr_data_completion(link_id, tx_finish_ns, links, events, next_event_seq);
}

fn advance_rr_data(link: &mut LinkRuntime, now_ns: u64) {
    // Equal-share model: distribute elapsed service time across active flows.
    if link.rr_active_data.is_empty() {
        link.rr_data_last_update_ns = link.rr_data_last_update_ns.max(now_ns);
        return;
    }
    if now_ns <= link.rr_data_last_update_ns {
        return;
    }
    let elapsed_ns = now_ns - link.rr_data_last_update_ns;
    let active_count = link.rr_active_data.len() as f64;
    let delivered_per_flow =
        elapsed_ns as f64 * link.bandwidth_bps as f64 / 8_000_000_000.0 / active_count;
    for active in link.rr_active_data.values_mut() {
        active.remaining_bytes = (active.remaining_bytes - delivered_per_flow).max(0.0);
        if !active.head_released {
            if let Some(remaining) = active.head_remaining_bytes.as_mut() {
                *remaining = (*remaining - delivered_per_flow).max(0.0);
            }
        }
    }
    link.busy_ns = link.busy_ns.saturating_add(elapsed_ns);
    link.rr_data_last_update_ns = now_ns;
}

fn schedule_rr_data_completion(
    link_id: usize,
    now_ns: u64,
    links: &mut [LinkRuntime],
    events: &mut BinaryHeap<Event>,
    next_event_seq: &mut u64,
) {
    // Schedule the next RR completion based on the smallest remaining head/tail.
    let link = &mut links[link_id];
    if link.data_queue_mode != DataQueueMode::RoundRobinQp || link.rr_active_data.is_empty() {
        return;
    }
    let service_start_ns = link.rr_data_last_update_ns.max(now_ns);
    let min_remaining = link
        .rr_active_data
        .values()
        .map(|active| match active.head_remaining_bytes {
            Some(head_remaining) if !active.head_released => {
                active.remaining_bytes.min(head_remaining)
            }
            _ => active.remaining_bytes,
        })
        .fold(f64::INFINITY, f64::min);
    let active_count = link.rr_active_data.len() as u64;
    let bytes_until_completion = ((min_remaining * active_count as f64).ceil() as u64).max(1);
    let duration_ns = serialization_duration_ns(bytes_until_completion, link.bandwidth_bps);
    link.rr_generation += 1;
    push_event(
        events,
        next_event_seq,
        service_start_ns + duration_ns,
        EventKind::RrDataComplete {
            link_id,
            generation: link.rr_generation,
        },
    );
}

fn pop_next_packet(link: &mut LinkRuntime) -> Option<QueuedPacket> {
    // ACKs are always dequeued before data packets.
    if let Some(ack) = link.ack_queue.pop_front() {
        return Some(ack);
    }
    match link.data_queue_mode {
        DataQueueMode::Fifo => link.data_fifo.pop_front(),
        DataQueueMode::RoundRobinQp => pop_next_data_qp_packet(link),
    }
}

fn has_pending_data(link: &LinkRuntime) -> bool {
    match link.data_queue_mode {
        DataQueueMode::Fifo => !link.data_fifo.is_empty(),
        DataQueueMode::RoundRobinQp => !link.rr_active_data.is_empty(),
    }
}

fn pop_next_data_qp_packet(link: &mut LinkRuntime) -> Option<QueuedPacket> {
    // Round-robin across per-flow queues for fairness.
    let selected_flow_id = match link.rr_last_flow_id {
        Some(last) => link
            .data_queues
            .range((last + 1)..)
            .next()
            .map(|(flow_id, _)| *flow_id)
            .or_else(|| link.data_queues.keys().next().copied()),
        None => link.data_queues.keys().next().copied(),
    }?;
    let queue = link
        .data_queues
        .get_mut(&selected_flow_id)
        .expect("selected data QP must exist");
    let packet = queue.pop_front();
    if queue.is_empty() {
        link.data_queues.remove(&selected_flow_id);
    }
    link.rr_last_flow_id = Some(selected_flow_id);
    packet
}

fn serialization_duration_ns(bytes: u64, bandwidth_bps: u64) -> u64 {
    // Convert bytes and bandwidth into a serialization duration.
    if bytes == 0 {
        return 0;
    }
    let bits = (bytes as u128) * 8;
    let serialized = bits * 1_000_000_000_u128;
    serialized.div_ceil(bandwidth_bps as u128) as u64
}

pub(crate) fn make_link_runtime(
    bandwidth_bps: u64,
    latency_ns: u64,
    data_queue_mode: DataQueueMode,
    path_kind: LinkPathKind,
) -> LinkRuntime {
    LinkRuntime {
        bandwidth_bps,
        latency_ns,
        busy_until_ns: 0,
        active: false,
        data_queue_mode,
        path_kind,
        data_fifo: VecDeque::new(),
        data_queues: BTreeMap::new(),
        rr_active_data: BTreeMap::new(),
        rr_data_last_update_ns: 0,
        rr_generation: 0,
        ack_queue: VecDeque::new(),
        rr_last_flow_id: None,
        transmissions: 0,
        busy_ns: 0,
        queue_wait_ns: 0,
        last_finish_ns: 0,
    }
}
