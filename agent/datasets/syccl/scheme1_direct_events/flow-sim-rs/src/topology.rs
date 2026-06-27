use std::collections::HashMap;

use anyhow::{anyhow, Result};

use crate::config::{spec_to_bps, spec_to_ns, SimConfig, TopologyKind};
use crate::simulator::{make_link_runtime, DataQueueMode, LinkKey, LinkPathKind, RouteKind};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Route {
    pub kind: RouteKind,
    pub links: Vec<LinkKey>,
}

#[derive(Debug, Clone)]
pub struct Topology {
    gpus_per_host: usize,
    host_count: usize,
    nics_per_host: usize,
    rail_count: usize,
    gpu_count: usize,
    nvswitch_base: usize,
    nic_base: usize,
    rail_base: usize,
    spine_base: usize,
    topology: TopologyKind,
    links: Vec<(LinkKey, u64, u64, DataQueueMode, LinkPathKind)>,
    link_ids: HashMap<LinkKey, usize>,
}

impl Topology {
    pub fn from_config(config: &SimConfig) -> Result<Self> {
        let gpu_count = config.hosts.host_num * config.hosts.gpus_per_host;
        if gpu_count == 0 {
            return Err(anyhow!("empty topology"));
        }
        let nvswitch_base = gpu_count;
        let nic_base = nvswitch_base + config.hosts.host_num;
        let rail_base = nic_base + config.hosts.host_num * config.hosts.nics_per_host;
        let spine_base = rail_base + config.rail_count;
        let mut topo = Self {
            gpus_per_host: config.hosts.gpus_per_host,
            host_count: config.hosts.host_num,
            nics_per_host: config.hosts.nics_per_host,
            rail_count: config.rail_count,
            gpu_count,
            nvswitch_base,
            nic_base,
            rail_base,
            spine_base,
            topology: config.topology,
            links: Vec::new(),
            link_ids: HashMap::new(),
        };

        let host_bw = spec_to_bps(config.host.bw_mbpus);
        let host_lat = spec_to_ns(config.host.lat_us);
        let nic_bw = spec_to_bps(config.nic.bw_mbpus);
        let nic_lat = spec_to_ns(config.nic.lat_us);
        let net_bw = spec_to_bps(config.net.bw_mbpus);
        let net_lat = spec_to_ns(config.net.lat_us);
        let spine_spec = config.spine.unwrap_or(config.net);
        let spine_bw = spec_to_bps(spine_spec.bw_mbpus);
        let spine_lat = spec_to_ns(spine_spec.lat_us);

        for host in 0..config.hosts.host_num {
            let nvswitch = topo.nvswitch_id(host);
            for nic_index in 0..config.hosts.nics_per_host {
                let nic = topo.nic_id(host, nic_index);
                let leaf = topo.leaf_id_for_host_nic(host, nic_index);
                topo.add_bidirectional(
                    nic,
                    leaf,
                    net_bw,
                    net_lat,
                    DataQueueMode::Fifo,
                    LinkPathKind::Net,
                );
            }
            for local_gpu in 0..config.hosts.gpus_per_host {
                let gpu = host * config.hosts.gpus_per_host + local_gpu;
                let nic = topo.nic_id(host, topo.nic_index_for_local_gpu(local_gpu));
                topo.add_bidirectional(
                    gpu,
                    nvswitch,
                    host_bw,
                    host_lat,
                    DataQueueMode::RoundRobinQp,
                    LinkPathKind::Host,
                );
                topo.add_bidirectional(
                    gpu,
                    nic,
                    nic_bw,
                    nic_lat,
                    DataQueueMode::RoundRobinQp,
                    LinkPathKind::Nic,
                );
            }
        }
        if config.topology == TopologyKind::Clos {
            for leaf in 0..config.rail_count {
                let leaf_id = topo.rail_id(leaf);
                let spine_id = topo.spine_id(0);
                topo.add_bidirectional(
                    leaf_id,
                    spine_id,
                    spine_bw,
                    spine_lat,
                    DataQueueMode::Fifo,
                    LinkPathKind::Net,
                );
            }
        }

        Ok(topo)
    }

    pub fn route(&self, src_gpu: usize, dst_gpu: usize) -> Result<Route> {
        if src_gpu >= self.gpu_count || dst_gpu >= self.gpu_count {
            return Err(anyhow!(
                "GPU route out of range: {} -> {}",
                src_gpu,
                dst_gpu
            ));
        }
        if src_gpu == dst_gpu {
            return Ok(Route {
                kind: RouteKind::SameHost,
                links: Vec::new(),
            });
        }

        let src_host = src_gpu / self.gpus_per_host;
        let dst_host = dst_gpu / self.gpus_per_host;
        // Same-host routes go via NVSwitch; cross-host routes go via NIC and rail.
        if src_host == dst_host {
            let nvswitch = self.nvswitch_id(src_host);
            Ok(Route {
                kind: RouteKind::SameHost,
                links: vec![
                    LinkKey::new(src_gpu, nvswitch),
                    LinkKey::new(nvswitch, dst_gpu),
                ],
            })
        } else {
            let src_local = src_gpu % self.gpus_per_host;
            let dst_local = dst_gpu % self.gpus_per_host;
            let src_nic = self.nic_id(src_host, self.nic_index_for_local_gpu(src_local));
            let dst_nic = self.nic_id(dst_host, self.nic_index_for_local_gpu(dst_local));
            match self.topology {
                TopologyKind::Multirail => {
                    let rail = self.rail_id(self.nic_index_for_local_gpu(src_local));
                    Ok(Route {
                        kind: RouteKind::CrossHost,
                        links: vec![
                            LinkKey::new(src_gpu, src_nic),
                            LinkKey::new(src_nic, rail),
                            LinkKey::new(rail, dst_nic),
                            LinkKey::new(dst_nic, dst_gpu),
                        ],
                    })
                }
                TopologyKind::Clos => {
                    let src_leaf = self.leaf_id_for_host(src_host);
                    let dst_leaf = self.leaf_id_for_host(dst_host);
                    let mut links = vec![
                        LinkKey::new(src_gpu, src_nic),
                        LinkKey::new(src_nic, src_leaf),
                    ];
                    if src_leaf != dst_leaf {
                        let spine = self.spine_id(0);
                        links.push(LinkKey::new(src_leaf, spine));
                        links.push(LinkKey::new(spine, dst_leaf));
                    }
                    links.push(LinkKey::new(dst_leaf, dst_nic));
                    links.push(LinkKey::new(dst_nic, dst_gpu));
                    Ok(Route {
                        kind: RouteKind::CrossHost,
                        links,
                    })
                }
            }
        }
    }

    pub(crate) fn route_link_ids(&self, src_gpu: usize, dst_gpu: usize) -> Result<Vec<usize>> {
        let route = self.route(src_gpu, dst_gpu)?;
        route
            .links
            .iter()
            .map(|key| {
                self.link_ids.get(key).copied().ok_or_else(|| {
                    anyhow!("route references missing link {} -> {}", key.src, key.dst)
                })
            })
            .collect()
    }

    pub(crate) fn link_runtimes(&self) -> Vec<(LinkKey, crate::simulator::LinkRuntime)> {
        self.links
            .iter()
            .map(|(key, bw, lat, mode, path_kind)| {
                (*key, make_link_runtime(*bw, *lat, *mode, *path_kind))
            })
            .collect()
    }

    fn add_bidirectional(
        &mut self,
        a: usize,
        b: usize,
        bandwidth_bps: u64,
        latency_ns: u64,
        data_queue_mode: DataQueueMode,
        path_kind: LinkPathKind,
    ) {
        self.add_directed(a, b, bandwidth_bps, latency_ns, data_queue_mode, path_kind);
        self.add_directed(b, a, bandwidth_bps, latency_ns, data_queue_mode, path_kind);
    }

    fn add_directed(
        &mut self,
        src: usize,
        dst: usize,
        bandwidth_bps: u64,
        latency_ns: u64,
        data_queue_mode: DataQueueMode,
        path_kind: LinkPathKind,
    ) {
        let key = LinkKey::new(src, dst);
        if self.link_ids.contains_key(&key) {
            return;
        }
        let id = self.links.len();
        self.links
            .push((key, bandwidth_bps, latency_ns, data_queue_mode, path_kind));
        self.link_ids.insert(key, id);
    }

    fn nvswitch_id(&self, host: usize) -> usize {
        self.nvswitch_base + host
    }

    fn nic_id(&self, host: usize, local_gpu: usize) -> usize {
        self.nic_base + host * self.nics_per_host + local_gpu
    }

    fn rail_id(&self, local_gpu: usize) -> usize {
        self.rail_base + local_gpu
    }

    fn spine_id(&self, spine: usize) -> usize {
        self.spine_base + spine
    }

    fn nic_index_for_local_gpu(&self, local_gpu: usize) -> usize {
        local_gpu * self.nics_per_host / self.gpus_per_host
    }

    fn leaf_id_for_host_nic(&self, host: usize, nic_index: usize) -> usize {
        match self.topology {
            TopologyKind::Multirail => self.rail_id(nic_index),
            TopologyKind::Clos => self.leaf_id_for_host(host),
        }
    }

    fn leaf_id_for_host(&self, host: usize) -> usize {
        match self.topology {
            TopologyKind::Multirail => self.rail_id(host % self.nics_per_host),
            TopologyKind::Clos => {
                let hosts_per_leaf = (self.host_count + self.rail_count - 1) / self.rail_count;
                self.rail_id(host / hosts_per_leaf)
            }
        }
    }
}
