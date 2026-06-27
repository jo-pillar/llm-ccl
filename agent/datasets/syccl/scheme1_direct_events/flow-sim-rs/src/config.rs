use std::collections::{BTreeSet, HashMap};
use std::io::{BufReader, Read};

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
struct RawConfig {
    coll: RawCollective,
    hosts: RawHosts,
    topo: Vec<RawTopoLayer>,
    link_spec: HashMap<String, LinkSpec>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawCollective {
    name: String,
    byte: u64,
}

#[derive(Debug, Clone, Deserialize)]
struct RawHosts {
    host_num: usize,
    host_gpu_num: usize,
    host_nic_num: usize,
    host_links: String,
}

#[derive(Debug, Clone, Deserialize)]
struct RawTopoLayer {
    #[allow(dead_code)]
    layer_id: usize,
    #[serde(rename = "type")]
    layer_type: String,
    link_spec: Option<String>,
    switch_topo: Option<String>,
    switch_num: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerGroup {
    pub layer_id: usize,
    pub group_id: usize,
    pub members: BTreeSet<usize>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct LinkSpec {
    pub bw_mbpus: f64,
    pub lat_us: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hosts {
    pub host_num: usize,
    pub gpus_per_host: usize,
    pub nics_per_host: usize,
    pub host_links: String,
}

#[derive(Debug, Clone)]
pub struct SimConfig {
    pub coll_name: String,
    pub coll_bytes: u64,
    pub hosts: Hosts,
    pub topology: TopologyKind,
    pub rail_count: usize,
    pub memcpy: LinkSpec,
    pub host: LinkSpec,
    pub nic: LinkSpec,
    pub net: LinkSpec,
    pub spine: Option<LinkSpec>,
    pub layer_groups: HashMap<usize, HashMap<usize, BTreeSet<usize>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyKind {
    Multirail,
    Clos,
}

pub fn parse_config<R: Read>(reader: R) -> Result<SimConfig> {
    let raw: RawConfig = serde_json::from_reader(BufReader::new(reader))
        .context("failed to parse SYCCL config JSON")?;
    if raw.coll.name != "allgather" && raw.coll.name != "alltoall" {
        return Err(anyhow!(
            "ConfigError: unsupported collective {}, flow-sim supports allgather and alltoall",
            raw.coll.name
        ));
    }
    if raw.hosts.host_num == 0 || raw.hosts.host_gpu_num == 0 {
        return Err(anyhow!(
            "ConfigError: host_num and host_gpu_num must be positive"
        ));
    }
    if raw.hosts.host_nic_num == 0 {
        return Err(anyhow!("ConfigError: host_nic_num must be positive"));
    }
    if raw.hosts.host_gpu_num % raw.hosts.host_nic_num != 0 {
        return Err(anyhow!(
            "ConfigError: host_gpu_num must be divisible by host_nic_num, got {} GPUs and {} NICs per host",
            raw.hosts.host_gpu_num,
            raw.hosts.host_nic_num
        ));
    }

    let host_layer = raw
        .topo
        .iter()
        .find(|layer| layer.layer_type == "host")
        .ok_or_else(|| anyhow!("ConfigError: missing host layer"))?;
    let nic_layer = raw
        .topo
        .iter()
        .find(|layer| layer.layer_type == "nic")
        .ok_or_else(|| anyhow!("ConfigError: missing nic layer"))?;
    let switch_layers: Vec<&RawTopoLayer> = raw
        .topo
        .iter()
        .filter(|layer| layer.layer_type == "switch")
        .collect();
    let first_switch = switch_layers
        .first()
        .copied()
        .ok_or_else(|| anyhow!("ConfigError: missing switch layer"))?;
    let topology = match first_switch.switch_topo.as_deref() {
        Some("multirail") => TopologyKind::Multirail,
        Some("pod") => TopologyKind::Clos,
        other => {
            return Err(anyhow!(
                "unsupported switch_topo {:?}, expected multirail or pod",
                other
            ))
        }
    };
    let rail_count = first_switch
        .switch_num
        .ok_or_else(|| anyhow!("missing switch_num on switch layer"))?;
    if topology == TopologyKind::Multirail && rail_count != raw.hosts.host_nic_num {
        return Err(anyhow!(
            "ConfigError: switch_num must match host_nic_num for multirail, got {} and {}",
            rail_count,
            raw.hosts.host_nic_num
        ));
    }

    let memcpy = find_spec(&raw.link_spec, "memcpy")?;
    let host = find_spec(
        &raw.link_spec,
        host_layer.link_spec.as_deref().unwrap_or("nvlink"),
    )?;
    let nic = find_spec(
        &raw.link_spec,
        nic_layer.link_spec.as_deref().unwrap_or("link_nic"),
    )?;
    let net = find_spec(
        &raw.link_spec,
        first_switch.link_spec.as_deref().unwrap_or("netlink"),
    )?;
    let spine = switch_layers
        .get(1)
        .map(|layer| {
            find_spec(
                &raw.link_spec,
                layer.link_spec.as_deref().unwrap_or("netlink"),
            )
        })
        .transpose()?;

    let layer_groups = derive_layer_groups(&raw)?;

    Ok(SimConfig {
        coll_name: raw.coll.name,
        coll_bytes: raw.coll.byte,
        topology,
        hosts: Hosts {
            host_num: raw.hosts.host_num,
            gpus_per_host: raw.hosts.host_gpu_num,
            nics_per_host: raw.hosts.host_nic_num,
            host_links: raw.hosts.host_links,
        },
        rail_count,
        memcpy,
        host,
        nic,
        net,
        spine,
        layer_groups,
    })
}

fn derive_layer_groups(raw: &RawConfig) -> Result<HashMap<usize, HashMap<usize, BTreeSet<usize>>>> {
    let mut groups = HashMap::new();
    let mut prev_layer_type = String::new();
    let mut prev_layer_id = None;
    for layer in &raw.topo {
        if layer.layer_type == "host" {
            let mut host_groups = HashMap::new();
            for host in 0..raw.hosts.host_num {
                host_groups.insert(host, host_range(host, raw.hosts.host_gpu_num));
            }
            groups.insert(layer.layer_id, host_groups);
        } else if layer.layer_type == "switch" {
            let switch_num = layer
                .switch_num
                .ok_or_else(|| anyhow!("missing switch_num on switch layer {}", layer.layer_id))?;
            match (prev_layer_type.as_str(), layer.switch_topo.as_deref()) {
                ("nic", Some("multirail")) => {
                    groups.insert(
                        layer.layer_id,
                        derive_multirail_groups(
                            raw.hosts.host_num,
                            raw.hosts.host_gpu_num,
                            raw.hosts.host_nic_num,
                            switch_num,
                        )?,
                    );
                }
                ("nic", Some("pod")) => {
                    groups.insert(
                        layer.layer_id,
                        derive_first_pod_switch_groups(
                            raw.hosts.host_num,
                            raw.hosts.host_gpu_num,
                            switch_num,
                        )?,
                    );
                }
                ("switch", Some("pod")) => {
                    if let Some(prev_id) = prev_layer_id {
                        if let Some(lower_groups) = groups.get(&prev_id) {
                            groups.insert(
                                layer.layer_id,
                                derive_upper_pod_switch_groups(lower_groups, switch_num)?,
                            );
                        }
                    }
                }
                _ => {}
            }
        }
        prev_layer_type = layer.layer_type.clone();
        prev_layer_id = Some(layer.layer_id);
    }
    Ok(groups)
}

fn host_range(host: usize, gpus_per_host: usize) -> BTreeSet<usize> {
    let start = host * gpus_per_host;
    (start..start + gpus_per_host).collect()
}

fn derive_multirail_groups(
    host_num: usize,
    host_gpu_num: usize,
    host_nic_num: usize,
    switch_num: usize,
) -> Result<HashMap<usize, BTreeSet<usize>>> {
    if host_gpu_num % host_nic_num != 0 {
        return Err(anyhow!(
            "host_gpu_num must be divisible by host_nic_num for multirail groups"
        ));
    }
    if host_nic_num % switch_num != 0 {
        return Err(anyhow!(
            "host_nic_num must be divisible by multirail switch_num"
        ));
    }
    let gpu_per_nic = host_gpu_num / host_nic_num;
    let nics_per_switch = host_nic_num / switch_num;
    let mut groups = HashMap::new();
    for switch_id in 0..switch_num {
        let mut members = BTreeSet::new();
        let nic_begin = switch_id * nics_per_switch;
        let nic_end = (switch_id + 1) * nics_per_switch;
        for host in 0..host_num {
            let host_base = host * host_gpu_num;
            for nic in nic_begin..nic_end {
                let local_begin = nic * gpu_per_nic;
                let local_end = local_begin + gpu_per_nic;
                members.extend(host_base + local_begin..host_base + local_end);
            }
        }
        groups.insert(switch_id, members);
    }
    Ok(groups)
}

fn derive_first_pod_switch_groups(
    host_num: usize,
    host_gpu_num: usize,
    switch_num: usize,
) -> Result<HashMap<usize, BTreeSet<usize>>> {
    if host_num % switch_num != 0 {
        return Err(anyhow!("host_num must be divisible by pod switch_num"));
    }
    let hosts_per_switch = host_num / switch_num;
    let mut groups = HashMap::new();
    for switch_id in 0..switch_num {
        let mut members = BTreeSet::new();
        for host in switch_id * hosts_per_switch..(switch_id + 1) * hosts_per_switch {
            members.extend(host_range(host, host_gpu_num));
        }
        groups.insert(switch_id, members);
    }
    Ok(groups)
}

fn derive_upper_pod_switch_groups(
    lower_groups: &HashMap<usize, BTreeSet<usize>>,
    switch_num: usize,
) -> Result<HashMap<usize, BTreeSet<usize>>> {
    if lower_groups.len() % switch_num != 0 {
        return Err(anyhow!(
            "lower pod switch count must be divisible by upper switch_num"
        ));
    }
    let lower_per_switch = lower_groups.len() / switch_num;
    let mut ordered_lower: Vec<_> = lower_groups.iter().collect();
    ordered_lower.sort_by_key(|(group_id, _)| **group_id);
    let mut groups = HashMap::new();
    for switch_id in 0..switch_num {
        let mut members = BTreeSet::new();
        for (_, group_members) in
            &ordered_lower[switch_id * lower_per_switch..(switch_id + 1) * lower_per_switch]
        {
            members.extend(group_members.iter().copied());
        }
        groups.insert(switch_id, members);
    }
    Ok(groups)
}

fn find_spec(specs: &HashMap<String, LinkSpec>, name: &str) -> Result<LinkSpec> {
    specs
        .get(name)
        .copied()
        .ok_or_else(|| anyhow!("missing link_spec {}", name))
}

pub fn spec_to_bps(bw_mbpus: f64) -> u64 {
    (bw_mbpus * 8.0 * 1024.0 * 1024.0 * 1_000_000.0).round() as u64
}

pub fn spec_to_ns(lat_us: f64) -> u64 {
    (lat_us * 1000.0).round() as u64
}
