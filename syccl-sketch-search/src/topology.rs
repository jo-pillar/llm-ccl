use crate::config::SycclConfig;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
pub struct Topology {
    num_gpus: usize,
    layers: Vec<LayerInfo>,
}

impl Topology {
    pub fn from_syccl_config(config: &SycclConfig) -> anyhow::Result<Self> {
        let host_num = config.hosts.host_num;
        let gpus_per_host = config.hosts.host_gpu_num;
        let nics_per_host = config.hosts.host_nic_num;
        let num_gpus = host_num * gpus_per_host;
        let mut layers = Vec::new();
        let mut nic_groups: Vec<Vec<usize>> = Vec::new();
        let mut previous_switch_groups: Vec<BTreeSet<usize>> = Vec::new();

        for (expected_id, layer_config) in config.topo.iter().enumerate() {
            if layer_config.layer_id != expected_id {
                anyhow::bail!(
                    "topo layer ids must be dense and ordered: expected {expected_id}, got {}",
                    layer_config.layer_id
                );
            }
            let groups = match layer_config.layer_type.as_str() {
                "local" => (0..num_gpus)
                    .map(|gpu| BTreeSet::from([gpu]))
                    .collect::<Vec<_>>(),
                "host" => host_layer_groups(config, host_num, gpus_per_host)?,
                "nic" => {
                    nic_groups = build_nic_groups(host_num, gpus_per_host, nics_per_host)?;
                    nic_groups
                        .iter()
                        .map(|group| group.iter().copied().collect())
                        .collect()
                }
                "switch" => {
                    let switch_num = layer_config.switch_num.ok_or_else(|| {
                        anyhow::anyhow!("switch layer {} missing switch_num", layer_config.layer_id)
                    })?;
                    let switch_topo = layer_config.switch_topo.as_deref().unwrap_or("pod");
                    let group_sets = if previous_switch_groups.is_empty() {
                        build_first_switch_groups(
                            switch_topo,
                            switch_num,
                            host_num,
                            nics_per_host,
                            &nic_groups,
                        )?
                    } else {
                        build_higher_switch_groups(
                            switch_topo,
                            switch_num,
                            &previous_switch_groups,
                        )?
                    };
                    previous_switch_groups = group_sets.clone();
                    group_sets
                }
                other => anyhow::bail!("unsupported topology layer type {other:?}"),
            };
            layers.push(LayerInfo::new(layer_config.layer_id, groups)?);
        }

        populate_layer_relationships(&mut layers);
        Ok(Self { num_gpus, layers })
    }

    pub fn num_gpus(&self) -> usize {
        self.num_gpus
    }

    pub fn layers(&self) -> &[LayerInfo] {
        &self.layers
    }

    pub(crate) fn layer(&self, layer: usize) -> &LayerInfo {
        &self.layers[layer]
    }
}

#[derive(Debug, Clone)]
pub struct LayerInfo {
    pub id: usize,
    pub groups: Vec<GroupInfo>,
    relationships: BTreeMap<usize, LayerRelation>,
}

impl LayerInfo {
    fn new(id: usize, groups: Vec<BTreeSet<usize>>) -> anyhow::Result<Self> {
        let mut group_infos = Vec::new();
        let mut gpu_to_group = BTreeMap::new();
        for (group_id, connected_gpus) in groups.into_iter().enumerate() {
            if connected_gpus.is_empty() {
                anyhow::bail!("layer {id} group {group_id} is empty");
            }
            for gpu in &connected_gpus {
                if gpu_to_group.insert(*gpu, group_id).is_some() {
                    anyhow::bail!("GPU {gpu} appears in multiple groups for layer {id}");
                }
            }
            group_infos.push(GroupInfo {
                id: group_id,
                connected_gpus,
            });
        }
        Ok(Self {
            id,
            groups: group_infos,
            relationships: BTreeMap::new(),
        })
    }

    pub(crate) fn relation_to(&self, other_layer: usize) -> LayerRelation {
        self.relationships
            .get(&other_layer)
            .copied()
            .unwrap_or(LayerRelation::Cross)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupInfo {
    pub id: usize,
    pub connected_gpus: BTreeSet<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum LayerRelation {
    Repeat,
    Contain,
    BeContained,
    Cross,
}

fn host_layer_groups(
    config: &SycclConfig,
    host_num: usize,
    gpus_per_host: usize,
) -> anyhow::Result<Vec<BTreeSet<usize>>> {
    match config.hosts.host_links.as_str() {
        "nvswitch" => Ok((0..host_num)
            .map(|host| {
                let start = host * gpus_per_host;
                (start..start + gpus_per_host).collect()
            })
            .collect()),
        "nvlink" => {
            if config.host_links.nvlink.len() != gpus_per_host {
                anyhow::bail!(
                    "host_links.nvlink row count {} does not match host_gpu_num {}",
                    config.host_links.nvlink.len(),
                    gpus_per_host
                );
            }
            let components = nvlink_components(&config.host_links.nvlink, gpus_per_host)?;
            let mut groups = Vec::new();
            for host in 0..host_num {
                let offset = host * gpus_per_host;
                for component in &components {
                    groups.push(component.iter().map(|gpu| gpu + offset).collect());
                }
            }
            Ok(groups)
        }
        other => anyhow::bail!("unsupported hosts.host_links value {other:?}"),
    }
}

fn nvlink_components(
    matrix: &[Vec<usize>],
    gpus_per_host: usize,
) -> anyhow::Result<Vec<BTreeSet<usize>>> {
    let mut adj = vec![BTreeSet::new(); gpus_per_host];
    for (src, dsts) in matrix.iter().enumerate() {
        for &dst in dsts {
            if dst >= gpus_per_host {
                anyhow::bail!("nvlink destination {dst} is out of host GPU range {gpus_per_host}");
            }
            if dst != src {
                adj[src].insert(dst);
                adj[dst].insert(src);
            }
        }
    }

    let mut visited = vec![false; gpus_per_host];
    let mut components = Vec::new();
    for start in 0..gpus_per_host {
        if visited[start] {
            continue;
        }
        let mut stack = vec![start];
        let mut component = BTreeSet::new();
        visited[start] = true;
        while let Some(gpu) = stack.pop() {
            component.insert(gpu);
            for &next in &adj[gpu] {
                if !visited[next] {
                    visited[next] = true;
                    stack.push(next);
                }
            }
        }
        components.push(component);
    }
    Ok(components)
}

fn build_nic_groups(
    host_num: usize,
    gpus_per_host: usize,
    nics_per_host: usize,
) -> anyhow::Result<Vec<Vec<usize>>> {
    if nics_per_host == 0 {
        anyhow::bail!("nic topology layer requires hosts.host_nic_num > 0");
    }
    if gpus_per_host % nics_per_host != 0 {
        anyhow::bail!(
            "host_gpu_num {gpus_per_host} must be divisible by host_nic_num {nics_per_host}"
        );
    }
    let gpus_per_nic = gpus_per_host / nics_per_host;
    let mut groups = Vec::new();
    for nic in 0..nics_per_host {
        for host in 0..host_num {
            let start = host * gpus_per_host + nic * gpus_per_nic;
            groups.push((start..start + gpus_per_nic).collect());
        }
    }
    Ok(groups)
}

fn build_first_switch_groups(
    switch_topo: &str,
    switch_num: usize,
    host_num: usize,
    nics_per_host: usize,
    nic_groups: &[Vec<usize>],
) -> anyhow::Result<Vec<BTreeSet<usize>>> {
    if switch_num == 0 {
        anyhow::bail!("switch_num must be non-zero");
    }
    let total_nics = nics_per_host * host_num;
    if total_nics != nic_groups.len() {
        anyhow::bail!("internal NIC group count mismatch");
    }
    if total_nics % switch_num != 0 {
        anyhow::bail!("total NICs {total_nics} must be divisible by switch_num {switch_num}");
    }
    let total_nic_per_switch = total_nics / switch_num;
    let mut result = Vec::new();

    match switch_topo {
        "multirail" => {
            if total_nic_per_switch % host_num != 0 {
                anyhow::bail!(
                    "multirail requires NICs per switch {total_nic_per_switch} divisible by hosts {host_num}"
                );
            }
            let per_switch_nics_per_host = total_nic_per_switch / host_num;
            for switch_id in 0..switch_num {
                let mut group = BTreeSet::new();
                for host in 0..host_num {
                    for nic in switch_id * per_switch_nics_per_host
                        ..(switch_id + 1) * per_switch_nics_per_host
                    {
                        let nic_group_index = nic * host_num + host;
                        group.extend(nic_groups[nic_group_index].iter().copied());
                    }
                }
                result.push(group);
            }
        }
        "pod" => {
            if total_nic_per_switch % nics_per_host != 0 {
                anyhow::bail!(
                    "pod requires NICs per switch {total_nic_per_switch} divisible by NICs per host {nics_per_host}"
                );
            }
            let hosts_per_switch = total_nic_per_switch / nics_per_host;
            let mut host = 0;
            for _switch_id in 0..switch_num {
                let mut group = BTreeSet::new();
                for _ in 0..hosts_per_switch {
                    for nic in 0..nics_per_host {
                        let nic_group_index = nic * host_num + host;
                        group.extend(nic_groups[nic_group_index].iter().copied());
                    }
                    host += 1;
                }
                result.push(group);
            }
        }
        other => anyhow::bail!("unsupported switch_topo {other:?}"),
    }
    Ok(result)
}

fn build_higher_switch_groups(
    switch_topo: &str,
    switch_num: usize,
    previous_groups: &[BTreeSet<usize>],
) -> anyhow::Result<Vec<BTreeSet<usize>>> {
    if switch_topo != "pod" {
        anyhow::bail!("higher switch layers currently support switch_topo=\"pod\" only");
    }
    if switch_num == 0 || previous_groups.len() % switch_num != 0 {
        anyhow::bail!(
            "previous switch group count {} must be divisible by switch_num {switch_num}",
            previous_groups.len()
        );
    }
    let groups_per_switch = previous_groups.len() / switch_num;
    let mut result = Vec::new();
    for switch_id in 0..switch_num {
        let mut group = BTreeSet::new();
        for lower_group in previous_groups
            .iter()
            .skip(switch_id * groups_per_switch)
            .take(groups_per_switch)
        {
            group.extend(lower_group.iter().copied());
        }
        result.push(group);
    }
    Ok(result)
}

fn populate_layer_relationships(layers: &mut [LayerInfo]) {
    for i in 0..layers.len() {
        for j in i + 1..layers.len() {
            let (left, right) = relation_between_layers(&layers[i], &layers[j]);
            layers[i].relationships.insert(j, left);
            layers[j].relationships.insert(i, right);
        }
    }
}

fn relation_between_layers(left: &LayerInfo, right: &LayerInfo) -> (LayerRelation, LayerRelation) {
    let mut left_to_right = BTreeSet::new();
    let mut right_to_left = BTreeSet::new();
    for left_group in &left.groups {
        for right_group in &right.groups {
            let relation =
                relation_between_sets(&left_group.connected_gpus, &right_group.connected_gpus);
            if relation != LayerRelation::Cross {
                left_to_right.insert(relation);
                right_to_left.insert(invert_relation(relation));
            }
        }
    }
    if left_to_right.len() == 1 {
        (
            *left_to_right.iter().next().unwrap(),
            *right_to_left.iter().next().unwrap(),
        )
    } else {
        (LayerRelation::Cross, LayerRelation::Cross)
    }
}

fn relation_between_sets(left: &BTreeSet<usize>, right: &BTreeSet<usize>) -> LayerRelation {
    if left == right {
        LayerRelation::Repeat
    } else if left.is_superset(right) {
        LayerRelation::Contain
    } else if left.is_subset(right) {
        LayerRelation::BeContained
    } else {
        LayerRelation::Cross
    }
}

fn invert_relation(relation: LayerRelation) -> LayerRelation {
    match relation {
        LayerRelation::Repeat => LayerRelation::Repeat,
        LayerRelation::Contain => LayerRelation::BeContained,
        LayerRelation::BeContained => LayerRelation::Contain,
        LayerRelation::Cross => LayerRelation::Cross,
    }
}
