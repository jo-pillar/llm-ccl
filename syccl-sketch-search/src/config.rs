use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct SycclConfig {
    pub coll: CollConfig,
    pub hosts: HostsConfig,
    pub topo: Vec<TopoLayerConfig>,
    #[serde(default)]
    pub host_links: HostLinksConfig,
    #[serde(default)]
    pub prune: PruneConfig,
    #[serde(default)]
    pub sketch: SketchConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CollConfig {
    pub name: String,
    pub byte: u64,
    #[serde(default = "minus_one")]
    pub root_sender: isize,
    #[serde(default = "minus_one")]
    pub root_receiver: isize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HostsConfig {
    pub host_num: usize,
    pub host_gpu_num: usize,
    #[serde(default)]
    pub host_nic_num: usize,
    pub host_links: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TopoLayerConfig {
    pub layer_id: usize,
    #[serde(rename = "type")]
    pub layer_type: String,
    #[serde(default)]
    pub switch_topo: Option<String>,
    #[serde(default)]
    pub switch_num: Option<usize>,
    #[serde(default)]
    pub link_spec: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct HostLinksConfig {
    #[serde(default)]
    pub nvlink: Vec<Vec<usize>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PruneConfig {
    #[serde(default = "minus_one")]
    pub max_comb_layer_num: isize,
    #[serde(default)]
    pub ignored_layers: BTreeSet<usize>,
    #[serde(default)]
    pub must_contain_layers: BTreeSet<usize>,
    #[serde(default)]
    pub layer_comm_ordering: BTreeMap<usize, BTreeSet<usize>>,
    #[serde(default)]
    pub layer_comm_uplimit: BTreeMap<usize, usize>,
    #[serde(default)]
    pub all_groups_used: bool,
    #[serde(default = "default_true")]
    pub allow_unequal_source_per_group: bool,
    #[serde(default = "default_true")]
    pub allow_unequal_dest_per_source: bool,
    #[serde(default = "default_true")]
    pub at_least_one_layer_no_source_with_0_dest: bool,
    #[serde(default)]
    pub allow_source_group_special: bool,
    #[serde(default = "minus_one")]
    pub limit_num_steps: isize,
    #[serde(default = "default_negative_ratio")]
    pub limit_num_steps_ratio: f64,
    #[serde(default = "default_true")]
    pub start_source_diff_layer: bool,
    #[serde(default = "default_true")]
    pub same_step_diff: bool,
    #[serde(default = "default_true")]
    pub prune_sources_by_compare_steps: bool,
    #[serde(default = "default_true")]
    pub allow_num_groups_arithmetic_per_step: bool,
    #[serde(default = "default_true")]
    pub allow_source_per_group_arithmetic_per_step: bool,
    #[serde(default = "default_true")]
    pub allow_num_groups_geometric_per_step: bool,
    #[serde(default = "default_true")]
    pub allow_source_per_group_geometric_per_step: bool,
    #[serde(default = "default_true")]
    pub drop_small_groups: bool,
    #[serde(default = "default_true")]
    pub drop_small_sources: bool,
    #[serde(default)]
    pub keep_large_groups: bool,
    #[serde(default)]
    pub keep_large_sources: bool,
    #[serde(default = "default_true")]
    pub partition_sources: bool,
    #[serde(default = "default_true")]
    pub partition_dests: bool,
    #[serde(default = "default_true")]
    pub prune_dests: bool,
    #[serde(default = "default_true")]
    pub allow_dest_balanced: bool,
    #[serde(default = "default_true")]
    pub allow_dest_centralized: bool,
    #[serde(default = "default_max_dest_combs")]
    pub max_dest_combs: isize,
}

impl Default for PruneConfig {
    fn default() -> Self {
        Self {
            max_comb_layer_num: -1,
            ignored_layers: BTreeSet::new(),
            must_contain_layers: BTreeSet::new(),
            layer_comm_ordering: BTreeMap::new(),
            layer_comm_uplimit: BTreeMap::new(),
            all_groups_used: false,
            allow_unequal_source_per_group: true,
            allow_unequal_dest_per_source: true,
            at_least_one_layer_no_source_with_0_dest: true,
            allow_source_group_special: false,
            limit_num_steps: -1,
            limit_num_steps_ratio: -1.0,
            start_source_diff_layer: true,
            same_step_diff: true,
            prune_sources_by_compare_steps: true,
            allow_num_groups_arithmetic_per_step: true,
            allow_source_per_group_arithmetic_per_step: true,
            allow_num_groups_geometric_per_step: true,
            allow_source_per_group_geometric_per_step: true,
            drop_small_groups: true,
            drop_small_sources: true,
            keep_large_groups: false,
            keep_large_sources: false,
            partition_sources: true,
            partition_dests: true,
            prune_dests: true,
            allow_dest_balanced: true,
            allow_dest_centralized: true,
            max_dest_combs: 100,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SketchConfig {
    #[serde(default)]
    pub customize_sketch: bool,
    #[serde(default)]
    pub use_sketch_input: bool,
    #[serde(default)]
    pub save_sketch: bool,
    #[serde(default)]
    pub sketch_path: String,
}

#[derive(Debug, Clone)]
pub struct SearchConfig {
    pub ngpus: usize,
    pub sender: usize,
    pub prune: PruneConfig,
}

impl TryFrom<&SycclConfig> for SearchConfig {
    type Error = anyhow::Error;

    fn try_from(config: &SycclConfig) -> Result<Self, Self::Error> {
        let ngpus = config.hosts.host_num * config.hosts.host_gpu_num;
        let sender = match config.coll.name.as_str() {
            "allgather" | "alltoall" => 0,
            "broadcast" | "scatter" => {
                if config.coll.root_sender < 0 {
                    anyhow::bail!(
                        "{} requires non-negative coll.root_sender",
                        config.coll.name
                    );
                }
                config.coll.root_sender as usize
            }
            other => anyhow::bail!("unsupported collective {other:?} for sketch search"),
        };
        if sender >= ngpus {
            anyhow::bail!("sender GPU {sender} is out of range for {ngpus} GPUs");
        }

        Ok(Self {
            ngpus,
            sender,
            prune: config.prune.clone(),
        })
    }
}

fn minus_one() -> isize {
    -1
}

fn default_true() -> bool {
    true
}

fn default_negative_ratio() -> f64 {
    -1.0
}

fn default_max_dest_combs() -> isize {
    100
}
