use anyhow::{anyhow, Result};
use serde_json::json;

use crate::topodsl::TopologyParams;

pub fn render_flow_sim_config(
    params: &TopologyParams,
    collective: &str,
    message_size: u64,
) -> Result<String> {
    let value = match params.family.as_str() {
        "clos" => render_clos(params, collective, message_size)?,
        "multirail" => render_multirail(params, collective, message_size)?,
        other => return Err(anyhow!("unsupported topology family {other}")),
    };
    serde_json::to_string_pretty(&value).map_err(Into::into)
}

fn render_clos(params: &TopologyParams, collective: &str, message_size: u64) -> Result<serde_json::Value> {
    let leaf_switches = params
        .leaf_switches
        .ok_or_else(|| anyhow!("clos TopoDSL requires leaf_switches"))?;
    let spine_switches = params.spine_switches.unwrap_or(1);
    Ok(json!({
        "coll": {
            "name": collective,
            "byte": message_size,
            "root_sender": -1,
            "root_receiver": -1
        },
        "hosts": {
            "host_num": params.hosts,
            "host_gpu_num": params.gpus_per_host,
            "host_nic_num": params.nics_per_host,
            "host_links": "nvswitch"
        },
        "topo": [
            {"layer_id": 0, "type": "local", "link_spec": "memcpy"},
            {"layer_id": 1, "type": "host", "link_spec": "nvlink"},
            {"layer_id": 2, "type": "nic", "link_spec": "link_nic"},
            {"layer_id": 3, "type": "switch", "switch_topo": "pod", "switch_num": leaf_switches, "link_spec": "netlink_leaf"},
            {"layer_id": 4, "type": "switch", "switch_topo": "pod", "switch_num": spine_switches, "link_spec": "netlink_spine"}
        ],
        "link_spec": {
            "memcpy": {"bw_mbpus": 10.0, "lat_us": 0.01},
            "nvlink": {"bw_mbpus": params.host_bw_mbpus, "lat_us": params.host_lat_us},
            "link_nic": {"bw_mbpus": params.nic_bw_mbpus, "lat_us": params.nic_lat_us},
            "netlink_leaf": {"bw_mbpus": params.leaf_bw_mbpus, "lat_us": params.leaf_lat_us},
            "netlink_spine": {"bw_mbpus": params.spine_bw_mbpus, "lat_us": params.spine_lat_us}
        }
    }))
}

fn render_multirail(
    params: &TopologyParams,
    collective: &str,
    message_size: u64,
) -> Result<serde_json::Value> {
    let rails = params.rails.unwrap_or(params.nics_per_host);
    Ok(json!({
        "coll": {
            "name": collective,
            "byte": message_size,
            "root_sender": -1,
            "root_receiver": -1
        },
        "hosts": {
            "host_num": params.hosts,
            "host_gpu_num": params.gpus_per_host,
            "host_nic_num": params.nics_per_host,
            "host_links": "nvswitch"
        },
        "topo": [
            {"layer_id": 0, "type": "local", "link_spec": "memcpy"},
            {"layer_id": 1, "type": "host", "link_spec": "nvlink"},
            {"layer_id": 2, "type": "nic", "link_spec": "link_nic"},
            {"layer_id": 3, "type": "switch", "switch_topo": "multirail", "switch_num": rails, "link_spec": "netlink"}
        ],
        "link_spec": {
            "memcpy": {"bw_mbpus": 10.0, "lat_us": 0.01},
            "nvlink": {"bw_mbpus": params.host_bw_mbpus, "lat_us": params.host_lat_us},
            "link_nic": {"bw_mbpus": params.nic_bw_mbpus, "lat_us": params.nic_lat_us},
            "netlink": {"bw_mbpus": params.leaf_bw_mbpus, "lat_us": params.leaf_lat_us}
        }
    }))
}
