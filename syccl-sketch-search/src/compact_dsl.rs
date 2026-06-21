use crate::graph::SketchGraph;
use std::collections::BTreeSet;

pub fn format_compact_dsl(sketches: &[SketchGraph]) -> String {
    let mut out = String::from("compact_sketches = [\n");
    for sketch in sketches {
        out.push_str("  [\n");
        for node in &sketch.nodes {
            out.push_str("    ");
            out.push_str(&format!(
                "({}, {}, {}, {}, {}),\n",
                node.step,
                node.layer,
                node.group,
                format_gpu_operand(&node.src_dest_pair.srcs),
                format_gpu_operand(&node.src_dest_pair.dsts),
            ));
        }
        out.push_str("  ],\n");
    }
    out.push_str("]\n");
    out
}

fn format_gpu_operand(gpus: &BTreeSet<usize>) -> String {
    if gpus.len() == 1 {
        return gpus.iter().next().expect("one gpu").to_string();
    }
    let items = gpus
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{items}]")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{SketchNode, SrcDestPair};

    #[test]
    fn formats_agent_style_compact_dsl() {
        let sketches = vec![SketchGraph {
            ngpus: 4,
            src_gpu: 0,
            nodes: vec![SketchNode {
                id: 0,
                step: 0,
                layer: 1,
                group: 0,
                src_dest_pair: SrcDestPair {
                    srcs: BTreeSet::from([0]),
                    dsts: BTreeSet::from([1, 2, 3]),
                },
                deps: vec![],
                next: vec![],
            }],
        }];

        assert_eq!(
            format_compact_dsl(&sketches),
            "compact_sketches = [\n  [\n    (0, 1, 0, 0, [1, 2, 3]),\n  ],\n]\n"
        );
    }
}
