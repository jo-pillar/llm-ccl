use std::collections::BTreeSet;

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct SketchGraph {
    pub ngpus: usize,
    pub src_gpu: usize,
    pub nodes: Vec<SketchNode>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SketchNode {
    pub id: usize,
    pub step: usize,
    pub layer: usize,
    pub group: usize,
    pub src_dest_pair: SrcDestPair,
    pub deps: Vec<usize>,
    pub next: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SrcDestPair {
    pub srcs: BTreeSet<usize>,
    pub dsts: BTreeSet<usize>,
}
