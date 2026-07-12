from syccl_agents.base_topology import BaseTopology, LinkSpec, NodeType, Node, CollectiveType, LayerSpec

###TopoBegin
class DGX2ClosTopology(BaseTopology):
    def __init__(self, gpu_num, coll_bytes, collective: CollectiveType, **kwargs: LayerSpec):
        self.connections = {}
        self.gpu_num = gpu_num
        for value in kwargs.values():
            if isinstance(value, LayerSpec):
                setattr(self, f"layer_spec_{value.layer_id}", value)
        self.coll_bytes = coll_bytes
        self.collective = collective
        self.build_topology()

    def build_topology(self):
        self.f_gpu2gpu()
        self.f_gpu2hostnic()
        self.f_hostnic2leaf()
        self.f_leaf2spine()
        return self.connections

    def f_gpu2gpu(self):
        """Layer 1: uniform DGX-2 host-local NVSwitch fabric."""
        host_local_spec = getattr(self, "layer_spec_1", None)
        assert host_local_spec is not None, "Layer 1 specification is missing."
        gpu_per_host = self.gpu_num // host_local_spec.group_num
        assert gpu_per_host == host_local_spec.node_num, "Mismatch between GPU count and host-local specification."
        for host_id in range(host_local_spec.group_num):
            host_base = host_id * gpu_per_host
            for local_src in range(gpu_per_host):
                for local_dst in range(local_src + 1, gpu_per_host):
                    src_node = Node(node_id=f"gpu[{host_base + local_src}]", node_type=NodeType.GPU)
                    dst_node = Node(node_id=f"gpu[{host_base + local_dst}]", node_type=NodeType.GPU)
                    self.connect(src_node, dst_node, host_local_spec.link_spec)

    def f_gpu2hostnic(self):
        """Layer 2: all GPUs in a DGX-2 share its host NIC."""
        host_local_spec = getattr(self, "layer_spec_1", None)
        host_nic_spec = getattr(self, "layer_spec_2", None)
        assert host_local_spec is not None, "Layer 1 specification is missing."
        assert host_nic_spec is not None, "Layer 2 specification is missing."
        gpu_per_host = host_local_spec.node_num
        assert host_nic_spec.node_num == 1, "DGX-2 Clos template expects one NIC per host."
        for host_id in range(host_local_spec.group_num):
            nic_node = Node(node_id=f"nic[{host_id}]", node_type=NodeType.NIC)
            for local_gpu in range(gpu_per_host):
                gpu_node = Node(node_id=f"gpu[{host_id * gpu_per_host + local_gpu}]", node_type=NodeType.GPU)
                self.connect(gpu_node, nic_node, host_nic_spec.link_spec)

    def f_hostnic2leaf(self):
        """Layer 3: one host-facing NIC is attached to each leaf."""
        host_nic_spec = getattr(self, "layer_spec_2", None)
        host_leaf_spec = getattr(self, "layer_spec_3", None)
        assert host_nic_spec is not None, "Layer 2 specification is missing."
        assert host_leaf_spec is not None, "Layer 3 specification is missing."
        host_num = host_nic_spec.group_num
        leaf_num = host_leaf_spec.group_num
        hosts_per_leaf = host_leaf_spec.node_num
        assert host_num == leaf_num * hosts_per_leaf, "Mismatch between host and leaf counts."
        for host_id in range(host_num):
            leaf_id = host_id // hosts_per_leaf
            nic_node = Node(node_id=f"nic[{host_id}]", node_type=NodeType.NIC)
            leaf_node = Node(node_id=f"leaf[{leaf_id}]", node_type=NodeType.SWITCH)
            self.connect(nic_node, leaf_node, host_leaf_spec.link_spec)

    def f_leaf2spine(self):
        """Layer 4: every leaf connects to every spine."""
        host_leaf_spec = getattr(self, "layer_spec_3", None)
        leaf_spine_spec = getattr(self, "layer_spec_4", None)
        assert host_leaf_spec is not None, "Layer 3 specification is missing."
        assert leaf_spine_spec is not None, "Layer 4 specification is missing."
        leaf_num = host_leaf_spec.group_num
        assert leaf_num == leaf_spine_spec.node_num, "Mismatch between leaf and spine specification."
        for leaf_id in range(leaf_num):
            for spine_id in range(leaf_spine_spec.group_num):
                leaf_node = Node(node_id=f"leaf[{leaf_id}]", node_type=NodeType.SWITCH)
                spine_node = Node(node_id=f"spine[{spine_id}]", node_type=NodeType.SWITCH)
                self.connect(leaf_node, spine_node, leaf_spine_spec.link_spec)


topology = DGX2ClosTopology(
    64,
    4294967296,
    CollectiveType.ALLGATHER,
    layer1=LayerSpec(1, LinkSpec("125GB/s", "3us"), group_num=4, node_num=16, node_type=NodeType.GPU),
    layer2=LayerSpec(2, LinkSpec("12.5GB/s", "0us"), group_num=4, node_num=1, node_type=NodeType.NIC),
    layer3=LayerSpec(3, LinkSpec("12.5GB/s", "25us"), group_num=4, node_num=1, node_type=NodeType.SWITCH),
    layer4=LayerSpec(4, LinkSpec("400GB/s", "25us"), group_num=1, node_num=4, node_type=NodeType.SWITCH),
)
###TopoEND
