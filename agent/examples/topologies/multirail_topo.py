from syccl_agents.base_topology import BaseTopology, LinkSpec, NodeType, Node, CollectiveType, LayerSpec

###TopoBegin
class MultiRailTopology(BaseTopology):
    def __init__(self, gpu_num, coll_bytes, collective: CollectiveType, **kwargs: LayerSpec):
        self.connections = {}
        self.gpu_num = gpu_num
        for key, value in kwargs.items():
            if isinstance(value, LayerSpec):
                setattr(self, f"layer_spec_{value.layer_id}", value)
        self.coll_bytes = coll_bytes
        self.collective = collective
        self.build_topology()

    def build_topology(self):
        self.f_gpu2gpu()
        self.f_gpu2nic()
        self.f_nic2rail()
        return self.connections
 
    def f_gpu2gpu(self):
        host_local_spec = getattr(self, "layer_spec_0", None)
        assert host_local_spec is not None, "Layer 0 specification is missing."
        gpu_per_host = self.gpu_num // host_local_spec.group_num
        assert gpu_per_host == host_local_spec.node_num, "Mismatch between GPU count and Layer 0 node specification."

        nvlink_matrix = [
            [1, 2, 3, 5],
            [0, 2, 3, 4],
            [0, 1, 3, 7],
            [0, 1, 2, 6],
            [5, 6, 7, 1],
            [4, 6, 7, 0],
            [4, 5, 7, 3],
            [4, 5, 6, 2],
        ]
        for host_id in range(host_local_spec.group_num):
            for gpu_i, neighbors in enumerate(nvlink_matrix):
                for gpu_j in neighbors:
                    if gpu_i < gpu_j:
                        src_node = Node(node_id=f"gpu[{host_id}][{gpu_i}]", node_type=NodeType.GPU)
                        dst_node = Node(node_id=f"gpu[{host_id}][{gpu_j}]", node_type=NodeType.GPU)
                        self.connect(src_node, dst_node, host_local_spec.link_spec)

    def f_gpu2nic(self):
        host_local_spec = getattr(self, "layer_spec_0", None)
        gpu_nic_spec = getattr(self, "layer_spec_1", None)
        assert host_local_spec is not None, "Layer 0 specification is missing."
        assert gpu_nic_spec is not None, "Layer 1 specification is missing."
        gpu_per_host = self.gpu_num // host_local_spec.group_num
        nic_per_host = gpu_nic_spec.node_num
        assert gpu_per_host % nic_per_host == 0, "GPU count per host must be divisible by NIC count."
        gpu_per_nic = gpu_per_host // nic_per_host

        for host_id in range(host_local_spec.group_num):
            for nic_id in range(nic_per_host):
                for local_gpu in range(nic_id * gpu_per_nic, (nic_id + 1) * gpu_per_nic):
                    src_node = Node(node_id=f"gpu[{host_id}][{local_gpu}]", node_type=NodeType.GPU)
                    dst_node = Node(node_id=f"nic[{host_id}][{nic_id}]", node_type=NodeType.NIC)
                    self.connect(src_node, dst_node, gpu_nic_spec.link_spec)

    def f_nic2rail(self):
        host_local_spec = getattr(self, "layer_spec_0", None)
        gpu_nic_spec = getattr(self, "layer_spec_1", None)
        rail_spec = getattr(self, "layer_spec_3", None)
        assert host_local_spec is not None, "Layer 0 specification is missing."
        assert gpu_nic_spec is not None, "Layer 1 specification is missing."
        assert rail_spec is not None, "Layer 3 specification is missing."
        host_num = host_local_spec.group_num
        nic_per_host = gpu_nic_spec.node_num
        rail_num = rail_spec.group_num
        nic_per_rail_per_host = rail_spec.node_num
        assert nic_per_host == rail_num * nic_per_rail_per_host, "Mismatch between NIC count and rail specification."

        for rail_id in range(rail_num):
            for host_id in range(host_num):
                for nic_offset in range(nic_per_rail_per_host):
                    nic_id = rail_id * nic_per_rail_per_host + nic_offset
                    src_node = Node(node_id=f"nic[{host_id}][{nic_id}]", node_type=NodeType.NIC)
                    dst_node = Node(node_id=f"rail[{rail_id}]", node_type=NodeType.SWITCH)
                    self.connect(src_node, dst_node, rail_spec.link_spec)


topology = MultiRailTopology(
    32,
    "1MB",
    CollectiveType.ALLGATHER,
    layer0=LayerSpec(0, LinkSpec("300GB/s", "9us"), group_num=4, node_num=8, node_type=NodeType.GPU),
    layer1=LayerSpec(1, LinkSpec("22.5GB/s", "0us"), group_num=4, node_num=8, node_type=NodeType.NIC),
    layer3=LayerSpec(3, LinkSpec("22.5GB/s", "25us"), group_num=8, node_num=1, node_type=NodeType.SWITCH),
)
###TopoEND
