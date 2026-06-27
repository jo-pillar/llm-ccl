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
        host_local_spec = getattr(self, "layer_spec_1", None)
        assert host_local_spec is not None, "Layer 1 specification is missing."
        gpu_per_host = self.gpu_num // host_local_spec.group_num
        assert gpu_per_host == host_local_spec.node_num, "Mismatch between GPU count and Layer 0 node specification."
        for host_id in range(host_local_spec.group_num):
                # 同一主机内GPU两两全连接（无重复）
                for gpu_i in range(gpu_per_host):
                    for gpu_j in range(gpu_i + 1, gpu_per_host):
                        src_node = Node(node_id=f"gpu[{host_id*gpu_per_host + gpu_i}]", node_type=NodeType.GPU)
                        dst_node = Node(node_id=f"gpu[{host_id*gpu_per_host + gpu_j}]", node_type=NodeType.GPU)
                        self.connect(src_node, dst_node, host_local_spec.link_spec)

    def f_gpu2nic(self):
        host_local_spec = getattr(self, "layer_spec_1", None)
        gpu_nic_spec = getattr(self, "layer_spec_2", None)
        assert host_local_spec is not None, "Layer 1 specification is missing."
        assert gpu_nic_spec is not None, "Layer 2 specification is missing."
        nic_num = gpu_nic_spec.group_num
        nic_per_gpu = gpu_nic_spec.node_num
        host_nic_num = nic_num // host_local_spec.group_num
        assert nic_num//nic_per_gpu == self.gpu_num, "NIC count devised by NICs per GPU must equal total GPU count."
        # connect each GPU to its corresponding NICs
        for gpu_id in range(self.gpu_num):
            for nic_offset in range(nic_per_gpu):
                nic_id = gpu_id * nic_per_gpu + nic_offset
                src_node = Node(node_id=f"gpu[{gpu_id}]", node_type=NodeType.GPU)
                dst_node = Node(node_id=f"nic[{nic_id}]", node_type=NodeType.NIC)
                self.connect(src_node, dst_node, gpu_nic_spec.link_spec)

    def f_nic2rail(self):
        host_local_spec = getattr(self, "layer_spec_1", None)
        rail_spec = getattr(self, "layer_spec_3", None)
        assert host_local_spec is not None, "Layer 1 specification is missing."
        assert rail_spec is not None, "Layer 3 specification is missing."
        host_num = host_local_spec.group_num
        gpu_per_host = host_local_spec.node_num
        rail_num = rail_spec.group_num
        gpu_per_rail = rail_spec.node_num

        for rail_id in range(rail_num):
            for host_id in range(host_num):
                for gpu in range(gpu_per_host):
                    # connect gpu to the same rail based on the rail_id and gpu index
                    if gpu==rail_id:
                        src_node = Node(node_id=f"gpu[{host_id*gpu_per_host + gpu}]", node_type=NodeType.GPU)
                        dst_node = Node(node_id=f"rail[{rail_id}]", node_type=NodeType.SWITCH)
                        self.connect(src_node, dst_node, rail_spec.link_spec)


topology = MultiRailTopology(
    512,
    "1MB",
    CollectiveType.ALLGATHER,
    layer1=LayerSpec(1, LinkSpec("150GB/s", "10.5us"), group_num=64, node_num=8, node_type=NodeType.GPU),
    layer2=LayerSpec(2, LinkSpec("45GB/s", "0us"), group_num=512, node_num=1, node_type=NodeType.NIC),
    layer3=LayerSpec(3, LinkSpec("45GB/s", "21.5us"), group_num=8, node_num=64, node_type=NodeType.SWITCH),
)
###TopoEND
