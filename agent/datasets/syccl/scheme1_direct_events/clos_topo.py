from syccl_agents.base_topology import BaseTopology, LinkSpec, NodeType, Node, CollectiveType, LayerSpec

###TopoBegin
class ClosTopology(BaseTopology):
    def __init__(self, gpu_num, coll_bytes, collective: CollectiveType, **kwargs: LayerSpec):
        # 集群规模参数（均为总数）
        self.connections = {}
        self.gpu_num = gpu_num
        # 提取每层拓扑参数
        for key, value in kwargs.items():
            if isinstance(value, LayerSpec):
                setattr(self, f"layer_spec_{value.layer_id}", value)
        self.coll_bytes = coll_bytes
        self.collective = collective
        self.build_topology()

    def build_topology(self):
        """按顺序构建层级拓扑"""
        self.f_gpu2gpu()
        self.f_gpu2hostnic()
        self.f_host_leaf()
        self.f_leaf_spine()
        return self.connections

    def f_gpu2gpu(self):
        """Layer 0: 同一主机内所有GPU之间NVLink全连接"""
        # 计算每个主机上的GPU数量（要求gpu_num能被host_num整除）
        host_local_spec = getattr(self, "layer_spec_0", None)
        assert host_local_spec is not None, "Layer 0 specification is missing."
        gpu_per_host = self.gpu_num // host_local_spec.group_num
        assert gpu_per_host == host_local_spec.node_num, "Mismatch between GPU count and Layer 0 node specification."
        for host_id in range(host_local_spec.group_num):
            # 同一主机内GPU两两全连接（无重复）
            for gpu_i in range(gpu_per_host):
                for gpu_j in range(gpu_i + 1, gpu_per_host):
                    src_node = Node(node_id=f"gpu[{host_id}][{gpu_i}]", node_type=NodeType.GPU)
                    dst_node = Node(node_id=f"gpu[{host_id}][{gpu_j}]", node_type=NodeType.GPU)
                    self.connect(src_node, dst_node, host_local_spec.link_spec)

    def f_gpu2hostnic(self):
        """Layer 1: GPU到主机侧交换节点的连接"""
        host_local_spec = getattr(self, "layer_spec_0", None)
        host_nic_spec = getattr(self, "layer_spec_1", None)
        assert host_local_spec is not None, "Layer 0 specification is missing."
        assert host_nic_spec is not None, "Layer 1 specification is missing."
        gpu_per_host = self.gpu_num // host_local_spec.group_num
        for host_id in range(host_local_spec.group_num):
            for gpu_id in range(gpu_per_host):
                src_node = Node(node_id=f"gpu[{host_id}][{gpu_id}]", node_type=NodeType.GPU)
                dst_node = Node(node_id=f"host[{host_id}]", node_type=host_nic_spec.node_type)
                self.connect(src_node, dst_node, host_nic_spec.link_spec)

    def f_host_leaf(self):
        """Layer 2: 主机侧交换节点均匀分配到各个Leaf交换机，单归连接"""
        host_nic_spec = getattr(self, "layer_spec_1", None)
        host_leaf_spec = getattr(self, "layer_spec_2", None)
        assert host_nic_spec is not None, "Layer 1 specification is missing."
        assert host_leaf_spec is not None, "Layer 2 specification is missing."
        host_num = host_nic_spec.group_num
        leaf_num = host_leaf_spec.group_num
        host_per_leaf = host_leaf_spec.node_num
        assert host_num == leaf_num * host_per_leaf, "Mismatch between host count and Layer 2 node specification."

        for leaf_id in range(leaf_num):
            start_host = leaf_id * host_per_leaf
            end_host = start_host + host_per_leaf
            for host_id in range(start_host, end_host):
                src_node = Node(node_id=f"host[{host_id}]", node_type=host_nic_spec.node_type)
                dst_node = Node(node_id=f"leaf[{leaf_id}]", node_type=NodeType.SWITCH)
                self.connect(src_node, dst_node, host_leaf_spec.link_spec)

    def f_leaf_spine(self):
        """Layer 3: Leaf与Spine全互联（标准Clos核心）"""
        host_leaf_spec = getattr(self, "layer_spec_2", None)
        leaf_spine_spec = getattr(self, "layer_spec_3", None)
        assert host_leaf_spec is not None, "Layer 2 specification is missing."
        assert leaf_spine_spec is not None, "Layer 3 specification is missing."
        leaf_num = host_leaf_spec.group_num
        spine_num = leaf_spine_spec.group_num
        leaf_per_spine = leaf_spine_spec.node_num
        assert leaf_num == leaf_per_spine, "Mismatch between leaf count and Layer 3 node specification."

        for leaf_id in range(leaf_num):
            for spine_id in range(spine_num):
                src_node = Node(node_id=f"leaf[{leaf_id}]", node_type=NodeType.SWITCH)
                dst_node = Node(node_id=f"spine[{spine_id}]", node_type=NodeType.SWITCH)
                self.connect(src_node, dst_node, leaf_spine_spec.link_spec)


topology = ClosTopology(
    64,
    "1MB",
    CollectiveType.ALLGATHER,
    layer0=LayerSpec(1, LinkSpec("32.5GBps", "9us"), group_num=8, node_num=8, node_type=NodeType.GPU),
    layer1=LayerSpec(2, LinkSpec("2.8125GBps", "25us"), group_num=8, node_num=1, node_type=NodeType.SWITCH),
    layer2=LayerSpec(3, LinkSpec("2.8125GBps", "25us"), group_num=2, node_num=4, node_type=NodeType.SWITCH),
    layer3=LayerSpec(4, LinkSpec("45GBps", "25us"), group_num=1, node_num=2, node_type=NodeType.SWITCH),
)
###TopoEND
