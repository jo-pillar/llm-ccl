from syccl_agents.base_topology import BaseTopology, LinkSpec, NodeType, Node, CollectiveType, LayerSpec

###TopoBegin
class TorusTopology(BaseTopology):
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
        self.f_gpu_torus()
        return self.connections

    def f_gpu_torus(self):
        torus_spec = getattr(self, "layer_spec_0", None)
        assert torus_spec is not None, "Layer 0 specification is missing."
        rows = torus_spec.group_num
        cols = torus_spec.node_num
        assert self.gpu_num == rows * cols, "GPU count must match torus dimensions."

        for row in range(rows):
            for col in range(cols):
                src = row * cols + col
                right = row * cols + ((col + 1) % cols)
                down = ((row + 1) % rows) * cols + col
                self.connect(
                    Node(node_id=f"gpu[{src}]", node_type=NodeType.GPU),
                    Node(node_id=f"gpu[{right}]", node_type=NodeType.GPU),
                    torus_spec.link_spec,
                )
                self.connect(
                    Node(node_id=f"gpu[{src}]", node_type=NodeType.GPU),
                    Node(node_id=f"gpu[{down}]", node_type=NodeType.GPU),
                    torus_spec.link_spec,
                )


topology = TorusTopology(
    16,
    "1MB",
    CollectiveType.ALLGATHER,
    layer0=LayerSpec(0, LinkSpec("400Gb/s", "100ns"), group_num=4, node_num=4, node_type=NodeType.GPU),
)
###TopoEND
