from enum import StrEnum

class NodeType(StrEnum):
    GPU = "GPU"
    SWITCH = "SWITCH"
    NIC = "NIC"
class CollectiveType(StrEnum):
    ALLREDUCE = "allreduce"
    ALLGATHER = "allgather"
    BROADCAST = "broadcast"
    ALLTOALL = "alltoall"
    
class Node:
    def __init__(self, node_id: str, node_type: NodeType):
        self.node_id = node_id
        self.node_type = node_type
class LinkSpec:
    def __init__(self, bandwidth: str, latency: str):
        self.bandwidth = bandwidth
        self.latency = latency
class LayerSpec:
    def __init__(self, layer_id: int, link_spec: LinkSpec,group_num:int, node_num: int,node_type: NodeType):
        self.layer_id = layer_id
        self.link_spec = link_spec
        self.group_num = group_num
        self.node_num = node_num
        self.node_type = node_type
class BaseTopology:
    def __init__(self, gpu_num, coll_bytes, collective, **kwargs:LayerSpec):
        self.connections={}
        pass
    def connect(self, src_node: Node, dst_node: Node, link_spec: LinkSpec):
        """Connect two nodes with a specified link specification."""
        self.connections[(src_node, dst_node)] = link_spec
        
    def build_topology(self):
        """Build the topology by connecting nodes according to the specified parameters."""
        return self.connections
    