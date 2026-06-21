class ClosTopology(BaseTopology):
    def __init__(self, gpu_num, host_num, leaf_num, spine_num,
                 gpu_host_bw, gpu_host_lat,
                 host_leaf_bw, host_leaf_lat,
                 leaf_spine_bw, leaf_spine_lat,
                 coll_bytes, collective):
        # 集群规模参数（均为总数）
        self.gpu_num = gpu_num
        self.host_num = host_num
        self.leaf_num = leaf_num
        self.spine_num = spine_num
        self.coll_bytes = coll_bytes
        self.collective = collective
        
        # 链路规格
        self.link_spec_0 =LinkSpec(bandwidth= gpu_host_bw, latency= gpu_host_lat)  # NVLink
        self.link_spec_1 = LinkSpec (bandwidth=host_leaf_bw, latency=host_leaf_lat)  # 主机-Leaf
        self.link_spec_2 =LinkSpec (bandwidth=leaf_spine_bw, latency=leaf_spine_lat)  # Leaf-Spine
    def build_topology(self):
        """按顺序构建三层拓扑"""
        self.f_gpu_host()
        self.f_host_leaf()
        self.f_leaf_spine()

    def f_gpu_host(self):
        """Layer 0: 同一主机内所有GPU之间NVLink全连接"""
        # 计算每个主机上的GPU数量（要求gpu_num能被host_num整除）
        gpu_per_host = self.gpu_num // self.host_num
        for host_id in range(self.host_num):
            # 同一主机内GPU两两全连接（无重复）
            for gpu_i in range(gpu_per_host):
                for gpu_j in range(gpu_i + 1, gpu_per_host):
                    src_node = f"gpu[{host_id}][{gpu_i}]"
                    dst_node = f"gpu[{host_id}][{gpu_j}]"
                    self.connect(src_node, dst_node, self.link_spec_0)
    
    def f_host_leaf(self):
        """Layer 1: 主机均匀分配到各个Leaf交换机，单归连接"""
        # 计算每个Leaf交换机连接的主机数量（要求host_num能被leaf_num整除）
        host_per_leaf = self.host_num // self.leaf_num
        
        for leaf_id in range(self.leaf_num):
            start_host = leaf_id * host_per_leaf
            end_host = start_host + host_per_leaf
            for host_id in range(start_host, end_host):
                src_node = f"host[{host_id}]"
                dst_node = f"leaf[{leaf_id}]"
                self.connect(src_node, dst_node, self.link_spec_1)
    
    def f_leaf_spine(self):
        """Layer 2: Leaf与Spine全互联（标准Clos核心）"""
        for leaf_id in range(self.leaf_num):
            for spine_id in range(self.spine_num):
                src_node = f"leaf[{leaf_id}]"
                dst_node = f"spine[{spine_id}]"
                self.connect(src_node, dst_node, self.link_spec_2)

ClosTopology(32,4,2,1,"32.5GBps","9us","2.8125GBps","25us","45GBps","25us", "1MB", "allgather")
