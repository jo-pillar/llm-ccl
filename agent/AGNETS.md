帮我构建一下用户的输入流程 整个流程应该是这样的，用户提供3个文件，一个topodsl用来描述拓扑和集合原语类型以及通信量大小，你可以假设当前只会提供2种类型的拓扑dsl，分别是clos multirail，你需要从topodsl中构造config 传递给流仿真器。一个--init-program，这个init program描述了一个ring算法 他接受一个参数为GPU_NUM,你需要使用从topodsl中解析出来的GPU num 来传参数给他 从而构造初始用来评估的sketch。一个instructions 会加载prompt_templete.txt,你需要用解析出来的参数去填充这些占位符。 TOPODSL 你可以使用环境变量来传递，如果不存在应视为fatal error 整个进程退出。
删除原先这些环境变量：SYCCL_TASK_HOST_NUM=4
SYCCL_TASK_HOST_GPU_NUM=8
SYCCL_TASK_NGPUS=32
SYCCL_TASK_TOPOLOGY=clos
SYCCL_TASK_CROSS_LAYER=4
SYCCL_TASK_CROSS_GROUP=0
你可以参考的文件有：
/home/antl/wzd/syccl/agent/scripts/run_syccl_simpletes.py
/home/antl/wzd/llm-ccl/agent/examples/topologies
/home/antl/wzd/llm-ccl/agent/syccl_agents/topodsl.py
/home/antl/wzd/llm-ccl/agent/syccl_agents/sketch_dsl.py
/home/antl/wzd/llm-ccl/agent/syccl_agents/prompts.py
/home/antl/wzd/llm-ccl/agent/syccl_agents/config_render.py
/home/antl/wzd/llm-ccl/agent/syccl_agents/base_topology.py
