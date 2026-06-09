import json

from datasets.syccl.common.config import load_syccl_config
from datasets.syccl.common.sketch import normalize_sketches
from datasets.syccl.common.translate import sketches_to_translated_schedule


def tiny_config():
  return {
      "coll": {"name": "allgather", "byte": 4096, "root_sender": -1, "root_receiver": -1},
      "hosts": {"host_num": 2, "host_gpu_num": 2, "host_nic_num": 2, "host_links": "nvswitch"},
      "topo": [
          {"layer_id": 0, "type": "local", "link_spec": "memcpy"},
          {"layer_id": 1, "type": "host", "link_spec": "nvlink"},
          {"layer_id": 2, "type": "nic", "link_spec": "link_nic"},
          {"layer_id": 3, "type": "switch", "switch_topo": "multirail", "switch_num": 2, "link_spec": "netlink"},
      ],
      "link_spec": {
          "memcpy": {"bw_mbpus": 10, "lat_us": 0.01},
          "nvlink": {"bw_mbpus": 0.15, "lat_us": 3},
          "link_nic": {"bw_mbpus": 0.0455, "lat_us": 0},
          "netlink": {"bw_mbpus": 0.0455, "lat_us": 10},
      },
  }


def tiny_a2a_config():
  config = tiny_config()
  config["coll"] = {"name": "alltoall", "byte": 4096, "root_sender": -1, "root_receiver": -1}
  return config


def test_translates_single_root_sketch_into_allgather_schedule_shape(tmp_path):
  config_path = tmp_path / "config.json"
  config_path.write_text(json.dumps(tiny_config()), encoding="utf-8")
  context = load_syccl_config(config_path)
  sketch = [
      (0, 1, 0, 0, 1),
      (0, 3, 0, 0, 2),
      (1, 1, 1, 2, 3),
  ]

  graphs = normalize_sketches([sketch], context)
  translated = sketches_to_translated_schedule(graphs, context)

  assert translated["coll_name"] == "allgather"
  assert translated["ngpus"] == 4
  assert translated["chunk_size_byte"] == 4096
  events = translated["algorithms"][0]["final_schedule"]["Schedule"]["Events"]
  assert [event["src_chunk"] for event in events] == ["(0, 0)", "(1, 0)", "(2, 0)", "(3, 0)"]
  assert events[0]["sends"] == [
      {"src_gpu": 0, "dst_gpu": 1, "epoch": 0, "layer_used": 1, "copy": True, "reduce": False},
      {"src_gpu": 0, "dst_gpu": 2, "epoch": 0, "layer_used": 3, "copy": True, "reduce": False},
      {"src_gpu": 2, "dst_gpu": 3, "epoch": 1, "layer_used": 1, "copy": True, "reduce": False},
  ]
  assert events[1]["sends"][0]["src_gpu"] == 1
  assert events[1]["sends"][0]["dst_gpu"] == 0
  assert events[1]["sends"][1]["src_gpu"] == 1
  assert events[1]["sends"][1]["dst_gpu"] == 3
  assert events[2]["sends"][0]["src_gpu"] == 2
  assert events[2]["sends"][0]["dst_gpu"] == 3
  assert events[2]["sends"][1]["src_gpu"] == 2
  assert events[2]["sends"][1]["dst_gpu"] == 0


def test_translator_rejects_multiple_candidate_graphs_for_flow_sim(tmp_path):
  config_path = tmp_path / "config.json"
  config_path.write_text(json.dumps(tiny_config()), encoding="utf-8")
  context = load_syccl_config(config_path)
  sketch = [
      (0, 1, 0, 0, 1),
      (0, 3, 0, 0, 2),
      (1, 1, 1, 2, 3),
  ]
  graphs = normalize_sketches([sketch], context)

  try:
    sketches_to_translated_schedule(graphs + graphs, context)
  except ValueError as exc:
    assert "exactly one sketch" in str(exc)
  else:
    raise AssertionError("expected ValueError")


def test_translates_alltoall_as_one_chunk_per_source_destination(tmp_path):
  config_path = tmp_path / "config.json"
  config_path.write_text(json.dumps(tiny_a2a_config()), encoding="utf-8")
  context = load_syccl_config(config_path)
  sketch = [
      (0, 1, 0, 0, 1),
      (0, 3, 0, 0, 2),
      (1, 1, 1, 2, 3),
  ]

  graphs = normalize_sketches([sketch], context)
  translated = sketches_to_translated_schedule(graphs, context)

  assert translated["coll_name"] == "alltoall"
  events = translated["algorithms"][0]["final_schedule"]["Schedule"]["Events"]
  assert len(events) == 16
  assert events[0]["src_chunk"] == "(0, 0)"
  assert events[1]["src_chunk"] == "(0, 1)"
  assert events[4]["src_chunk"] == "(1, 0)"
  direct = next(event for event in events if event["src_chunk"] == "(0, 1)")
  assert direct["sends"] == [
      {"src_gpu": 0, "dst_gpu": 1, "epoch": 0, "layer_used": 1, "copy": True, "reduce": False}
  ]
  self_chunk = next(event for event in events if event["src_chunk"] == "(2, 2)")
  assert self_chunk["sends"] == []
