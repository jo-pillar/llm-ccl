import pytest

from topodsl import BaseTopology, LinkSpec


class LineTopology(BaseTopology):
    def validate_config(self):
        errors = []
        if self.params["node_num"] <= 0:
            errors.append("node_num must be positive")
        return errors

    def build_topology(self):
        self.f_line()

    def f_line(self):
        for node_id in range(self.params["node_num"] - 1):
            self.connect(
                f"node[{node_id}]",
                f"node[{node_id + 1}]",
                self.link_specs["line"],
            )


def test_base_topology_builds_connections_during_initialization():
    topology = LineTopology(
        params={"node_num": 3},
        link_specs={"line": LinkSpec(bandwidth="100Gbps", latency="1us")},
    )

    assert topology.connections == [
        {
            "source": "node[0]",
            "destination": "node[1]",
            "bandwidth": "100Gbps",
            "latency": "1us",
        },
        {
            "source": "node[1]",
            "destination": "node[2]",
            "bandwidth": "100Gbps",
            "latency": "1us",
        },
    ]


def test_base_topology_rejects_invalid_subclass_config():
    with pytest.raises(ValueError, match="node_num must be positive"):
        LineTopology(
            params={"node_num": 0},
            link_specs={"line": LinkSpec(bandwidth="100Gbps", latency="1us")},
        )


def test_base_topology_requires_subclasses_to_implement_hooks():
    with pytest.raises(TypeError):
        BaseTopology(params={}, link_specs={})


def test_print_connections_uses_clos_style_output(capsys):
    topology = LineTopology(
        params={"node_num": 2},
        link_specs={"line": LinkSpec(bandwidth="100Gbps", latency="1us")},
    )

    topology.print_connections()

    output = capsys.readouterr().out
    assert "总连接数: 1" in output
    assert "node[0] <--> node[1]" in output
    assert "带宽: 100Gbps" in output
    assert "延迟: 1us" in output


def test_to_dict_exports_params_link_specs_and_connections():
    topology = LineTopology(
        params={"node_num": 2},
        link_specs={"line": LinkSpec(bandwidth="100Gbps", latency="1us")},
    )

    assert topology.to_dict() == {
        "topology": "LineTopology",
        "params": {"node_num": 2},
        "link_specs": {"line": {"bandwidth": "100Gbps", "latency": "1us"}},
        "connections": [
            {
                "source": "node[0]",
                "destination": "node[1]",
                "bandwidth": "100Gbps",
                "latency": "1us",
            }
        ],
    }
