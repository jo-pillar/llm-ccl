import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simpletes.node import Node, NodeDatabase, Status
from simpletes.policies.llm_elite import LLMElitePolicy


def _done_node(
    node_id: str,
    score: float,
    code: str,
    reflection: str | None = None,
    bottleneck_profile: dict | None = None,
) -> Node:
    metrics = {"combined_score": score, "validity": 1.0}
    if bottleneck_profile is not None:
        metrics["bottleneck_profile"] = bottleneck_profile
    return Node(
        id=node_id,
        code=code,
        metrics=metrics,
        score=score,
        status=Status.DONE,
        reflection=reflection,
    )


def test_all_strategy_uses_only_best_elite_as_full_code_inspiration():
    policy = LLMElitePolicy(
        num_chains=1,
        elite_selection_strategy="all",
        llm_policy_pool_size=10,
    )
    weak = _done_node("weak", 1.0, "WEAK_CODE")
    best = _done_node("best", 3.0, "BEST_CODE")
    policy.elite_sets[0] = [weak, best]

    selected = policy._select_from_chain(0, [weak, best], n=5)

    assert [node.id for node in selected] == ["best"]


def test_all_strategy_policy_context_keeps_only_best_bottleneck_and_other_reflections():
    policy = LLMElitePolicy(
        num_chains=1,
        elite_selection_strategy="all",
        llm_policy_pool_size=10,
    )
    best = _done_node(
        "best",
        3.0,
        "BEST_CODE_SHOULD_BE_SENT_AS_INSPIRATION_NOT_POLICY_CONTEXT",
        reflection="best reflection",
        bottleneck_profile={"marker": "best-profile", "critical_flow_chain": {"latest_flow_id": 7}},
    )
    other = _done_node(
        "other",
        2.0,
        "OTHER_CODE_SHOULD_NOT_APPEAR",
        reflection="alternate relay layout with lower contention",
        bottleneck_profile={"marker": "other-profile"},
    )
    policy.elite_sets[0] = [other, best]

    context = policy.get_policy_context(0, NodeDatabase())

    assert "[ELITE POOL LEDGER]" in context
    assert "[BEST ELITE BOTTLENECK PROFILE]" in context
    assert "best-profile" in context
    assert "alternate relay layout with lower contention" in context
    assert "OTHER_CODE_SHOULD_NOT_APPEAR" not in context
    assert "BEST_CODE_SHOULD_BE_SENT_AS_INSPIRATION_NOT_POLICY_CONTEXT" not in context
    assert "other-profile" not in context
