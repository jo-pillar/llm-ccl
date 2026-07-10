"""
Elite context template for LLMElitePolicy.
"""

ELITE_CONTEXT_TEMPLATE = """\
[ELITE POOL OVERVIEW] ({num_elites} diverse solutions, scores and insights only)
This shows the breadth of approaches already explored. Use this to:
- Understand what directions have been tried
- Avoid duplicating existing approaches
- Identify gaps for novel solutions

{entries}"""

ELITE_ENTRY_TEMPLATE = """\
#{index} [Score: {score}]
{metrics_line}{reflection_line}
"""

ELITE_ALL_CONTEXT_TEMPLATE = """\
[ELITE POOL LEDGER] ({num_elites} accepted elite solutions)
The full elite pool is maintained by the llm_elite curator, but the generator
only receives the best solution as full code in [SAMPLED INSPIRATIONS].
Use this compact ledger to avoid duplicate directions and identify gaps.

[BEST ELITE BOTTLENECK PROFILE] id={best_node_id} score={best_score}
```text
{best_bottleneck_profile}
```

[OTHER ELITE REFLECTION RECORDS]
{reflection_records}"""

ELITE_REFLECTION_RECORD_TEMPLATE = """\
#{index} [Score: {score}] id={node_id}
Reflection: {reflection}
"""
