from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any


def parse_sketch_dsl(text: str) -> list[dict[str, Any]]:
    parsed_function = _try_construct_sketches(text)
    if parsed_function is not None:
        candidate = _unwrap_candidate(parsed_function)
        return [_normalize_transmission(item, index) for index, item in enumerate(candidate)]
    payload = _extract_payload(text)
    parsed = _loads(payload)
    candidate = _unwrap_candidate(parsed)
    return [_normalize_transmission(item, index) for index, item in enumerate(candidate)]


def write_compact_sketch(transmissions: list[dict[str, Any]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(transmissions, indent=2), encoding="utf-8")
    return output


def _extract_payload(text: str) -> str:
    fenced = re.search(r"```(?:json|python|text)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    stripped = text.strip()
    if stripped.startswith(("[", "{")):
        return stripped
    extracted = _extract_balanced_literal(stripped)
    return extracted if extracted is not None else stripped


def _try_construct_sketches(text: str) -> Any | None:
    payload = _extract_code_payload(text)
    if "def construct_sketches" not in payload:
        return None
    namespace: dict[str, Any] = {"__builtins__": {}}
    try:
        exec(payload, namespace, namespace)
    except Exception as exc:
        raise ValueError(f"could not execute construct_sketches(): {exc}") from exc
    construct = namespace.get("construct_sketches")
    if not callable(construct):
        raise ValueError("construct_sketches is not callable")
    try:
        return construct()
    except Exception as exc:
        raise ValueError(f"construct_sketches() failed: {exc}") from exc


def _extract_code_payload(text: str) -> str:
    fenced = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def _extract_balanced_literal(text: str) -> str | None:
    for start, opener in ((text.find("["), "["), (text.find("{"), "{")):
        if start < 0:
            continue
        closer = "]" if opener == "[" else "}"
        depth = 0
        in_string = False
        quote = ""
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == quote:
                    in_string = False
                continue
            if char in {"'", '"'}:
                in_string = True
                quote = char
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]
    return None


def _loads(payload: str) -> Any:
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(payload)
        except Exception as exc:
            raise ValueError(f"could not parse SketchDSL as JSON or Python literal: {exc}") from exc


def _unwrap_candidate(parsed: Any) -> list[Any]:
    if isinstance(parsed, dict) and {"step", "layer", "group"} <= set(parsed):
        return [parsed]
    if isinstance(parsed, dict) and "sketch" in parsed:
        return _unwrap_candidate(parsed["sketch"])
    if isinstance(parsed, dict) and "transmissions" in parsed:
        return _unwrap_candidate(parsed["transmissions"])
    if isinstance(parsed, list):
        if parsed and isinstance(parsed[0], dict) and "nodes" in parsed[0]:
            return _unwrap_candidate(parsed[0])
        return parsed
    if isinstance(parsed, dict) and "nodes" in parsed:
        nodes = parsed["nodes"]
        if not isinstance(nodes, list):
            raise ValueError("native sketch nodes must be a list")
        return [
            {
                "step": node["step"],
                "layer": node["layer"],
                "group": node["group"],
                "srcs": node["src_dest_pair"]["srcs"],
                "dsts": node["src_dest_pair"]["dsts"],
            }
            for node in nodes
        ]
    raise ValueError("SketchDSL must be a list of transmissions")


def _normalize_transmission(item: Any, index: int) -> dict[str, Any]:
    if isinstance(item, (list, tuple)) and len(item) == 5:
        step, layer, group, srcs, dsts = item
    elif isinstance(item, dict):
        step = item.get("step")
        layer = item.get("layer")
        group = item.get("group")
        srcs = item.get("srcs", item.get("src"))
        dsts = item.get("dsts", item.get("dst"))
    else:
        raise ValueError(f"transmission {index} must be a dict or 5-tuple")
    if step is None or layer is None or group is None or srcs is None or dsts is None:
        raise ValueError(f"transmission {index} is missing one of step/layer/group/srcs/dsts")
    return {
        "step": int(step),
        "layer": int(layer),
        "group": int(group),
        "srcs": _as_int_list(srcs, f"transmission {index}.srcs"),
        "dsts": _as_int_list(dsts, f"transmission {index}.dsts"),
    }


def _as_int_list(value: Any, field: str) -> list[int]:
    values = value if isinstance(value, list) else [value]
    if not values:
        raise ValueError(f"{field} must not be empty")
    return [int(item) for item in values]
