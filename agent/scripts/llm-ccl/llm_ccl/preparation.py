from __future__ import annotations

import hashlib
import json
import shutil
import string
import uuid
from datetime import datetime, timezone
from pathlib import Path

from syccl_agents.config_render import render_syccl_config

from .manifest import ManifestStore
from .models import CaseSpec, ExperimentSpec
from .topology import render_case_topology


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_instruction(case: CaseSpec, template: str, topology_source: str, hosts: int, gpus_per_host: int, nics: int) -> str:
    values = {
        "GPU_NUM": str(case.scale.gpu_count),
        "Collective": case.collective,
        "COLLECTIVE": case.collective,
        "TOPOLOGY": topology_source,
        "TOPODSL": topology_source,
        "MESSAGE_SIZE": str(case.coll_byte),
        "TOTAL_MESSAGE_SIZE": str(case.total_message_size),
        "HOST_NUM": str(hosts),
        "HOST_GPU_NUM": str(gpus_per_host),
        "NIC_NUM": str(nics),
    }
    return string.Template(template).safe_substitute(values)


def _render_initial_program(case: CaseSpec, source: str) -> str:
    start_marker = "# EVOLVE-BLOCK-START"
    end_marker = "# EVOLVE-BLOCK-END"
    start = source.find(start_marker)
    end = source.find(end_marker)
    if start < 0 or end < start:
        raise ValueError("initial program must contain one EVOLVE-BLOCK")
    block = source[start + len(start_marker):end].strip("\n")
    return (
        f"GPU_NUM = {case.scale.gpu_count}\n\n"
        f"{start_marker}\n{block}\n{end_marker}\n\n"
        "def run_code():\n"
        "  return construct_sketches(GPU_NUM)\n"
    )


def _case_payload(case: CaseSpec) -> dict:
    return {
        "scale": case.scale.name,
        "gpu_count": case.scale.gpu_count,
        "collective": case.collective,
        "total_message_size": case.total_message_size,
        "coll_byte": case.coll_byte,
        "search": {"status": "pending", "attempts": [], "latest_successful_attempt": None},
        "selection": {"status": "pending"},
        "resim": {"status": "pending"},
    }


def prepare_bundle(
    project: ExperimentSpec,
    bundle_root: Path,
    launch_id: str,
    case_ids: set[str] | None = None,
) -> Path:
    bundle = bundle_root.expanduser().resolve() / project.name / launch_id
    if bundle.exists():
        raise FileExistsError(bundle)
    selected = [case for case in project.cases if case_ids is None or case.case_id in case_ids]
    unknown = (case_ids or set()) - {case.case_id for case in project.cases}
    if unknown:
        raise ValueError(f"unknown case IDs: {sorted(unknown)}")
    if not selected:
        raise ValueError("no cases selected")

    instruction_template = project.instruction_template.read_text(encoding="utf-8")
    initial_source = project.initial_program.read_text(encoding="utf-8")
    project_root = bundle.parent
    project_root.mkdir(parents=True, exist_ok=True)
    staging = project_root / f".{launch_id}.tmp-{uuid.uuid4().hex}"
    try:
        staging.mkdir()
        cases_payload: dict[str, dict] = {}
        for case in selected:
            rendered = render_case_topology(case, project.topology_template)
            config = render_syccl_config(rendered.topo.params)
            case_dir = staging / "cases" / case.case_id
            case_dir.mkdir(parents=True)
            (case_dir / "topodsl.py").write_text(rendered.source, encoding="utf-8")
            (case_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            (case_dir / "instruction.txt").write_text(
                _render_instruction(
                    case,
                    instruction_template,
                    rendered.topo.prompt_source,
                    rendered.topo.params.hosts,
                    rendered.topo.params.gpus_per_host,
                    rendered.topo.params.nics_per_host,
                ),
                encoding="utf-8",
            )
            (case_dir / "init_program.py").write_text(
                _render_initial_program(case, initial_source), encoding="utf-8"
            )
            cases_payload[case.case_id] = _case_payload(case)

        manifest = {
            "schema_version": 1,
            "project": project.name,
            "launch_id": launch_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sources": {
                "topology": {"path": str(project.topology_template), "sha256": _sha256(project.topology_template)},
                "instruction": {"path": str(project.instruction_template), "sha256": _sha256(project.instruction_template)},
                "initial_program": {"path": str(project.initial_program), "sha256": _sha256(project.initial_program)},
            },
            "cases": cases_payload,
        }
        ManifestStore(staging / "manifest.json").create(manifest)
        staging.rename(bundle)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return bundle
