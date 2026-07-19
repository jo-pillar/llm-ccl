#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from llm_ccl.preparation import prepare_bundle
from llm_ccl.project_loader import discover_projects
from llm_ccl.reporting import write_report
from llm_ccl.resim import resim_selected
from llm_ccl.searching import DEFAULT_FLOW_SIM_BIN, search_all
from llm_ccl.selection import select_all


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BUNDLE_ROOT = REPO_ROOT / "experiments" / "llm-ccl"


def _add_search_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--jobs", type=int, default=1, help="number of cases searched concurrently")
    parser.add_argument("--max-generations", type=int, default=10000)
    parser.add_argument("--k-candidates", type=int, default=4)
    parser.add_argument("--eval-concurrency", type=int, default=1)
    parser.add_argument("--gen-concurrency", type=int, default=1)
    parser.add_argument("--llm-policy-pool-size", type=int, default=100)
    parser.add_argument("--flow-sim-bin", type=Path, default=DEFAULT_FLOW_SIM_BIN)
    parser.add_argument("--search-timeout", type=float)
    parser.add_argument("--api-base")
    parser.add_argument("--api-key")
    parser.add_argument("--max-tokens", type=int)


def _search(bundle: Path, args: argparse.Namespace) -> None:
    extra: list[str] = []
    for flag, value in (
        ("--api-base", args.api_base),
        ("--api-key", args.api_key),
        ("--max-tokens", args.max_tokens),
    ):
        if value is not None:
            extra.extend((flag, str(value)))
    search_all(
        bundle,
        model=args.model,
        jobs=args.jobs,
        max_generations=args.max_generations,
        k_candidates=args.k_candidates,
        eval_concurrency=args.eval_concurrency,
        gen_concurrency=args.gen_concurrency,
        llm_policy_pool_size=args.llm_policy_pool_size,
        flow_sim_bin=args.flow_sim_bin,
        timeout=args.search_timeout,
        extra_args=extra,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LLM-CCL: prepare, search all cases, select each best, then resim once at the end"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("projects", help="list available experiment projects")

    prepare = sub.add_parser("prepare", help="render a new experiment bundle")
    prepare.add_argument("project")
    prepare.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    prepare.add_argument("--launch-id", default=f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}")
    prepare.add_argument("--case", action="append", dest="cases")

    search = sub.add_parser("search", help="search every unfinished case")
    search.add_argument("bundle", type=Path)
    _add_search_args(search)

    select = sub.add_parser("select", help="select the fastest valid candidate per case")
    select.add_argument("bundle", type=Path)

    resim = sub.add_parser("resim", help="resim selected best candidates after all searches finish")
    resim.add_argument("bundle", type=Path)
    resim.add_argument("--synthesize-bin", type=Path, required=True)
    resim.add_argument("--flow-sim-bin", type=Path, default=DEFAULT_FLOW_SIM_BIN)
    resim.add_argument("--resim-timeout", type=float)

    report = sub.add_parser("report", help="write JSON and CSV reports")
    report.add_argument("bundle", type=Path)

    run = sub.add_parser("run", help="run the complete pipeline in the required order")
    run.add_argument("project")
    run.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    run.add_argument("--launch-id", default=f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}")
    run.add_argument("--case", action="append", dest="cases")
    run.add_argument("--synthesize-bin", type=Path, required=True)
    run.add_argument("--resim-timeout", type=float)
    _add_search_args(run)
    return parser


def main() -> None:
    args = _parser().parse_args()
    projects = discover_projects()
    if args.command == "projects":
        for name, project in projects.items():
            print(f"{name}\t{len(project.cases)} cases")
        return

    if args.command in {"prepare", "run"}:
        if args.project not in projects:
            raise SystemExit(f"unknown project {args.project!r}; use 'projects' to list choices")
        bundle = prepare_bundle(
            projects[args.project],
            args.bundle_root,
            args.launch_id,
            set(args.cases) if args.cases else None,
        )
        print(bundle)
        if args.command == "prepare":
            return
        _search(bundle, args)
        select_all(bundle)
        resim_selected(
            bundle,
            synthesize_bin=args.synthesize_bin,
            flow_sim_bin=args.flow_sim_bin,
            timeout=args.resim_timeout,
        )
        json_path, csv_path = write_report(bundle)
        print(json_path)
        print(csv_path)
        return

    bundle = args.bundle.resolve()
    if args.command == "search":
        _search(bundle, args)
    elif args.command == "select":
        select_all(bundle)
    elif args.command == "resim":
        resim_selected(
            bundle,
            synthesize_bin=args.synthesize_bin,
            flow_sim_bin=args.flow_sim_bin,
            timeout=args.resim_timeout,
        )
    elif args.command == "report":
        json_path, csv_path = write_report(bundle)
        print(json_path)
        print(csv_path)


if __name__ == "__main__":
    main()
