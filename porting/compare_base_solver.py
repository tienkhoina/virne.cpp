#!/usr/bin/env python3
"""AST-isolated exact differential for the pinned BaseSolver leaf."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import pathlib
import subprocess
from types import SimpleNamespace
from typing import Any, Dict, Optional, Type


SOURCE_SHA256 = (
    "196573e631654a6d14888685b93977601412f110a9b5620398850cce121a2806"
)
WORKERS = (1, 2, 8)


def load_leaf(source: pathlib.Path):
    source = source.resolve()
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"BaseSolver source hash drift: {actual}")

    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    names = [node.name for node in classes]
    if names != ["Solver", "SolverRegistry"]:
        raise RuntimeError(f"BaseSolver class inventory drift: {names}")

    class Dummy:
        pass

    isolated = ast.fix_missing_locations(
        ast.Module(body=classes, type_ignores=[]))
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "abc": __import__("abc"),
        "os": os,
        "Controller": Dummy,
        "Recorder": Dummy,
        "Counter": Dummy,
        "Logger": Dummy,
        "Solution": Dummy,
        "SolutionStepEnvironment": Dummy,
        "Optional": Optional,
        "Dict": Dict,
        "Type": Type,
    }
    exec(compile(isolated, str(source), "exec"), namespace)
    return namespace["Solver"], namespace["SolverRegistry"]


def make_config(
    root: str = "/root/save",
    solver_name: str = "demo",
    run_id: str = "r1",
):
    return SimpleNamespace(
        experiment=SimpleNamespace(
            seed=17,
            save_root_dir=root,
            run_id=run_id,
        ),
        solver=SimpleNamespace(
            solver_name=solver_name,
            reusable=True,
            node_ranking_method="nps",
            link_ranking_method="ffd",
            matching_mathod="l2s2",
            shortest_method="all_shortest",
            k_shortest=-7,
            allow_rejection=True,
            allow_revocable=False,
        ),
    )


def python_payload(Solver, SolverRegistry):
    controller = object()
    recorder = object()
    counter = object()
    logger = object()
    config = make_config()
    solver = Solver(
        controller,
        recorder,
        counter,
        logger,
        config,
        verbose=7,
        ignored_keyword="ignored",
    )
    explicit = {
        "seed": solver.seed,
        "verbose": solver.verbose,
        "num_arrived": solver.num_arrived_v_nets,
        "save_dir": solver.save_dir,
        "reusable": solver.reusable,
        "node_ranking_method": solver.node_ranking_method,
        "link_ranking_method": solver.link_ranking_method,
        "matching_mathod": solver.matching_mathod,
        "shortest_method": solver.shortest_method,
        "k_shortest": solver.k_shortest,
        "allow_rejection": solver.allow_rejection,
        "allow_revocable": solver.allow_revocable,
    }

    default_solver = Solver(
        controller, recorder, counter, logger, make_config())
    ready = "none" if default_solver.ready() is None else "wrong"
    try:
        default_solver.solve({})
    except NotImplementedError:
        solve = "not_implemented"
    else:
        solve = "wrong"

    SolverRegistry._registry = {}

    class Alpha(Solver):
        pass

    class Beta(Solver):
        pass

    ignored_environment = object()
    returned_alpha = SolverRegistry.register(
        "alpha", "heuristic", env_cls=ignored_environment)(Alpha)
    returned_beta = SolverRegistry.register("beta", "exact")(Beta)
    registry_order = list(SolverRegistry.list_registered())
    categories = [returned_alpha.type, returned_beta.type]
    copied = SolverRegistry.list_registered()
    copied.clear()
    copy_isolated = len(SolverRegistry.list_registered()) == 2

    class DuplicateCandidate(Solver):
        pass

    try:
        SolverRegistry.register("alpha", "meta_heuristic")(
            DuplicateCandidate)
    except ValueError:
        duplicate = (
            "duplicate" if len(SolverRegistry._registry) == 2 else "wrong")
    else:
        duplicate = "wrong"

    try:
        SolverRegistry.get("missing")
    except NotImplementedError:
        missing = "not_implemented"
    else:
        missing = "wrong"

    created = SolverRegistry.get("alpha")(
        controller, recorder, counter, logger, config, verbose=7)
    factory_identity = (
        created.controller is controller
        and created.recorder is recorder
        and created.counter is counter
        and created.logger is logger
    )

    return {
        "constructor_explicit": explicit,
        "constructor_default_verbose": default_solver.verbose,
        "path_absolute_solver": Solver(
            controller,
            recorder,
            counter,
            logger,
            make_config("/root", "/absolute", "run"),
        ).save_dir,
        "path_absolute_run": Solver(
            controller,
            recorder,
            counter,
            logger,
            make_config("/root", "demo", "/absolute-run"),
        ).save_dir,
        "path_empty": Solver(
            controller,
            recorder,
            counter,
            logger,
            make_config("/root", "", ""),
        ).save_dir,
        "ready": ready,
        "solve": solve,
        "registry_order": registry_order,
        "registry_categories": categories,
        "registry_copy": copy_isolated,
        "registry_duplicate": duplicate,
        "registry_missing": missing,
        "registry_factory_identity": factory_identity,
    }


def run_native(harness: pathlib.Path, workers: int):
    completed = subprocess.run(
        [str(harness), "--workers", str(workers)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return json.loads(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=pathlib.Path)
    parser.add_argument("--harness", required=True, type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    Solver, SolverRegistry = load_leaf(args.source)
    expected = python_payload(Solver, SolverRegistry)
    for workers in WORKERS:
        actual = run_native(args.harness.resolve(), workers)
        if actual != expected:
            raise RuntimeError(
                f"workers={workers} mismatch\n"
                f"python={json.dumps(expected, sort_keys=True)}\n"
                f"native={json.dumps(actual, sort_keys=True)}")

    result = {
        "component": "solver.base_solver",
        "status": "PASS",
        "source_sha256": SOURCE_SHA256.upper(),
        "shared_case_count": len(expected),
        "native_workers": list(WORKERS),
        "boundaries": {
            "config_identity": (
                "Python retains the dynamic config object identity; native "
                "copies one typed fixed-field snapshot before solve loops."),
            "dynamic_kwargs": (
                "Python ignores every constructor kwarg except verbose. "
                "Native accepts only the typed SolverConfigInput boundary."),
            "environment_class": (
                "Python registry env_cls is dead input. Native reserves no "
                "environment dependency in the base registry."),
            "registry_freeze": (
                "Native explicitly freezes registration, then reads compact "
                "SolverId slots concurrently without name lookup."),
            "learning": (
                "Learning categories are enum seams only; no RL/Torch/CUDA "
                "factory or import is linked."),
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
