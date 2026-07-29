#!/usr/bin/env python3
"""AST-isolated differential for the pinned Python Counter leaf."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import struct
import subprocess
from typing import Any, Dict, List, Optional, Union

import numpy as np


SOURCE_SHA256 = "574745f86e99cd656cb0165330e3196b4f5ebf4eaa0687b076b8d9602db4d637"

BOUNDARIES = {
    "pandas_summary_records":
        "The pinned oracle image has no Pandas; summary_records parity is classified here and covered by the native typed-record unit gate.",
    "legacy_summary_csv":
        "Python summary_csv reaches an unbound instance-method call after CSV parsing; the native legacy CSV behavior is covered by its focused unit gate.",
    "dynamic_protocols":
        "Python accepts arbitrary mappings, custom numerics, monkey patches, and arbitrary-precision aggregate integers; native Counter uses frozen typed IDs and int64/double lanes.",
    "native_workers":
        "Configured deterministic gather workers are a native extension; the pinned Python methods expose scalar execution only.",
}


class FakePandas:
    class DataFrame:
        pass


class FakeAttribute:
    def __init__(self, name: str, attr_type: str = "resource"):
        self.name = name
        self.type = attr_type


def create_attrs(settings):
    return {attr.name: attr for attr in settings}


def load_counter(source: pathlib.Path):
    source = source.resolve()
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"Counter source hash drift: {actual}")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    matches = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Counter"
    ]
    if len(matches) != 1:
        raise RuntimeError("Counter class inventory drift")
    isolated = ast.fix_missing_locations(
        ast.Module(body=matches, type_ignores=[]))
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "BaseNetwork": Any,
        "Dict": Dict,
        "DictConfig": Any,
        "List": List,
        "OmegaConf": Any,
        "Optional": Optional,
        "Solution": Any,
        "Union": Union,
        "VirtualNetwork": Any,
        "create_link_attrs_from_setting": create_attrs,
        "create_node_attrs_from_setting": create_attrs,
        "np": np,
        "pd": FakePandas,
    }
    exec(compile(isolated, str(source), "exec"), namespace)
    return namespace["Counter"]


class LinkView:
    def __init__(self, values):
        self.values = {
            tuple(sorted(link)): attributes
            for link, attributes in values.items()
        }

    def __getitem__(self, link):
        return self.values[tuple(sorted(link))]

    def __iter__(self):
        return iter(self.values)


class FakeNetwork:
    def __init__(self, nodes, links, lifetime_marker=3.0):
        self.nodes = nodes
        self.links = LinkView(links)
        if lifetime_marker is not None:
            self.lifetime = lifetime_marker

    def get_node_attrs_data(self, attrs):
        return [
            [values[attr.name] for values in self.nodes.values()]
            for attr in attrs
        ]

    def get_link_attrs_data(self, attrs):
        return [
            [values[attr.name] for values in self.links.values.values()]
            for attr in attrs
        ]


class FakeSolution(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def to_dict(self):
        return dict(self)


NODE_ATTRS = [FakeAttribute("cpu"), FakeAttribute("memory")]
LINK_ATTRS = [FakeAttribute("bw")]


def make_counter(Counter, node_attrs=None, link_attrs=None):
    return Counter(
        NODE_ATTRS if node_attrs is None else node_attrs,
        LINK_ATTRS if link_attrs is None else link_attrs,
        [],
        {})


def make_network(with_lifetime=True):
    return FakeNetwork(
        {
            0: {"cpu": 2, "memory": 1.5},
            1: {"cpu": 3, "memory": 2.5},
            2: {"cpu": 4, "memory": 3.5},
        },
        {
            (0, 1): {"bw": 5},
            (1, 2): {"bw": 7},
        },
        3.0 if with_lifetime else None)


def make_solution():
    return FakeSolution({
        "result": False,
        "place_result": True,
        "route_result": True,
        "early_rejection": False,
        "node_slots": {0: 10, 2: 12},
        "link_paths": {
            (0, 1): [(0, 1), (1, 2)],
            (1, 2): [],
        },
        "link_paths_info": {
            ((0, 1), (0, 1)): {"bw": 5},
            ((0, 1), (1, 2)): {"bw": 5},
        },
        "v_net_demand": 0.0,
        "v_net_node_demand": 0.0,
        "v_net_link_demand": 0.0,
        "v_net_node_revenue": 0.0,
        "v_net_link_revenue": 0.0,
        "v_net_revenue": 0.0,
        "v_net_node_cost": 0.0,
        "v_net_link_cost": 0.0,
        "v_net_path_cost": 0.0,
        "v_net_cost": 0.0,
        "v_net_r2c_ratio": 0.0,
        "v_net_time_revenue": 0.0,
        "v_net_time_cost": 0.0,
        "v_net_time_rc_ratio": 0.0,
    })


def seed_metrics(solution):
    solution.update({
        "v_net_demand": 4.0,
        "v_net_node_demand": 41.0,
        "v_net_link_demand": 42.0,
        "v_net_node_revenue": 43.0,
        "v_net_link_revenue": 44.0,
        "v_net_revenue": 45.0,
        "v_net_node_cost": 46.0,
        "v_net_link_cost": 47.0,
        "v_net_path_cost": 48.0,
        "v_net_cost": 49.0,
        "v_net_r2c_ratio": 50.0,
        "v_net_time_revenue": 51.0,
        "v_net_time_cost": 52.0,
        "v_net_time_rc_ratio": 53.0,
    })


def double_token(value) -> str:
    return "d:" + struct.pack(">d", float(value)).hex()


def number_token(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        return f"i:{int(value)}"
    if isinstance(value, (int, np.integer)):
        return f"i:{int(value)}"
    if isinstance(value, (float, np.floating)):
        return double_token(value)
    raise RuntimeError(f"unexpected Counter numeric lane: {type(value)!r}")


def optional_size(solution, name) -> str:
    return str(solution[name]) if name in solution else "none"


def solution_payload(solution) -> str:
    return (
        f"counts={optional_size(solution, 'num_placed_nodes')},"
        f"{optional_size(solution, 'num_routed_links')};"
        f"demand={double_token(solution['v_net_node_demand'])},"
        f"{double_token(solution['v_net_link_demand'])},"
        f"{double_token(solution['v_net_demand'])};"
        f"revenue={double_token(solution['v_net_node_revenue'])},"
        f"{double_token(solution['v_net_link_revenue'])},"
        f"{double_token(solution['v_net_revenue'])};"
        f"cost={double_token(solution['v_net_node_cost'])},"
        f"{double_token(solution['v_net_link_cost'])},"
        f"{double_token(solution['v_net_path_cost'])},"
        f"{double_token(solution['v_net_cost'])};"
        f"ratio={double_token(solution['v_net_r2c_ratio'])};"
        f"time={double_token(solution['v_net_time_revenue'])},"
        f"{double_token(solution['v_net_time_cost'])},"
        f"{double_token(solution['v_net_time_rc_ratio'])};"
        f"flags={1 if solution['result'] else 0},"
        f"{1 if solution['place_result'] else 0},"
        f"{1 if solution['route_result'] else 0},"
        f"{1 if solution['early_rejection'] else 0}"
    )


def run_count_success(Counter):
    counter = make_counter(Counter)
    network = make_network()
    solution = make_solution()
    seed_metrics(solution)
    solution.update({
        "result": True,
        "place_result": False,
        "route_result": False,
        "early_rejection": True,
    })
    counter.count_solution(network, solution)
    return solution_payload(solution)


def python_cases(Counter):
    counter = make_counter(Counter)
    network = make_network()
    cases = {
        "sum_node_mixed": number_token(
            counter.calculate_sum_node_resource(network)),
        "sum_link_integer": number_token(
            counter.calculate_sum_link_resource(network)),
        "sum_network_both": number_token(
            counter.calculate_sum_network_resource(network, True, True)),
        "sum_network_node_only": number_token(
            counter.calculate_sum_network_resource(network, True, False)),
        "sum_network_link_only": number_token(
            counter.calculate_sum_network_resource(network, False, True)),
        "sum_network_disabled": number_token(
            counter.calculate_sum_network_resource(network, False, False)),
    }

    numeric_counter = make_counter(
        Counter, node_attrs=[FakeAttribute("value")], link_attrs=[])
    bool_int = FakeNetwork(
        {0: {"value": True}, 1: {"value": 2}}, {}, 3.0)
    cases["sum_bool_int_promotion"] = number_token(
        numeric_counter.calculate_sum_node_resource(bool_int))
    wrap = FakeNetwork(
        {
            0: {"value": np.iinfo(np.int64).max},
            1: {"value": 1},
        },
        {},
        3.0)
    cases["sum_int64_wrap"] = number_token(
        numeric_counter.calculate_sum_node_resource(wrap))

    solution = make_solution()
    cases["helper_link_cost"] = number_token(
        counter.calculate_v_net_link_cost(network, solution))
    cases["helper_total_cost"] = number_token(
        counter.calculate_v_net_cost(network, solution))
    cases["helper_revenue"] = number_token(
        counter.calculate_v_net_revenue(network, solution))

    solution = make_solution()
    counter.count_partial_solution(network, solution)
    cases["partial_success"] = solution_payload(solution)

    solution = make_solution()
    seed_metrics(solution)
    del solution["link_paths_info"][((0, 1), (1, 2))]
    error = "none"
    try:
        counter.count_partial_solution(network, solution)
    except KeyError:
        error = "missing_info"
    cases["partial_missing_info"] = (
        f"error={error};" + solution_payload(solution))

    empty_node = make_counter(Counter, node_attrs=[], link_attrs=LINK_ATTRS)
    solution = make_solution()
    seed_metrics(solution)
    error = "none"
    try:
        empty_node.count_partial_solution(network, solution)
    except ZeroDivisionError:
        error = "empty_node"
    cases["partial_empty_node_selection"] = (
        f"error={error};" + solution_payload(solution))

    cases["count_success_demand_bug"] = run_count_success(Counter)

    solution = make_solution()
    seed_metrics(solution)
    solution.update({
        "result": False,
        "place_result": False,
        "route_result": True,
        "early_rejection": True,
    })
    counter.count_solution(network, solution)
    cases["count_failure_stale_fields"] = solution_payload(solution)

    no_lifetime = make_network(with_lifetime=False)
    solution = make_solution()
    seed_metrics(solution)
    solution.update({
        "result": True,
        "place_result": False,
        "route_result": False,
        "early_rejection": True,
    })
    error = "none"
    try:
        counter.count_solution(no_lifetime, solution)
    except AttributeError:
        error = "missing_lifetime"
    cases["count_missing_lifetime_partial"] = (
        f"error={error};" + solution_payload(solution))

    solution = make_solution()
    seed_metrics(solution)
    solution.update({
        "result": True,
        "place_result": False,
        "route_result": False,
        "early_rejection": True,
    })
    del solution["link_paths_info"][((0, 1), (1, 2))]
    error = "none"
    try:
        counter.count_solution(network, solution)
    except KeyError:
        error = "missing_info"
    cases["count_missing_info_partial"] = (
        f"error={error};" + solution_payload(solution))

    canonical_sum = number_token(
        counter.calculate_sum_network_resource(network, True, True))
    cases["sum_workers_0_1_2_8"] = "|".join(
        [canonical_sum, canonical_sum, canonical_sum, canonical_sum])
    canonical_count = run_count_success(Counter)
    cases["count_workers_0_1_2_8"] = "|".join(
        [canonical_count, canonical_count, canonical_count, canonical_count])
    return cases


def cpp_cases(harness: pathlib.Path):
    process = subprocess.run(
        [str(harness), "differential"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"Counter harness failed: {process.stderr.strip()}")
    result = {}
    for line in process.stdout.splitlines():
        name, payload = line.split("\t", 1)
        if name in result:
            raise RuntimeError(f"duplicate native case: {name}")
        result[name] = payload
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    Counter = load_counter(args.source)
    expected = python_cases(Counter)
    actual = cpp_cases(args.harness)
    if actual.keys() != expected.keys():
        raise RuntimeError(
            f"case inventory drift: Python={list(expected)}, C++={list(actual)}")
    mismatches = {
        name: {"python": expected[name], "cpp": actual[name]}
        for name in expected
        if expected[name] != actual[name]
    }
    if mismatches:
        raise RuntimeError(
            "Counter differential mismatch:\n" +
            json.dumps(mismatches, indent=2, sort_keys=True))

    report = {
        "component": "core.Counter",
        "source_sha256": SOURCE_SHA256.upper(),
        "shared_case_count": len(expected),
        "python_only_boundaries": BOUNDARIES,
        "status": "PASS",
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
