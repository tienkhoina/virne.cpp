#!/usr/bin/env python3
"""Pinned direct-source differential for non-ML VirtualNetwork."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import struct
import subprocess
import sys
from typing import Any

import compare_base_network as base_oracle


SOURCE_SHA256 = "0b73e73feb43793559976f08fd93ed227698b810ed8741ea5bcc1534adb3768c"
BOUNDARIES = {
    "arbitrary_reflection":
        "Python permits arbitrary attributes and descriptors; fixed native request fields are direct optionals.",
    "arbitrary_numpy_dtype":
        "Object/string/custom NumPy dtype promotion and user numeric protocols are outside AttrValue numeric lanes.",
    "numpy_pairwise_extremes":
        "Adversarial overflow and payload-sensitive NumPy reduction details remain a dynamic numeric boundary.",
    "sparse_hashable_labels":
        "Python accepts arbitrary hashable labels; the frozen native graph uses dense vertices.",
    "cached_property_deletion":
        "Python deletes a cached_property through __dict__; native code exposes explicit invalidation.",
}


def verify(path: pathlib.Path) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"VirtualNetwork source hash drift: {actual}")


def double_token(value: float) -> str:
    return "d:" + struct.pack(">d", float(value)).hex()


def load_oracle(source: pathlib.Path, base_source: pathlib.Path):
    source = source.resolve()
    verify(source)
    base = base_oracle.load_oracle(base_source)
    try:
        module = base_oracle.execute_module("virne.network.virtual_network", source)
    except Exception:
        base_oracle.unload_oracle()
        raise
    return module


def unload_oracle() -> None:
    sys.modules.pop("virne.network.virtual_network", None)
    base_oracle.unload_oracle()


def make_fixture(module):
    graph = module.nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
    config = {
        "node_attrs_setting": [
            {"name": "cpu", "owner": "node", "type": "resource"},
            {"name": "peak", "owner": "node", "type": "resource"},
        ],
        "link_attrs_setting": [
            {"name": "bw", "owner": "link", "type": "resource"},
            {"name": "peak_bw", "owner": "link", "type": "resource"},
        ],
    }
    value = module.VirtualNetwork(
        graph, config=config, id=7, arrival_time=2.5, lifetime=9
    )
    module.nx.set_node_attributes(value, {0: 1, 1: 2, 2: 3, 3: 4}, "cpu")
    module.nx.set_node_attributes(
        value, {0: 0.5, 1: 1.5, 2: 2.5, 3: 3.5}, "peak"
    )
    module.nx.set_edge_attributes(value, {(0, 1): 2, (1, 2): 3, (2, 3): 4}, "bw")
    module.nx.set_edge_attributes(
        value, {(0, 1): 0.25, (1, 2): 1.25, (2, 3): 2.25}, "peak_bw"
    )
    return value


def demand_payload(value) -> str:
    return (
        "node=" + double_token(value.total_node_resource_demand)
        + ";link=" + double_token(value.total_link_resource_demand)
        + ";total=" + double_token(value.total_resource_demand)
    )


def oracle_cases(module) -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []
    empty = module.VirtualNetwork()
    cases.append((
        "empty",
        "id=" + ("1" if hasattr(empty, "id") else "0")
        + ";arrival=" + ("1" if hasattr(empty, "arrival_time") else "0")
        + ";lifetime=" + ("1" if hasattr(empty, "lifetime") else "0")
        + ";" + demand_payload(empty),
    ))

    fixed = make_fixture(module)
    cases.append((
        "fixed_metadata",
        f"id={fixed.id};arrival={double_token(fixed.arrival_time)};"
        f"lifetime={double_token(fixed.lifetime)}",
    ))
    fixed.id = 11
    fixed.arrival_time = 4.25
    fixed.lifetime = 12.5
    cases.append((
        "fixed_assignment",
        f"id={fixed.id};arrival={double_token(fixed.arrival_time)};"
        f"lifetime={double_token(fixed.lifetime)}",
    ))

    expected_demand = demand_payload(make_fixture(module))
    for workers in (0, 1, 2, 8):
        cases.append((f"demand_w{workers}", expected_demand))

    cached = make_fixture(module)
    before = cached.total_resource_demand
    module.nx.set_node_attributes(cached, {node: 100 for node in cached.nodes}, "cpu")
    cases.append((
        "cached_stale",
        "before=" + double_token(before)
        + ";node=" + double_token(cached.total_node_resource_demand)
        + ";total=" + double_token(cached.total_resource_demand),
    ))
    cached.__dict__.pop("total_resource_demand", None)
    cases.append(("native_cache_invalidate", demand_payload(cached)))

    cases.append(("no_resources", demand_payload(module.VirtualNetwork())))

    graph = module.nx.Graph()
    graph.add_edge(0, 1)
    lanes = module.VirtualNetwork(
        graph,
        config={
            "node_attrs_setting": [
                {"name": "flag", "owner": "node", "type": "resource"}
            ],
            "link_attrs_setting": [
                {"name": "units", "owner": "link", "type": "resource"}
            ],
        },
    )
    module.nx.set_node_attributes(lanes, {0: True, 1: False}, "flag")
    module.nx.set_edge_attributes(lanes, {(0, 1): -3}, "units")
    cases.append(("bool_integer", demand_payload(lanes)))

    ragged = make_fixture(module)
    del ragged.nodes[3]["cpu"]
    cases.append((
        "ragged_zero",
        "node=" + double_token(ragged.total_node_resource_demand)
        + ";total=" + double_token(ragged.total_resource_demand),
    ))

    nonnumeric = make_fixture(module)
    nonnumeric.edges[0, 1]["bw"] = "bad"
    cases.append((
        "nonnumeric_zero",
        "link=" + double_token(nonnumeric.total_link_resource_demand)
        + ";total=" + double_token(nonnumeric.total_resource_demand),
    ))

    source = make_fixture(module)
    source.id = 42
    source_cached = source.total_resource_demand
    clone = source.clone()
    module.nx.set_node_attributes(clone, {node: 0 for node in clone.nodes}, "cpu")
    cases.append((
        "clone",
        f"id={clone.id};clone_cached={double_token(clone.total_resource_demand)};"
        f"source_cached={double_token(source_cached)};"
        f"source_node={double_token(source.total_node_resource_demand)}",
    ))

    cases.append(("native_move", "node_binding=1;id=42"))
    return cases


def parse_cpp(path: pathlib.Path) -> list[tuple[str, str]]:
    process = subprocess.run(
        [str(path)], check=False, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True
    )
    if process.returncode != 0:
        raise RuntimeError(f"VirtualNetwork harness failed: {process.stderr.strip()}")
    result: list[tuple[str, str]] = []
    for line in process.stdout.splitlines():
        fields = line.split("|")
        if len(fields) != 3 or not fields[0].startswith("case=") or fields[1] != "ok":
            raise RuntimeError(f"malformed VirtualNetwork line: {line!r}")
        result.append((fields[0][5:], bytes.fromhex(fields[2]).decode("utf-8")))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--base-source", type=pathlib.Path, required=True)
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    module = load_oracle(args.source, args.base_source)
    try:
        expected = oracle_cases(module)
        numpy_version = module.np.__version__
        networkx_version = module.nx.__version__
    finally:
        unload_oracle()
    actual = parse_cpp(args.harness)
    if actual != expected:
        for got, wanted in zip(actual, expected):
            if got != wanted:
                raise RuntimeError(
                    f"VirtualNetwork mismatch {wanted[0]}: C++={got!r}, Python={wanted!r}"
                )
        raise RuntimeError(
            f"VirtualNetwork inventory mismatch: C++={actual!r}, Python={expected!r}"
        )

    payload: dict[str, Any] = {
        "source_sha256": SOURCE_SHA256,
        "base_source_sha256": base_oracle.SOURCE_SHA256,
        "numpy_version": numpy_version,
        "networkx_version": networkx_version,
        "shared_cases": [name for name, _ in expected[:-1]],
        "native_extension_cases": [expected[-1][0]],
        "python_only_boundaries": BOUNDARIES,
        "case_count": len(expected) + len(BOUNDARIES),
        "result": "PASS",
    }
    if args.output:
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        f"VirtualNetwork differential: PASS ({len(expected) - 1} shared + "
        f"1 native + {len(BOUNDARIES)} boundaries)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
