#!/usr/bin/env python3
"""AST-isolated differential for the pinned Python ResourceUpdator leaf."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import struct
import subprocess
from typing import Any


SOURCE_SHA256 = "9b7c14f8c6eaa5e8bc50a723b727fec7eff8f1c7d05f7bef0da0cc941c15ac85"

BOUNDARIES = {
    "assert_optimization":
        "Python -O can disable operator/owner/safety asserts; native typed validation is stable in every build.",
    "dynamic_protocols":
        "Python accepts arbitrary mappings, numeric objects, monkey patches, and truthiness; native values use frozen bool/int64/double lanes.",
    "unbounded_values":
        "Python integers and node labels are unbounded; native range failures are typed and never invoke signed-overflow UB.",
    "cold_strings":
        "Python owner/operator/attribute strings remain compatibility inputs; native operations are enums and prepared resource IDs.",
}


def load_updator(source: pathlib.Path):
    source = source.resolve()
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"ResourceUpdator source hash drift: {actual}")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    matches = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ResourceUpdator"
    ]
    if len(matches) != 1:
        raise RuntimeError("ResourceUpdator class inventory drift")
    isolated = ast.fix_missing_locations(
        ast.Module(body=matches, type_ignores=[]))
    namespace: dict[str, Any] = {"__name__": "__main__", "Optional": Any}
    exec(compile(isolated, str(source), "exec"), namespace)
    return namespace["ResourceUpdator"]


class NodeView:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, node):
        return self.values[node]


class LinkView:
    def __init__(self, values):
        self.values = {
            tuple(sorted(link)): attributes for link, attributes in values.items()
        }

    def __getitem__(self, link):
        return self.values[tuple(sorted(link))]


class FakeNetwork:
    def __init__(self, nodes, links):
        self.nodes = NodeView(nodes)
        self.links = LinkView(links)


class FakeLinkResource:
    def __init__(self, name: str):
        self.name = name

    def update_path(self, virtual_link, physical, path, operator, safe=True):
        if len(path) <= 1:
            raise ValueError("short path")
        for link in zip(path[:-1], path[1:]):
            target = physical.links[link]
            value = virtual_link[self.name]
            if operator in ("+", "add"):
                target[self.name] += value
            elif operator in ("-", "sub"):
                if safe:
                    assert target[self.name] >= value
                target[self.name] -= value
            else:
                raise NotImplementedError


def fixture(ResourceUpdator, duplicate_bandwidth=False, empty=False):
    virtual = FakeNetwork(
        nodes={
            0: {"cpu": 2, "memory": 1.5, "flag": True},
            1: {"cpu": 3, "memory": 2.5, "flag": False},
        },
        links={(0, 1): {"bandwidth": 3, "burst": 1.5}},
    )
    physical = FakeNetwork(
        nodes={
            0: {"cpu": 10, "memory": 10.0, "flag": True},
            1: {"cpu": 20, "memory": 20.0, "flag": False},
            2: {"cpu": 30, "memory": 30.0, "flag": True},
        },
        links={
            (0, 1): {"bandwidth": 10, "burst": 8.0},
            (1, 2): {"bandwidth": 12, "burst": 9.0},
        },
    )
    attrs = [] if empty else [
        FakeLinkResource("bandwidth"),
        FakeLinkResource("burst"),
    ]
    if duplicate_bandwidth:
        attrs.append(FakeLinkResource("bandwidth"))
    return ResourceUpdator(attrs), virtual, physical


def number_token(value) -> str:
    if isinstance(value, bool):
        return "b:1" if value else "b:0"
    if isinstance(value, int):
        return f"i:{value}"
    return "d:" + struct.pack(">d", float(value)).hex()


def snapshot(physical) -> str:
    nodes = ",".join(
        f"{node}{{{number_token(values['cpu'])},"
        f"{number_token(values['memory'])},{number_token(values['flag'])}}}"
        for node, values in physical.nodes.values.items()
    )
    links = ",".join(
        f"{link[0]}:{link[1]}{{{number_token(values['bandwidth'])},"
        f"{number_token(values['burst'])}}}"
        for link, values in physical.links.values.items()
    )
    return f"nodes=[{nodes}];links=[{links}]"


def python_cases(ResourceUpdator):
    cases = {}

    updator, _, physical = fixture(ResourceUpdator)
    updator.update_resource(physical, "node", 0, "cpu", 3, "-", True)
    cases["node_sub"] = snapshot(physical)

    updator, _, physical = fixture(ResourceUpdator)
    updator.update_resource(physical, "node", 0, "memory", 2.5, "add", True)
    cases["node_add"] = snapshot(physical)

    updator, _, physical = fixture(ResourceUpdator)
    updator.update_node_resources(
        physical, 1, {"cpu": 2, "memory": 1.5}, "sub", True)
    cases["node_list"] = snapshot(physical)

    updator, _, physical = fixture(ResourceUpdator)
    error = "none"
    try:
        updator.update_node_resources(
            physical, 1, {"cpu": 2, "memory": 999.0}, "-", True)
    except AssertionError:
        error = "insufficient"
    cases["node_partial"] = f"error={error};{snapshot(physical)}"

    updator, _, physical = fixture(ResourceUpdator)
    updator.update_link_resources(
        physical, (1, 0), {"bandwidth": 2, "burst": 0.5}, "-", True)
    cases["link_sub"] = snapshot(physical)

    updator, _, physical = fixture(ResourceUpdator)
    updator.update_resource(
        physical, "link", (0, 1), "bandwidth", 13, "sub", False)
    cases["link_unsafe"] = snapshot(physical)

    updator, virtual, physical = fixture(
        ResourceUpdator, duplicate_bandwidth=True)
    updator.update_path_resources(
        virtual, physical, (0, 1), [0, 1, 2], "-", True)
    cases["path"] = snapshot(physical)

    updator, virtual, physical = fixture(ResourceUpdator)
    physical.links[(1, 2)]["bandwidth"] = 1
    error = "none"
    try:
        updator.update_path_resources(
            virtual, physical, (0, 1), [0, 1, 2], "-", True)
    except AssertionError:
        error = "insufficient"
    cases["path_partial"] = f"error={error};{snapshot(physical)}"

    updator, virtual, physical = fixture(ResourceUpdator, empty=True)
    updator.update_path_resources(
        virtual, physical, (99, 100), [0], "-", True)
    cases["empty"] = snapshot(physical)

    updator, _, physical = fixture(ResourceUpdator)
    updator.update_resource(
        physical, "node", 0, "flag", True, "+", False)
    cases["bool_add"] = snapshot(physical)
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
        raise RuntimeError(
            f"ResourceUpdator harness failed: {process.stderr.strip()}")
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

    ResourceUpdator = load_updator(args.source)
    expected = python_cases(ResourceUpdator)
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
            "ResourceUpdator differential mismatch:\n" +
            json.dumps(mismatches, indent=2, sort_keys=True))

    report = {
        "component": "core.controller.ResourceUpdator",
        "source_sha256": SOURCE_SHA256.upper(),
        "shared_case_count": len(expected),
        "native_unit_case_groups": 7,
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
