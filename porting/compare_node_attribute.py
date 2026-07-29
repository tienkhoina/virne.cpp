#!/usr/bin/env python3
"""Direct-source Python/C++ differential for node attributes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import struct
import subprocess
import sys
import types
from typing import Any

import compare_dataset_core


SOURCE_SHA256 = "e90e286320e59ebbba9957b701f6f12acdc1785e4821de80ccc5a0ab0f3ee56a"
BASE_SHA256 = "103c5c16126ca76e782c2191ff0811b95ed88a0c3637f3a61e84c0b22df42e8e"
METHOD_SHA256 = "e17499af8e6ffbdb12f2100dd58abeab48dd06d92feb442ef192f8b9310b6b4f"


def bits(value: float) -> str:
    return f"{struct.unpack('>Q', struct.pack('>d', float(value)))[0]:016x}"


def hex_text(value: str) -> str:
    return value.encode("utf-8").hex()


def scalar(value: Any) -> str:
    if isinstance(value, (bool,)):
        return "b:1" if bool(value) else "b:0"
    if isinstance(value, (int,)):
        return f"i:{int(value)}"
    if isinstance(value, (float,)) or type(value).__module__.startswith("numpy"):
        return "d:" + bits(value)
    if isinstance(value, str):
        return "s:" + hex_text(value)
    raise RuntimeError(f"unsupported oracle scalar: {type(value)!r}")


def scalar_list(items) -> str:
    return ",".join(scalar(item) for item in items)


def position_list(items) -> str:
    return ";".join(
        f"{scalar(x)},{scalar(y)},d:{bits(radius)}" for x, y, radius in items
    )


def verify(path: pathlib.Path, expected: str, label: str):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"{label} source hash drift: {actual}")


def execute_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_oracle(
    source: pathlib.Path,
    base_source: pathlib.Path,
    method_source: pathlib.Path,
    dataset_source: pathlib.Path,
):
    verify(source, SOURCE_SHA256, "node_attribute")
    verify(base_source, BASE_SHA256, "base_attribute")
    verify(method_source, METHOD_SHA256, "attribute_method")
    dataset = compare_dataset_core.load_oracle(dataset_source)

    import networkx as nx

    virne = types.ModuleType("virne")
    virne.__path__ = []
    utils = types.ModuleType("virne.utils")
    utils.path_to_links = lambda path: path
    utils.generate_data_with_distribution = dataset.generate_data_with_distribution
    network = types.ModuleType("virne.network")
    network.__path__ = []
    attribute_package = types.ModuleType("virne.network.attribute")
    attribute_package.__path__ = []
    base_network_module = types.ModuleType("virne.network.base_network")

    class BaseNetwork(nx.Graph):
        def __init__(self, node_count: int = 0):
            super().__init__()
            self.add_nodes_from(range(node_count))
            self.node_attrs = {}
            self.link_attrs = {}

        @property
        def num_nodes(self):
            return self.number_of_nodes()

        @property
        def num_links(self):
            return self.number_of_edges()

    base_network_module.BaseNetwork = BaseNetwork
    replacements = {
        "virne": virne,
        "virne.utils": utils,
        "virne.network": network,
        "virne.network.attribute": attribute_package,
        "virne.network.base_network": base_network_module,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    names = [
        "virne.network.attribute.base_attribute",
        "virne.network.attribute.attribute_method",
        "virne.network.attribute.node_attribute",
    ]
    try:
        base = execute_module(names[0], base_source)
        method = execute_module(names[1], method_source)
        node = execute_module(names[2], source)
    except Exception:
        for name in names:
            sys.modules.pop(name, None)
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
        raise
    return node, base, method, dataset, BaseNetwork, previous, names


def restore(previous, names):
    for name in names:
        sys.modules.pop(name, None)
    for name, old in previous.items():
        if old is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old


def parse_harness(path: pathlib.Path) -> dict[str, list[str]]:
    process = subprocess.run(
        [str(path)], check=False, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"node harness failed: {process.stderr.strip()}")
    result: dict[str, list[str]] = {}
    passed = False
    for line in process.stdout.splitlines():
        if line == "status=PASS":
            passed = True
            continue
        if not line.startswith("case="):
            raise RuntimeError(f"malformed harness line: {line!r}")
        fields = line.split("|")
        name = fields[0][5:]
        if name in result:
            raise RuntimeError(f"duplicate harness case: {name}")
        result[name] = fields[1:]
    if not passed:
        raise RuntimeError("node harness did not report PASS")
    return result


def graph_adapter_cases(node_module, BaseNetwork) -> dict[str, list[str]]:
    result = {}
    for prefix in ("graph", "digraph"):
        graph = BaseNetwork(5)
        item = node_module.NodeAttribute("mixed", "node", "status")
        item.set_data(graph, [1, 2.5, True, "four", -5, 999])
        result[f"{prefix}_dense"] = ["ok", scalar_list(item.get_data(graph))]

        sparse = BaseNetwork(5)
        item.set_data(sparse, {3: 31, 9: 90, 1: "one"})
        result[f"{prefix}_sparse"] = ["ok", scalar_list(item.get_data(sparse))]
        try:
            item.get(sparse, 0)
        except KeyError as error:
            result[f"{prefix}_missing_get"] = [
                "error", "missing_attribute", None, type(error).__name__
            ]
        else:
            raise RuntimeError("Python missing node attribute unexpectedly succeeded")
        before = dict(sparse.nodes(data="mixed", default=None))
        try:
            item.set_data(sparse, [0.0] * 4)
        except IndexError as error:
            after = dict(sparse.nodes(data="mixed", default=None))
            if before != after:
                raise RuntimeError("Python short dense input partially mutated graph")
            result[f"{prefix}_short_dense"] = [
                "error", "dense_data_too_short", None, type(error).__name__
            ]
        else:
            raise RuntimeError("Python short dense input unexpectedly succeeded")
    return result


def construction_cases(node_module) -> dict[str, list[str]]:
    result = {}
    status = node_module.NodeStatusAttribute("alive")
    result["status_fields"] = [
        "ok",
        f"{status.name},{status.owner},{status.type},"
        f"{int(status.generative)},{int(status.is_constraint)}",
    ]
    try:
        node_module.NodeExtremaAttribute("cpu_max")
    except ValueError as error:
        result["extrema_missing_originator"] = [
            "error", "missing_originator", hex_text(str(error))
        ]
    extrema = node_module.NodeExtremaAttribute(
        "cpu_max", config={"originator": "cpu"}
    )
    result["extrema_fields"] = [
        "ok", f"{extrema.name},{extrema.owner},{extrema.type},"
        f"{extrema.originator},17,{int(extrema.is_constraint)}"
    ]
    resource = node_module.NodeResourceAttribute(
        "cpu", config={"constraint_restrictions": "soft"}
    )
    result["resource_fields"] = [
        "ok", f"{resource.name},{resource.owner},{resource.type},"
        f"{resource.constraint_restrictions},{resource.checking_level},"
        f"{int(resource.is_constraint)}"
    ]

    position = node_module.NodePositionAttribute("where")
    position.generative = True
    position.min_r = -0.25
    position.max_r = 0.75
    result["position_fields"] = [
        "ok", f"{position.name},{position.owner},{position.type},1,"
        f"{bits(position.min_r)},{bits(position.max_r)},"
        f"{position.constraint_restrictions},{int(position.is_constraint)}"
    ]
    return result


def resource_cases(node_module) -> dict[str, list[str]]:
    hard = node_module.NodeResourceAttribute("cpu")
    soft = node_module.NodeResourceAttribute(
        "cpu", config={"constraint_restrictions": "soft"}
    )
    result = {}
    for name, virtual, physical, policy in (
        ("resource_hard_pass", 7, 10.5, hard),
        ("resource_hard_fail", 12.0, 10.5, hard),
        ("resource_soft_fail", 12.0, 10.5, soft),
    ):
        flag, offset = policy.check_constraint_satisfiability(
            {"cpu": virtual}, {"cpu": physical}, "le"
        )
        result[name] = ["ok", f"{int(flag)},{scalar(offset)}"]

    virtual = {"cpu": 3}
    physical = {"cpu": 10}
    hard.update(virtual, physical, "-", True)
    hard.update(virtual, physical, "+", True)
    result["resource_update_roundtrip"] = ["ok", scalar(physical["cpu"])]
    try:
        hard.check_constraint_satisfiability({}, {"cpu": 10})
    except ValueError as error:
        result["resource_missing"] = [
            "error", "missing_resource_value", hex_text(str(error))
        ]
    try:
        hard.check_constraint_satisfiability({"cpu": "bad"}, {"cpu": 10})
    except TypeError as error:
        result["resource_nonnumeric"] = [
            "error", "non_numeric_resource", None, type(error).__name__
        ]
    try:
        hard.check_constraint_satisfiability({"cpu": "bad"}, {})
    except ValueError:
        result["resource_check_missing_precedence"] = [
            "error", "missing_resource_value"
        ]
    else:
        raise RuntimeError("resource check precedence unexpectedly succeeded")
    for name, method, safe in (
        ("resource_add_missing_precedence", "+", True),
        ("resource_sub_unsafe_missing_precedence", "-", False),
    ):
        try:
            hard.update({}, {"cpu": "bad"}, method, safe)
        except KeyError:
            result[name] = ["error", "missing_resource_value"]
        else:
            raise RuntimeError(f"{name} unexpectedly succeeded")
    try:
        hard.update({"cpu": "bad"}, {}, "-", True)
    except KeyError:
        result["resource_sub_safe_missing_precedence"] = [
            "error", "missing_resource_value"
        ]
    else:
        raise RuntimeError("safe subtract precedence unexpectedly succeeded")
    return result


def configure_position(
    node_module,
    distribution: str,
    dtype: str,
    minimum: float,
    maximum: float,
    **parameters,
):
    value = node_module.NodePositionAttribute()
    value.generative = True
    value.distribution = distribution
    value.dtype = dtype
    value.min_r = minimum
    value.max_r = maximum
    for key, item in parameters.items():
        setattr(value, key, item)
    return value


def generated_position_case(
    node_module,
    BaseNetwork,
    name: str,
    value,
    count: int,
    seed: int,
    use_public_generate: bool = True,
) -> list[str]:
    node_module.np.random.seed(seed)
    try:
        if use_public_generate:
            output = value.generate_data(BaseNetwork(count))
        else:
            output = value._generate_data(BaseNetwork(count))
    except Exception as error:
        next_bits = bits(node_module.np.random.random_sample())
        return ["error", type(error).__name__, hex_text(str(error)), next_bits]
    next_bits = bits(node_module.np.random.random_sample())
    return ["ok", position_list(output), next_bits]


def position_cases(node_module, BaseNetwork) -> dict[str, list[str]]:
    result = {}
    for workers in (0, 1, 2, 8):
        value = configure_position(
            node_module, "uniform", "float", -0.25, 0.75,
            low=-2.0, high=3.0,
        )
        result[f"position_uniform_float_w{workers}"] = generated_position_case(
            node_module, BaseNetwork,
            f"position_uniform_float_w{workers}", value, 17, 101,
        )
    integer = configure_position(
        node_module, "uniform", "int", -1.5, 2.5, low=-3, high=4
    )
    result["position_uniform_int"] = generated_position_case(
        node_module, BaseNetwork, "position_uniform_int", integer, 13, 102
    )
    reversed_clip = configure_position(
        node_module, "uniform", "float", 2.0, -1.0, low=-2.0, high=3.0
    )
    result["position_reversed_clip"] = generated_position_case(
        node_module, BaseNetwork, "position_reversed_clip", reversed_clip, 11, 103
    )
    signed_zero = configure_position(
        node_module, "normal", "float", -0.0, 0.0, loc=-0.0, scale=0.0
    )
    result["position_signed_zero"] = generated_position_case(
        node_module, BaseNetwork, "position_signed_zero", signed_zero, 7, 104
    )
    nan_minimum = configure_position(
        node_module, "normal", "float", float("nan"), 1.0,
        loc=0.0, scale=1.0,
    )
    result["position_nan_minimum"] = generated_position_case(
        node_module, BaseNetwork, "position_nan_minimum", nan_minimum, 5, 106
    )
    nan_maximum = configure_position(
        node_module, "normal", "float", -1.0, float("nan"),
        loc=0.0, scale=1.0,
    )
    result["position_nan_maximum"] = generated_position_case(
        node_module, BaseNetwork, "position_nan_maximum", nan_maximum, 5, 107
    )
    boolean = configure_position(
        node_module, "normal", "bool", 0.0, 1.0, loc=-1.0, scale=2.0
    )
    result["position_normal_bool"] = generated_position_case(
        node_module, BaseNetwork, "position_normal_bool", boolean, 9, 108
    )

    not_generative = node_module.NodePositionAttribute()
    result["position_not_generative"] = generated_position_case(
        node_module, BaseNetwork, "position_not_generative", not_generative,
        5, 105, use_public_generate=False,
    )

    empty = node_module.NodePositionAttribute("other_name")
    try:
        empty.generate_data(BaseNetwork(0))
    except IndexError as error:
        result["position_existing_empty"] = [
            "error", "empty_position_network", None, type(error).__name__
        ]
    missing = BaseNetwork(4)
    try:
        empty.generate_data(missing)
    except AttributeError as error:
        result["position_existing_missing_first"] = [
            "error", "missing_position_data", hex_text(str(error))
        ]
    partial = BaseNetwork(4)
    partial.nodes[0]["pos"] = "p0"
    partial.nodes[2]["pos"] = "p2"
    result["position_existing_partial"] = [
        "ok", scalar_list(empty.generate_data(partial))
    ]
    return result


def characterization_cases(node_module, BaseNetwork) -> int:
    count = 0
    # These dynamic Python boundaries are intentionally not represented by a
    # fixed-field C++ map, but remain locked so future factories do not guess.
    for callable_, expected in (
        (lambda: node_module.NodePositionAttribute(generative=True), TypeError),
        (lambda: node_module.NodePositionAttribute(config={"min_r": 0.2}), TypeError),
        (lambda: node_module.NodeStatusAttribute(config={"name": "x"}), TypeError),
    ):
        try:
            callable_()
        except expected:
            count += 1
        else:
            raise RuntimeError("Python duplicate-key constructor boundary drift")
    empty_originator = node_module.NodeExtremaAttribute(
        "empty", config={"originator": ""}
    )
    if empty_originator.originator != "":
        raise RuntimeError("empty extrema originator boundary drift")
    count += 1
    graph = BaseNetwork(2)
    item = node_module.NodeAttribute("x", "node", "status")
    item.set_data(graph, {9: 1})
    if item.get_data(graph) != []:
        raise RuntimeError("unknown sparse node ignore boundary drift")
    count += 1
    return count


def compare_cases(cpp: dict[str, list[str]], python: dict[str, list[str]]):
    if set(cpp) != set(python):
        raise RuntimeError(
            f"case inventory mismatch: missing={sorted(set(python)-set(cpp))}, "
            f"extra={sorted(set(cpp)-set(python))}"
        )
    for name, expected in python.items():
        actual = cpp[name]
        if expected[0] == "ok":
            if actual != expected:
                raise RuntimeError(
                    f"{name} output mismatch:\npython={expected!r}\ncpp={actual!r}"
                )
            continue
        if actual[0] != "error":
            raise RuntimeError(f"{name} expected error, got {actual!r}")
        expected_code = expected[1]
        # Position generation exposes the owning Base/Dataset domain instead
        # of a NodeAttribute code; the Python exception/message remains direct.
        if name == "position_not_generative":
            if actual[1] != "base" or actual[-1] != expected[-1]:
                raise RuntimeError(f"{name} error/state mismatch: {actual!r}")
            continue
        if actual[1] != expected_code:
            raise RuntimeError(
                f"{name} typed error mismatch: expected={expected_code}, actual={actual!r}"
            )
        expected_message = expected[2] if len(expected) > 2 else None
        if expected_message is not None and actual[2] != expected_message:
            raise RuntimeError(
                f"{name} error message mismatch: "
                f"python={expected_message!r}, cpp={actual[2]!r}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--base-source", type=pathlib.Path, required=True)
    parser.add_argument("--method-source", type=pathlib.Path, required=True)
    parser.add_argument("--dataset-source", type=pathlib.Path, required=True)
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    node, base, method, dataset, BaseNetwork, previous, names = load_oracle(
        args.source, args.base_source, args.method_source, args.dataset_source
    )
    try:
        python_cases = {}
        python_cases.update(construction_cases(node))
        python_cases.update(graph_adapter_cases(node, BaseNetwork))
        python_cases.update(resource_cases(node))
        python_cases.update(position_cases(node, BaseNetwork))
        boundary_count = characterization_cases(node, BaseNetwork)
        cpp_cases = parse_harness(args.harness)
        compare_cases(cpp_cases, python_cases)
    finally:
        restore(previous, names)

    payload = {
        "source_sha256": SOURCE_SHA256,
        "base_source_sha256": BASE_SHA256,
        "method_source_sha256": METHOD_SHA256,
        "dataset_source_sha256": compare_dataset_core.SOURCE_SHA256,
        "harness_sha256": hashlib.sha256(args.harness.read_bytes()).hexdigest(),
        "numpy_version": dataset.np.__version__,
        "networkx_version": node.nx.__version__,
        "differential_cases": len(python_cases),
        "python_boundary_cases": boundary_count,
        "total_cases": len(python_cases) + boundary_count,
        "result": "PASS",
    }
    if args.output:
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"node_attribute differential: PASS ({payload['total_cases']} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
