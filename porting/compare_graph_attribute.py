#!/usr/bin/env python3
"""Direct-source Python/C++ differential for graph attributes."""

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


SOURCE_SHA256 = "dfa858918068a792cb0a673b400ee6f3f93107f1d17baad28f72546f50fbcede"
BASE_SHA256 = "103c5c16126ca76e782c2191ff0811b95ed88a0c3637f3a61e84c0b22df42e8e"
METHOD_SHA256 = "e17499af8e6ffbdb12f2100dd58abeab48dd06d92feb442ef192f8b9310b6b4f"


def bits(value: float) -> str:
    return f"{struct.unpack('>Q', struct.pack('>d', float(value)))[0]:016x}"


def from_bits(value: int) -> float:
    return struct.unpack(">d", struct.pack(">Q", value))[0]


def hex_text(value: str) -> str:
    return value.encode("utf-8").hex()


def scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "b:1" if value else "b:0"
    if isinstance(value, int):
        return f"i:{value}"
    if isinstance(value, float) or type(value).__module__.startswith("numpy"):
        return "d:" + bits(value)
    if isinstance(value, str):
        return "s:" + hex_text(value)
    if isinstance(value, (list, tuple)):
        return "l:[" + ";".join(scalar(item) for item in value) + "]"
    raise RuntimeError(f"unsupported oracle scalar: {type(value)!r}")


def scalar_list(values) -> str:
    return ",".join(scalar(value) for value in values)


def verify(path: pathlib.Path, expected: str, label: str) -> None:
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


def load_oracle(args):
    verify(args.source, SOURCE_SHA256, "graph_attribute")
    verify(args.base_source, BASE_SHA256, "base_attribute")
    verify(args.method_source, METHOD_SHA256, "attribute_method")
    dataset = compare_dataset_core.load_oracle(args.dataset_source)

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

    class DirectedBaseNetwork(nx.DiGraph):
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
        "virne.network.attribute.graph_attribute",
    ]
    try:
        base = execute_module(names[0], args.base_source)
        method = execute_module(names[1], args.method_source)
        graph = execute_module(names[2], args.source)
    except Exception:
        restore(previous, names)
        raise
    return (
        graph,
        base,
        method,
        dataset,
        nx,
        BaseNetwork,
        DirectedBaseNetwork,
        previous,
        names,
    )


def restore(previous, names) -> None:
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
        raise RuntimeError(f"graph harness failed: {process.stderr.strip()}")
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
        raise RuntimeError("graph harness did not report PASS")
    return result


class _OriginatorConfig(dict):
    """Expose originator through get() without duplicating it through **config."""

    def __init__(self, originator: str):
        super().__init__({"oracle_marker": True})
        self._originator = originator

    def get(self, key, default=None):
        if key == "originator":
            return self._originator
        return super().get(key, default)


def make_extrema(graph_module):
    return graph_module.GraphExtremaAttribute(
        "capacity_max", config=_OriginatorConfig("capacity")
    )


class _RestrictionConfig(dict):
    """Probe config precedence without duplicating the explicit restriction."""

    def __init__(self):
        super().__init__({"oracle_marker": True})

    def get(self, key, default=None):
        if key == "constraint_restrictions":
            return "soft"
        if key == "restriction":
            return "hard"
        return super().get(key, default)


def construction_cases(graph_module) -> dict[str, list[str]]:
    result = {}
    status = graph_module.GraphStatusAttribute("alive")
    result["status_fields"] = [
        "ok",
        f"{status.name},{status.owner},{status.type},"
        f"{int(status.generative)},{int(status.is_constraint)}",
    ]
    try:
        graph_module.GraphExtremaAttribute("capacity_max")
    except ValueError:
        result["extrema_missing_originator"] = ["error", "missing_originator"]
    else:
        raise RuntimeError("missing GraphExtrema originator unexpectedly succeeded")
    extrema = make_extrema(graph_module)
    result["extrema_fields"] = [
        "ok",
        f"{extrema.name},{extrema.owner},{extrema.type},"
        f"{extrema.originator},17,{int(extrema.is_constraint)},link",
    ]
    hard = graph_module.GraphResourceAttribute("capacity")
    result["resource_fields_hard"] = [
        "ok",
        f"{hard.name},{hard.owner},{hard.type},"
        f"{hard.constraint_restrictions},{hard.checking_level},"
        f"{int(hard.is_constraint)}",
    ]
    precedence = graph_module.GraphResourceAttribute(
        "capacity",
        config=_RestrictionConfig(),
    )
    result["resource_fields_precedence"] = [
        "ok",
        f"{precedence.name},{precedence.owner},{precedence.type},"
        f"{precedence.constraint_restrictions},{precedence.checking_level},"
        f"{int(precedence.is_constraint)}",
    ]
    return result


def adapter_cases(graph_module, BaseNetwork, DirectedBaseNetwork):
    result = {}
    lanes = [
        -(1 << 63),
        -0.0,
        from_bits(0x7FF8000000001234),
        True,
        "a|b",
        [7, 2.5, False, "z"],
    ]
    for prefix, graph_type in (
        ("graph", BaseNetwork),
        ("digraph", DirectedBaseNetwork),
    ):
        network = graph_type(2)
        item = graph_module.GraphAttribute("mixed", "graph", "status")
        observed = []
        for value in lanes:
            item.set_data(network, value)
            observed.append(item.get_data(network))
        result[f"{prefix}_lanes"] = ["ok", scalar_list(observed)]

        identity_value = [1, -0.0, "live"]
        item.set_data(network, identity_value)
        result[f"{prefix}_identity"] = [
            "ok",
            f"{int(item.get(network) is item.get_data(network))},"
            f"{int(item.get(network) is identity_value)},"
            f"{scalar(item.get(network))}",
        ]
        item.set_data(network, 11)
        item.set_data(network, "overwritten")
        result[f"{prefix}_overwrite"] = ["ok", scalar(item.get_data(network))]

        empty = graph_module.GraphAttribute("", "graph", "status")
        empty.set_data(network, False)
        result[f"{prefix}_empty_name"] = ["ok", scalar(empty.get(network))]

        missing = graph_type()
        try:
            item.get_data(missing)
        except KeyError:
            result[f"{prefix}_missing"] = ["error", "missing_attribute"]
        else:
            raise RuntimeError("missing graph attribute unexpectedly succeeded")

    first = BaseNetwork()
    second = BaseNetwork()
    first.graph["padding"] = 1
    second.graph["other_padding"] = 2
    item = graph_module.GraphAttribute("mixed", "graph", "status")
    item.set_data(first, 31)
    item.set_data(second, 47)
    result["different_registries"] = [
        "ok", f"{scalar(item.get(first))},{scalar(item.get(second))}"
    ]
    return result


def batch_cases(graph_module, BaseNetwork):
    result = {}
    item = graph_module.GraphAttribute("batch_value", "graph", "status")
    values = [1, -0.0, from_bits(0x7FF8000000004321), True, "five", [6, 7.5]]
    for workers in (0, 1, 2, 8):
        networks = [BaseNetwork() for _ in values]
        for index, network in enumerate(networks):
            network.graph[f"padding_{index}"] = index
        for network, value in zip(networks, values):
            item.set_data(network, value)
        output = [item.get_data(network) for network in networks]
        result[f"batch_w{workers}"] = ["ok", scalar_list(output)]
    result["batch_empty"] = ["ok", ""]

    duplicate = BaseNetwork()
    for value in (1, 2.5, True):
        item.set_data(duplicate, value)
    result["batch_duplicate_map"] = ["ok", scalar(item.get_data(duplicate))]
    return result


class _Originator:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def get_data(self, network):
        self.calls.append(network)
        return self.output


class _BombRegistry(dict):
    def __getitem__(self, key):
        raise RuntimeError(f"wrong origin registry read: {key!r}")


def extrema_cases(graph_module, BaseNetwork, DirectedBaseNetwork):
    result = {}
    values = [1, -0.0, True, "link"]
    extrema = make_extrema(graph_module)
    for prefix, graph_type, workers in (
        ("graph", BaseNetwork, 0),
        ("graph", BaseNetwork, 1),
        ("graph", BaseNetwork, 2),
        ("graph", BaseNetwork, 8),
        ("digraph", DirectedBaseNetwork, 8),
    ):
        network = graph_type(4)
        network.add_edges_from([(0, 1), (2, 3), (1, 3), (3, 3)])
        originator = _Originator(values)
        network.node_attrs = _BombRegistry()
        network.link_attrs["capacity"] = originator
        output = extrema.generate_data(network)
        if originator.calls != [network]:
            raise RuntimeError("GraphExtrema did not delegate exactly once")
        result[f"extrema_{prefix}_w{workers}"] = ["ok", scalar_list(output)]
    return result


def _network_value(network_type, value, present: bool = True):
    network = network_type()
    if present:
        network.graph["capacity"] = value
    return network


def resource_cases(graph_module, BaseNetwork):
    hard = graph_module.GraphResourceAttribute("capacity")
    soft = graph_module.GraphResourceAttribute(
        "capacity", config={"constraint_restrictions": "soft"}
    )
    result = {}
    checks = (
        ("resource_le_hard_pass", hard, 7, 10.5, "le"),
        ("resource_le_hard_fail", hard, 12.0, 10.5, "le"),
        ("resource_le_soft_fail", soft, 12.0, 10.5, "le"),
        ("resource_ge_hard_pass", hard, 11, 10, "ge"),
        ("resource_ge_hard_fail", hard, 9, 10, "ge"),
        ("resource_eq_hard_pass", hard, -0.0, 0.0, "eq"),
        ("resource_eq_soft_fail", soft, 4.5, 4.0, "eq"),
        ("resource_bool_promotion", hard, True, False, "le"),
    )
    for name, policy, virtual, physical, method in checks:
        checked = policy.check_constraint_satisfiability(
            _network_value(BaseNetwork, virtual),
            _network_value(BaseNetwork, physical),
            method,
        )
        result[name] = ["ok", f"{int(checked[0])},{scalar(checked[1])}"]

    try:
        hard.check_constraint_satisfiability(
            _network_value(BaseNetwork, "bad"),
            _network_value(BaseNetwork, 0, present=False),
        )
    except ValueError:
        result["resource_check_physical_missing_precedence"] = [
            "error", "missing_resource_value"
        ]
    else:
        raise RuntimeError("physical missing precedence unexpectedly succeeded")
    try:
        hard.check_constraint_satisfiability(
            _network_value(BaseNetwork, 0, present=False),
            _network_value(BaseNetwork, "bad"),
        )
    except ValueError:
        result["resource_check_virtual_missing_precedence"] = [
            "error", "missing_resource_value"
        ]
    else:
        raise RuntimeError("virtual missing precedence unexpectedly succeeded")
    try:
        hard.check_constraint_satisfiability(
            _network_value(BaseNetwork, "bad"),
            _network_value(BaseNetwork, 10),
        )
    except TypeError:
        result["resource_check_nonnumeric"] = [
            "error", "non_numeric_resource"
        ]
    else:
        raise RuntimeError("nonnumeric graph resource unexpectedly succeeded")

    target = _network_value(BaseNetwork, 10)
    operand = _network_value(BaseNetwork, 3)
    hard.update(target, operand, "-", True)
    hard.update(target, operand, "+", True)
    result["resource_update_roundtrip"] = [
        "ok", scalar(target.graph["capacity"])
    ]
    target = _network_value(BaseNetwork, 2)
    hard.update(target, operand, "sub", True)
    result["resource_update_negative_safe_ignored"] = [
        "ok", scalar(target.graph["capacity"])
    ]
    target = _network_value(BaseNetwork, False)
    boolean_operand = _network_value(BaseNetwork, True)
    hard.update(target, boolean_operand, "add", False)
    result["resource_update_bool_promotion"] = [
        "ok", scalar(target.graph["capacity"])
    ]
    alias = _network_value(BaseNetwork, 7)
    hard.update(alias, alias, "+", True)
    doubled = scalar(alias.graph["capacity"])
    hard.update(alias, alias, "-", True)
    result["resource_update_self_alias"] = [
        "ok", f"{doubled},{scalar(alias.graph['capacity'])}"
    ]

    try:
        hard.update(BaseNetwork(), _network_value(BaseNetwork, "bad"), "+", True)
    except KeyError:
        result["resource_update_target_missing_precedence"] = [
            "error", "missing_resource_value"
        ]
    else:
        raise RuntimeError("target missing precedence unexpectedly succeeded")
    try:
        hard.update(_network_value(BaseNetwork, "bad"), BaseNetwork(), "+", True)
    except KeyError:
        result["resource_update_operand_missing_precedence"] = [
            "error", "missing_resource_value"
        ]
    else:
        raise RuntimeError("operand missing precedence unexpectedly succeeded")
    try:
        hard.update(BaseNetwork(), BaseNetwork(), "multiply", True)
    except NotImplementedError:
        result["resource_update_invalid_operation"] = [
            "error", "unsupported_update_operation"
        ]
    else:
        raise RuntimeError("invalid graph update unexpectedly succeeded")
    return result


def native_extension_cases() -> dict[str, list[str]]:
    return {
        "batch_get_null_slot": ["error", "null_batch_slot"],
        "batch_set_null_slot": ["error", "null_batch_slot"],
        "batch_shape_mismatch": ["error", "invalid_batch_shape"],
        "resource_update_overflow": ["error", "numeric_range"],
    }


def characterization_cases(graph_module, BaseNetwork):
    names = []
    none_name = graph_module.GraphAttribute(None, "graph", "status")
    try:
        none_name.get(BaseNetwork())
    except AttributeError:
        names.append("none_name")
    else:
        raise RuntimeError("None graph name boundary drift")

    class Arbitrary:
        pass

    arbitrary = Arbitrary()
    item = graph_module.GraphAttribute("dynamic", "graph", "status")
    network = BaseNetwork()
    item.set_data(network, arbitrary)
    if item.get_data(network) is not arbitrary:
        raise RuntimeError("arbitrary-object identity boundary drift")
    names.append("arbitrary_object")

    resource = graph_module.GraphResourceAttribute("capacity")
    target = _network_value(BaseNetwork, 1 << 100)
    operand = _network_value(BaseNetwork, 1 << 100)
    resource.update(target, operand, "+")
    if target.graph["capacity"] != 1 << 101:
        raise RuntimeError("unbounded integer boundary drift")
    names.append("unbounded_integer")

    try:
        graph_module.GraphExtremaAttribute(
            "maximum", config={"originator": "first"}, originator="second"
        )
    except TypeError:
        names.append("duplicate_originator_argument")
    else:
        raise RuntimeError("duplicate originator argument boundary drift")

    status = graph_module.GraphStatusAttribute("alive")
    status.owner = "node"
    del status.name
    if status.owner != "node" or hasattr(status, "name"):
        raise RuntimeError("reflection boundary drift")
    names.append("reflection_mutation")
    return names


def compare_cases(cpp, expected) -> None:
    if set(cpp) != set(expected):
        raise RuntimeError(
            f"case inventory mismatch: missing={sorted(set(expected)-set(cpp))}, "
            f"extra={sorted(set(cpp)-set(expected))}"
        )
    for name, wanted in expected.items():
        actual = cpp[name]
        if wanted[0] == "ok":
            if actual != wanted:
                raise RuntimeError(
                    f"{name} output mismatch:\npython={wanted!r}\ncpp={actual!r}"
                )
            continue
        if actual[0] != "error" or actual[1] != wanted[1]:
            raise RuntimeError(
                f"{name} typed error mismatch: python={wanted!r}, cpp={actual!r}"
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

    (
        graph,
        _,
        _,
        dataset,
        nx,
        BaseNetwork,
        DirectedBaseNetwork,
        previous,
        names,
    ) = load_oracle(args)
    try:
        differential = {}
        differential.update(construction_cases(graph))
        differential.update(adapter_cases(graph, BaseNetwork, DirectedBaseNetwork))
        differential.update(batch_cases(graph, BaseNetwork))
        differential.update(extrema_cases(graph, BaseNetwork, DirectedBaseNetwork))
        differential.update(resource_cases(graph, BaseNetwork))
        native = native_extension_cases()
        boundaries = characterization_cases(graph, BaseNetwork)
        expected = dict(differential)
        expected.update(native)
        cpp = parse_harness(args.harness)
        compare_cases(cpp, expected)
    finally:
        restore(previous, names)

    payload = {
        "source_sha256": SOURCE_SHA256,
        "base_source_sha256": BASE_SHA256,
        "method_source_sha256": METHOD_SHA256,
        "dataset_source_sha256": compare_dataset_core.SOURCE_SHA256,
        "harness_sha256": hashlib.sha256(args.harness.read_bytes()).hexdigest(),
        "numpy_version": dataset.np.__version__,
        "networkx_version": nx.__version__,
        "differential_cases": len(differential),
        "native_extension_cases": len(native),
        "python_boundary_cases": len(boundaries),
        "python_boundary_names": boundaries,
        "total_cases": len(differential) + len(native) + len(boundaries),
        "result": "PASS",
    }
    if args.output:
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"graph_attribute differential: PASS ({payload['total_cases']} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
