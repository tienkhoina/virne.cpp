#!/usr/bin/env python3
"""Pinned direct-source differential for the non-ML BaseNetwork core."""

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
from collections.abc import Callable
from typing import Any

import compare_attribute_factory as factory_oracle


SOURCE_SHA256 = "94cb1185b5e6f0046d9393f97af1ee0dd1c0d688596b102a0ec3cc98314addce"
MANAGER_SHA256 = "8ad9bab52a40342eaf331e3ac71b3d5f6db0326742d3d3c7c6aad7d34ad52d9e"

PYTHON_BOUNDARY_DESCRIPTIONS = {
    "omegaconf_reflection":
        "Arbitrary OmegaConf truthiness/interpolation remains a cold Python boundary.",
    "sparse_or_hashable_labels":
        "Python accepts arbitrary hashable labels; the native graph uses dense vertices.",
    "arbitrary_attribute_objects":
        "Monkey-patched attributes, descriptors, and arbitrary setattr are Python-only.",
    "getitem_type_object":
        "Python returns the TypeError class for unsupported keys; C++ exposes typed APIs.",
    "assert_under_optimization":
        "Python -O may remove attribute assertions; native typed errors are invariant.",
    "mixed_name_object_lists":
        "Python dispatches from only element zero; native loops accept resolved IDs.",
    "numpy_scipy_identity":
        "NumPy/SciPy object identity, object dtypes, and ragged arrays are not native lanes.",
    "callable_view_predicates":
        "Arbitrary Python callables outside the frozen typed filter surface are excluded.",
    "arbitrary_gml_labels":
        "Arbitrary GML labels and str/repr side effects remain serialization boundaries.",
    "null_dynamic_metadata":
        "Python None graph metadata has no frozen AttrValue lane.",
    "manager_none_originator_key":
        "With an all-types extrema request Python uses a present None originator "
        "as a dictionary key; the frozen native string-key domain applies the "
        "documented attribute-name fallback.",
}

_LOADED_NAMES = (
    "virne.utils.config",
    "virne.network.topology",
    "virne.network.base_network",
)
_MISSING = object()
_RESTORE_STATE: tuple[dict[str, Any], Any, Any] | None = None


def verify(path: pathlib.Path, expected: str, label: str) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"{label} source hash drift: {actual}")


def execute_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot direct-load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_oracle(source: pathlib.Path):
    """Load exact BaseNetwork and non-ML leaves without importing Virne."""
    global _RESTORE_STATE
    if _RESTORE_STATE is not None:
        raise RuntimeError("BaseNetwork oracle is already loaded")

    source = source.resolve()
    verify(source, SOURCE_SHA256, "BaseNetwork")
    attribute_source = source.parent / "attribute" / "__init__.py"
    manager_source = source.parent / "attribute" / "attribute_benchmark_manager.py"
    verify(manager_source, MANAGER_SHA256, "AttributeBenchmarkManager")

    factory = factory_oracle.load_oracle(attribute_source)
    previous = {name: sys.modules.get(name) for name in _LOADED_NAMES}
    try:
        utils = sys.modules["virne.utils"]
        network = sys.modules["virne.network"]

        def write_setting(_value, _path):
            raise RuntimeError("BaseNetwork differential unexpectedly serialized settings")

        def flatten_dict_list_for_gml(values):
            result: dict[str, Any] = {}
            for index, value in enumerate(values):
                for key, item in value.items():
                    result[f"{index}___{key}"] = item
            return result

        utils.write_setting = write_setting
        utils.flatten_dict_list_for_gml = flatten_dict_list_for_gml

        config_module = types.ModuleType("virne.utils.config")
        config_module.resolve_config_to_dict = (
            lambda config: dict(config) if config is not None else None
        )
        sys.modules["virne.utils.config"] = config_module

        topology_module = types.ModuleType("virne.network.topology")

        class TopologyGenerator:
            @staticmethod
            def generate(kind, num_nodes, **kwargs):
                nx = sys.modules["networkx"]
                if kind == "path":
                    return nx.path_graph(num_nodes)
                if kind == "star":
                    return nx.star_graph(num_nodes - 1)
                raise NotImplementedError((kind, num_nodes, kwargs))

        topology_module.TopologyGenerator = TopologyGenerator
        sys.modules["virne.network.topology"] = topology_module

        manager = execute_module(
            "virne.network.attribute.attribute_benchmark_manager", manager_source
        )
        factory.AttributeBenchmarkManager = manager.AttributeBenchmarkManager
        factory.AttributeBenchmarks = manager.AttributeBenchmarks

        base = execute_module("virne.network.base_network", source)

        # The pinned non-ML oracle image intentionally excludes SciPy. This
        # facade provides only the dense materialization consumed by the
        # original LinkAttribute aggregation method; the frozen graph sparse
        # library retains its own complete oracle and is not reimplemented.
        previous_sparse = getattr(base.nx, "attr_sparse_matrix", _MISSING)
        previous_sparse_matrix = getattr(
            base.nx, "to_scipy_sparse_matrix", _MISSING
        )

        class DenseSparseFacade:
            def __init__(self, matrix):
                self._matrix = matrix

            def toarray(self):
                return self._matrix.copy()

        def attr_sparse_matrix(
            graph, edge_attr=None, node_attr=None, normalized=False,
            rc_order=None, dtype=None, order=None,
        ):
            del node_attr, dtype, order
            nodes = list(graph.nodes) if rc_order is None else list(rc_order)
            indices = {node: index for index, node in enumerate(nodes)}
            matrix = base.np.zeros((len(nodes), len(nodes)), dtype=float)
            for source_node, target_node, data in graph.edges(data=True):
                if edge_attr not in data:
                    if edge_attr != "weight":
                        raise KeyError(edge_attr)
                    value = 1.0
                else:
                    value = data[edge_attr]
                row = indices[source_node]
                column = indices[target_node]
                matrix[row, column] += value
                if not graph.is_directed() and source_node != target_node:
                    matrix[column, row] += value
            if normalized:
                for row in range(len(nodes)):
                    row_sum = 0.0
                    for column in range(len(nodes)):
                        row_sum += matrix[row, column]
                    if row_sum != 0.0:
                        reciprocal = 1.0 / row_sum
                        for column in range(len(nodes)):
                            matrix[row, column] *= reciprocal
            return DenseSparseFacade(matrix)

        base.nx.attr_sparse_matrix = attr_sparse_matrix
        base._factory_oracle = factory
        base._benchmark_oracle = manager
        network.base_network = base
    except Exception:
        for name in _LOADED_NAMES:
            sys.modules.pop(name, None)
        for name, value in previous.items():
            if value is not None:
                sys.modules[name] = value
        factory_oracle.unload_oracle()
        raise

    _RESTORE_STATE = previous, previous_sparse, previous_sparse_matrix
    return base


def unload_oracle() -> None:
    global _RESTORE_STATE
    if _RESTORE_STATE is None:
        return
    previous, previous_sparse, previous_sparse_matrix = _RESTORE_STATE
    import networkx as nx
    if previous_sparse is _MISSING:
        nx.__dict__.pop("attr_sparse_matrix", None)
    else:
        nx.attr_sparse_matrix = previous_sparse
    if previous_sparse_matrix is _MISSING:
        nx.__dict__.pop("to_scipy_sparse_matrix", None)
    else:
        nx.to_scipy_sparse_matrix = previous_sparse_matrix
    for name in _LOADED_NAMES:
        sys.modules.pop(name, None)
    for name, value in previous.items():
        if value is not None:
            sys.modules[name] = value
    factory_oracle.unload_oracle()
    _RESTORE_STATE = None


def hex_text(value: str) -> str:
    return value.encode("utf-8").hex()


def double_token(value: float) -> str:
    return "d:" + struct.pack(">d", float(value)).hex()


def float_token(value: Any) -> str:
    return struct.pack(">f", float(value)).hex()


def value_token(value: Any) -> str:
    if isinstance(value, bool):
        return "b:1" if value else "b:0"
    if isinstance(value, int):
        return f"i:{value}"
    if isinstance(value, float):
        return double_token(value)
    if isinstance(value, str):
        return "s:" + hex_text(value)
    if type(value).__module__.startswith("numpy"):
        return double_token(value)
    raise RuntimeError(f"unsupported canonical value: {type(value)!r}")


def canonical_rows(rows) -> str:
    return "[" + ",".join(
        "[" + ",".join(value_token(value) for value in row) + "]"
        for row in rows
    ) + "]"


def canonical_names(values) -> str:
    return "[" + ",".join(hex_text(value) for value in values) + "]"


def canonical_map(value) -> str:
    if value is None:
        return "none"
    return "{" + ",".join(
        f"{hex_text(str(name))}={double_token(item)}"
        for name, item in value.items()
    ) + "}"


def canonical_benchmarks(value) -> str:
    return (
        "node=" + canonical_map(value.node_attr_benchmarks)
        + ";link=" + canonical_map(value.link_attr_benchmarks)
        + ";link_sum=" + canonical_map(value.link_sum_attr_benchmarks)
    )


def canonical_all_type_benchmarks(value) -> str:
    """Apply the frozen string-key translation for Python's None-key boundary."""
    node = value.node_attr_benchmarks
    link = value.link_attr_benchmarks
    link_sum = value.link_sum_attr_benchmarks
    if node is None or link is None or link_sum is None:
        raise RuntimeError("all-types manager unexpectedly disabled a group")
    translated_node = {
        ("state" if name is None else name): item for name, item in node.items()
    }
    translated_link = {
        ("delay" if name is None else name): item for name, item in link.items()
    }
    translated_link_sum = {
        ("delay" if name is None else name): item for name, item in link_sum.items()
    }
    translated = types.SimpleNamespace(
        node_attr_benchmarks=translated_node,
        link_attr_benchmarks=translated_link,
        link_sum_attr_benchmarks=translated_link_sum,
    )
    return canonical_benchmarks(translated)


def fixture_specs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return (
        [
            {"name": "cpu", "owner": "node", "type": "resource"},
            {
                "name": "peak",
                "owner": "node",
                "type": "extrema",
                "originator": "cpu",
            },
            {"name": "state", "owner": "node", "type": "status"},
        ],
        [
            {"name": "bw", "owner": "link", "type": "resource"},
            {
                "name": "peak_bw",
                "owner": "link",
                "type": "extrema",
                "originator": "bw",
            },
            {"name": "delay", "owner": "link", "type": "latency"},
        ],
    )


def make_fixture(module, count: int = 4):
    if count < 4:
        raise RuntimeError("fixture requires at least four nodes")
    graph = module.nx.Graph()
    graph.add_nodes_from(range(count))
    graph.add_edges_from((index, index + 1) for index in range(count - 1))
    graph.add_edges_from((index, index + 2) for index in range(count - 2))
    node_specs, link_specs = fixture_specs()
    config = {
        "node_attrs_setting": node_specs,
        "link_attrs_setting": link_specs,
        "topology": "fixture",
        "output": "memory",
        "graph_attrs_setting": {"zeta": 7, "label": "initial"},
    }
    network = module.BaseNetwork(
        incoming_graph_data=graph, config=config, label="override", flag=True
    )
    nodes = list(network.nodes)
    edges = list(network.edges)
    module.nx.set_node_attributes(
        network, {node: 10 * (index + 1) for index, node in enumerate(nodes)}, "cpu"
    )
    module.nx.set_node_attributes(
        network,
        {node: 10.0 * (index + 1) + 1.25 for index, node in enumerate(nodes)},
        "peak",
    )
    module.nx.set_node_attributes(
        network, {node: index % 2 == 0 for index, node in enumerate(nodes)}, "state"
    )
    module.nx.set_edge_attributes(
        network, {edge: 2 * index + 5 for index, edge in enumerate(edges)}, "bw"
    )
    module.nx.set_edge_attributes(
        network,
        {edge: 2.0 * index + 6.5 for index, edge in enumerate(edges)},
        "peak_bw",
    )
    module.nx.set_edge_attributes(
        network,
        {edge: 0.25 * index + 0.5 for index, edge in enumerate(edges)},
        "delay",
    )
    return network


def benchmark_fixture(module, count: int):
    if count < 4:
        raise RuntimeError("benchmark fixture requires at least four nodes")
    network = make_fixture(module, count)
    nodes = list(network.nodes)
    edges = list(network.edges)
    module.nx.set_node_attributes(
        network, {node: (index * 17) % 1000 for index, node in enumerate(nodes)}, "cpu"
    )
    module.nx.set_node_attributes(
        network,
        {node: float((index * 17) % 1000) + 0.25 for index, node in enumerate(nodes)},
        "peak",
    )
    module.nx.set_edge_attributes(
        network,
        {edge: (index * 13) % 1000 + 1 for index, edge in enumerate(edges)},
        "bw",
    )
    module.nx.set_edge_attributes(
        network,
        {edge: float((index * 13) % 1000 + 1) + 0.5 for index, edge in enumerate(edges)},
        "peak_bw",
    )
    return network


def fixture_inventory(network) -> str:
    metadata_names = ["topology", "output", "zeta", "label", "flag"]
    metadata = ",".join(
        f"{hex_text(name)}={value_token(network.graph[name])}"
        for name in metadata_names if name in network.graph
    )
    return (
        f"nodes={network.number_of_nodes()};links={network.number_of_edges()}"
        f";node_names={canonical_names(list(network.node_attrs))}"
        f";node_types={canonical_names(network.get_node_attr_types())}"
        f";link_names={canonical_names(list(network.link_attrs))}"
        f";link_types={canonical_names(network.get_link_attr_types())}"
        f";features={network.num_node_features},{network.num_link_features},"
        f"{network.num_node_resource_features},{network.num_link_resource_features}"
        f";metadata={{{metadata}}}"
    )


def canonical_prepared_group(group) -> str:
    if group is None:
        return "none"
    descriptors, matrix, extrema, repetitions = group
    attrs = ",".join(
        f"{item[0]}:{item[1]}:{hex_text(item[2])}:"
        f"{hex_text(item[3]) if item[3] is not None else 'none'}"
        for item in descriptors
    )
    bits = "".join(float_token(value) for value in matrix.ravel(order="C"))
    return (
        f"attrs=[{attrs}];shape={matrix.shape[0]},{matrix.shape[1]}"
        f";bits={bits};extrema={1 if extrema else 0};rep={repetitions}"
    )


def python_prepared_request(module, network, workers: int = 1) -> str:
    np = module.np
    node_types = ["resource", "extrema"]
    link_types = ["resource", "extrema"]
    node_attrs = network.get_node_attrs(node_types)
    link_attrs = network.get_link_attrs(link_types)

    def descriptors(attrs, registry):
        ids = {name: index for index, name in enumerate(registry)}
        return [
            (ids[attr.name], attr.type, attr.name, getattr(attr, "originator", None))
            for attr in attrs
        ]

    node_matrix = np.asarray(network.get_node_attrs_data(node_attrs), dtype=np.float32)
    link_matrix = np.asarray(network.get_link_attrs_data(link_attrs), dtype=np.float32)
    link_sum_matrix = np.asarray(
        network.get_aggregation_attrs_data(link_attrs, aggr="sum"), dtype=np.float32
    )
    node_group = (
        descriptors(node_attrs, network.node_attrs),
        node_matrix,
        True,
        1,
    )
    link_group = (
        descriptors(link_attrs, network.link_attrs),
        link_matrix,
        True,
        2,
    )
    link_sum_group = (
        descriptors(link_attrs, network.link_attrs),
        link_sum_matrix,
        True,
        1,
    )
    return (
        f"workers={workers};node={canonical_prepared_group(node_group)}"
        f";link={canonical_prepared_group(link_group)}"
        f";link_sum={canonical_prepared_group(link_sum_group)}"
    )


def ok(callable_: Callable[[], str]) -> tuple[str, ...]:
    return ("ok", callable_())


def expected_error(
    callable_: Callable[[], Any],
    exception_type: type[BaseException] | tuple[type[BaseException], ...],
    code: str,
    operation: str,
    index: str = "-",
) -> tuple[str, ...]:
    try:
        callable_()
    except exception_type:
        return ("error", code, operation, index)
    raise RuntimeError(f"Python oracle did not raise {exception_type!r}")


def oracle_cases(module) -> list[tuple[str, tuple[str, ...]]]:
    cases: list[tuple[str, tuple[str, ...]]] = []
    empty = module.BaseNetwork()
    cases.append(("empty", ok(lambda: fixture_inventory(empty))))

    fixture = make_fixture(module)
    cases.append(("fixture_inventory", ok(lambda: fixture_inventory(fixture))))

    duplicate_graph = module.nx.Graph()
    duplicate_graph.add_nodes_from(range(2))
    duplicate_graph.add_edge(0, 1)
    duplicate = module.BaseNetwork(
        duplicate_graph,
        config={
            "node_attrs_setting": [
                {"name": "a", "owner": "node", "type": "status"},
                {"name": "b", "owner": "node", "type": "resource"},
                {"name": "a", "owner": "node", "type": "position"},
            ],
            "link_attrs_setting": [
                {"name": "x", "owner": "link", "type": "status"},
                {"name": "y", "owner": "link", "type": "resource"},
                {"name": "x", "owner": "link", "type": "latency"},
            ],
        },
    )
    cases.append(("duplicate_first_order", ok(lambda: fixture_inventory(duplicate))))

    cached = make_fixture(module)
    before_nodes = cached.num_nodes
    before_links = cached.num_links
    cached.add_node(4)
    cached.add_edge(0, 4)
    cases.append((
        "cardinality_cache",
        ("ok", f"before={before_nodes},{before_links};stale={cached.num_nodes},{cached.num_links};edge={cached.num_edges}"),
    ))

    selected = make_fixture(module)
    resource_names = [value.name for value in selected.get_node_attrs(["resource"], ["state"])]
    empty_names = [value.name for value in selected.get_node_attrs([], ["state"])]
    all_names = [value.name for value in selected.get_node_attrs()]
    cases.append((
        "selection_precedence",
        ("ok", f"resource={canonical_names(resource_names)};empty={canonical_names(empty_names)};all={canonical_names(all_names)}"),
    ))

    for workers in (0, 1, 2, 8):
        rows = make_fixture(module)
        node_attrs = rows.get_node_attrs(names=["cpu", "peak"])
        link_attrs = rows.get_link_attrs(names=["bw", "peak_bw"])
        value = (
            "node=" + canonical_rows(rows.get_node_attrs_data(node_attrs))
            + ";link=" + canonical_rows(rows.get_link_attrs_data(link_attrs))
        )
        cases.append((f"rows_workers_{workers}", ("ok", value)))

    set_case = make_fixture(module)
    node_updates = {
        set_case.node_attrs["cpu"]: [1, 2, 3, 4],
        set_case.node_attrs["peak"]: [1.5, 2.5, 3.5, 4.5],
    }
    link_count = set_case.number_of_edges()
    link_updates = {
        set_case.link_attrs["bw"]: list(range(20, 20 + link_count)),
        set_case.link_attrs["peak_bw"]: [30.5 + index for index in range(link_count)],
    }
    set_case.set_node_attrs_data(node_updates)
    set_case.set_link_attrs_data(link_updates)
    cases.append((
        "ordered_setters",
        ("ok", "node=" + canonical_rows(set_case.get_node_attrs_data(list(node_updates)))
         + ";link=" + canonical_rows(set_case.get_link_attrs_data(list(link_updates)))),
    ))

    empty_rows = make_fixture(module)
    cases.append((
        "empty_node_rows",
        expected_error(lambda: empty_rows.get_node_attrs_data([]), IndexError,
                       "empty_attribute_selection", "get_attribute_data", "0"),
    ))
    cases.append((
        "empty_link_rows",
        expected_error(lambda: empty_rows.get_link_attrs_data([]), IndexError,
                       "empty_attribute_selection", "get_attribute_data", "0"),
    ))

    aggregation = make_fixture(module)
    link_attrs = aggregation.get_link_attrs(names=["bw", "peak_bw"])
    cases.append((
        "aggregation_sum",
        ok(lambda: canonical_rows(aggregation.get_aggregation_attrs_data(link_attrs, aggr="sum"))),
    ))

    sample = make_fixture(module)
    later_node = list(sample.nodes)[-1]
    later_link = list(sample.edges)[-1]
    del sample.nodes[later_node]["cpu"]
    del sample.edges[later_link]["bw"]
    sample.check_attrs_existence()
    cases.append(("existence_sample_only", ("ok", "pass")))

    missing_first = make_fixture(module)
    del missing_first.nodes[list(missing_first.nodes)[0]]["cpu"]
    cases.append((
        "existence_missing_first_node",
        expected_error(lambda: missing_first.check_attrs_existence(), AssertionError,
                       "missing_node_attribute", "check_attributes", "0"),
    ))

    no_nodes = module.BaseNetwork()
    cases.append((
        "existence_no_nodes",
        expected_error(lambda: no_nodes.check_attrs_existence(), ValueError,
                       "no_nodes", "check_attributes", "0"),
    ))
    no_links_graph = module.nx.Graph()
    no_links_graph.add_node(0)
    no_links = module.BaseNetwork(no_links_graph)
    cases.append((
        "existence_no_links",
        expected_error(lambda: no_links.check_attrs_existence(), ValueError,
                       "no_links", "check_attributes", "0"),
    ))

    view_source = make_fixture(module)
    view = view_source.subgraph([0, 1, 2])
    cases.append((
        "induced_view",
        ("ok", f"nodes={view.number_of_nodes()};links={view.number_of_edges()};shared={int(view.node_attrs is view_source.node_attrs and view.link_attrs is view_source.link_attrs)}"),
    ))

    clone_source = make_fixture(module)
    clone = clone_source.clone()
    clone.nodes[0]["cpu"] = 999
    cases.append((
        "clone_depth",
        ("ok", f"source={value_token(clone_source.nodes[0]['cpu'])};clone={value_token(clone.nodes[0]['cpu'])};registries={int(clone.node_attrs is not clone_source.node_attrs and clone.link_attrs is not clone_source.link_attrs)}"),
    ))

    prepared = make_fixture(module)
    cases.append((
        "prepared_default",
        ok(lambda: python_prepared_request(module, prepared, 1)),
    ))

    manager = module._benchmark_oracle.AttributeBenchmarkManager
    for workers in (0, 1, 2, 8):
        manager_network = make_fixture(module)
        cases.append((
            f"manager_default_workers_{workers}",
            ok(lambda network=manager_network: canonical_benchmarks(manager.get_benchmarks(network))),
        ))

    all_network = make_fixture(module)
    cases.append((
        "manager_all_types",
        ok(lambda: canonical_all_type_benchmarks(manager.get_benchmarks(
            all_network, node_attr_types=None, link_attr_types=None))),
    ))
    disabled_network = make_fixture(module)
    cases.append((
        "manager_disabled",
        ok(lambda: canonical_benchmarks(manager.get_benchmarks(
            disabled_network, node_attrs=False, link_attrs=False,
            link_sum_attrs=False))),
    ))
    empty_manager = module.BaseNetwork()
    cases.append((
        "manager_empty_link_sum",
        ok(lambda: canonical_benchmarks(manager.get_benchmarks(
            empty_manager, node_attrs=False, link_attrs=False,
            link_sum_attrs=True))),
    ))
    cases.append((
        "manager_empty_node",
        expected_error(lambda: manager.get_benchmarks(
            empty_manager, node_attrs=True, link_attrs=False,
            link_sum_attrs=False), IndexError,
            "empty_attribute_selection", "get_attribute_data", "0"),
    ))

    nonnumeric = make_fixture(module)
    nonnumeric.nodes[0]["peak"] = "bad"
    cases.append((
        "manager_nonnumeric",
        expected_error(lambda: manager.get_benchmarks(
            nonnumeric, link_attrs=False, link_sum_attrs=False), ValueError,
            "non_numeric_benchmark_value", "prepare_benchmarks", "1"),
    ))
    return cases


def parse_cpp(path: pathlib.Path) -> list[tuple[str, tuple[str, ...]]]:
    process = subprocess.run(
        [str(path)], check=False, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True
    )
    if process.returncode != 0:
        raise RuntimeError(f"C++ BaseNetwork harness failed: {process.stderr.strip()}")
    result: list[tuple[str, tuple[str, ...]]] = []
    seen: set[str] = set()
    for raw_line in process.stdout.splitlines():
        fields = raw_line.split("|")
        if len(fields) < 3 or not fields[0].startswith("case="):
            raise RuntimeError(f"malformed C++ BaseNetwork line: {raw_line!r}")
        name = fields[0][5:]
        if name in seen:
            raise RuntimeError(f"duplicate C++ BaseNetwork case: {name}")
        seen.add(name)
        if fields[1] == "ok" and len(fields) == 3:
            try:
                payload = bytes.fromhex(fields[2]).decode("utf-8")
            except (ValueError, UnicodeDecodeError) as error:
                raise RuntimeError(f"invalid C++ payload for {name}") from error
            result.append((name, ("ok", payload)))
        elif fields[1] == "error" and len(fields) == 5:
            result.append((name, tuple(fields[1:])))
        else:
            raise RuntimeError(f"malformed C++ result for {name}: {fields!r}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    module = load_oracle(args.source)
    try:
        expected = oracle_cases(module)
        numpy_version = module.np.__version__
        networkx_version = module.nx.__version__
    finally:
        unload_oracle()
    actual = parse_cpp(args.harness)
    if [name for name, _ in actual] != [name for name, _ in expected]:
        raise RuntimeError(
            "BaseNetwork case inventory/order mismatch: "
            f"C++={[name for name, _ in actual]!r}, "
            f"Python={[name for name, _ in expected]!r}"
        )
    mismatches = [
        (name, actual_value, expected_value)
        for (name, actual_value), (_, expected_value) in zip(actual, expected)
        if actual_value != expected_value
    ]
    if mismatches:
        name, actual_value, expected_value = mismatches[0]
        raise RuntimeError(
            f"BaseNetwork differential mismatch in {name}: "
            f"C++={actual_value!r}, Python={expected_value!r}"
        )

    payload = {
        "source_sha256": SOURCE_SHA256,
        "manager_source_sha256": MANAGER_SHA256,
        "numpy_version": numpy_version,
        "networkx_version": networkx_version,
        "shared_cases": [name for name, _ in expected],
        "python_only_boundaries": PYTHON_BOUNDARY_DESCRIPTIONS,
        "case_count": len(expected) + len(PYTHON_BOUNDARY_DESCRIPTIONS),
        "result": "PASS",
    }
    if args.output:
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        f"BaseNetwork differential: PASS ({len(expected)} shared + "
        f"{len(PYTHON_BOUNDARY_DESCRIPTIONS)} boundaries)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
