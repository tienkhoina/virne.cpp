#!/usr/bin/env python3
"""Direct-source Python/C++ differential for link attributes."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import struct
import sys
from typing import Any

import compare_dataset_core
import compare_node_attribute


SOURCE_SHA256 = "a95cfd2b8e2b46d4de23f70934ca942de502e674e2a7eaa139482df640e8646f"


def bits(value: float) -> str:
    return f"{struct.unpack('>Q', struct.pack('>d', float(value)))[0]:016x}"


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
    raise RuntimeError(f"unsupported oracle scalar: {type(value)!r}")


def scalar_list(values) -> str:
    return ",".join(scalar(value) for value in values)


def double_list(values) -> str:
    return ",".join(bits(value) for value in values)


def matrix_bits(value) -> str:
    return double_list(value.reshape(-1))


def load_oracle(args):
    actual = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"link_attribute source hash drift: {actual}")
    node, base, method, dataset, BaseNetwork, previous, names = (
        compare_node_attribute.load_oracle(
            args.node_source,
            args.base_source,
            args.method_source,
            args.dataset_source,
        )
    )
    link_name = "virne.network.attribute.link_attribute"
    try:
        link = compare_node_attribute.execute_module(link_name, args.source)
    except Exception:
        compare_node_attribute.restore(previous, names)
        raise
    names.append(link_name)
    BaseNetwork.links = property(lambda self: self.edges)
    link.path_to_links = lambda path: list(zip(path[:-1], path[1:]))

    # The pinned non-ML image deliberately excludes SciPy.  This controlled
    # facade supplies only the exact dense behavior consumed by this leaf;
    # the frozen C++ graph sparse implementation has its own full differential.
    class DenseSparseFacade:
        def __init__(self, matrix):
            self._matrix = matrix

        def toarray(self):
            return self._matrix.copy()

    def attr_sparse_matrix(
        graph, edge_attr=None, node_attr=None, normalized=False, rc_order=None,
        dtype=None, order=None,
    ):
        del node_attr, dtype, order
        nodes = list(graph.nodes) if rc_order is None else list(rc_order)
        index = {node: offset for offset, node in enumerate(nodes)}
        matrix = dataset.np.zeros((len(nodes), len(nodes)), dtype=float)
        for source, target, data in graph.edges(data=True):
            if edge_attr not in data:
                if edge_attr == "weight":
                    value = 1.0
                else:
                    raise KeyError(edge_attr)
            else:
                value = data[edge_attr]
            row = index[source]
            column = index[target]
            matrix[row, column] += value
            if not graph.is_directed() and source != target:
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

    link.nx.attr_sparse_matrix = attr_sparse_matrix

    class DirectedBaseNetwork(link.nx.DiGraph):
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

        @property
        def links(self):
            return self.edges

    return link, dataset, BaseNetwork, DirectedBaseNetwork, previous, names


def construction_cases(link):
    result = {}
    status = link.LinkStatusAttribute("alive")
    result["status_fields"] = [
        "ok",
        f"{status.name},{status.owner},{status.type},"
        f"{int(status.generative)},{int(status.is_constraint)}",
    ]
    try:
        link.LinkExtremaAttribute("bandwidth_max")
    except ValueError as error:
        result["extrema_missing_originator"] = [
            "error", "missing_originator", hex_text(str(error))
        ]
    else:
        raise RuntimeError("missing LinkExtrema originator unexpectedly succeeded")
    extrema = link.LinkExtremaAttribute(
        "bandwidth_max", config={"originator": "bandwidth"}
    )
    result["extrema_fields"] = [
        "ok",
        f"{extrema.name},{extrema.owner},{extrema.type},"
        f"{extrema.originator},17,{int(extrema.is_constraint)}",
    ]
    resource = link.LinkResourceAttribute(
        "bandwidth", config={"constraint_restrictions": "soft"}
    )
    result["resource_fields"] = [
        "ok",
        f"{resource.name},{resource.owner},{resource.type},"
        f"{resource.constraint_restrictions},{resource.checking_level},"
        f"{int(resource.is_constraint)}",
    ]
    latency = link.LinkLatencyAttribute(
        "latency",
        config={
            "generative": True,
            "distribution": "position",
            "min": -0.25,
            "max": 0.75,
        },
    )
    result["latency_fields"] = [
        "ok",
        f"{latency.name},{latency.owner},{latency.type},1,position,"
        f"{scalar(latency.min)},{scalar(latency.max)},"
        f"{latency.constraint_restrictions},{latency.checking_level},"
        f"{int(latency.is_constraint)}",
    ]
    return result


def adapter_cases(link, BaseNetwork, DirectedBaseNetwork):
    result = {}
    edges = [(3, 4), (0, 2), (1, 4), (0, 1), (2, 2)]
    for prefix, graph_type in (
        ("graph", BaseNetwork),
        ("digraph", DirectedBaseNetwork),
    ):
        graph = graph_type(5)
        graph.add_edges_from(edges)
        item = link.LinkAttribute("mixed", "link", "status")
        item.set_data(graph, [1, 2.5, True, "four", -5, 999])
        result[f"{prefix}_dense"] = ["ok", scalar_list(item.get_data(graph))]

        sparse = graph_type(3)
        sparse.add_edges_from([(0, 1), (1, 2)])
        item.set_data(sparse, {(1, 2): 31, (9, 9): 90, (1, 2): 32})
        result[f"{prefix}_sparse"] = ["ok", scalar_list(item.get_data(sparse))]
        try:
            item.get(sparse, (0, 1))
        except KeyError:
            result[f"{prefix}_missing_get"] = ["error", "missing_attribute"]
        else:
            raise RuntimeError("missing link attribute unexpectedly succeeded")
        before = list(sparse.edges(data="mixed", default=None))
        try:
            item.set_data(sparse, [7])
        except IndexError:
            after = list(sparse.edges(data="mixed", default=None))
            if before != after:
                raise RuntimeError("short link data partially mutated graph")
            result[f"{prefix}_short_dense"] = [
                "error", "dense_data_too_short"
            ]
        else:
            raise RuntimeError("short link data unexpectedly succeeded")
    return result


def matrix_cases(link, BaseNetwork):
    graph = BaseNetwork(3)
    graph.add_edges_from([(0, 1), (1, 2)])
    item = link.LinkAttribute("capacity", "link", "resource")
    item.set_data(graph, [2.0, 4.0])
    result = {
        "matrix_raw": ["ok", matrix_bits(item.get_adjacency_data(graph, False))],
        "matrix_normalized": [
            "ok", matrix_bits(item.get_adjacency_data(graph, True))
        ],
    }
    for name in ("sum", "mean", "max", "min"):
        result[f"aggregation_{name}"] = [
            "ok", double_list(item.get_aggregation_data(graph, name, False))
        ]

    signed_zero = BaseNetwork(2)
    signed_zero.add_edge(0, 1)
    item.set_data(signed_zero, [-0.0])
    result["aggregation_signed_zero_max"] = [
        "ok", double_list(item.get_aggregation_data(signed_zero, "max", False))
    ]
    result["aggregation_signed_zero_min"] = [
        "ok", double_list(item.get_aggregation_data(signed_zero, "min", False))
    ]
    nan_graph = BaseNetwork(2)
    nan_graph.add_edge(0, 1)
    item.set_data(nan_graph, [float("nan")])
    result["aggregation_nan_max"] = [
        "ok", double_list(item.get_aggregation_data(nan_graph, "max", False))
    ]
    result["aggregation_nan_min"] = [
        "ok", double_list(item.get_aggregation_data(nan_graph, "min", False))
    ]
    return result


def resource_cases(link, BaseNetwork):
    hard = link.LinkResourceAttribute("bandwidth")
    soft = link.LinkResourceAttribute(
        "bandwidth", config={"constraint_restrictions": "soft"}
    )
    flag, offset = hard.check_constraint_satisfiability(
        {"bandwidth": 7}, {"bandwidth": 10.5}, "le"
    )
    result = {
        "resource_hard_pass": ["ok", f"{int(flag)},{scalar(offset)}"]
    }
    flag, offset = soft.check_constraint_satisfiability(
        {"bandwidth": 12.0}, {"bandwidth": 10.5}, "le"
    )
    result["resource_soft_fail"] = ["ok", f"{int(flag)},{scalar(offset)}"]

    graph = BaseNetwork(3)
    graph.add_edges_from([(0, 1), (1, 2)])
    graph.edges[0, 1]["bandwidth"] = 10
    graph.edges[1, 2]["bandwidth"] = 8
    virtual = {"bandwidth": 3}
    hard.update_path(virtual, graph, [0, 1, 2], "sub", safe=True)
    result["resource_path_update"] = [
        "ok",
        f"{scalar(graph.edges[0, 1]['bandwidth'])},"
        f"{scalar(graph.edges[1, 2]['bandwidth'])}",
    ]
    try:
        hard.update_path(virtual, graph, [0], "add", safe=True)
    except ValueError:
        result["resource_short_path"] = ["error", "path_too_short"]
    else:
        raise RuntimeError("short resource path unexpectedly succeeded")

    graph.edges[0, 1]["bandwidth"] = 10
    graph.edges[1, 2]["bandwidth"] = 2
    try:
        hard.update_path(virtual, graph, [0, 1, 2], "sub", safe=True)
    except ValueError:
        result["resource_partial_path"] = ["error", "insufficient_resource"]
    else:
        raise RuntimeError("insufficient resource path unexpectedly succeeded")
    result["resource_partial_state"] = [
        "ok",
        f"{scalar(graph.edges[0, 1]['bandwidth'])},"
        f"{scalar(graph.edges[1, 2]['bandwidth'])}",
    ]
    return result


def latency_cases(link, BaseNetwork):
    latency = link.LinkLatencyAttribute(
        "latency",
        config={
            "generative": True,
            "distribution": "position",
            "min": 10,
            "max": 12,
        },
    )
    graph = BaseNetwork(3)
    graph.add_edges_from([(0, 1), (1, 2)])
    first = [(0, 0), (3, 4), (3, 0)]
    for node, coordinates in enumerate(first):
        graph.nodes[node]["node_pos_first"] = coordinates
        graph.nodes[node]["pos_second"] = (0.0, 0.0)
    generated = latency.generate_data(graph)
    result = {}
    for workers in (0, 1, 2, 8):
        result[f"latency_position_w{workers}"] = [
            "ok", "0", double_list(generated)
        ]

    flag, offset = latency.check_constraint_satisfiability(
        {"latency": 10.0}, [{"latency": 3.0}, {"latency": 4.0}], "ge"
    )
    result["latency_path_check"] = [
        "ok", f"{int(flag)},{scalar(offset)}"
    ]
    non_generative = link.LinkLatencyAttribute(
        "latency", config={"distribution": "position"}
    )
    try:
        non_generative.generate_data(graph)
    except NotImplementedError:
        result["latency_non_generative"] = [
            "error", "non_generative_latency"
        ]
    else:
        raise RuntimeError("non-generative latency unexpectedly succeeded")
    return result


def characterization_cases(link, BaseNetwork):
    count = 0
    base = link.LinkAttribute("value", "link", "status")
    try:
        base.update_path({}, BaseNetwork(0), [])
    except NotImplementedError:
        count += 1
    else:
        raise RuntimeError("abstract update_path boundary drift")
    try:
        base.get_aggregation_data(BaseNetwork(0), "median")
    except NotImplementedError:
        count += 1
    else:
        raise RuntimeError("unsupported aggregation boundary drift")
    try:
        link.LinkStatusAttribute("alive", config={"owner": "link"})
    except TypeError:
        count += 1
    else:
        raise RuntimeError("duplicate constructor field boundary drift")
    position = link.LinkLatencyAttribute(
        "latency",
        config={"generative": True, "distribution": "position"},
    )
    try:
        position.generate_data(BaseNetwork(0))
    except IndexError:
        count += 1
    else:
        raise RuntimeError("empty position graph boundary drift")
    missing = BaseNetwork(2)
    missing.add_edge(0, 1)
    try:
        position.generate_data(missing)
    except AttributeError:
        count += 1
    else:
        raise RuntimeError("missing position data boundary drift")
    return count


def compare_cases(cpp, python):
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
        if actual[0] != "error" or actual[1] != expected[1]:
            raise RuntimeError(
                f"{name} typed error mismatch: python={expected!r}, cpp={actual!r}"
            )
        if len(expected) > 2 and actual[2] != expected[2]:
            raise RuntimeError(
                f"{name} message mismatch: python={expected!r}, cpp={actual!r}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--node-source", type=pathlib.Path, required=True)
    parser.add_argument("--base-source", type=pathlib.Path, required=True)
    parser.add_argument("--method-source", type=pathlib.Path, required=True)
    parser.add_argument("--dataset-source", type=pathlib.Path, required=True)
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    link, dataset, BaseNetwork, DirectedBaseNetwork, previous, names = (
        load_oracle(args)
    )
    try:
        python_cases = {}
        python_cases.update(construction_cases(link))
        python_cases.update(adapter_cases(link, BaseNetwork, DirectedBaseNetwork))
        python_cases.update(matrix_cases(link, BaseNetwork))
        python_cases.update(resource_cases(link, BaseNetwork))
        python_cases.update(latency_cases(link, BaseNetwork))
        boundary_count = characterization_cases(link, BaseNetwork)
        cpp_cases = compare_node_attribute.parse_harness(args.harness)
        compare_cases(cpp_cases, python_cases)
    finally:
        compare_node_attribute.restore(previous, names)

    payload = {
        "source_sha256": SOURCE_SHA256,
        "node_source_sha256": compare_node_attribute.SOURCE_SHA256,
        "base_source_sha256": compare_node_attribute.BASE_SHA256,
        "method_source_sha256": compare_node_attribute.METHOD_SHA256,
        "dataset_source_sha256": compare_dataset_core.SOURCE_SHA256,
        "harness_sha256": hashlib.sha256(args.harness.read_bytes()).hexdigest(),
        "numpy_version": dataset.np.__version__,
        "networkx_version": link.nx.__version__,
        "differential_cases": len(python_cases),
        "python_boundary_cases": boundary_count,
        "total_cases": len(python_cases) + boundary_count,
        "result": "PASS",
    }
    if args.output:
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"link_attribute differential: PASS ({payload['total_cases']} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
