#!/usr/bin/env python3
"""AST-isolated exact differential for the pinned non-ML NodeRank leaf."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import struct
import subprocess
from typing import Any, Dict, Type, Union

import numpy as np
import networkx as nx


SOURCE_SHA256 = "428f0deb188e685a9f3eb6177a70df533519f62c560284063fc4cf6ec7624fff"
WORKERS = (1, 2, 8)
CASES = (
    "order",
    "random_sequence",
    "ffd_int",
    "ffd_mixed_nan",
    "nrm",
    "nrm_no_links",
    "nea",
    "grc",
    "grc_inf",
    "rw",
    "nps_unsorted",
    "nps_sorted",
    "nps_weighted",
)


def load_node_rank(source: pathlib.Path):
    source = source.resolve()
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"NodeRank source hash drift: {actual}")

    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    expected = [
        "NodeRankRegistry",
        "NodeRank",
        "OrderNodeRank",
        "RandomNodeRank",
        "FFDNodeRank",
        "NRMNodeRank",
        "DegreeWeightedResoureNodeRank",
        "GRCNodeRank",
        "RWNodeRank",
        "NPSNodeRank",
    ]
    names = [node.name for node in classes]
    if names != expected:
        raise RuntimeError(
            f"NodeRank class inventory drift: expected {expected}, got {names}")

    isolated = ast.fix_missing_locations(
        ast.Module(body=classes, type_ignores=[]))
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "abc": __import__("abc"),
        "Any": Any,
        "BaseNetwork": Any,
        "Dict": Dict,
        "Type": Type,
        "Union": Union,
        "np": np,
        "nx": nx,
    }
    exec(compile(isolated, str(source), "exec"), namespace)
    registry = namespace["NodeRankRegistry"]._registry
    if list(registry) != [
        "order", "random", "ffd", "nrm", "nea", "grc", "rw", "nps"
    ]:
        raise RuntimeError(f"NodeRank registry order drift: {list(registry)}")
    return registry


class FakeNetwork(nx.Graph):
    def __init__(
        self,
        node_count: int,
        edges,
        node_rows=(),
        aggregation_rows=(),
        adjacency_matrices=(),
    ):
        super().__init__()
        self.add_nodes_from(range(node_count))
        self.add_edges_from(edges)
        self.num_nodes = node_count
        self._node_rows = [list(row) for row in node_rows]
        self._aggregation_rows = [list(row) for row in aggregation_rows]
        self._adjacency_matrices = [
            np.asarray(matrix, dtype=np.float64)
            for matrix in adjacency_matrices
        ]
        self._node_token = object()
        self._link_token = object()

    def get_node_attrs(self, _types):
        return self._node_token

    def get_link_attrs(self, _types):
        return self._link_token

    def get_node_attrs_data(self, token):
        if token is not self._node_token:
            raise RuntimeError("unexpected node attribute token")
        return [list(row) for row in self._node_rows]

    def get_aggregation_attrs_data(
        self, token, aggr="sum", normalized=False
    ):
        if token is not self._link_token or aggr != "sum" or normalized:
            raise RuntimeError("unexpected aggregation request")
        return [list(row) for row in self._aggregation_rows]

    def get_adjacency_attrs_data(self, token, normalized=False):
        if token is not self._link_token:
            raise RuntimeError("unexpected adjacency request")
        return [matrix.copy() for matrix in self._adjacency_matrices]


class FakeCOO:
    def __init__(self, network: FakeNetwork):
        rows: list[int] = []
        columns: list[int] = []
        data: list[float] = []
        for source in network.nodes:
            for target, attributes in network.adj[source].items():
                rows.append(int(source))
                columns.append(int(target))
                data.append(float(attributes.get("weight", 1.0)))
        self.row = np.asarray(rows, dtype=np.int64)
        self.col = np.asarray(columns, dtype=np.int64)
        self.data = np.asarray(data, dtype=np.float64)
        self._shape = (network.num_nodes, network.num_nodes)

    def tocoo(self):
        return self

    def nonzero(self):
        keep = self.data != 0
        return self.row[keep], self.col[keep]

    def toarray(self):
        result = np.zeros(self._shape, dtype=np.float64)
        for row, column, value in zip(self.row, self.col, self.data):
            result[int(row), int(column)] += value
        return result


class FakeSparseMatrix:
    def __init__(self, network: FakeNetwork):
        self._coo = FakeCOO(network)

    def tocoo(self):
        return self._coo


def binary64(value: Any) -> str:
    return struct.pack(">d", float(value)).hex()


def canonical_ranking(ranking) -> list[list[Any]]:
    result: list[list[Any]] = []
    for node, value in ranking.items():
        if isinstance(value, tuple):
            distance, score = value
            result.append([int(node), "p", binary64(score), binary64(distance)])
        else:
            result.append([int(node), "s", binary64(value), binary64(0.0)])
    return result


def make_case(name: str) -> tuple[FakeNetwork, bool, dict[str, Any]]:
    if name == "order":
        return FakeNetwork(4, [(2, 3), (0, 1)]), True, {}
    if name == "ffd_int":
        return FakeNetwork(
            4,
            [(0, 1), (1, 2), (2, 3)],
            node_rows=[
                [5, np.iinfo(np.int64).max, -3, 7],
                [True, 1, False, -2],
            ],
        ), True, {}
    if name == "ffd_mixed_nan":
        return FakeNetwork(
            3,
            [(0, 1), (1, 2)],
            node_rows=[[2, -2, 1], [0.5, -0.0, float("nan")]],
        ), True, {}
    if name in {"nrm", "nea"}:
        return FakeNetwork(
            4,
            [(0, 1), (1, 1), (1, 2)],
            node_rows=[[2, 3, -4, 5]],
            aggregation_rows=[[1.0, 6.0, 3.0, 0.0]],
        ), False, {}
    if name == "nrm_no_links":
        return FakeNetwork(
            2,
            [(0, 1)],
            node_rows=[[-2, 3]],
            aggregation_rows=[],
        ), False, {}
    if name in {"grc", "grc_inf", "rw"}:
        matrix = (
            [[0.0, 5.0], [5.0, 0.0]]
            if name == "rw"
            else [[0.0, 1.0], [1.0, 0.0]]
        )
        return FakeNetwork(
            2,
            [(0, 1)],
            node_rows=[[2, 3]],
            adjacency_matrices=[matrix],
        ), False, ({"sigma": float("inf")} if name == "grc_inf" else {})
    if name in {"nps_unsorted", "nps_sorted", "nps_weighted"}:
        network = FakeNetwork(
            4,
            [(0, 2), (0, 1)],
            node_rows=[[1, 5, 2, 9]],
            aggregation_rows=[[2.0, 1.0, 1.0, 0.0]],
        )
        if name == "nps_weighted":
            network[0][2]["weight"] = 3.0
            network[0][1]["weight"] = 1.0
        return network, name != "nps_unsorted", {}
    raise KeyError(name)


def python_case(registry, name: str):
    if name == "random_sequence":
        network = FakeNetwork(6, [(0, 1), (1, 2)])
        np.random.seed(42)
        ranker = registry["random"]()
        first = ranker.rank(network, sort=False)
        second = ranker.rank(network, sort=False)
        next_value = int(np.random.randint(0, 2**31))
        return {
            "first": canonical_ranking(first),
            "second": canonical_ranking(second),
            "next": next_value,
        }

    network, sort, constructor = make_case(name)
    method = name.split("_", 1)[0]
    if name == "grc_inf":
        method = "grc"
    ranker = registry[method](**constructor)
    if name == "rw":
        original = nx.adjacency_matrix
        nx.adjacency_matrix = lambda graph: FakeSparseMatrix(graph)
        try:
            ranking = ranker.rank(network, sort=sort)
        finally:
            nx.adjacency_matrix = original
    else:
        ranking = ranker.rank(network, sort=sort)
    return {"ranking": canonical_ranking(ranking)}


def native_case(harness: pathlib.Path, name: str, workers: int):
    completed = subprocess.run(
        [str(harness), name, str(workers)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    args = parser.parse_args()

    registry = load_node_rank(args.source)
    expected = {name: python_case(registry, name) for name in CASES}
    payload = json.dumps(
        expected, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    payload_sha = hashlib.sha256(payload).hexdigest().upper()

    checked = 0
    for name in CASES:
        for workers in WORKERS:
            actual = native_case(args.harness.resolve(), name, workers)
            if actual != expected[name]:
                raise AssertionError(
                    f"{name} workers={workers} mismatch\n"
                    f"python={expected[name]}\ncpp={actual}")
        checked += 1

    result = {
        "component": "solver.rank.NodeRank",
        "source_sha256": SOURCE_SHA256.upper(),
        "shared_case_count": checked,
        "native_workers": list(WORKERS),
        "case_payload_sha256": payload_sha,
        "status": "PASS",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
