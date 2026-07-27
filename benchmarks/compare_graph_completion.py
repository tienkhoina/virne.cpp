#!/usr/bin/env python3
"""Parity checks and focused benchmarks for the completed graph surface."""

from __future__ import annotations

import itertools
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import networkx as nx
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "build" / "graph" / "graph_completion_harness"
MASK = (1 << 64) - 1
FNV_OFFSET = 1469598103934665603


def mix(value: int, item: int) -> int:
    value ^= item
    return (value * 1099511628211) & MASK


def hash_path(value: int, path: list[int]) -> int:
    value = mix(value, len(path))
    for vertex in path:
        value = mix(value, vertex + 1)
    return value


def benchmark_graph(n: int) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    def add(u: int, v: int, weight: float) -> None:
        if v >= n or graph.has_edge(u, v):
            return
        edge_id = graph.number_of_edges()
        graph.add_edge(
            u,
            v,
            weight=weight,
            capacity=float(edge_id % 9 + 1),
        )

    for u in range(n):
        add(u, u + 1, 1.0 + (u % 7) * 0.013)
        add(u, u + 5, 2.2 + (u % 11) * 0.017)
        add(u, u + 13, 3.7 + (u % 13) * 0.019)
    return graph


def dense_matrix_fixture(n: int) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=np.float64)
    for u in range(n):
        if u % 17 == 0:
            matrix[u, u] = float(u % 13 + 1)
        for offset in (1, 7, 23):
            v = u + offset
            if v >= n:
                continue
            matrix[u, v] = float((u * 11 + v) % 31 + 1)
            if (u + offset) % 3 != 0:
                matrix[v, u] = float((u * 7 + v * 3) % 29 + 1)
    return matrix


def layered_graph(layers: int) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(range(2 * layers + 2))
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    for layer in range(layers - 1):
        left = 1 + 2 * layer
        next_left = left + 2
        graph.add_edge(left, next_left)
        graph.add_edge(left, next_left + 1)
        graph.add_edge(left + 1, next_left)
        graph.add_edge(left + 1, next_left + 1)
    last = 1 + 2 * (layers - 1)
    target = 2 * layers + 1
    graph.add_edge(last, target)
    graph.add_edge(last + 1, target)
    return graph


def reverse_layer_graph(graph_type: type[nx.Graph]) -> nx.Graph:
    graph = graph_type()
    graph.add_nodes_from(range(7))
    graph.add_edges_from(
        [
            (0, 3),
            (0, 2),
            (0, 1),
            (3, 5),
            (3, 4),
            (2, 5),
            (2, 4),
            (1, 5),
            (1, 4),
            (5, 6),
            (4, 6),
        ]
    )
    return graph


def ordered_result_graph(graph_type: type[nx.Graph]) -> nx.Graph:
    graph = graph_type()
    graph.add_nodes_from(range(7))
    graph.add_weighted_edges_from(
        [
            (0, 4, 2.0),
            (0, 2, 1.0),
            (0, 5, 1.0),
            (4, 3, 1.0),
            (2, 3, 2.0),
            (5, 6, 1.0),
            (3, 1, 1.0),
            (6, 1, 1.0),
        ]
    )
    return graph


def all_pairs_length_checksum(graph: nx.Graph) -> int:
    lengths = dict(nx.shortest_path_length(graph))
    checksum = FNV_OFFSET
    for source in graph.nodes:
        for target in graph.nodes:
            if target not in lengths[source]:
                continue
            checksum = mix(checksum, source + 1)
            checksum = mix(checksum, target + 1)
            checksum = mix(checksum, lengths[source][target] + 1)
    return checksum


def all_pairs_path_checksum(graph: nx.Graph) -> int:
    paths = nx.shortest_path(graph)
    checksum = FNV_OFFSET
    for source in graph.nodes:
        for target in graph.nodes:
            checksum = mix(checksum, source + 1)
            checksum = mix(checksum, target + 1)
            checksum = hash_path(checksum, paths[source][target])
    return checksum


def all_shortest_checksum(graph: nx.DiGraph) -> int:
    paths = nx.all_shortest_paths(graph, 0, len(graph) - 1)
    checksum = FNV_OFFSET
    for path in paths:
        checksum = hash_path(checksum, path)
    return checksum


def all_shortest_weighted_order_checksum(graph: nx.Graph) -> int:
    checksum = FNV_OFFSET
    for path in nx.all_shortest_paths(graph, 0, 6, weight="weight"):
        checksum = hash_path(checksum, path)
    return checksum


def filtered_path_checksum(view: nx.Graph) -> int:
    path = nx.dijkstra_path(view, 0, len(view) - 1, weight="weight")
    return hash_path(FNV_OFFSET, path)


def lazy_yen_checksum(graph: nx.Graph) -> int:
    paths = itertools.islice(
        nx.shortest_simple_paths(graph, 0, len(graph) - 1, weight="weight"),
        20,
    )
    checksum = FNV_OFFSET
    for path in paths:
        checksum = hash_path(checksum, path)
    return checksum


def unweighted_yen_checksum(graph: nx.Graph) -> int:
    checksum = FNV_OFFSET
    for path in nx.shortest_simple_paths(graph, 0, 6):
        checksum = hash_path(checksum, path)
    return checksum


def score_checksum(scores: dict[int, float]) -> int:
    checksum = FNV_OFFSET
    for vertex in scores:
        checksum = mix(checksum, round(scores[vertex] * 1e12))
    return checksum


def ordered_unweighted_map_checksum(graph: nx.Graph) -> int:
    result = nx.single_source_shortest_path_length(graph, 0)
    checksum = FNV_OFFSET
    for vertex, distance in result.items():
        checksum = mix(checksum, vertex + 1)
        checksum = mix(checksum, distance + 1)
    return checksum


def ordered_weighted_map_checksum(graph: nx.Graph) -> int:
    result = nx.single_source_dijkstra_path_length(
        graph, 0, weight="weight"
    )
    checksum = FNV_OFFSET
    for vertex, distance in result.items():
        checksum = mix(checksum, vertex + 1)
        checksum = mix(checksum, round(distance * 1e6))
    return checksum


def ordered_all_pairs_checksum(graph: nx.Graph) -> int:
    result = dict(nx.all_pairs_shortest_path_length(graph))
    checksum = FNV_OFFSET
    for source, lengths in result.items():
        checksum = mix(checksum, source + 1)
        for target, distance in lengths.items():
            checksum = mix(checksum, target + 1)
            checksum = mix(checksum, distance + 1)
    return checksum


def sparse_checksum(graph: nx.Graph, order: list[int]) -> int:
    matrix = nx.attr_sparse_matrix(
        graph,
        edge_attr="capacity",
        normalized=True,
        rc_order=order,
    ).tocoo()
    checksum = FNV_OFFSET
    for row, column, value in zip(matrix.row, matrix.col, matrix.data):
        scaled = int(float(value) * 1e12 + 0.5)
        checksum = mix(checksum, int(row) + 1)
        checksum = mix(checksum, int(column) + 1)
        checksum = mix(checksum, scaled)
    return checksum


def directed_checksum(graph: nx.Graph) -> int:
    directed = graph.to_directed()
    checksum = len(directed) * 1000003 + directed.number_of_edges()
    for source, target in directed.edges:
        checksum += (source + 1) * 1000033 + (target + 1) * 1000037
    return checksum & MASK


def in_out_checksum(graph: nx.DiGraph) -> int:
    checksum = 0
    for vertex in graph.nodes:
        checksum += (vertex + 1) * (
            graph.in_degree(vertex) * 31 + graph.out_degree(vertex) * 37
        )
    return checksum & MASK


def graph_shape_checksum(graph: nx.Graph) -> int:
    checksum = len(graph) * 1000003 + graph.number_of_edges()
    for source, target in graph.edges:
        checksum = mix(checksum, source + 1)
        checksum = mix(checksum, target + 1)
    return checksum


def dense_matrix_checksum(matrix: np.ndarray) -> int:
    graph = nx.Graph(matrix)
    checksum = graph_shape_checksum(graph)
    for source, target in graph.edges:
        checksum = mix(
            checksum,
            round(float(graph[source][target]["weight"]) * 1e6),
        )
    return checksum


def endpoint_attribute_checksum(graph: nx.Graph) -> int:
    values = nx.get_edge_attributes(graph, "capacity")
    checksum = FNV_OFFSET
    for (source, target), value in values.items():
        checksum = mix(checksum, source + 1)
        checksum = mix(checksum, target + 1)
        checksum = mix(checksum, round(float(value) * 1e6))
    return checksum


def unweighted_dijkstra_checksum(graph: nx.Graph) -> int:
    return hash_path(
        FNV_OFFSET,
        nx.dijkstra_path(
            graph, 0, len(graph) - 1, weight=None
        ),
    )


def cutoff_dijkstra_checksum(graph: nx.Graph) -> int:
    distances = nx.single_source_dijkstra_path_length(
        graph, 0, cutoff=4.0, weight="weight"
    )
    checksum = FNV_OFFSET
    for vertex, distance in distances.items():
        checksum = mix(checksum, vertex + 1)
        checksum = mix(checksum, round(float(distance) * 1e6))
    return checksum


def cutoff_unweighted_checksum(graph: nx.Graph) -> int:
    distances = nx.single_source_shortest_path_length(
        graph, 0, cutoff=2.1
    )
    checksum = FNV_OFFSET
    for vertex, distance in distances.items():
        checksum = mix(checksum, vertex + 1)
        checksum = mix(checksum, distance)
    return checksum


def waxman_checksum(graph: nx.Graph) -> int:
    checksum = graph_shape_checksum(graph)
    for vertex in graph.nodes:
        for coordinate in graph.nodes[vertex]["pos"]:
            bits = struct.unpack("=Q", struct.pack("=d", coordinate))[0]
            checksum = mix(checksum, bits)
    return checksum


@dataclass(frozen=True)
class CppResult:
    checksum: int
    milliseconds: float
    repeats: int


def cpp_results() -> dict[str, CppResult]:
    completed = subprocess.run(
        [str(HARNESS)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    results: dict[str, CppResult] = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5 or fields[0] != "RESULT":
            raise AssertionError(f"unknown C++ output: {line}")
        results[fields[1]] = CppResult(
            int(fields[2]),
            float(fields[3]),
            int(fields[4]),
        )
    return results


def time_case(repeats: int, function: Callable[[], int]) -> tuple[int, float]:
    checksum = function()
    start = time.perf_counter()
    sink = 0
    for _ in range(repeats):
        sink ^= function()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if sink == 0x12345678:
        raise AssertionError("unreachable benchmark sink")
    return checksum, elapsed_ms


def run() -> None:
    if nx.__version__ != "3.4.2":
        raise RuntimeError(f"NetworkX 3.4.2 required, got {nx.__version__}")
    if not HARNESS.is_file():
        raise RuntimeError(f"build the harness first: {HARNESS}")

    cpp = cpp_results()
    graph = benchmark_graph(120)
    dense_matrix = dense_matrix_fixture(120)
    path = nx.path_graph(100)
    layered = layered_graph(10)
    view = nx.subgraph_view(
        graph,
        filter_edge=lambda u, v: abs(u - v) == 1 or (u + v) % 4 != 0,
    )
    directed = graph.to_directed()
    tie_graph = reverse_layer_graph(nx.Graph)
    tie_digraph = reverse_layer_graph(nx.DiGraph)
    tie_view = nx.subgraph_view(
        tie_digraph,
        filter_edge=lambda u, v: (u, v) != (2, 5),
    )
    ordered_graph = ordered_result_graph(nx.Graph)
    ordered_digraph = ordered_result_graph(nx.DiGraph)
    ordered_graph_view = nx.subgraph_view(
        ordered_graph,
        filter_node=lambda vertex: vertex != 4,
    )
    ordered_digraph_view = nx.subgraph_view(
        ordered_digraph,
        filter_node=lambda vertex: vertex != 4,
    )
    reverse_order = list(reversed(list(graph.nodes)))

    functions: dict[str, Callable[[], int]] = {
        "all_pairs_length": lambda: all_pairs_length_checksum(graph),
        "all_pairs_path": lambda: all_pairs_path_checksum(path),
        "ordered_unweighted_graph": lambda: ordered_unweighted_map_checksum(
            ordered_graph
        ),
        "ordered_unweighted_digraph": lambda: ordered_unweighted_map_checksum(
            ordered_digraph
        ),
        "ordered_unweighted_graph_view": lambda: (
            ordered_unweighted_map_checksum(ordered_graph_view)
        ),
        "ordered_unweighted_digraph_view": lambda: (
            ordered_unweighted_map_checksum(ordered_digraph_view)
        ),
        "ordered_weighted_graph": lambda: ordered_weighted_map_checksum(
            ordered_graph
        ),
        "ordered_weighted_digraph": lambda: ordered_weighted_map_checksum(
            ordered_digraph
        ),
        "ordered_weighted_graph_view": lambda: ordered_weighted_map_checksum(
            ordered_graph_view
        ),
        "ordered_weighted_digraph_view": lambda: ordered_weighted_map_checksum(
            ordered_digraph_view
        ),
        "ordered_all_pairs_graph_view": lambda: ordered_all_pairs_checksum(
            ordered_graph_view
        ),
        "ordered_all_pairs_digraph_view": lambda: ordered_all_pairs_checksum(
            ordered_digraph_view
        ),
        "all_shortest_paths": lambda: all_shortest_checksum(layered),
        "all_shortest_weighted_graph_order": lambda: (
            all_shortest_weighted_order_checksum(tie_graph)
        ),
        "all_shortest_weighted_digraph_order": lambda: (
            all_shortest_weighted_order_checksum(tie_digraph)
        ),
        "all_shortest_weighted_view_order": lambda: (
            all_shortest_weighted_order_checksum(tie_view)
        ),
        "filtered_dijkstra": lambda: filtered_path_checksum(view),
        "lazy_yen_20": lambda: lazy_yen_checksum(graph),
        "yen_unweighted_graph_order": lambda: unweighted_yen_checksum(tie_graph),
        "yen_unweighted_digraph_order": lambda: unweighted_yen_checksum(
            tie_digraph
        ),
        "yen_unweighted_view_order": lambda: unweighted_yen_checksum(tie_view),
        "attr_sparse_normalized": lambda: sparse_checksum(graph, reverse_order),
        "dense_matrix_constructor": lambda: dense_matrix_checksum(dense_matrix),
        "edge_attributes_endpoint": lambda: endpoint_attribute_checksum(graph),
        "dijkstra_weight_none": lambda: unweighted_dijkstra_checksum(graph),
        "single_source_dijkstra_cutoff": lambda: cutoff_dijkstra_checksum(graph),
        "single_source_shortest_cutoff": lambda: cutoff_unweighted_checksum(graph),
        "to_directed": lambda: directed_checksum(graph),
        "digraph_in_out_fast": lambda: in_out_checksum(directed),
        "betweenness_unweighted": lambda: score_checksum(
            nx.betweenness_centrality(graph)
        ),
        "betweenness_weighted": lambda: score_checksum(
            nx.betweenness_centrality(graph, weight="weight")
        ),
        "erdos_renyi": lambda: graph_shape_checksum(
            nx.erdos_renyi_graph(220, 0.08, seed=42)
        ),
        "connected_erdos_renyi": lambda: graph_shape_checksum(
            _connected_erdos(160, 0.035, 42)
        ),
        "waxman": lambda: waxman_checksum(
            nx.waxman_graph(150, beta=0.65, alpha=0.35, seed=42)
        ),
    }

    print("| API | parity | C++ ms | NetworkX ms | speedup |")
    print("|---|---:|---:|---:|---:|")
    for name, result in cpp.items():
        checksum, python_ms = time_case(result.repeats, functions[name])
        if checksum != result.checksum:
            raise AssertionError(
                f"{name}: checksum {result.checksum} != NetworkX {checksum}"
            )
        speedup = python_ms / result.milliseconds
        if speedup <= 1.0:
            raise AssertionError(
                f"{name}: C++ {result.milliseconds:.3f} ms is not faster "
                f"than NetworkX {python_ms:.3f} ms"
            )
        print(
            f"| {name} | exact | {result.milliseconds:.3f} | "
            f"{python_ms:.3f} | {speedup:.2f}x |"
        )


def _connected_erdos(n: int, probability: float, seed: int) -> nx.Graph:
    import random

    stream = random.Random(seed)
    for _ in range(10_000):
        graph = nx.erdos_renyi_graph(n, probability, seed=stream)
        if nx.is_connected(graph):
            return graph
    raise RuntimeError("connected Erdős-Rényi retry exhausted")


if __name__ == "__main__":
    run()
