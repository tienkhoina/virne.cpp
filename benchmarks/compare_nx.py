#!/usr/bin/env python3
"""Compare the local DiGraph implementation with NetworkX 3.4.2.

Run this file with the repository-local interpreter after installing the
small, pinned requirements file::

    .venv/bin/python benchmarks/compare_nx.py

The C++ binary only emits deterministic text, so this script is also useful
on machines where scipy is unavailable: ``--parity-only`` does not import
scipy and still checks the core, centrality, shortest-path, and adjacency
results.  The benchmark uses the same edge recipe in both languages and
constructs both graphs before timing.
"""

from __future__ import annotations

import argparse
import itertools
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = ROOT / "build" / "graph_nx_harness"
FIXTURE_EDGES = [
    (0, 1, 1.0),
    (0, 2, 4.0),
    (1, 2, 2.0),
    (1, 3, 5.0),
    (2, 3, 1.0),
    (3, 1, 1.0),
    (3, 4, 3.0),
    (4, 0, 7.0),
]
OFFSETS = (1, 7, 31, 127, 509)


def fixture_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(range(5))
    graph.add_weighted_edges_from(FIXTURE_EDGES, weight="weight")
    return graph


def benchmark_graph(nodes: int) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(range(nodes))
    for u in range(nodes):
        for offset in OFFSETS:
            v = (u + offset) % nodes
            if u == v or graph.has_edge(u, v):
                continue
            graph.add_edge(u, v, weight=1.0 + ((u * 13 + offset) % 997) / 997.0)
    return graph


def parse_cpp_output(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise AssertionError(f"malformed C++ output line: {line!r}")
        key, value = line.split("=", 1)
        if key in values:
            raise AssertionError(f"duplicate C++ output key: {key}")
        values[key] = value
    return values


def as_float(value: str) -> float:
    return math.inf if value == "inf" else float(value)


def as_path(value: str) -> list[int]:
    return [] if not value else [int(item) for item in value.split(",")]


def assert_close(actual: float, expected: float, key: str) -> None:
    # NetworkX and the C++ implementation use different operation orders in
    # a few iterative centralities.  This is still tight enough to catch a
    # wrong directed convention or a missing edge.
    if not math.isclose(actual, expected, rel_tol=5e-6, abs_tol=2e-6):
        raise AssertionError(f"{key}: C++={actual!r}, NetworkX={expected!r}")


def compare_fixture(cpp: dict[str, str]) -> int:
    graph = fixture_graph()
    expected: dict[str, Any] = {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "has_edge_0_1": int(graph.has_edge(0, 1)),
        "has_edge_1_0": int(graph.has_edge(1, 0)),
        "has_edge_3_1": int(graph.has_edge(3, 1)),
        "has_edge_1_3": int(graph.has_edge(1, 3)),
        "shortest_path_length_0_4": nx.shortest_path_length(graph, 0, 4),
        "shortest_path_0_4": nx.shortest_path(graph, 0, 4),
        "dijkstra_path_length_0_4": nx.dijkstra_path_length(graph, 0, 4, weight="weight"),
        "dijkstra_path_0_4": nx.dijkstra_path(graph, 0, 4, weight="weight"),
        "is_connected_weak": int(nx.is_weakly_connected(graph)),
        "yen_count": 3,
    }

    degree = nx.degree_centrality(graph)
    eigen = nx.eigenvector_centrality(graph, max_iter=10000, tol=1e-6)
    closeness = nx.closeness_centrality(graph)
    between = nx.betweenness_centrality(graph, normalized=True, weight="weight")
    bfs = dict(nx.single_source_shortest_path_length(graph, 0))
    dijkstra = dict(nx.single_source_dijkstra_path_length(graph, 0, weight="weight"))
    floyd = dict(nx.floyd_warshall(graph, weight="weight"))
    yen = list(nx.shortest_simple_paths(graph, 0, 4, weight="weight"))[:3]
    adjacency = {
        (u, v): float(data.get("weight", 1.0))
        for u, v, data in graph.edges(data=True)
    }

    for vertex in graph:
        expected[f"degree_centrality[{vertex}]"] = degree[vertex]
        expected[f"eigenvector[{vertex}]"] = eigen[vertex]
        expected[f"closeness[{vertex}]"] = closeness[vertex]
        expected[f"betweenness[{vertex}]"] = between[vertex]
        expected[f"bfs_distance[{vertex}]"] = bfs[vertex]
        expected[f"bfs_map[{vertex}]"] = bfs[vertex]
        expected[f"dijkstra_distance[{vertex}]"] = dijkstra[vertex]
        expected[f"dijkstra_map[{vertex}]"] = dijkstra[vertex]

    for u in graph:
        for v in graph:
            expected[f"floyd[{u},{v}]"] = floyd[u][v]
    for i, path in enumerate(yen):
        expected[f"yen_path[{i}]"] = path
        expected[f"yen_cost[{i}]"] = sum(graph[a][b]["weight"] for a, b in zip(path, path[1:]))
    for (u, v), value in adjacency.items():
        expected[f"adj[{u},{v}]"] = value

    missing = sorted(set(expected) - set(cpp))
    if missing:
        raise AssertionError(f"C++ parity output is missing keys: {missing}")

    for key, want in expected.items():
        got_raw = cpp[key]
        if isinstance(want, list):
            got = as_path(got_raw)
            if got != want:
                raise AssertionError(f"{key}: C++={got!r}, NetworkX={want!r}")
        elif isinstance(want, int):
            got = int(got_raw)
            if got != int(want):
                raise AssertionError(f"{key}: C++={got!r}, NetworkX={want!r}")
        else:
            assert_close(as_float(got_raw), float(want), key)

    # SparseMatrix is a coordinate list.  A directed fixture has one entry
    # per arc, so checking the complete set catches accidental symmetrization.
    actual_adj_keys = {key for key in cpp if key.startswith("adj[")}
    expected_adj_keys = {key for key in expected if key.startswith("adj[")}
    if actual_adj_keys != expected_adj_keys:
        raise AssertionError(
            f"adjacency coordinates differ: C++={sorted(actual_adj_keys)}, "
            f"NetworkX={sorted(expected_adj_keys)}"
        )

    print(f"PARITY PASS: {len(expected)} values match NetworkX {nx.__version__}")
    return len(expected)


def run_cpp(binary: Path, *args: str) -> dict[str, str]:
    if not binary.exists():
        raise FileNotFoundError(
            f"{binary} does not exist; configure/build the graph_nx_harness target first"
        )
    completed = subprocess.run(
        [str(binary), *args],
        check=False,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"C++ harness failed with exit {completed.returncode}:\n{completed.stderr}"
        )
    return parse_cpp_output(completed.stdout)


def result_checksum(result: Any) -> int:
    """Small consumption barrier kept outside the measured Python interval."""
    if isinstance(result, bool):
        return int(result)
    if isinstance(result, (int, float)):
        return int(abs(float(result)) * 1_000_000) & 0xFFFFFFFF
    if isinstance(result, dict):
        return len(result)
    if hasattr(result, "nnz"):
        return int(result.nnz)
    try:
        return len(result)
    except TypeError:
        return hash(repr(result)) & 0xFFFFFFFF


def timed_median(
    fn: Any, warmups: int, reps: int, calls: int = 1
) -> tuple[float, int]:
    checksum = 0
    for _ in range(warmups):
        for _ in range(calls):
            result = fn()
        checksum += result_checksum(result)
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        for _ in range(calls):
            result = fn()
        samples.append((time.perf_counter() - start) * 1000.0 / calls)
        checksum += result_checksum(result)
    return statistics.median(samples), checksum


def path_cost_nx(graph: nx.DiGraph, path: list[int]) -> float:
    return sum(graph[u][v].get("weight", 1.0) for u, v in zip(path, path[1:]))


def path_prefix_costs_nx(graph: nx.DiGraph, path: list[int]) -> list[float]:
    prefix = [0.0]
    for u, v in zip(path, path[1:]):
        prefix.append(prefix[-1] + graph[u][v].get("weight", 1.0))
    return prefix


def reconstruct_path_nx(
    predecessors: dict[int, list[int]], source: int, target: int
) -> list[int]:
    path = [target]
    current = target
    while current != source:
        current = predecessors[current][0]
        path.append(current)
    path.reverse()
    return path


def generate_candidates_nx(
    graph: nx.DiGraph, shortest: list[int], target: int
) -> list[tuple[float, list[int]]]:
    """One Yen deviation round, composed only from public NetworkX APIs."""
    candidates: list[tuple[float, list[int]]] = []
    seen: set[tuple[int, ...]] = set()
    for spur_index in range(len(shortest) - 1):
        root = shortest[: spur_index + 1]
        hidden_nodes = frozenset(root[:-1])
        hidden_edge = (shortest[spur_index], shortest[spur_index + 1])
        view = nx.subgraph_view(
            graph,
            filter_node=lambda node, hidden=hidden_nodes: node not in hidden,
            filter_edge=lambda u, v, banned=hidden_edge: (u, v) != banned,
        )
        try:
            cost, spur = nx.bidirectional_dijkstra(
                view, root[-1], target, weight="weight"
            )
        except nx.NetworkXNoPath:
            continue
        full = root[:-1] + spur
        key = tuple(full)
        if key not in seen:
            seen.add(key)
            candidates.append((cost, full))
    return candidates


def compare_benchmark(cpp: dict[str, str], nodes: int, warmups: int, reps: int) -> None:
    count = int(cpp["bench_count"])
    cpp_reps = int(cpp["bench_reps"])
    if cpp_reps != reps:
        raise AssertionError(f"C++ benchmark used {cpp_reps} reps, expected {reps}")

    metadata: list[tuple[str, int, int, int, float]] = []
    for index in range(count):
        prefix = f"bench[{index}]."
        metadata.append(
            (
                cpp[prefix + "name"],
                int(cpp[prefix + "nodes"]),
                int(cpp[prefix + "edges"]),
                int(cpp[prefix + "calls"]),
                as_float(cpp[prefix + "cpp_ms"]),
            )
        )

    # All graph construction and all reusable views/fixtures stay outside
    # the timing loops, exactly as in the C++ harness.
    graphs = {size: benchmark_graph(size) for _, size, _, _, _ in metadata}
    for name, size, edges, _, _ in metadata:
        actual_edges = graphs[size].number_of_edges()
        if actual_edges != edges:
            raise AssertionError(
                f"{name}: graph mismatch C++=({size},{edges}), "
                f"NetworkX=({size},{actual_edges})"
            )

    main_graph = graphs[max(graphs)]
    main_target = main_graph.number_of_nodes() - 1
    hidden_nodes = frozenset({main_graph.number_of_nodes() // 3})
    hidden_edges = frozenset({(0, 1)})
    masked_graph = nx.subgraph_view(
        main_graph,
        filter_node=lambda node: node not in hidden_nodes,
        filter_edge=lambda u, v: (u, v) not in hidden_edges,
    )
    prepared_path = nx.dijkstra_path(
        main_graph, 0, main_target, weight="weight"
    )
    prepared_predecessors, _ = nx.dijkstra_predecessor_and_distance(
        main_graph, 0, weight="weight"
    )
    split = len(prepared_path) // 2
    join_root = prepared_path[: split + 1]
    join_spur = prepared_path[split:]

    yen_size = next(
        size
        for name, size, _, _, _ in metadata
        if name == "yen_k_shortest_paths"
    )
    yen_graph = graphs[yen_size]
    yen_target = yen_size - 1
    _, first_yen_path = nx.bidirectional_dijkstra(
        yen_graph, 0, yen_target, weight="weight"
    )

    def graph_for(name: str, size: int) -> nx.DiGraph:
        graph = graphs[size]
        if name in {
            "raw_neighbor_scan", "weight_cache_scan", "bfs", "bfs_nx", "bfs_stats",
            "bidirectional_bfs", "nx.single_source_shortest_path_length",
            "nx.shortest_path_length", "nx.shortest_path", "dijkstra",
            "dijkstra_masked", "bidirectional_dijkstra",
            "edge_cost", "path_cost", "path_prefix_costs", "build_path",
            "join_paths",
            "nx.single_source_dijkstra_path_length", "nx.dijkstra_path_length",
            "nx.dijkstra_path", "nx.adjacency_matrix", "nx.attr_sparse_matrix",
            "nx.is_connected", "nx.degree_centrality",
        }:
            return main_graph
        return graph

    def operation(name: str, size: int) -> Any:
        graph = graph_for(name, size)
        target = graph.number_of_nodes() - 1
        operations: dict[str, Any] = {
            "raw_neighbor_scan": lambda: sum(
                int(v) for u in graph for v in graph.adj[u]
            ),
            "weight_cache_scan": lambda: sum(
                data.get("weight", 1.0)
                for _, _, data in graph.edges(data=True)
            ),
            "bfs": lambda: dict(nx.single_source_shortest_path_length(graph, 0)),
            "bfs_nx": lambda: dict(nx.single_source_shortest_path_length(graph, 0)),
            "bfs_stats": lambda: (
                lambda distances: (sum(distances.values()), len(distances) - 1)
            )(dict(nx.single_source_shortest_path_length(graph, 0))),
            "bidirectional_bfs": lambda: nx.bidirectional_shortest_path(graph, 0, target),
            "nx.single_source_shortest_path_length": lambda: dict(
                nx.single_source_shortest_path_length(graph, 0)
            ),
            "nx.shortest_path_length": lambda: nx.shortest_path_length(graph, 0, target),
            "nx.shortest_path": lambda: nx.shortest_path(graph, 0, target),
            "dijkstra": lambda: dict(
                nx.single_source_dijkstra_path_length(graph, 0, weight="weight")
            ),
            "dijkstra_masked": lambda: dict(
                nx.single_source_dijkstra_path_length(masked_graph, 0, weight="weight")
            ),
            "bidirectional_dijkstra": lambda: nx.bidirectional_dijkstra(
                graph, 0, target, weight="weight"
            ),
            "edge_cost": lambda: graph[0][1].get("weight", 1.0),
            "path_cost": lambda: path_cost_nx(graph, prepared_path),
            "path_prefix_costs": lambda: path_prefix_costs_nx(
                graph, prepared_path
            ),
            "build_path": lambda: reconstruct_path_nx(
                prepared_predecessors, 0, main_target
            ),
            "join_paths": lambda: join_root + join_spur[1:],
            "nx.single_source_dijkstra_path_length": lambda: dict(
                nx.single_source_dijkstra_path_length(graph, 0, weight="weight")
            ),
            "nx.dijkstra_path_length": lambda: nx.dijkstra_path_length(
                graph, 0, target, weight="weight"
            ),
            "nx.dijkstra_path": lambda: nx.dijkstra_path(
                graph, 0, target, weight="weight"
            ),
            "floyd_warshall": lambda: nx.floyd_warshall(graph, weight="weight"),
            "nx.floyd_warshall": lambda: nx.floyd_warshall(graph, weight="weight"),
            "yen_k_shortest_paths": lambda: tuple(
                itertools.islice(
                    nx.shortest_simple_paths(graph, 0, target, weight="weight"), 5
                )
            ),
            "generate_candidates": lambda: generate_candidates_nx(
                yen_graph, first_yen_path, yen_target
            ),
            "nx.shortest_simple_paths": lambda: tuple(
                itertools.islice(
                    nx.shortest_simple_paths(graph, 0, target, weight="weight"), 5
                )
            ),
            "nx.adjacency_matrix": lambda: nx.adjacency_matrix(
                graph, weight="weight"
            ),
            "nx.attr_sparse_matrix": lambda: [
                (u, v, data["weight"])
                for u, v, data in graph.edges(data=True)
                if "weight" in data
            ],
            "nx.is_connected": lambda: nx.is_weakly_connected(graph),
            "nx.degree_centrality": lambda: nx.degree_centrality(graph),
            "nx.eigenvector_centrality": lambda: nx.eigenvector_centrality(
                graph, max_iter=10000, tol=1e-6
            ),
            "nx.closeness_centrality": lambda: nx.closeness_centrality(graph),
            "nx.betweenness_centrality": lambda: nx.betweenness_centrality(
                graph, normalized=True, weight="weight"
            ),
        }
        try:
            return operations[name]
        except KeyError as error:
            raise AssertionError(f"no NetworkX benchmark equivalent for {name}") from error

    results: list[tuple[str, int, int, int, float, float, float, str]] = []
    failures: list[str] = []
    checksum = 0
    for name, size, edges, calls, cpp_ms in metadata:
        py_ms, py_checksum = timed_median(
            operation(name, size), warmups, reps, calls
        )
        checksum += py_checksum
        speedup = py_ms / cpp_ms
        passed = cpp_ms < py_ms
        status = "PASS" if passed else "FAIL"
        results.append((name, size, edges, calls, cpp_ms, py_ms, speedup, status))
        if not passed:
            failures.append(name)

    print("\n| Algorithm / API | Nodes | Edges | Calls/sample | C++ ms/call | NetworkX ms/call | Speedup | Status |")
    print("|---|---:|---:|---:|---:|---:|---:|:---:|")
    for name, size, edges, calls, cpp_ms, py_ms, speedup, status in results:
        print(
            f"| `{name}` | {size} | {edges} | {calls} | {cpp_ms:.6f} | "
            f"{py_ms:.6f} | {speedup:.2f}x | {status} |"
        )
    print(f"\nBENCH CHECKSUM: Python={checksum}, C++={cpp['bench_sink']}")

    if failures:
        raise AssertionError(
            "strict benchmark gate requires C++ < NetworkX for: "
            + ", ".join(failures)
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--parity-only", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--nodes", type=int, default=5000)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--reps", type=int, default=15)
    args = parser.parse_args()
    if nx.__version__ != "3.4.2":
        raise RuntimeError(
            f"this oracle requires NetworkX 3.4.2, found {nx.__version__}; "
            "run it with .venv/bin/python after installing benchmarks/requirements.txt"
        )
    if args.parity_only and args.benchmark_only:
        parser.error("--parity-only and --benchmark-only are mutually exclusive")
    if args.nodes < 1 or args.reps < 1 or args.warmups < 0:
        parser.error("nodes/reps must be positive and warmups must be non-negative")

    if not args.benchmark_only:
        compare_fixture(run_cpp(args.binary, "--parity"))
    if not args.parity_only:
        compare_benchmark(
            run_cpp(
                args.binary,
                "--bench",
                "--nodes",
                str(args.nodes),
                "--warmups",
                str(args.warmups),
                "--reps",
                str(args.reps),
            ),
            args.nodes,
            args.warmups,
            args.reps,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, FileNotFoundError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
