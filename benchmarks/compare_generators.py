#!/usr/bin/env python3
"""Differential topology-generator parity against NetworkX 3.4.2."""

from __future__ import annotations

import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "build" / "graph" / "graph_generator_parity_harness"


@dataclass(frozen=True)
class Dump:
    nodes: list[int]
    edges: list[tuple[int, int]]
    positions: dict[int, tuple[float, float]]
    attr_counts: tuple[int, int]


def parse_dump(text: str) -> Dump:
    nodes: list[int] = []
    edges: list[tuple[int, int]] = []
    positions: dict[int, tuple[float, float]] = {}
    attr_counts = (-1, -1)
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "NODES":
            nodes = [int(value) for value in fields[1:]]
        elif fields[0] == "EDGE":
            edges.append((int(fields[1]), int(fields[2])))
        elif fields[0] == "POS":
            positions[int(fields[1])] = (
                float(fields[2]),
                float(fields[3]),
            )
        elif fields[0] == "ATTR_COUNTS":
            attr_counts = (int(fields[1]), int(fields[2]))
        else:
            raise AssertionError(f"unknown harness record: {line}")
    return Dump(nodes, edges, positions, attr_counts)


def cpp_dump(*arguments: object) -> Dump:
    completed = subprocess.run(
        [str(HARNESS), *(str(value) for value in arguments)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return parse_dump(completed.stdout)


def cpp_dump_sequence(*arguments: object) -> tuple[Dump, Dump]:
    completed = subprocess.run(
        [str(HARNESS), *(str(value) for value in arguments)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    parts = completed.stdout.split("\nNEXT\n")
    if len(parts) != 2:
        raise AssertionError(f"expected two graph dumps, got {len(parts)}")
    return parse_dump(parts[0]), parse_dump(parts[1])


def python_dump(graph: nx.Graph) -> Dump:
    positions = {
        int(node): (float(data["pos"][0]), float(data["pos"][1]))
        for node, data in graph.nodes(data=True)
        if "pos" in data
    }
    node_attr_count = sum(len(data) for _, data in graph.nodes(data=True))
    edge_attr_count = sum(len(data) for _, _, data in graph.edges(data=True))
    return Dump(
        [int(node) for node in graph.nodes],
        [(int(u), int(v)) for u, v in graph.edges],
        positions,
        (node_attr_count, edge_attr_count),
    )


def connected_retry(factory: Callable[[], nx.Graph], attempts: int) -> nx.Graph:
    for _ in range(attempts):
        graph = factory()
        if nx.is_connected(graph):
            return graph
    raise RuntimeError("NetworkX connected retry exhausted")


def assert_equal(label: str, actual: Dump, expected: Dump) -> None:
    if actual != expected:
        if actual.nodes != expected.nodes:
            detail = f"nodes {actual.nodes!r} != {expected.nodes!r}"
        elif actual.edges != expected.edges:
            detail = f"edges {actual.edges!r} != {expected.edges!r}"
        elif actual.positions != expected.positions:
            detail = f"positions {actual.positions!r} != {expected.positions!r}"
        else:
            detail = f"attr_counts {actual.attr_counts!r} != {expected.attr_counts!r}"
        raise AssertionError(f"{label}: {detail}")


def run() -> int:
    if nx.__version__ != "3.4.2":
        raise RuntimeError(f"NetworkX 3.4.2 required, got {nx.__version__}")
    if not HARNESS.is_file():
        raise RuntimeError(f"build the harness first: {HARNESS}")

    checks = 0
    deterministic = [
        ("path-0", ("path", 0), nx.path_graph(0)),
        ("path-7", ("path", 7), nx.path_graph(7)),
        ("star-0", ("star", 0), nx.star_graph(0)),
        ("star-6", ("star", 6), nx.star_graph(6)),
    ]
    for label, arguments, graph in deterministic:
        assert_equal(label, cpp_dump(*arguments), python_dump(graph))
        checks += 1

    for rows, columns in [(0, 3), (1, 4), (2, 3), (4, 5)]:
        graph = nx.grid_2d_graph(rows, columns, periodic=False)
        graph = nx.convert_node_labels_to_integers(graph, ordering="default")
        label = f"grid-{rows}x{columns}"
        assert_equal(label, cpp_dump("grid", rows, columns), python_dump(graph))
        checks += 1

    for rows, columns in [(1, 4), (2, 3), (3, 4), (4, 5)]:
        graph = nx.grid_2d_graph(rows, columns, periodic=True)
        graph = nx.convert_node_labels_to_integers(graph, ordering="default")
        label = f"grid-periodic-{rows}x{columns}"
        assert_equal(
            label,
            cpp_dump("grid_periodic", rows, columns),
            python_dump(graph),
        )
        checks += 1

    seeds = [0, 1, 2, 42, 4_294_967_303]
    for directed in (False, True):
        name = "erdos_di" if directed else "erdos"
        for probability in (0.0, 0.2, 0.65, 1.0):
            for seed in seeds:
                graph = nx.erdos_renyi_graph(
                    11,
                    probability,
                    seed=seed,
                    directed=directed,
                )
                label = f"{name}-p{probability}-seed{seed}"
                assert_equal(
                    label,
                    cpp_dump(name, 11, probability, seed),
                    python_dump(graph),
                )
                checks += 1

    for seed in seeds:
        graph = nx.waxman_graph(
            12,
            beta=0.65,
            alpha=0.35,
            seed=seed,
        )
        label = f"waxman-seed{seed}"
        assert_equal(
            label,
            cpp_dump("waxman", 12, 0.35, 0.65, seed),
            python_dump(graph),
        )
        checks += 1

    attempts = 10_000
    sequence_seeds = [0, 42, 4_294_967_303]
    for seed in sequence_seeds:
        stream = random.Random(seed)
        expected = tuple(
            python_dump(
                nx.erdos_renyi_graph(
                    13,
                    0.31,
                    seed=stream,
                )
            )
            for _ in range(2)
        )
        actual = cpp_dump_sequence(
            "erdos_global", 13, 0.31, seed
        )
        for index in range(2):
            assert_equal(
                f"erdos-global-seed{seed}-graph{index}",
                actual[index],
                expected[index],
            )
        checks += 1

    for seed in sequence_seeds:
        stream = random.Random(seed)
        expected = tuple(
            python_dump(
                nx.erdos_renyi_graph(
                    13,
                    0.31,
                    seed=stream,
                    directed=True,
                )
            )
            for _ in range(2)
        )
        actual = cpp_dump_sequence(
            "erdos_di_global", 13, 0.31, seed
        )
        for index in range(2):
            assert_equal(
                f"erdos-di-global-seed{seed}-graph{index}",
                actual[index],
                expected[index],
            )
        checks += 1

    for directed in (False, True):
        name = "erdos_di_stream" if directed else "erdos_stream"
        for seed in sequence_seeds:
            stream = random.Random(seed)
            expected = tuple(
                python_dump(
                    nx.erdos_renyi_graph(
                        13,
                        0.31,
                        seed=stream,
                        directed=directed,
                    )
                )
                for _ in range(2)
            )
            actual = cpp_dump_sequence(name, 13, 0.31, seed)
            for index in range(2):
                assert_equal(
                    f"{name}-seed{seed}-graph{index}",
                    actual[index],
                    expected[index],
                )
            checks += 1

    for seed in sequence_seeds:
        stream = random.Random(seed)
        expected = tuple(
            python_dump(
                connected_retry(
                    lambda: nx.erdos_renyi_graph(
                        10, 0.23, seed=stream
                    ),
                    attempts,
                )
            )
            for _ in range(2)
        )
        actual = cpp_dump_sequence(
            "connected_erdos_stream", 10, 0.23, seed, attempts
        )
        for index in range(2):
            assert_equal(
                f"connected-erdos-stream-seed{seed}-graph{index}",
                actual[index],
                expected[index],
            )
        checks += 1

    for seed in sequence_seeds:
        stream = random.Random(seed)
        expected = tuple(
            python_dump(
                nx.waxman_graph(
                    12,
                    seed=stream,
                )
            )
            for _ in range(2)
        )
        actual = cpp_dump_sequence(
            "waxman_global", 12, seed
        )
        for index in range(2):
            assert_equal(
                f"waxman-global-defaults-seed{seed}-graph{index}",
                actual[index],
                expected[index],
            )
        checks += 1

    for seed in sequence_seeds:
        stream = random.Random(seed)
        expected = tuple(
            python_dump(
                nx.waxman_graph(
                    12,
                    beta=0.65,
                    alpha=0.35,
                    seed=stream,
                )
            )
            for _ in range(2)
        )
        actual = cpp_dump_sequence(
            "waxman_stream", 12, 0.35, 0.65, seed
        )
        for index in range(2):
            assert_equal(
                f"waxman-stream-seed{seed}-graph{index}",
                actual[index],
                expected[index],
            )
        checks += 1

    for seed in sequence_seeds:
        stream = random.Random(seed)
        expected = tuple(
            python_dump(
                connected_retry(
                    lambda: nx.waxman_graph(
                        12,
                        beta=0.8,
                        alpha=0.45,
                        seed=stream,
                    ),
                    attempts,
                )
            )
            for _ in range(2)
        )
        actual = cpp_dump_sequence(
            "connected_waxman_stream",
            12,
            0.45,
            0.8,
            seed,
            attempts,
        )
        for index in range(2):
            assert_equal(
                f"connected-waxman-stream-seed{seed}-graph{index}",
                actual[index],
                expected[index],
            )
        checks += 1

    for seed in seeds:
        stream = random.Random(seed)
        graph = connected_retry(
            lambda: nx.erdos_renyi_graph(10, 0.23, seed=stream),
            attempts,
        )
        label = f"connected-erdos-seed{seed}"
        assert_equal(
            label,
            cpp_dump("connected_erdos", 10, 0.23, seed, attempts),
            python_dump(graph),
        )
        checks += 1

    for seed in seeds:
        stream = random.Random(seed)
        graph = connected_retry(
            lambda: nx.waxman_graph(
                12,
                beta=0.8,
                alpha=0.45,
                seed=stream,
            ),
            attempts,
        )
        label = f"connected-waxman-seed{seed}"
        assert_equal(
            label,
            cpp_dump(
                "connected_waxman",
                12,
                0.45,
                0.8,
                seed,
                attempts,
            ),
            python_dump(graph),
        )
        checks += 1

    print(f"generator parity: {checks}/{checks} exact cases PASS")
    return checks


if __name__ == "__main__":
    run()
