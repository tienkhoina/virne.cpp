#!/usr/bin/env python3
"""Compact Python-vs-C++ benchmark for the frozen OrderRank solve pipeline."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import networkx as nx


EXPECTED_SOURCE_SHA256 = (
    "44A6F63F1A1798935453C057C6981A692F5DC4F03BDE6A6535607AECBDC61389"
)
VIRTUAL_NODE_COUNT = 16
PHYSICAL_NODE_COUNT = 32
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
U64_MASK = (1 << 64) - 1


def virtual_edges() -> list[tuple[int, int]]:
    return [
        (0, 2),
        (0, 5),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
        (5, 7),
        (5, 10),
        (6, 8),
        (7, 9),
        (8, 10),
        (9, 11),
        (10, 12),
        (10, 15),
        (11, 13),
        (12, 14),
        (13, 15),
    ]


def physical_edges() -> list[tuple[int, int]]:
    return [(source, source + 1) for source in range(PHYSICAL_NODE_COUNT - 1)]


class BenchmarkNetwork(nx.Graph):
    def __init__(
        self,
        node_count: int,
        ordered_links: list[tuple[int, int]],
    ) -> None:
        super().__init__()
        self.add_nodes_from(range(node_count))
        self.add_edges_from(ordered_links)
        self._ordered_links = tuple(ordered_links)

    @property
    def links(self) -> tuple[tuple[int, int], ...]:
        return self._ordered_links


class PythonSolution(dict[str, bool]):
    def __init__(self) -> None:
        super().__init__(result=False, place_result=True, route_result=True)
        self.node_slots: dict[int, int] = {}
        self.link_paths: dict[tuple[int, int], list[tuple[int, int]]] = {}

    @classmethod
    def from_v_net(cls, _virtual_network: BenchmarkNetwork) -> "PythonSolution":
        return cls()


class PythonSolverBase:
    def __init__(
        self,
        controller: Any,
        recorder: Any,
        counter: Any,
        logger: Any,
        config: Any,
        **_kwargs: Any,
    ) -> None:
        self.controller = controller
        self.recorder = recorder
        self.counter = counter
        self.logger = logger
        self.config = config


class OrderNodeRank:
    def __call__(self, network: BenchmarkNetwork) -> dict[int, int]:
        return {node_id: node_id for node_id in network.nodes}


class NoConstraintNodeMapper:
    def node_mapping(
        self,
        _virtual_network: BenchmarkNetwork,
        _physical_network: BenchmarkNetwork,
        *,
        sorted_v_nodes: list[int],
        sorted_p_nodes: list[int],
        solution: PythonSolution,
        reusable: bool,
        inplace: bool,
        matching_mathod: str,
    ) -> bool:
        del inplace, matching_mathod
        if not reusable and len(sorted_p_nodes) < len(sorted_v_nodes):
            return False
        for index, virtual_node in enumerate(sorted_v_nodes):
            physical_index = 0 if reusable else index
            solution.node_slots[virtual_node] = sorted_p_nodes[physical_index]
        return True


class NetworkXShortestLinkMapper:
    def link_mapping(
        self,
        _virtual_network: BenchmarkNetwork,
        physical_network: BenchmarkNetwork,
        *,
        solution: PythonSolution,
        sorted_v_links: tuple[tuple[int, int], ...],
        shortest_method: str,
        k: int,
        inplace: bool,
    ) -> bool:
        del shortest_method, k, inplace
        try:
            for virtual_link in sorted_v_links:
                source = solution.node_slots[virtual_link[0]]
                target = solution.node_slots[virtual_link[1]]
                path = nx.shortest_path(physical_network, source, target)
                solution.link_paths[virtual_link] = list(zip(path, path[1:]))
        except nx.NetworkXNoPath:
            return False
        return True


class NoConstraintController:
    def __init__(self) -> None:
        self.node_mapper = NoConstraintNodeMapper()
        self.link_mapper = NetworkXShortestLinkMapper()


def load_exact_order_rank_solver(source_path: Path) -> type:
    source_bytes = source_path.read_bytes()
    actual_hash = hashlib.sha256(source_bytes).hexdigest().upper()
    if actual_hash != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "refusing to benchmark changed Python target: "
            f"expected {EXPECTED_SOURCE_SHA256}, got {actual_hash}"
        )

    syntax_tree = ast.parse(source_bytes, filename=str(source_path))
    target_names = {"BaseNodeRankSolver", "OrderRankSolver"}
    selected_classes: list[ast.ClassDef] = []
    for statement in syntax_tree.body:
        if isinstance(statement, ast.ClassDef) and statement.name in target_names:
            statement.decorator_list = []
            selected_classes.append(statement)
    if [node.name for node in selected_classes] != [
        "BaseNodeRankSolver",
        "OrderRankSolver",
    ]:
        raise RuntimeError("could not isolate the exact OrderRank class pair")

    isolated_module = ast.fix_missing_locations(
        ast.Module(body=selected_classes, type_ignores=[])
    )
    namespace: dict[str, Any] = {
        "__name__": "_virne_order_rank_benchmark_target",
        "Solver": PythonSolverBase,
        "Solution": PythonSolution,
        "Controller": object,
        "Recorder": object,
        "Counter": object,
        "Logger": object,
        "VirtualNetwork": BenchmarkNetwork,
        "PhysicalNetwork": BenchmarkNetwork,
        "OrderNodeRank": OrderNodeRank,
    }
    exec(compile(isolated_module, str(source_path), "exec"), namespace)
    return namespace["OrderRankSolver"]


def prepare_fixture() -> tuple[
    BenchmarkNetwork,
    BenchmarkNetwork,
    list[tuple[int, int]],
]:
    expected_links = virtual_edges()
    virtual_network = BenchmarkNetwork(VIRTUAL_NODE_COUNT, expected_links)
    physical_network = BenchmarkNetwork(PHYSICAL_NODE_COUNT, physical_edges())
    return virtual_network, physical_network, expected_links


def validate_solution(
    solution: PythonSolution,
    expected_links: list[tuple[int, int]],
) -> None:
    if not (
        solution["result"]
        and solution["place_result"]
        and solution["route_result"]
    ):
        raise RuntimeError("Python OrderRank did not produce a successful solution")

    expected_slots = [(node_id, node_id) for node_id in range(VIRTUAL_NODE_COUNT)]
    if list(solution.node_slots.items()) != expected_slots:
        raise RuntimeError("Python node slots are not in exact OrderRank order")

    if list(solution.link_paths) != expected_links:
        raise RuntimeError("Python routed links are not in exact graph order")
    for link in expected_links:
        expected_path = [
            (source, source + 1) for source in range(link[0], link[1])
        ]
        if solution.link_paths[link] != expected_path:
            raise RuntimeError("Python shortest-path edge order differs")


class HashWriter:
    def __init__(self) -> None:
        self.checksum = FNV_OFFSET
        self.byte_count = 0

    def byte(self, value: int) -> None:
        self.checksum ^= value & 0xFF
        self.checksum = (self.checksum * FNV_PRIME) & U64_MASK
        self.byte_count += 1

    def boolean(self, value: bool) -> None:
        self.byte(1 if value else 0)

    def u64(self, value: int) -> None:
        value &= U64_MASK
        for shift in range(0, 64, 8):
            self.byte(value >> shift)

    def i64(self, value: int) -> None:
        self.u64(value)


def append_solution(writer: HashWriter, solution: PythonSolution) -> None:
    writer.boolean(solution["result"])
    writer.boolean(solution["place_result"])
    writer.boolean(solution["route_result"])

    writer.u64(len(solution.node_slots))
    for virtual_node, physical_node in solution.node_slots.items():
        writer.i64(virtual_node)
        writer.i64(physical_node)

    writer.u64(len(solution.link_paths))
    for virtual_link, path in solution.link_paths.items():
        writer.i64(virtual_link[0])
        writer.i64(virtual_link[1])
        writer.u64(len(path))
        for edge in path:
            writer.i64(edge[0])
            writer.i64(edge[1])


def gate_outputs(
    outputs: list[PythonSolution],
    expected_links: list[tuple[int, int]],
) -> dict[str, int]:
    writer = HashWriter()
    writer.u64(len(outputs))
    for output in outputs:
        validate_solution(output, expected_links)
        append_solution(writer, output)
    return {
        "entries": len(outputs),
        "bytes": writer.byte_count,
        "checksum": writer.checksum,
    }


def solve_batch(
    order_solver: Any,
    instance: dict[str, BenchmarkNetwork],
    repetitions: int,
    measure: bool,
) -> tuple[int, list[PythonSolution]]:
    elapsed_ns = 0
    outputs: list[PythonSolution] = []
    for _ in range(repetitions):
        if not measure:
            outputs.append(order_solver.solve(instance))
            continue
        begin = time.perf_counter_ns()
        solution = order_solver.solve(instance)
        elapsed_ns += time.perf_counter_ns() - begin
        outputs.append(solution)
    return elapsed_ns, outputs


def upper_median(values: list[int]) -> int:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def benchmark_python(
    source_path: Path,
    warmups: int,
    samples: int,
    repetitions: int,
) -> dict[str, Any]:
    order_rank_type = load_exact_order_rank_solver(source_path)
    virtual_network, physical_network, expected_links = prepare_fixture()
    fixture_nodes = (
        tuple(virtual_network.nodes),
        tuple(physical_network.nodes),
    )
    fixture_edges = (
        tuple(virtual_network.links),
        tuple(physical_network.links),
    )

    order_solver = order_rank_type(
        NoConstraintController(),
        object(),
        object(),
        object(),
        {},
        matching_mathod="greedy",
        shortest_method="bfs_shortest",
        k_shortest=1,
    )
    instance = {"v_net": virtual_network, "p_net": physical_network}

    for _ in range(warmups):
        _, outputs = solve_batch(order_solver, instance, repetitions, False)
        gate_outputs(outputs, expected_links)

    sample_times: list[int] = []
    reference_gate: dict[str, int] | None = None
    for _ in range(samples):
        elapsed_ns, outputs = solve_batch(order_solver, instance, repetitions, True)
        gate = gate_outputs(outputs, expected_links)
        if reference_gate is not None and gate != reference_gate:
            raise RuntimeError("Python output gate changed between samples")
        reference_gate = gate
        sample_times.append(elapsed_ns)

    if (
        fixture_nodes
        != (tuple(virtual_network.nodes), tuple(physical_network.nodes))
        or fixture_edges
        != (tuple(virtual_network.links), tuple(physical_network.links))
    ):
        raise RuntimeError("prepared Python fixture was mutated")
    if reference_gate is None:
        raise RuntimeError("no Python sample was measured")
    return {
        **reference_gate,
        "median_ns": upper_median(sample_times),
        "samples_ns": sample_times,
    }


def run_native(
    executable: Path,
    workers: int,
    warmups: int,
    samples: int,
    repetitions: int,
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            str(executable),
            str(workers),
            str(warmups),
            str(samples),
            str(repetitions),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in reversed(completed.stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict) and result.get("workers") == workers:
            return result
    raise RuntimeError(
        f"native worker-{workers} run did not emit its benchmark JSON"
    )


def positive_integer(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def nonnegative_integer(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path, help="native benchmark executable")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=positive_integer,
        nargs="+",
        default=[1, 2, 8],
        help="caller-selected native worker widths (default: 1 2 8)",
    )
    parser.add_argument("--warmups", type=nonnegative_integer, default=1)
    parser.add_argument("--samples", type=positive_integer, default=3)
    parser.add_argument("--repetitions", type=positive_integer, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    python_result = benchmark_python(
        args.source,
        args.warmups,
        args.samples,
        args.repetitions,
    )
    expected_gate = {
        key: python_result[key] for key in ("entries", "bytes", "checksum")
    }

    native_results: list[dict[str, Any]] = []
    for workers in args.workers:
        native = run_native(
            args.benchmark,
            workers,
            args.warmups,
            args.samples,
            args.repetitions,
        )
        native_gate = {key: native.get(key) for key in expected_gate}
        if native_gate != expected_gate:
            raise RuntimeError(
                f"worker-{workers} differential gate failed: "
                f"Python={expected_gate}, C++={native_gate}"
            )
        native_results.append(native)

    report_native = []
    for native in native_results:
        median_ns = int(native["median_ns"])
        report_native.append(
            {
                **native,
                "ns_per_solve": median_ns / args.repetitions,
                "speedup_vs_python": python_result["median_ns"] / median_ns,
            }
        )

    report = {
        "component": "heuristic_node_rank_order_solve",
        "status": "PASS",
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "fixture": {
            "virtual_nodes": VIRTUAL_NODE_COUNT,
            "virtual_links": len(virtual_edges()),
            "physical_nodes": PHYSICAL_NODE_COUNT,
            "physical_links": len(physical_edges()),
            "native_hard_constraints": 2,
            "native_resource_updates": 0,
            "python_dependency_model": (
                "AST-isolated target with lightweight deterministic "
                "rank/mapper doubles"
            ),
            "matching": "greedy",
            "shortest_path": "bfs_shortest",
            "warmups": args.warmups,
            "samples": args.samples,
            "repetitions": args.repetitions,
        },
        "output_gate": expected_gate,
        "python_sequential": {
            **python_result,
            "ns_per_solve": python_result["median_ns"] / args.repetitions,
        },
        "cpp": report_native,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
