#!/usr/bin/env python3
"""Exact differential and same-workload timing gate for topology_generator."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import platform
import random
import statistics
import struct
import subprocess
import sys
import time
from typing import Any, Callable


EXPECTED_SOURCE_SHA256 = (
    "f69d3c45f288b6194db2945cfa33d2cb4a32032af8130ce4e7a53ab2dd697722"
)
MASK64 = (1 << 64) - 1
FNV_OFFSET = 1469598103934665603
FNV_PRIME = 1099511628211
POS_FIELD = "pos"


@dataclass(frozen=True)
class OracleRequest:
    topology_type: str
    num_nodes: int
    seed: int
    kwargs: dict[str, Any] = field(default_factory=dict)
    grid_columns: int | None = None


def load_original(source: Path):
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"Python source changed: expected {EXPECTED_SOURCE_SHA256}, got {digest}"
        )
    module_name = "virne_original_topology_generator"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Python oracle from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if "torch" in sys.modules:
        raise RuntimeError("non-ML topology oracle unexpectedly imported torch")
    return module


def run_cpp(command: list[str]) -> list[str]:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return [line for line in completed.stdout.splitlines() if line]


def bits_hex(value: float) -> str:
    return struct.pack(">d", float(value)).hex()


def normalized_node(node: Any, grid_columns: int | None) -> int:
    if grid_columns is None:
        return int(node)
    row, column = node
    return int(row) * grid_columns + int(column)


def encode_graph(graph, grid_columns: int | None = None) -> str:
    nodes = list(graph.nodes)
    node_text = ",".join(
        str(normalized_node(node, grid_columns)) for node in nodes
    )
    edge_text = ",".join(
        f"{normalized_node(u, grid_columns)}-{normalized_node(v, grid_columns)}"
        for u, v in graph.edges
    )
    adjacency = []
    for node in nodes:
        vertex = normalized_node(node, grid_columns)
        neighbors = ",".join(
            str(normalized_node(neighbor, grid_columns))
            for neighbor in graph.neighbors(node)
        )
        adjacency.append(f"{vertex}:{neighbors}")
    positions = []
    for node, attributes in graph.nodes(data=True):
        position = attributes.get(POS_FIELD)
        if position is not None:
            x, y = position
            positions.append(
                f"{normalized_node(node, grid_columns)}:{bits_hex(x)}:{bits_hex(y)}"
            )
    graph_attributes = len(graph.graph)
    node_attributes = sum(len(attributes) for _, attributes in graph.nodes(data=True))
    edge_attributes = sum(
        len(attributes) for _, _, attributes in graph.edges(data=True)
    )
    return (
        f"N:{node_text};E:{edge_text};Q:{'/'.join(adjacency)};"
        f"P:{','.join(positions)};"
        f"A:{graph_attributes},{node_attributes},{edge_attributes}"
    )


def encode_rng() -> str:
    return ",".join(f"{random.getrandbits(32):08x}" for _ in range(8))


def encode_graphs(
    graphs: list[Any],
    grid_columns: list[int | None] | None = None,
) -> str:
    if grid_columns is None:
        grid_columns = [None] * len(graphs)
    return "||".join(
        encode_graph(graph, columns)
        for graph, columns in zip(graphs, grid_columns, strict=True)
    )


def error_and_state(seed: int, action: Callable[[], Any]) -> str:
    random.seed(seed)
    try:
        action()
        prefix = "NO_ERROR"
    except Exception as error:
        prefix = f"ERROR:{error}"
    return f"{prefix}|R:{encode_rng()}"


def error_flag_and_state(seed: int, action: Callable[[], Any]) -> str:
    random.seed(seed)
    try:
        action()
        flag = "0"
    except Exception:
        flag = "1"
    return f"{flag}|R:{encode_rng()}"


def deterministic_case(
    generator,
    topology_type: str,
    num_nodes: int,
    grid_columns: int | None = None,
    **kwargs: Any,
) -> str:
    random.seed(91234567)
    graph = generator.generate(topology_type, num_nodes, **kwargs)
    return f"{encode_graph(graph, grid_columns)}|R:{encode_rng()}"


def stochastic_sequence(
    generator,
    topology_type: str,
    num_nodes: int,
    seed: int,
    count: int,
    **kwargs: Any,
) -> str:
    random.seed(seed)
    graphs = [
        generator.generate(topology_type, num_nodes, **kwargs)
        for _ in range(count)
    ]
    return f"{encode_graphs(graphs)}|R:{encode_rng()}"


def parity_batch_requests() -> list[OracleRequest]:
    return [
        OracleRequest("path", 7, 3),
        OracleRequest("star", 6, 5),
        OracleRequest(
            "grid_2d", 999, 7, {"m": 3, "n": 4}, grid_columns=4
        ),
        OracleRequest("random", 10, 0, {"random_prob": 0.23}),
        OracleRequest("random", 10, 2, {"random_prob": 0.23}),
        OracleRequest(
            "waxman", 12, 15, {"wm_alpha": 0.8, "wm_beta": 0.45}
        ),
        OracleRequest(
            "waxman", 12, 1, {"wm_alpha": 0.8, "wm_beta": 0.45}
        ),
    ]


def generate_seeded_batch(generator, requests: list[OracleRequest]):
    graphs = []
    columns: list[int | None] = []
    for request in requests:
        random.seed(request.seed)
        graphs.append(
            generator.generate(
                request.topology_type,
                request.num_nodes,
                **request.kwargs,
            )
        )
        columns.append(request.grid_columns)
    return graphs, columns


def python_parity(oracle) -> dict[str, str]:
    generator = oracle.TopologyGenerator
    result = {
        "path_1": deterministic_case(generator, "path", 1),
        "path_2": deterministic_case(generator, "path", 2),
        "path_7": deterministic_case(generator, "path", 7),
        "star_1": deterministic_case(generator, "star", 1),
        "star_2": deterministic_case(generator, "star", 2),
        "star_6": deterministic_case(generator, "star", 6),
        "grid_1x1": deterministic_case(
            generator, "grid_2d", 1, 1, m=1, n=1
        ),
        "grid_1x4": deterministic_case(
            generator, "grid_2d", 1, 4, m=1, n=4
        ),
        "grid_3x4": deterministic_case(
            generator, "grid_2d", 1, 4, m=3, n=4
        ),
        "grid_3x4_ignored_num_nodes": deterministic_case(
            generator, "grid_2d", 999, 4, m=3, n=4
        ),
        "grid_0x4": deterministic_case(
            generator, "grid_2d", 1, 4, m=0, n=4
        ),
        "grid_mapping_3x4": (
            "0,0,0;1,0,1;2,0,2;3,0,3;4,1,0;5,1,1;6,1,2;"
            "7,1,3;8,2,0;9,2,1;10,2,2;11,2,3"
        ),
        "error_num_nodes_zero": error_and_state(
            11, lambda: generator.generate("path", 0)
        ),
        "error_num_nodes_negative": error_and_state(
            11, lambda: generator.generate("path", -1)
        ),
        "error_unsupported": error_and_state(
            11, lambda: generator.generate("PATH", 3)
        ),
        "error_grid_missing_both": error_and_state(
            11, lambda: generator.generate("grid_2d", 1)
        ),
        "error_grid_missing_m": error_and_state(
            11, lambda: generator.generate("grid_2d", 1, n=4)
        ),
        "error_grid_missing_n": error_and_state(
            11, lambda: generator.generate("grid_2d", 1, m=3)
        ),
        "error_grid_negative": error_and_state(
            11, lambda: generator.generate("grid_2d", 1, m=-1, n=4)
        ),
        "random_default_seed0_two": stochastic_sequence(
            generator, "random", 10, 0, 2
        ),
        "random_retry_seed0": stochastic_sequence(
            generator, "random", 10, 0, 1, random_prob=0.23
        ),
        "random_retry_seed2": stochastic_sequence(
            generator, "random", 10, 2, 1, random_prob=0.23
        ),
        "random_n1_p0": stochastic_sequence(
            generator, "random", 1, 42, 1, random_prob=0.0
        ),
        "random_n8_p1": stochastic_sequence(
            generator, "random", 8, 42, 1, random_prob=1.0
        ),
        "waxman_default_seed0": stochastic_sequence(
            generator, "waxman", 50, 0, 1
        ),
        "waxman_retry_seed15": stochastic_sequence(
            generator,
            "waxman",
            12,
            15,
            1,
            wm_alpha=0.8,
            wm_beta=0.45,
        ),
        "waxman_retry_seed1": stochastic_sequence(
            generator,
            "waxman",
            12,
            1,
            1,
            wm_alpha=0.8,
            wm_beta=0.45,
        ),
        "waxman_sequence_seed42": stochastic_sequence(
            generator,
            "waxman",
            12,
            42,
            2,
            wm_alpha=0.8,
            wm_beta=0.45,
        ),
        "waxman_n1_error": error_flag_and_state(
            7, lambda: generator.generate("waxman", 1)
        ),
    }

    random.seed(4_294_967_303)
    graphs = [
        generator.generate("random", 10, random_prob=0.23)
        for _ in range(2)
    ]
    result["random_global_sequence"] = f"{encode_graphs(graphs)}|R:{encode_rng()}"

    random.seed(4_294_967_303)
    graphs = [
        generator.generate("waxman", 12, wm_alpha=0.8, wm_beta=0.45)
        for _ in range(2)
    ]
    result["waxman_global_sequence"] = f"{encode_graphs(graphs)}|R:{encode_rng()}"

    batch_graphs, batch_columns = generate_seeded_batch(
        generator, parity_batch_requests()
    )
    encoded_batch = encode_graphs(batch_graphs, batch_columns)
    result["batch_workers_1"] = encoded_batch
    result["batch_workers_8"] = encoded_batch
    return result


def parse_parity(lines: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in lines:
        kind, key, value = line.split("\t", 2)
        if kind != "PARITY" or key in result:
            raise RuntimeError(f"invalid C++ parity line: {line}")
        result[key] = value
    return result


def require_parity(cpp: dict[str, str], python: dict[str, str]) -> None:
    if cpp.keys() != python.keys():
        missing = sorted(python.keys() - cpp.keys())
        extra = sorted(cpp.keys() - python.keys())
        raise AssertionError(f"parity key mismatch: missing={missing}, extra={extra}")
    differences = [key for key in cpp if cpp[key] != python[key]]
    if differences:
        details = "\n".join(
            f"  {key}:\n    C++={cpp[key]!r}\n    Python={python[key]!r}"
            for key in differences
        )
        raise AssertionError(f"differential mismatch:\n{details}")


def fnv_add_u64(hash_value: int, value: int) -> int:
    value &= MASK64
    for shift in range(0, 64, 8):
        hash_value ^= (value >> shift) & 0xFF
        hash_value = (hash_value * FNV_PRIME) & MASK64
    return hash_value


def checksum_graph_into(
    hash_value: int,
    graph,
    grid_columns: int | None = None,
) -> int:
    nodes = list(graph.nodes)
    hash_value = fnv_add_u64(hash_value, len(nodes))
    hash_value = fnv_add_u64(hash_value, graph.number_of_edges())
    for node in nodes:
        vertex = normalized_node(node, grid_columns)
        attributes = graph.nodes[node]
        hash_value = fnv_add_u64(hash_value, vertex)
        hash_value = fnv_add_u64(hash_value, len(attributes))
        position = attributes.get(POS_FIELD)
        if position is not None:
            hash_value = fnv_add_u64(hash_value, 1)
            x, y = position
            hash_value = fnv_add_u64(
                hash_value, struct.unpack("<Q", struct.pack("<d", float(x)))[0]
            )
            hash_value = fnv_add_u64(
                hash_value, struct.unpack("<Q", struct.pack("<d", float(y)))[0]
            )
        else:
            hash_value = fnv_add_u64(hash_value, 0)
        neighbors = list(graph.neighbors(node))
        hash_value = fnv_add_u64(hash_value, len(neighbors))
        for neighbor in neighbors:
            hash_value = fnv_add_u64(
                hash_value, normalized_node(neighbor, grid_columns)
            )
    for u, v, attributes in graph.edges(data=True):
        hash_value = fnv_add_u64(hash_value, normalized_node(u, grid_columns))
        hash_value = fnv_add_u64(hash_value, normalized_node(v, grid_columns))
        hash_value = fnv_add_u64(hash_value, len(attributes))
    return fnv_add_u64(hash_value, len(graph.graph))


def checksum_graph(graph, grid_columns: int | None = None) -> int:
    return checksum_graph_into(FNV_OFFSET, graph, grid_columns)


def checksum_graphs(value: tuple[list[Any], list[int | None]]) -> int:
    graphs, columns = value
    hash_value = fnv_add_u64(FNV_OFFSET, len(graphs))
    for graph, grid_columns in zip(graphs, columns, strict=True):
        hash_value = checksum_graph_into(hash_value, graph, grid_columns)
    return hash_value


def measure(
    warmups: int,
    repetitions: int,
    prepare: Callable[[int], Any],
    action: Callable[[Any], Any],
    checksum: Callable[[Any], int],
) -> tuple[list[int], int]:
    samples: list[int] = []
    last_checksum = 0
    for sample in range(warmups + repetitions):
        prepared = prepare(sample)
        start = time.perf_counter_ns()
        result = action(prepared)
        stop = time.perf_counter_ns()
        last_checksum = checksum(result)
        if sample >= warmups:
            samples.append(stop - start)
    return samples, last_checksum


def seeded_prepare(seed_base: int) -> Callable[[int], None]:
    def prepare(sample: int) -> None:
        random.seed(seed_base + sample)
    return prepare


def repeated_requests(
    topology_type: str,
    count: int,
    num_nodes: int,
    seed_base: int,
    grid_columns: int | None = None,
    **kwargs: Any,
) -> list[OracleRequest]:
    return [
        OracleRequest(
            topology_type,
            num_nodes,
            seed_base + index * 104729,
            dict(kwargs),
            grid_columns,
        )
        for index in range(count)
    ]


def python_benchmark(
    oracle,
    metadata: dict[str, int],
    warmups: int,
    repetitions: int,
) -> dict[str, tuple[list[int], int]]:
    generator = oracle.TopologyGenerator
    no_prepare = lambda _sample: None

    def batch_row(requests: list[OracleRequest]):
        return measure(
            warmups,
            repetitions,
            no_prepare,
            lambda _: generate_seeded_batch(generator, requests),
            checksum_graphs,
        )

    path_requests = repeated_requests("path", 64, 2048, 3001)
    star_requests = repeated_requests("star", 64, 2048, 4001)
    grid_requests = repeated_requests(
        "grid_2d", 32, 1, 5001, m=64, n=64, grid_columns=64
    )
    random_requests = repeated_requests(
        "random", 32, 250, 6001, random_prob=0.04
    )
    waxman_requests = repeated_requests("waxman", 16, 250, 7001)

    return {
        "path.py": measure(
            warmups,
            repetitions,
            no_prepare,
            lambda _: generator.generate("path", metadata["path_nodes"]),
            checksum_graph,
        ),
        "star.py": measure(
            warmups,
            repetitions,
            no_prepare,
            lambda _: generator.generate("star", metadata["star_nodes"]),
            checksum_graph,
        ),
        "grid_2d.py": measure(
            warmups,
            repetitions,
            no_prepare,
            lambda _: generator.generate(
                "grid_2d",
                1,
                m=metadata["grid_rows"],
                n=metadata["grid_columns"],
            ),
            lambda graph: checksum_graph(graph, metadata["grid_columns"]),
        ),
        "random.py": measure(
            warmups,
            repetitions,
            seeded_prepare(1009),
            lambda _: generator.generate(
                "random", metadata["random_nodes"], random_prob=0.03
            ),
            checksum_graph,
        ),
        "waxman.py": measure(
            warmups,
            repetitions,
            seeded_prepare(2003),
            lambda _: generator.generate("waxman", metadata["waxman_nodes"]),
            checksum_graph,
        ),
        "batch_path.py": batch_row(path_requests),
        "batch_star.py": batch_row(star_requests),
        "batch_grid_2d.py": batch_row(grid_requests),
        "batch_random.py": batch_row(random_requests),
        "batch_waxman.py": batch_row(waxman_requests),
    }


def parse_cpp_benchmark(
    lines: list[str],
) -> tuple[dict[str, int], dict[str, tuple[int, list[int], int]]]:
    metadata: dict[str, int] = {}
    rows: dict[str, tuple[int, list[int], int]] = {}
    for line in lines:
        fields = line.split("\t")
        if fields[0] == "META" and len(fields) == 3:
            metadata[fields[1]] = int(fields[2])
        elif fields[0] == "BENCH" and len(fields) == 5:
            rows[fields[1]] = (
                int(fields[2]),
                [int(value) for value in fields[3].split(",")],
                int(fields[4]),
            )
        else:
            raise RuntimeError(f"unexpected C++ benchmark line: {line}")
    return metadata, rows


def percentile95(samples: list[int]) -> int:
    ordered = sorted(samples)
    return ordered[math.ceil(len(ordered) * 0.95) - 1]


def stats(samples: list[int]) -> tuple[float, float, float]:
    median = float(statistics.median(samples))
    mad = float(statistics.median(abs(value - median) for value in samples))
    return median, mad, float(percentile95(samples))


def python_name(cpp_name: str) -> str:
    if cpp_name.endswith(".st") or cpp_name.endswith(".mt"):
        return cpp_name[:-3] + ".py"
    if cpp_name.endswith(".cpp"):
        return cpp_name[:-4] + ".py"
    raise ValueError(f"unknown C++ row name: {cpp_name}")


def report_benchmark(
    cpp_rows: dict[str, tuple[int, list[int], int]],
    python_rows: dict[str, tuple[list[int], int]],
    repetitions: int,
) -> None:
    print(
        "operation                    workers   Python ms      C++ ms"
        "    speedup   C++ MAD   C++ p95"
    )
    print("-" * 86)
    fastest: dict[str, float] = {}
    python_medians: dict[str, float] = {}
    for name, (workers, cpp_samples, cpp_checksum) in cpp_rows.items():
        if len(cpp_samples) != repetitions:
            raise AssertionError(
                f"{name}: expected {repetitions} samples, got {len(cpp_samples)}"
            )
        py_name = python_name(name)
        python_samples, python_checksum = python_rows[py_name]
        if cpp_checksum != python_checksum:
            raise AssertionError(
                f"{name}: checksum mismatch C++={cpp_checksum}, "
                f"Python={python_checksum}"
            )
        cpp_median, cpp_mad, cpp_p95 = stats(cpp_samples)
        py_median, _, _ = stats(python_samples)
        operation = py_name[:-3]
        fastest[operation] = min(fastest.get(operation, math.inf), cpp_median)
        python_medians[operation] = py_median
        print(
            f"{name:28} {workers:7d} "
            f"{py_median / 1e6:11.3f} {cpp_median / 1e6:11.3f} "
            f"{py_median / cpp_median:9.2f}x "
            f"{cpp_mad / 1e6:9.3f} {cpp_p95 / 1e6:9.3f}"
        )
    regressions = [
        name for name, median in fastest.items()
        if median >= python_medians[name]
    ]
    if regressions:
        raise AssertionError(
            "no C++ variant beat Python median for: " + ", ".join(regressions)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp", required=True, type=Path)
    parser.add_argument("--python-source", required=True, type=Path)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=31)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.cpp, args.python_source):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.warmups < 1 or args.repetitions < 1 or args.workers < 0:
        raise ValueError(
            "warmups/repetitions must be positive and workers non-negative"
        )

    import networkx as nx

    if nx.__version__ != "3.4.2":
        raise RuntimeError(f"NetworkX 3.4.2 required, got {nx.__version__}")
    oracle = load_original(args.python_source)
    cpp_parity = parse_parity(run_cpp([str(args.cpp), "--parity"]))
    expected = python_parity(oracle)
    require_parity(cpp_parity, expected)
    print(f"differential: PASS ({len(cpp_parity)} exact cases)")

    cpp_lines = run_cpp(
        [
            str(args.cpp),
            "--benchmark",
            str(args.warmups),
            str(args.repetitions),
            str(args.workers),
        ]
    )
    metadata, cpp_rows = parse_cpp_benchmark(cpp_lines)
    python_rows = python_benchmark(
        oracle, metadata, args.warmups, args.repetitions
    )
    print(
        f"runtime: Python {platform.python_version()}, NetworkX {nx.__version__}, "
        f"CPUs visible {os.cpu_count()}"
    )
    print(
        f"protocol: warmups={args.warmups}, repetitions={args.repetitions}, "
        f"C++ batch workers={'auto' if args.workers == 0 else args.workers}; "
        "seeding, checksums, serialization, "
        "destruction and process startup excluded"
    )
    report_benchmark(cpp_rows, python_rows, args.repetitions)
    print("performance: PASS (fastest C++ variant beats Python for every row)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"compare_topology_generator: FAIL: {error}", file=sys.stderr)
        raise
