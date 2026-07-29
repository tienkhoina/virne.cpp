#!/usr/bin/env python3
"""Differential and same-workload timing gate for virne.utils.network."""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
from pathlib import Path
import platform
import statistics
import struct
import subprocess
import sys
import time
from typing import Any, Callable


MASK64 = (1 << 64) - 1
FNV_OFFSET = 1469598103934665603
FNV_PRIME = 1099511628211


def load_original(source: Path):
    module_name = "virne_original_utils_network"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Python oracle from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if "torch" in sys.modules:
        raise RuntimeError("non-ML oracle unexpectedly imported torch")
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


def hex_text(value: str) -> str:
    return value.encode("utf-8").hex()


def encode_path(links: list[tuple[int, int]]) -> str:
    return ";".join(f"{u},{v}" for u, v in links)


def encode_levels(levels: list[list[int]]) -> str:
    return "|".join(",".join(str(node) for node in level) for level in levels)


def encode_scalar(value: Any) -> str:
    if value is None:
        return "n"
    if isinstance(value, bool):
        return "b1" if value else "b0"
    if isinstance(value, int):
        return f"i{value}"
    if isinstance(value, float):
        return f"f{format(value, '.17g')}"
    if isinstance(value, str):
        return f"s{hex_text(value)}"
    raise TypeError(f"parity encoder expected scalar, got {type(value)!r}")


def encode_flat(values: list[Any]) -> str:
    return ";".join(encode_scalar(value) for value in values)


def encode_dicts(dicts: list[dict[str, Any]]) -> str:
    encoded_dicts = []
    for entry in dicts:
        encoded_dicts.append(
            ";".join(
                f"{hex_text(str(key))}={encode_scalar(value)}"
                for key, value in entry.items()
            )
        )
    return "|".join(encoded_dicts)


def catches(function: Callable[[], Any]) -> str:
    try:
        function()
    except Exception:
        return "1"
    return "0"


def python_parity(oracle) -> dict[str, str]:
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(7))
    graph.add_edges_from(
        [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5)]
    )
    sources = [0, 1, 2, 6]
    sequential = [oracle.get_bfs_tree_level(graph, source) for source in sources]

    digraph = nx.DiGraph()
    digraph.add_nodes_from(range(4))
    digraph.add_edges_from([(0, 1), (2, 1), (1, 3)])

    nested = {
        "numbers": [1, 2.5],
        "nested": {"name": "end", "flag": True, "negative": -4},
    }
    numeric_collision = {True: "truth"}
    numeric_collision[1] = "integer-last"
    gml_input = [
        {
            "label": "cpu",
            "active": True,
            "none": None,
            "ratio": 1.5,
            "items": [1, "x"],
            "quote": ["a'b"],
            "control": ["a\0"],
            "fixed6": 1e6,
            "fixed15": 1e15,
            "scientific16": 1e16,
            "fixed_negative4": 1e-4,
            "scientific_negative5": 1e-5,
            "negative_zero": -0.0,
            "positive_inf": math.inf,
            "negative_inf": -math.inf,
            "nan": math.nan,
            "denorm_min": 5e-324,
            "min_normal": 2.2250738585072014e-308,
            "max": 1.7976931348623157e308,
            "roundtrip": 1.2345678901234567,
            "both_quotes": ["a'\"b\\c\n"],
            7: "numeric-key",
            1: "first-collision",
            "1": "last-collision",
        },
        numeric_collision,
    ]
    settings = [
        {"name": "cpu", "low": " -7 ", "high": "+12"},
        {"name": "bw", "low": 3.8, "high": True},
        {"name": "ram", "low": "1_000", "high": "2_000"},
    ]
    settings_identity = oracle.sanitize_attr_setting(settings) is settings

    float_corpus = []
    float_bits = 0x9E3779B97F4A7C15
    for _ in range(512):
        float_bits = (
            float_bits * 6364136223846793005 + 1442695040888963407
        ) & MASK64
        value = struct.unpack("<d", float_bits.to_bytes(8, "little"))[0]
        float_corpus.append({"value": value})

    return {
        "path": encode_path(oracle.path_to_links([4, 1, 7, 9])),
        "bfs": encode_levels(oracle.get_bfs_tree_level(graph, 0)),
        "bfs_isolated": encode_levels(oracle.get_bfs_tree_level(graph, 6)),
        "bfs_parallel_same": "1" if sequential == sequential else "0",
        "digraph_bfs": encode_levels(oracle.get_bfs_tree_level(digraph, 2)),
        "flatten": encode_flat(list(oracle.flatten_recurrent_dict(nested))),
        "flatten_empty": encode_flat(list(oracle.flatten_recurrent_dict({}))),
        "gml": encode_dicts(oracle.flatten_dict_list_for_gml(gml_input)),
        "gml_float_corpus": encode_dicts(
            oracle.flatten_dict_list_for_gml(float_corpus)
        ),
        "sanitize_identity": "1" if settings_identity else "0",
        "sanitize": encode_dicts(settings),
        "path_short_error": catches(lambda: oracle.path_to_links([1])),
        "bfs_missing_error": catches(
            lambda: oracle.get_bfs_tree_level(graph, 99)
        ),
        "flatten_none_error": catches(
            lambda: list(oracle.flatten_recurrent_dict([1, None]))
        ),
        "sanitize_invalid_error": catches(
            lambda: oracle.sanitize_attr_setting([{"low": "1.5"}])
        ),
    }


def parse_parity(lines: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in lines:
        kind, key, value = line.split("\t", 2)
        if kind != "PARITY":
            raise RuntimeError(f"unexpected C++ parity line: {line}")
        result[key] = value
    return result


def require_parity(cpp: dict[str, str], python: dict[str, str]) -> None:
    if cpp.keys() != python.keys():
        missing = sorted(python.keys() - cpp.keys())
        extra = sorted(cpp.keys() - python.keys())
        raise AssertionError(f"parity key mismatch: missing={missing}, extra={extra}")
    differences = [
        key for key in cpp if cpp[key] != python[key]
    ]
    if differences:
        details = "\n".join(
            f"  {key}:\n    C++={cpp[key]!r}\n    Python={python[key]!r}"
            for key in differences
        )
        raise AssertionError(f"differential mismatch:\n{details}")


def checksum_path(links: list[tuple[int, int]]) -> int:
    checksum = 0
    for u, v in links:
        checksum = (checksum + (u + 1) * 17 + (v + 1) * 31) & MASK64
    return checksum


def checksum_bfs(batches: list[list[list[int]]]) -> int:
    checksum = 0
    for batch_index, levels in enumerate(batches):
        for depth, level in enumerate(levels):
            for vertex in level:
                checksum = (
                    checksum
                    + (batch_index + 1) * 1000003
                    + (depth + 1) * 1009
                    + vertex
                ) & MASK64
    return checksum


def checksum_flat(values: list[int]) -> int:
    checksum = 0
    for index, value in enumerate(values):
        checksum = (checksum + (index + 1) * (value + 1)) & MASK64
    return checksum


def fnv_add(hash_value: int, value: str) -> int:
    for byte in value.encode("utf-8"):
        hash_value ^= byte
        hash_value = (hash_value * FNV_PRIME) & MASK64
    hash_value ^= 0xFF
    return (hash_value * FNV_PRIME) & MASK64


def checksum_dicts(dicts: list[dict[str, str]]) -> int:
    hash_value = FNV_OFFSET
    for entry in dicts:
        for key, value in entry.items():
            hash_value = fnv_add(hash_value, key)
            hash_value = fnv_add(hash_value, value)
    return hash_value


def checksum_sanitized(dicts: list[dict[str, Any]]) -> int:
    checksum = 0
    for index, entry in enumerate(dicts):
        for key, value in entry.items():
            if key in ("low", "high"):
                checksum = (
                    checksum + (index + 1) * (value + 100000)
                ) & MASK64
    return checksum


def measure(
    warmups: int,
    repetitions: int,
    prepare: Callable[[], Any],
    action: Callable[[Any], Any],
    checksum: Callable[[Any], int],
) -> tuple[list[int], int]:
    samples: list[int] = []
    last_checksum = 0
    for sample in range(warmups + repetitions):
        prepared = prepare()
        start = time.perf_counter_ns()
        result = action(prepared)
        stop = time.perf_counter_ns()
        last_checksum = checksum(result)
        if sample >= warmups:
            samples.append(stop - start)
    return samples, last_checksum


def parse_cpp_benchmark(
    lines: list[str],
) -> tuple[dict[str, int], dict[str, tuple[int, list[int], int]]]:
    metadata: dict[str, int] = {}
    rows: dict[str, tuple[int, list[int], int]] = {}
    for line in lines:
        parts = line.split("\t")
        if parts[0] == "META" and len(parts) == 3:
            metadata[parts[1]] = int(parts[2])
        elif parts[0] == "BENCH" and len(parts) == 5:
            rows[parts[1]] = (
                int(parts[2]),
                [int(value) for value in parts[3].split(",")],
                int(parts[4]),
            )
        else:
            raise RuntimeError(f"unexpected C++ benchmark line: {line}")
    return metadata, rows


def python_benchmark(
    oracle,
    fixture: Path,
    metadata: dict[str, int],
    warmups: int,
    repetitions: int,
) -> dict[str, tuple[list[int], int]]:
    import networkx as nx

    path = list(range(metadata["path_nodes"]))
    graph = nx.read_gml(fixture, label="id")
    sources = [
        (index * 11) % graph.number_of_nodes()
        for index in range(metadata["bfs_sources"])
    ]
    full_sources = list(graph.nodes)
    flatten_input = [
        {
            "values": [
                block + offset for offset in range(metadata["flatten_width"])
            ]
            + [block]
        }
        for block in range(metadata["flatten_blocks"])
    ]
    gml_input = [
        {
            "name": "cpu",
            "index": index,
            "active": index % 2 == 0,
            "ratio": 1.5,
            "none": None,
            "items": [1, "x"],
        }
        for index in range(metadata["gml_dicts"])
    ]
    sanitize_base = [
        {
            "name": "resource",
            "low": str(index % 97),
            "high": str(100 + index % 193),
        }
        for index in range(metadata["sanitize_dicts"])
    ]

    no_prepare = lambda: None
    return {
        "path_to_links.py": measure(
            warmups,
            repetitions,
            no_prepare,
            lambda _: oracle.path_to_links(path),
            checksum_path,
        ),
        "get_bfs_tree_level.py": measure(
            warmups,
            repetitions,
            no_prepare,
            lambda _: [
                oracle.get_bfs_tree_level(graph, source) for source in sources
            ],
            checksum_bfs,
        ),
        "get_bfs_tree_level_full.py": measure(
            warmups,
            repetitions,
            no_prepare,
            lambda _: [
                oracle.get_bfs_tree_level(graph, source)
                for source in full_sources
            ],
            checksum_bfs,
        ),
        "flatten_recurrent_dict.py": measure(
            warmups,
            repetitions,
            no_prepare,
            lambda _: list(oracle.flatten_recurrent_dict(flatten_input)),
            checksum_flat,
        ),
        "flatten_dict_list_for_gml.py": measure(
            warmups,
            repetitions,
            no_prepare,
            lambda _: oracle.flatten_dict_list_for_gml(gml_input),
            checksum_dicts,
        ),
        "sanitize_attr_setting.py": measure(
            warmups,
            repetitions,
            lambda: [entry.copy() for entry in sanitize_base],
            lambda prepared: oracle.sanitize_attr_setting(prepared),
            checksum_sanitized,
        ),
    }


def percentile95(samples: list[int]) -> int:
    ordered = sorted(samples)
    return ordered[math.ceil(len(ordered) * 0.95) - 1]


def stats(samples: list[int]) -> tuple[float, float, float]:
    median = float(statistics.median(samples))
    mad = float(statistics.median(abs(value - median) for value in samples))
    return median, mad, float(percentile95(samples))


def python_row_for(cpp_name: str) -> str:
    return cpp_name.rsplit(".", 1)[0] + ".py"


def report_benchmark(
    cpp_rows: dict[str, tuple[int, list[int], int]],
    python_rows: dict[str, tuple[list[int], int]],
    repetitions: int,
) -> None:
    print(
        "operation                            workers   Python ms      C++ ms"
        "    speedup   C++ MAD   C++ p95"
    )
    print("-" * 94)
    fastest: dict[str, float] = {}
    python_medians: dict[str, float] = {}
    for name, (workers, cpp_samples, cpp_checksum) in cpp_rows.items():
        if len(cpp_samples) != repetitions:
            raise AssertionError(
                f"{name}: expected {repetitions} C++ samples, got {len(cpp_samples)}"
            )
        python_name = python_row_for(name)
        if python_name not in python_rows:
            raise AssertionError(f"{name}: no Python timing row {python_name}")
        python_samples, python_checksum = python_rows[python_name]
        if cpp_checksum != python_checksum:
            raise AssertionError(
                f"{name}: benchmark checksum mismatch "
                f"C++={cpp_checksum}, Python={python_checksum}"
            )
        cpp_median, cpp_mad, cpp_p95 = stats(cpp_samples)
        python_median, _, _ = stats(python_samples)
        speedup = python_median / cpp_median
        base_name = name.rsplit(".", 1)[0]
        fastest[base_name] = min(fastest.get(base_name, math.inf), cpp_median)
        python_medians[base_name] = python_median
        print(
            f"{name:36} {workers:7d} "
            f"{python_median / 1e6:11.3f} {cpp_median / 1e6:11.3f} "
            f"{speedup:9.2f}x {cpp_mad / 1e6:9.3f} {cpp_p95 / 1e6:9.3f}"
        )

    regressions = [
        name
        for name, cpp_median in fastest.items()
        if cpp_median >= python_medians[name]
    ]
    if regressions:
        raise AssertionError(
            "no C++ variant beat Python median for: " + ", ".join(regressions)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp", required=True, type=Path)
    parser.add_argument("--python-source", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=31)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.cpp, args.python_source, args.fixture):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.warmups < 1 or args.repetitions < 1 or args.workers < 1:
        raise ValueError("warmups, repetitions and workers must be positive")

    oracle = load_original(args.python_source)
    cpp_parity = parse_parity(run_cpp([str(args.cpp), "--parity"]))
    expected_parity = python_parity(oracle)
    require_parity(cpp_parity, expected_parity)
    print(f"differential: PASS ({len(cpp_parity)} cases)")

    cpp_lines = run_cpp(
        [
            str(args.cpp),
            "--benchmark",
            str(args.fixture),
            str(args.warmups),
            str(args.repetitions),
            str(args.workers),
        ]
    )
    metadata, cpp_rows = parse_cpp_benchmark(cpp_lines)
    python_rows = python_benchmark(
        oracle,
        args.fixture,
        metadata,
        args.warmups,
        args.repetitions,
    )

    print(
        f"runtime: Python {platform.python_version()}, "
        f"NetworkX {sys.modules['networkx'].__version__}, "
        f"CPUs visible {os.cpu_count()}"
    )
    print(
        f"protocol: warmups={args.warmups}, repetitions={args.repetitions}, "
        f"C++ workers={args.workers}; fixture loading and checksums excluded"
    )
    report_benchmark(cpp_rows, python_rows, args.repetitions)
    print("performance: PASS (every Python function has a faster C++ variant)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"compare_utils_network: FAIL: {error}", file=sys.stderr)
        raise
