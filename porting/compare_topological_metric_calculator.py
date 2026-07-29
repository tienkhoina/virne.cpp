#!/usr/bin/env python3
"""Exact differential and same-workload timing gate for topology metrics."""

from __future__ import annotations

import argparse
import builtins
from dataclasses import dataclass
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Callable


EXPECTED_SOURCE_SHA256 = (
    "4bc760a97ab9b7a46ea115db981747a82fdbfed32096e4b327257270625bcf8d"
)
EXPECTED_NETWORKX_VERSION = "3.4.2"
EXPECTED_NUMPY_VERSION = "2.2.6"

MASK64 = (1 << 64) - 1
FNV_OFFSET = 1469598103934665603
FNV_PRIME = 1099511628211
CORPUS_LCG_MULTIPLIER = 6364136223846793005
CORPUS_LCG_INCREMENT = 1442695040888963407
CORPUS_SEED = 0x9E3779B97F4A7C15
CORPUS_INDEX_MIX = 0xD1B54A32D192ED03

DEGREE = 1 << 0
CLOSENESS = 1 << 1
EIGENVECTOR = 1 << 2
BETWEENNESS = 1 << 3
ALL_METRICS = DEGREE | CLOSENESS | EIGENVECTOR | BETWEENNESS

METRIC_FIELDS = (
    ("degree", "node_degree_centrality"),
    ("closeness", "node_closeness_centrality"),
    ("eigenvector", "node_eigenvector_centrality"),
    ("betweenness", "node_betweenness_centrality"),
)


@dataclass(frozen=True)
class DifferentialCase:
    name: str
    fixture: str
    metric_bits: int
    normalize: bool
    workers: int
    constructor_defaults: bool = False


@dataclass(frozen=True)
class CorpusCase:
    name: str
    case_index: int
    normalize: bool
    workers: int


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    node_count: int
    metric_bits: int


@dataclass(frozen=True)
class ColumnSnapshot:
    present: bool
    rows: int | None = None
    columns: int | None = None
    dtype: str | None = None
    bits: tuple[int, ...] = ()


@dataclass(frozen=True)
class CaseSnapshot:
    status: str
    error_kind: str | None
    degree: ColumnSnapshot | None = None
    closeness: ColumnSnapshot | None = None
    eigenvector: ColumnSnapshot | None = None
    betweenness: ColumnSnapshot | None = None


@dataclass(frozen=True)
class BenchmarkStats:
    median_ms: float
    mad_ms: float
    p95_ms: float
    checksum: int


DIFFERENTIAL_CASES = (
    DifferentialCase("empty_none", "empty", 0, True, 1),
    DifferentialCase("empty_degree", "empty", DEGREE, True, 1),
    DifferentialCase("empty_closeness", "empty", CLOSENESS, True, 4),
    DifferentialCase(
        "empty_betweenness", "empty", BETWEENNESS, True, 4
    ),
    DifferentialCase(
        "empty_eigenvector_error", "empty", EIGENVECTOR, True, 1
    ),
    DifferentialCase("empty_all_error", "empty", ALL_METRICS, True, 4),
    DifferentialCase(
        "singleton_all_raw", "singleton", ALL_METRICS, False, 1
    ),
    DifferentialCase(
        "singleton_all_normalized", "singleton", ALL_METRICS, True, 8
    ),
    DifferentialCase(
        "branch_degree_normalized", "branch", DEGREE, True, 1
    ),
    DifferentialCase("branch_all_raw", "branch", ALL_METRICS, False, 1),
    DifferentialCase(
        "branch_all_normalized_w1", "branch", ALL_METRICS, True, 1
    ),
    DifferentialCase(
        "branch_all_normalized_w2", "branch", ALL_METRICS, True, 2
    ),
    DifferentialCase(
        "branch_all_normalized_w4", "branch", ALL_METRICS, True, 4
    ),
    DifferentialCase(
        "branch_all_normalized_w8", "branch", ALL_METRICS, True, 8
    ),
    DifferentialCase(
        "disconnected_dcb_raw",
        "disconnected",
        DEGREE | CLOSENESS | BETWEENNESS,
        False,
        4,
    ),
    DifferentialCase(
        "disconnected_dcb_normalized",
        "disconnected",
        DEGREE | CLOSENESS | BETWEENNESS,
        True,
        8,
    ),
    DifferentialCase(
        "self_loop_all_raw", "self_loop", ALL_METRICS, False, 4
    ),
    DifferentialCase(
        "self_loop_all_normalized", "self_loop", ALL_METRICS, True, 8
    ),
    DifferentialCase(
        "cycle_chord_all_raw", "cycle_chord", ALL_METRICS, False, 1
    ),
    DifferentialCase(
        "cycle_chord_all_normalized",
        "cycle_chord",
        ALL_METRICS,
        True,
        4,
    ),
    DifferentialCase(
        "complete_all_normalized", "complete", ALL_METRICS, True, 8
    ),
    DifferentialCase(
        "path20_eigenvector_error", "path20", EIGENVECTOR, False, 1
    ),
    DifferentialCase("path20_none", "path20", 0, True, 8),
    DifferentialCase(
        "ordered_all_raw_w1", "ordered", ALL_METRICS, False, 1
    ),
    DifferentialCase(
        "ordered_all_raw_w8", "ordered", ALL_METRICS, False, 8
    ),
    DifferentialCase(
        "ordered_cb_normalized_w2",
        "ordered",
        CLOSENESS | BETWEENNESS,
        True,
        2,
    ),
    DifferentialCase(
        "ordered_cb_normalized_w8",
        "ordered",
        CLOSENESS | BETWEENNESS,
        True,
        8,
    ),
    DifferentialCase(
        "branch_constructor_defaults",
        "branch",
        DEGREE,
        True,
        0,
        constructor_defaults=True,
    ),
)


CORPUS_CASES = tuple(
    CorpusCase(
        name=f"generated_{case_index}",
        case_index=case_index,
        normalize=bool(case_index & 1),
        workers=(1, 2, 4, 8)[case_index % 4],
    )
    for case_index in range(64)
)


BENCHMARK_CASES = (
    BenchmarkCase("degree_20000", 20000, DEGREE),
    BenchmarkCase("eigenvector_4000", 4000, EIGENVECTOR),
    BenchmarkCase("closeness_500", 500, CLOSENESS),
    BenchmarkCase("betweenness_240", 240, BETWEENNESS),
    BenchmarkCase("all_metrics_240", 240, ALL_METRICS),
)


def torch_is_loaded() -> bool:
    return any(
        name == "torch" or name.startswith("torch.") for name in sys.modules
    )


def load_original(source: Path):
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "Python source changed: "
            f"expected {EXPECTED_SOURCE_SHA256}, got {digest}"
        )
    if torch_is_loaded():
        raise RuntimeError("torch was loaded before the non-ML oracle")

    module_name = "virne_original_topological_metric_calculator"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Python oracle from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals_value: dict[str, Any] | None = None,
        locals_value: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ):
        if name == "torch" or name.startswith("torch."):
            raise RuntimeError("non-ML oracle attempted to import torch")
        return original_import(
            name, globals_value, locals_value, fromlist, level
        )

    builtins.__import__ = guarded_import
    try:
        spec.loader.exec_module(module)
    finally:
        builtins.__import__ = original_import

    if torch_is_loaded():
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
    return completed.stdout.splitlines()


def parse_key_values(lines: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in lines:
        if not line or "=" not in line:
            raise RuntimeError(f"invalid C++ protocol line: {line!r}")
        key, value = line.split("=", 1)
        if not key or key in values:
            raise RuntimeError(f"duplicate/empty C++ protocol key: {key!r}")
        values[key] = value
    return values


def required(values: dict[str, str], key: str) -> str:
    try:
        return values[key]
    except KeyError as error:
        raise RuntimeError(f"C++ protocol omitted {key}") from error


def make_graph(nx, fixture: str):
    graph = nx.Graph()
    if fixture == "empty":
        return graph

    node_counts = {
        "singleton": 1,
        "branch": 6,
        "disconnected": 8,
        "self_loop": 5,
        "cycle_chord": 7,
        "complete": 6,
        "path20": 20,
        "ordered": 9,
    }
    try:
        node_count = node_counts[fixture]
    except KeyError as error:
        raise ValueError(f"unknown fixture: {fixture}") from error
    graph.add_nodes_from(range(node_count))

    edges: tuple[tuple[int, int], ...]
    if fixture == "singleton":
        edges = ()
    elif fixture == "branch":
        edges = ((0, 1), (1, 2), (2, 3), (2, 4), (4, 5))
    elif fixture == "disconnected":
        edges = ((0, 1), (1, 2), (3, 4), (4, 5), (5, 3))
    elif fixture == "self_loop":
        edges = ((0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 1))
    elif fixture == "cycle_chord":
        edges = (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 0),
            (0, 3),
            (2, 5),
        )
    elif fixture == "complete":
        edges = tuple(
            (left, right)
            for left in range(node_count)
            for right in range(left + 1, node_count)
        )
    elif fixture == "path20":
        edges = tuple((node - 1, node) for node in range(1, node_count))
    elif fixture == "ordered":
        edges = (
            (4, 1),
            (4, 7),
            (1, 0),
            (7, 8),
            (1, 2),
            (7, 6),
            (2, 5),
            (6, 3),
            (5, 3),
            (0, 8),
            (2, 6),
            (1, 7),
        )
    else:
        raise AssertionError(f"unhandled fixture: {fixture}")
    graph.add_edges_from(edges)
    return graph


def corpus_random(state: int) -> int:
    """Mirror unsigned 64-bit overflow in the harness LCG exactly."""
    return (
        state * CORPUS_LCG_MULTIPLIER + CORPUS_LCG_INCREMENT
    ) & MASK64


def corpus_shuffle(edges: list[tuple[int, int]], state: int) -> int:
    """Mirror the harness's in-place Fisher-Yates shuffle exactly."""
    for remaining in range(len(edges), 1, -1):
        state = corpus_random(state)
        swap_index = state % remaining
        edges[remaining - 1], edges[swap_index] = (
            edges[swap_index],
            edges[remaining - 1],
        )
    return state


def make_corpus_graph(nx, case_index: int):
    """Recreate make_corpus_graph from the C++ differential harness."""
    node_count = 3 + case_index % 19
    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    state = CORPUS_SEED ^ ((case_index * CORPUS_INDEX_MIX) & MASK64)

    ring = [
        (node, (node + 1) % node_count) for node in range(node_count)
    ]
    state = corpus_shuffle(ring, state)
    for left, right in ring:
        if not graph.has_edge(left, right):
            graph.add_edge(left, right)

    include_self_loops = case_index % 5 == 0
    candidates = [
        (left, right)
        for left in range(node_count)
        for right in range(left if include_self_loops else left + 1, node_count)
    ]
    corpus_shuffle(candidates, state)
    extra_target = node_count + (case_index * 5) % (node_count * 2 + 1)
    added = 0
    for left, right in candidates:
        if added == extra_target:
            break
        if not graph.has_edge(left, right):
            graph.add_edge(left, right)
            added += 1
    return graph


def make_benchmark_graph(nx, node_count: int):
    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    for node in range(node_count):
        for offset in (1, 7, 31, 127, 509):
            neighbor = (node + offset) % node_count
            if neighbor != node and not graph.has_edge(node, neighbor):
                graph.add_edge(node, neighbor)
    return graph


def metric_kwargs(metric_bits: int, normalize: bool) -> dict[str, bool]:
    return {
        "degree": bool(metric_bits & DEGREE),
        "closeness": bool(metric_bits & CLOSENESS),
        "eigenvector": bool(metric_bits & EIGENVECTOR),
        "betweenness": bool(metric_bits & BETWEENNESS),
        "normalize": normalize,
    }


def python_column_snapshot(np, value: Any, node_count: int) -> ColumnSnapshot:
    if value is None:
        return ColumnSnapshot(present=False)
    if not isinstance(value, np.ndarray):
        raise AssertionError(f"Python metric is not ndarray: {type(value)!r}")
    expected_shape = (node_count, 1)
    if value.shape != expected_shape:
        raise AssertionError(
            f"Python metric shape {value.shape}, expected {expected_shape}"
        )
    if value.dtype != np.dtype(np.float32):
        raise AssertionError(
            f"Python metric dtype {value.dtype}, expected float32"
        )
    payload = value.reshape(-1).view(np.uint32)
    return ColumnSnapshot(
        present=True,
        rows=node_count,
        columns=1,
        dtype="float32",
        bits=tuple(int(item) for item in payload),
    )


def python_graph_snapshot(
    oracle,
    nx,
    np,
    graph,
    name: str,
    metric_bits: int,
    normalize: bool,
    constructor_defaults: bool = False,
) -> CaseSnapshot:
    try:
        if constructor_defaults:
            metrics = oracle.TopologicalMetricCalculator(graph).metrics
        else:
            metrics = oracle.TopologicalMetricCalculator.calculate(
                graph, **metric_kwargs(metric_bits, normalize)
            )
    except nx.NetworkXPointlessConcept:
        return CaseSnapshot(status="error", error_kind="null_graph")
    except nx.PowerIterationFailedConvergence:
        return CaseSnapshot(status="error", error_kind="nonconvergence")
    except Exception as error:
        raise RuntimeError(
            f"unexpected Python error in {name}: "
            f"{type(error).__name__}: {error}"
        ) from error

    columns = {
        short_name: python_column_snapshot(
            np, getattr(metrics, attribute), graph.number_of_nodes()
        )
        for short_name, attribute in METRIC_FIELDS
    }
    return CaseSnapshot(
        status="ok",
        error_kind=None,
        degree=columns["degree"],
        closeness=columns["closeness"],
        eigenvector=columns["eigenvector"],
        betweenness=columns["betweenness"],
    )


def python_case_snapshot(oracle, nx, np, item: DifferentialCase) -> CaseSnapshot:
    return python_graph_snapshot(
        oracle,
        nx,
        np,
        make_graph(nx, item.fixture),
        item.name,
        item.metric_bits,
        item.normalize,
        item.constructor_defaults,
    )


def python_corpus_snapshot(oracle, nx, np, item: CorpusCase) -> CaseSnapshot:
    return python_graph_snapshot(
        oracle,
        nx,
        np,
        make_corpus_graph(nx, item.case_index),
        item.name,
        ALL_METRICS,
        item.normalize,
    )


def cpp_column_snapshot(
    values: dict[str, str], prefix: str
) -> ColumnSnapshot:
    present_text = required(values, prefix + ".present")
    if present_text not in ("0", "1"):
        raise RuntimeError(f"{prefix}.present is not 0/1: {present_text!r}")
    if present_text == "0":
        for suffix in (".rows", ".columns", ".bits"):
            if prefix + suffix in values:
                raise RuntimeError(
                    f"absent C++ column emitted {prefix + suffix}"
                )
        return ColumnSnapshot(present=False)

    rows = int(required(values, prefix + ".rows"))
    columns = int(required(values, prefix + ".columns"))
    bits_text = required(values, prefix + ".bits")
    bits = tuple(int(value) for value in bits_text.split(",")) if bits_text else ()
    if len(bits) != rows:
        raise RuntimeError(
            f"{prefix}: rows={rows}, but {len(bits)} float payloads emitted"
        )
    if any(value < 0 or value > 0xFFFFFFFF for value in bits):
        raise RuntimeError(f"{prefix}: uint32 payload out of range")
    return ColumnSnapshot(
        present=True,
        rows=rows,
        columns=columns,
        dtype="float32",
        bits=bits,
    )


def cpp_case_snapshot(values: dict[str, str], prefix: str) -> CaseSnapshot:
    status = required(values, prefix + ".status")
    if status == "error":
        return CaseSnapshot(
            status="error",
            error_kind=required(values, prefix + ".error_kind"),
        )
    if status == "ok":
        return CaseSnapshot(
            status="ok",
            error_kind=None,
            degree=cpp_column_snapshot(values, prefix + ".degree"),
            closeness=cpp_column_snapshot(values, prefix + ".closeness"),
            eigenvector=cpp_column_snapshot(values, prefix + ".eigenvector"),
            betweenness=cpp_column_snapshot(values, prefix + ".betweenness"),
        )
    raise RuntimeError(f"{prefix}.status is invalid: {status!r}")


def parse_cpp_differential(
    lines: list[str],
    cases: tuple[DifferentialCase, ...],
    corpus_cases: tuple[CorpusCase, ...],
) -> tuple[tuple[CaseSnapshot, ...], tuple[CaseSnapshot, ...]]:
    values = parse_key_values(lines)
    count = int(required(values, "diff_count"))
    if count != len(cases):
        raise AssertionError(
            f"C++ diff_count={count}, expected {len(cases)}"
        )

    snapshots: list[CaseSnapshot] = []
    for index, item in enumerate(cases):
        prefix = f"diff[{index}]"
        cpp_name = required(values, prefix + ".name")
        if cpp_name != item.name:
            raise AssertionError(
                f"case {index}: C++ name {cpp_name!r}, expected {item.name!r}"
            )
        snapshots.append(cpp_case_snapshot(values, prefix))

    corpus_count = int(required(values, "corpus_count"))
    if corpus_count != len(corpus_cases):
        raise AssertionError(
            f"C++ corpus_count={corpus_count}, expected {len(corpus_cases)}"
        )

    corpus_snapshots: list[CaseSnapshot] = []
    for index, item in enumerate(corpus_cases):
        prefix = f"corpus[{index}]"
        cpp_name = required(values, prefix + ".name")
        if cpp_name != item.name:
            raise AssertionError(
                f"corpus case {index}: C++ name {cpp_name!r}, "
                f"expected {item.name!r}"
            )
        cpp_normalize = required(values, prefix + ".normalize")
        expected_normalize = "1" if item.normalize else "0"
        if cpp_normalize != expected_normalize:
            raise AssertionError(
                f"{item.name}: C++ normalize={cpp_normalize!r}, "
                f"expected {expected_normalize!r}"
            )
        cpp_workers = required(values, prefix + ".workers")
        if cpp_workers != str(item.workers):
            raise AssertionError(
                f"{item.name}: C++ workers={cpp_workers!r}, "
                f"expected {item.workers}"
            )
        corpus_snapshots.append(cpp_case_snapshot(values, prefix))

    return tuple(snapshots), tuple(corpus_snapshots)


def require_differential(
    cpp: tuple[CaseSnapshot, ...],
    python: tuple[CaseSnapshot, ...],
    cases: tuple[DifferentialCase, ...] | tuple[CorpusCase, ...],
) -> None:
    differences: list[str] = []
    for item, cpp_case, python_case in zip(cases, cpp, python, strict=True):
        if cpp_case.status != python_case.status:
            differences.append(
                f"{item.name}.status: C++={cpp_case.status!r}, "
                f"Python={python_case.status!r}"
            )
            continue
        if cpp_case.error_kind != python_case.error_kind:
            differences.append(
                f"{item.name}.error_kind: C++={cpp_case.error_kind!r}, "
                f"Python={python_case.error_kind!r}"
            )
        if cpp_case.status != "ok":
            continue
        for short_name, _attribute in METRIC_FIELDS:
            cpp_column = getattr(cpp_case, short_name)
            python_column = getattr(python_case, short_name)
            if cpp_column != python_column:
                differences.append(
                    f"{item.name}.{short_name}:\n"
                    f"    C++={cpp_column!r}\n"
                    f"    Python={python_column!r}"
                )
    if differences:
        raise AssertionError(
            "topological metric differential mismatch:\n  "
            + "\n  ".join(differences)
        )


def hash_u64(hash_value: int, value: int) -> int:
    value &= MASK64
    for shift in range(0, 64, 8):
        hash_value ^= (value >> shift) & 0xFF
        hash_value = (hash_value * FNV_PRIME) & MASK64
    return hash_value


def float32_bits(np, value: Any) -> int:
    scalar = np.asarray(value, dtype=np.float32).reshape(1)
    return int(scalar.view(np.uint32)[0])


def hash_column(hash_value: int, np, column: Any) -> int:
    hash_value = hash_u64(hash_value, 0 if column is None else 1)
    if column is None:
        return hash_value
    if not isinstance(column, np.ndarray):
        raise AssertionError(f"checksum expected ndarray, got {type(column)!r}")
    if column.ndim != 2 or column.shape[1] != 1:
        raise AssertionError(f"checksum expected (N,1), got {column.shape}")
    if column.dtype != np.dtype(np.float32):
        raise AssertionError(f"checksum expected float32, got {column.dtype}")
    hash_value = hash_u64(hash_value, column.shape[0])
    for value in column.reshape(-1):
        hash_value = hash_u64(hash_value, float32_bits(np, value))
    return hash_value


def metrics_checksum(np, metrics: Any) -> int:
    hash_value = FNV_OFFSET
    for _short_name, attribute in METRIC_FIELDS:
        hash_value = hash_column(
            hash_value, np, getattr(metrics, attribute)
        )
    return hash_value


def median(values: list[int]) -> float:
    return float(statistics.median(values))


def percentile95(values: list[int]) -> float:
    ordered = sorted(values)
    return float(ordered[math.ceil(0.95 * len(ordered)) - 1])


def sample_stats_ms(samples_ns: list[int], checksum: int) -> BenchmarkStats:
    sample_median = median(samples_ns)
    deviations = [abs(value - sample_median) for value in samples_ns]
    return BenchmarkStats(
        median_ms=sample_median / 1e6,
        mad_ms=median(deviations) / 1e6,
        p95_ms=percentile95(samples_ns) / 1e6,
        checksum=checksum,
    )


def measure_python(
    action: Callable[[], Any],
    checksum_function: Callable[[Any], int],
    warmups: int,
    repetitions: int,
) -> BenchmarkStats:
    samples: list[int] = []
    expected_checksum: int | None = None
    previous_result: Any = None
    for sample in range(warmups + repetitions):
        # Drop the preceding result before the timer, matching the C++ scope
        # where result destruction occurs after the recorded stop time.
        previous_result = None
        start = time.perf_counter_ns()
        result = action()
        stop = time.perf_counter_ns()
        current_checksum = checksum_function(result)
        if expected_checksum is None:
            expected_checksum = current_checksum
        elif current_checksum != expected_checksum:
            raise AssertionError(
                "Python benchmark result changed between repetitions: "
                f"{expected_checksum} != {current_checksum}"
            )
        if sample >= warmups:
            samples.append(stop - start)
        previous_result = result
        del result

    if expected_checksum is None:
        raise AssertionError("benchmark produced no checksum")
    return sample_stats_ms(samples, expected_checksum)


def parse_cpp_benchmark(
    lines: list[str],
    cases: tuple[BenchmarkCase, ...],
    expected_workers: int,
    expected_warmups: int,
    expected_repetitions: int,
) -> tuple[dict[str, BenchmarkStats], dict[str, tuple[int, int]]]:
    values = parse_key_values(lines)
    count = int(required(values, "bench_count"))
    if count != len(cases):
        raise AssertionError(
            f"C++ bench_count={count}, expected {len(cases)}"
        )
    metadata = {
        "workers": int(required(values, "bench_workers")),
        "warmups": int(required(values, "bench_warmups")),
        "repetitions": int(required(values, "bench_repetitions")),
    }
    expected_metadata = {
        "workers": expected_workers,
        "warmups": expected_warmups,
        "repetitions": expected_repetitions,
    }
    if metadata != expected_metadata:
        raise AssertionError(
            f"C++ benchmark metadata {metadata}, expected {expected_metadata}"
        )

    rows: dict[str, BenchmarkStats] = {}
    shapes: dict[str, tuple[int, int]] = {}
    for index, item in enumerate(cases):
        prefix = f"bench[{index}]"
        name = required(values, prefix + ".name")
        if name != item.name:
            raise AssertionError(
                f"benchmark {index}: C++ name {name!r}, expected {item.name!r}"
            )
        shapes[name] = (
            int(required(values, prefix + ".nodes")),
            int(required(values, prefix + ".edges")),
        )
        rows[name] = BenchmarkStats(
            median_ms=float(required(values, prefix + ".cpp_median_ms")),
            mad_ms=float(required(values, prefix + ".cpp_mad_ms")),
            p95_ms=float(required(values, prefix + ".cpp_p95_ms")),
            checksum=int(required(values, prefix + ".checksum")),
        )
    return rows, shapes


def python_benchmark(
    oracle,
    nx,
    np,
    cases: tuple[BenchmarkCase, ...],
    warmups: int,
    repetitions: int,
) -> tuple[dict[str, BenchmarkStats], dict[str, tuple[int, int]]]:
    # Construct every canonical graph before any timing begins.
    graphs = {
        item.name: make_benchmark_graph(nx, item.node_count) for item in cases
    }
    shapes = {
        name: (graph.number_of_nodes(), graph.number_of_edges())
        for name, graph in graphs.items()
    }
    rows: dict[str, BenchmarkStats] = {}
    for item in cases:
        graph = graphs[item.name]
        kwargs = metric_kwargs(item.metric_bits, True)
        rows[item.name] = measure_python(
            lambda graph=graph, kwargs=kwargs: (
                oracle.TopologicalMetricCalculator.calculate(graph, **kwargs)
            ),
            lambda metrics: metrics_checksum(np, metrics),
            warmups,
            repetitions,
        )
    return rows, shapes


def report_benchmark(
    cpp_rows: dict[str, BenchmarkStats],
    python_rows: dict[str, BenchmarkStats],
    cases: tuple[BenchmarkCase, ...],
    workers: int,
) -> None:
    print(
        "operation             workers | Python median/MAD/p95 ms | "
        "C++ median/MAD/p95 ms | speedup"
    )
    print("-" * 101)
    slower: list[str] = []
    for item in cases:
        cpp = cpp_rows[item.name]
        python = python_rows[item.name]
        if cpp.checksum != python.checksum:
            raise AssertionError(
                f"{item.name}: FNV checksum mismatch "
                f"C++={cpp.checksum}, Python={python.checksum}"
            )
        speedup = python.median_ms / cpp.median_ms
        print(
            f"{item.name:21} {workers:7d} | "
            f"{python.median_ms:9.3f}/{python.mad_ms:7.3f}/"
            f"{python.p95_ms:7.3f} | "
            f"{cpp.median_ms:9.3f}/{cpp.mad_ms:7.3f}/"
            f"{cpp.p95_ms:7.3f} | {speedup:7.2f}x"
        )
        if cpp.median_ms >= python.median_ms:
            slower.append(item.name)
    if slower:
        raise AssertionError(
            "C++ median must beat Python for every canonical benchmark: "
            + ", ".join(slower)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact Python/C++ gate for TopologicalMetricCalculator"
    )
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
    if (
        len(DIFFERENTIAL_CASES) != 28
        or len(CORPUS_CASES) != 64
        or len(BENCHMARK_CASES) != 5
    ):
        raise AssertionError("canonical case table was changed")

    import networkx as nx
    import numpy as np

    if nx.__version__ != EXPECTED_NETWORKX_VERSION:
        raise RuntimeError(
            f"NetworkX {EXPECTED_NETWORKX_VERSION} required, got {nx.__version__}"
        )
    if np.__version__ != EXPECTED_NUMPY_VERSION:
        raise RuntimeError(
            f"NumPy {EXPECTED_NUMPY_VERSION} required, got {np.__version__}"
        )
    if torch_is_loaded():
        raise RuntimeError("NetworkX/NumPy unexpectedly imported torch")

    oracle = load_original(args.python_source)
    python_differential = tuple(
        python_case_snapshot(oracle, nx, np, item)
        for item in DIFFERENTIAL_CASES
    )
    python_corpus = tuple(
        python_corpus_snapshot(oracle, nx, np, item) for item in CORPUS_CASES
    )
    cpp_differential, cpp_corpus = parse_cpp_differential(
        run_cpp([str(args.cpp), "--differential"]),
        DIFFERENTIAL_CASES,
        CORPUS_CASES,
    )
    require_differential(
        cpp_differential, python_differential, DIFFERENTIAL_CASES
    )
    require_differential(cpp_corpus, python_corpus, CORPUS_CASES)
    exact_case_count = len(DIFFERENTIAL_CASES) + len(CORPUS_CASES)
    print(f"differential: PASS ({exact_case_count} exact cases)")

    cpp_rows, cpp_shapes = parse_cpp_benchmark(
        run_cpp(
            [
                str(args.cpp),
                "--benchmark",
                "--workers",
                str(args.workers),
                "--warmups",
                str(args.warmups),
                "--repetitions",
                str(args.repetitions),
            ]
        ),
        BENCHMARK_CASES,
        args.workers,
        args.warmups,
        args.repetitions,
    )
    python_rows, python_shapes = python_benchmark(
        oracle,
        nx,
        np,
        BENCHMARK_CASES,
        args.warmups,
        args.repetitions,
    )
    if cpp_shapes != python_shapes:
        raise AssertionError(
            f"benchmark graph recipe mismatch: C++={cpp_shapes}, "
            f"Python={python_shapes}"
        )

    print(
        f"runtime: Python {platform.python_version()}, "
        f"NetworkX {nx.__version__}, NumPy {np.__version__}, "
        f"CPUs visible {os.cpu_count()}"
    )
    print(
        f"protocol: warmups={args.warmups}, repetitions={args.repetitions}, "
        f"C++ workers={'auto' if args.workers == 0 else args.workers}; "
        "graph construction, checksums and process startup excluded"
    )
    report_benchmark(cpp_rows, python_rows, BENCHMARK_CASES, args.workers)
    print("performance: PASS (C++ median beats Python for all 5 cases)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"compare_topological_metric_calculator: FAIL: {error}",
            file=sys.stderr,
        )
        raise
