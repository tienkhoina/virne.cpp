#!/usr/bin/env python3
"""Compact checksum-gated greedy NodeMapper benchmark; not run on import."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import statistics
import subprocess
import time

import compare_node_mapper as oracle


FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


def fingerprint(value: str) -> tuple[int, int]:
    encoded = value.encode("utf-8")
    checksum = FNV_OFFSET
    for byte in encoded:
        checksum = ((checksum ^ byte) * FNV_PRIME) & MASK64
    return checksum, len(encoded)


def make_python_fixture(
    NodeMapper,
    virtual_node_count: int,
    low_candidate_count: int,
):
    virtual = oracle.FakeNetwork({
        node: {"cpu": 2} for node in range(virtual_node_count)
    })
    physical_node_count = low_candidate_count + virtual_node_count
    physical = oracle.FakeNetwork({
        node: {"cpu": 1 if node < low_candidate_count else 2}
        for node in range(physical_node_count)
    })
    cpu = oracle.FakeNodeResource("cpu")
    mapper = NodeMapper(
        oracle.FakeConstraintChecker([cpu]),
        oracle.FakeResourceUpdator(),
        [cpu],
        ["cpu"],
    )
    solution = oracle.make_solution()
    virtual_nodes = list(range(virtual_node_count))
    physical_nodes = list(range(physical_node_count))
    return mapper, virtual, physical, solution, virtual_nodes, physical_nodes


def run_python_once(
    NodeMapper,
    virtual_node_count: int,
    low_candidate_count: int,
) -> tuple[int, str]:
    # Fixture, input vectors, and dependency preparation are intentionally
    # outside the timed region. The original mapper still copies its candidate
    # vector and clears the two Solution tables inside node_mapping.
    mapper, virtual, physical, solution, virtual_nodes, physical_nodes = (
        make_python_fixture(
            NodeMapper, virtual_node_count, low_candidate_count
        )
    )
    begin = time.perf_counter_ns()
    mapped = mapper.node_mapping(
        virtual,
        physical,
        virtual_nodes,
        physical_nodes,
        solution,
        reusable=False,
        inplace=True,
        matching_mathod="greedy",
        if_allow_constraint_violation=False,
    )
    end = time.perf_counter_ns()
    if not mapped:
        raise RuntimeError("valid Python greedy mapping unexpectedly failed")
    payload = oracle.mapping_payload(mapped, physical, solution)
    return end - begin, payload


def parse_cpp(
    benchmark: pathlib.Path,
    virtual_node_count: int,
    low_candidate_count: int,
    candidate_workers: int,
) -> dict[str, int]:
    process = subprocess.run(
        [
            str(benchmark),
            str(virtual_node_count),
            str(low_candidate_count),
            str(candidate_workers),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"C++ NodeMapper benchmark failed: {process.stderr.strip()}"
        )
    fields: dict[str, str] = {}
    for line in process.stdout.splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in fields:
            raise RuntimeError(f"malformed C++ benchmark output: {line!r}")
        fields[key] = value
    required = {
        "protocol",
        "kind",
        "virtual_nodes",
        "low_candidates",
        "physical_nodes",
        "minimum_candidate_checks",
        "candidate_workers",
        "type_tag",
        "elapsed_ns",
        "checksum",
        "output_bytes",
        "entry_count",
        "status",
    }
    physical_nodes = low_candidate_count + virtual_node_count
    minimum_checks = virtual_node_count * (low_candidate_count + 1)
    if (
        set(fields) != required
        or fields["protocol"] != "1"
        or fields["kind"] != "node_mapper_greedy"
        or fields["type_tag"] != "ordered_numeric_mapping_v1"
        or fields["status"] != "PASS"
        or int(fields["virtual_nodes"]) != virtual_node_count
        or int(fields["low_candidates"]) != low_candidate_count
        or int(fields["physical_nodes"]) != physical_nodes
        or int(fields["minimum_candidate_checks"]) != minimum_checks
        or int(fields["candidate_workers"]) != candidate_workers
    ):
        raise RuntimeError(f"invalid C++ benchmark protocol: {fields!r}")
    return {
        "elapsed_ns": int(fields["elapsed_ns"]),
        "checksum": int(fields["checksum"]),
        "output_bytes": int(fields["output_bytes"]),
        "entry_count": int(fields["entry_count"]),
    }


def validate_output(
    current: dict[str, int],
    checksum: int,
    output_bytes: int,
    entry_count: int,
    label: str,
) -> None:
    if (
        current["checksum"] != checksum
        or current["output_bytes"] != output_bytes
        or current["entry_count"] != entry_count
    ):
        raise RuntimeError(f"{label} ordered numeric mapping output drift")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark", type=pathlib.Path, required=True)
    parser.add_argument("--virtual-nodes", type=int, default=32)
    parser.add_argument("--low-candidates", type=int, default=2048)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 8])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if (
        args.virtual_nodes <= 0
        or args.low_candidates <= 0
        or not args.workers
        or any(worker < 0 for worker in args.workers)
        or args.warmups < 0
        or args.repetitions <= 0
    ):
        raise RuntimeError("invalid NodeMapper benchmark protocol")

    NodeMapper = oracle.load_mapper(args.source)
    expected_entries = (
        args.low_candidates + args.virtual_nodes
        + 4 * args.virtual_nodes
    )
    _reference_ns, reference_payload = run_python_once(
        NodeMapper, args.virtual_nodes, args.low_candidates
    )
    expected_checksum, expected_bytes = fingerprint(reference_payload)

    # Exact ordered slots/info/offsets/violations, numeric lanes, result flags,
    # and every final physical capacity are gated before timing samples.
    for workers in args.workers:
        validate_output(
            parse_cpp(
                args.benchmark,
                args.virtual_nodes,
                args.low_candidates,
                workers,
            ),
            expected_checksum,
            expected_bytes,
            expected_entries,
            f"worker {workers} pre-timing gate",
        )

    for _ in range(args.warmups):
        _elapsed, payload = run_python_once(
            NodeMapper, args.virtual_nodes, args.low_candidates
        )
        if fingerprint(payload) != (expected_checksum, expected_bytes):
            raise RuntimeError("Python warm-up output drift")
    python_samples: list[int] = []
    for _ in range(args.repetitions):
        elapsed, payload = run_python_once(
            NodeMapper, args.virtual_nodes, args.low_candidates
        )
        if fingerprint(payload) != (expected_checksum, expected_bytes):
            raise RuntimeError("Python timed output drift")
        python_samples.append(elapsed)
    python_ns = int(statistics.median(python_samples))

    rows = []
    for workers in args.workers:
        for _ in range(args.warmups):
            validate_output(
                parse_cpp(
                    args.benchmark,
                    args.virtual_nodes,
                    args.low_candidates,
                    workers,
                ),
                expected_checksum,
                expected_bytes,
                expected_entries,
                f"worker {workers} warm-up",
            )
        cpp_samples: list[int] = []
        for _ in range(args.repetitions):
            current = parse_cpp(
                args.benchmark,
                args.virtual_nodes,
                args.low_candidates,
                workers,
            )
            validate_output(
                current,
                expected_checksum,
                expected_bytes,
                expected_entries,
                f"worker {workers} sample",
            )
            cpp_samples.append(current["elapsed_ns"])
        cpp_ns = int(statistics.median(cpp_samples))
        speedup = python_ns / cpp_ns
        if speedup <= 1.0:
            raise RuntimeError(
                f"worker {workers} C++ did not beat Python: {speedup:.3f}x"
            )
        row = {
            "kind": "node_mapper_greedy",
            "virtual_nodes": args.virtual_nodes,
            "low_candidates": args.low_candidates,
            "physical_nodes": args.low_candidates + args.virtual_nodes,
            "minimum_candidate_checks": (
                args.virtual_nodes * (args.low_candidates + 1)
            ),
            "candidate_workers": workers,
            "type_tag": "ordered_numeric_mapping_v1",
            "python_median_ms": python_ns / 1e6,
            "cpp_median_ms": cpp_ns / 1e6,
            "speedup": speedup,
            "checksum": expected_checksum,
            "output_bytes": expected_bytes,
            "entry_count": expected_entries,
        }
        rows.append(row)
        print(
            f"node_mapper_greedy/w{workers}: "
            f"python={row['python_median_ms']:.6f} ms, "
            f"cpp={row['cpp_median_ms']:.6f} ms, "
            f"speedup={speedup:.3f}x"
        )

    payload = {
        "source_sha256": oracle.SOURCE_SHA256,
        "benchmark_sha256": hashlib.sha256(
            args.benchmark.read_bytes()
        ).hexdigest(),
        "oracle_loader_sha256": hashlib.sha256(
            pathlib.Path(oracle.__file__).read_bytes()
        ).hexdigest(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "protocol": {
            "virtual_nodes": args.virtual_nodes,
            "low_candidates": args.low_candidates,
            "physical_nodes": args.low_candidates + args.virtual_nodes,
            "minimum_candidate_checks": (
                args.virtual_nodes * (args.low_candidates + 1)
            ),
            "workers": args.workers,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "matching_method": "greedy",
            "reusable": False,
            "inplace": True,
            "allow_constraint_violation": False,
            "fixture_and_preparation_excluded": True,
            "cpp_process_startup_excluded": True,
            "fingerprint_excluded": True,
            "ordered_numeric_output_gate_before_timing": True,
            "caller_configured_candidate_workers": True,
        },
        "rows": rows,
        "result": "PASS",
    }
    if args.output:
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"NodeMapper benchmark: PASS ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
