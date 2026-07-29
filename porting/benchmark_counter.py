#!/usr/bin/env python3
"""One checksum-gated Counter benchmark; do not broaden after acceptance.

The Python side imports only the existing AST leaf loader from
``compare_counter``. That loader hashes the exact Counter source, extracts the
single ``Counter`` class, and supplies narrow fake attribute/network objects.
The fakes intentionally implement only the constructor metadata and the two
resource-major row getters used by this benchmark; they are not substitutes
for BaseNetwork, graph, Solution, Pandas, or any system/solver component.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import statistics
import subprocess
import time

import compare_counter as oracle


SOURCE_SHA256 = "574745f86e99cd656cb0165330e3196b4f5ebf4eaa0687b076b8d9602db4d637"
WORKERS = (1, 2, 8)
WARMUPS = 1
REPETITIONS = 3
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


def fingerprint(value: str) -> tuple[int, int]:
    encoded = value.encode("utf-8")
    checksum = FNV_OFFSET
    for byte in encoded:
        checksum = ((checksum ^ byte) * FNV_PRIME) & MASK64
    return checksum, len(encoded)


def make_fixture(Counter, node_count: int):
    node_attributes = [
        oracle.FakeAttribute("node_integer"),
        oracle.FakeAttribute("node_floating"),
        oracle.FakeAttribute("node_boolean"),
    ]
    link_attributes = [
        oracle.FakeAttribute("link_integer"),
        oracle.FakeAttribute("link_floating"),
        oracle.FakeAttribute("link_boolean"),
    ]
    counter = oracle.make_counter(Counter, node_attributes, link_attributes)

    # Dict insertion order is the graph node/edge order consumed by the fake
    # row getters. Each getter emits one complete resource row at a time, so
    # np.array(rows).sum() observes resource-major C-order flattening.
    nodes = {
        index: {
            "node_integer": (index * 17) % 1009,
            "node_floating": float((index * 19) % 1013) + 0.25,
            "node_boolean": index % 3 == 0,
        }
        for index in range(node_count)
    }
    links = {
        (index, index + 1): {
            "link_integer": (index * 23) % 1019,
            "link_floating": float((index * 29) % 1021) + 0.125,
            "link_boolean": index % 5 == 0,
        }
        for index in range(node_count - 1)
    }
    return counter, oracle.FakeNetwork(nodes, links)


def result_payload(value) -> str:
    token = oracle.number_token(value)
    if not token.startswith("d:"):
        raise RuntimeError(f"mixed Counter result used the wrong lane: {token}")
    return "sum=" + token


def time_python(counter, network) -> tuple[int, str]:
    warm_value = None
    for _ in range(WARMUPS):
        warm_value = counter.calculate_sum_network_resource(
            network, node=True, link=True)
    assert warm_value is not None
    expected_payload = result_payload(warm_value)

    samples: list[int] = []
    for _ in range(REPETITIONS):
        started = time.perf_counter_ns()
        value = counter.calculate_sum_network_resource(
            network, node=True, link=True)
        stopped = time.perf_counter_ns()
        if result_payload(value) != expected_payload:
            raise RuntimeError("Python Counter output changed between samples")
        samples.append(stopped - started)
    return int(statistics.median(samples)), expected_payload


def parse_cpp(
    executable: pathlib.Path,
    node_count: int,
    workers: int,
) -> dict[str, int]:
    process = subprocess.run(
        [str(executable), str(node_count), str(workers)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"Counter benchmark failed at workers={workers}: "
            f"{process.stderr.strip()}")

    fields: dict[str, str] = {}
    for line in process.stdout.splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in fields:
            raise RuntimeError(f"malformed Counter benchmark line: {line!r}")
        fields[key] = value
    expected = {
        "protocol": "1",
        "kind": "counter_mixed_whole_network_sum",
        "semantics": "resource_major_numpy_flatten_v1",
        "node_count": str(node_count),
        "workers": str(workers),
        "type_tag": "counter_number_double_raw64",
        "status": "PASS",
    }
    for key, value in expected.items():
        if fields.get(key) != value:
            raise RuntimeError(
                f"Counter benchmark field {key}: "
                f"{fields.get(key)!r} != {value!r}")
    required_numeric = {"elapsed_ns", "checksum", "output_bytes", "entry_count"}
    if set(fields) != set(expected) | required_numeric:
        raise RuntimeError(
            f"Counter benchmark field inventory drift: {sorted(fields)}")
    return {key: int(fields[key]) for key in required_numeric}


def validate_native(
    result: dict[str, int],
    expected_checksum: int,
    expected_size: int,
    expected_entries: int,
    label: str,
) -> None:
    fingerprint_fields = {
        "checksum": expected_checksum,
        "output_bytes": expected_size,
        "entry_count": expected_entries,
    }
    mismatches = {
        key: {"expected": expected, "actual": result[key]}
        for key, expected in fingerprint_fields.items()
        if result[key] != expected
    }
    if mismatches:
        raise RuntimeError(
            f"{label} Counter type/bits/checksum mismatch: {mismatches}")


def time_cpp(
    executable: pathlib.Path,
    node_count: int,
    workers: int,
    expected_checksum: int,
    expected_size: int,
    expected_entries: int,
) -> int:
    for _ in range(WARMUPS):
        validate_native(
            parse_cpp(executable, node_count, workers),
            expected_checksum,
            expected_size,
            expected_entries,
            f"workers={workers} warm-up",
        )

    samples: list[int] = []
    for repetition in range(REPETITIONS):
        result = parse_cpp(executable, node_count, workers)
        validate_native(
            result,
            expected_checksum,
            expected_size,
            expected_entries,
            f"workers={workers} repetition={repetition}",
        )
        samples.append(result["elapsed_ns"])
    return int(statistics.median(samples))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark", type=pathlib.Path, required=True)
    parser.add_argument("--node-count", type=int, default=131072)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.node_count < 128:
        raise ValueError("node-count must be at least 128")
    if oracle.SOURCE_SHA256 != SOURCE_SHA256:
        raise RuntimeError("compare_counter oracle hash constant drift")

    Counter = oracle.load_counter(args.source)
    counter, network = make_fixture(Counter, args.node_count)
    python_ns, payload = time_python(counter, network)
    checksum, output_bytes = fingerprint(payload)
    entry_count = 3 * args.node_count + 3 * (args.node_count - 1)

    rows = []
    for workers in WORKERS:
        cpp_ns = time_cpp(
            args.benchmark,
            args.node_count,
            workers,
            checksum,
            output_bytes,
            entry_count,
        )
        speedup = python_ns / cpp_ns
        if speedup <= 1.0:
            raise RuntimeError(
                f"Counter C++ workers={workers} did not beat Python: "
                f"{speedup:.3f}x")
        rows.append({
            "workers": workers,
            "cpp_median_ns": cpp_ns,
            "speedup_vs_python": speedup,
        })

    report = {
        "component": "core.Counter",
        "case": "mixed_whole_network_resource_sum",
        "source_sha256": SOURCE_SHA256.upper(),
        "benchmark_sha256": hashlib.sha256(
            args.benchmark.read_bytes()).hexdigest().upper(),
        "oracle": {
            "loader": "exact single-class AST isolation",
            "fake_boundary": (
                "FakeAttribute plus FakeNetwork resource-major row getters only; "
                "no BaseNetwork, graph, Solution, Pandas, solver, or system emulation"
            ),
            "numpy_version": oracle.np.__version__,
        },
        "protocol": {
            "node_count": args.node_count,
            "entry_count": entry_count,
            "workers": list(WORKERS),
            "warmups": WARMUPS,
            "repetitions": REPETITIONS,
            "fixture_and_prepare_excluded": True,
            "process_startup_excluded": True,
            "fingerprint_excluded": True,
            "native_timed_loop_uses_only_prepared_ids": True,
            "result_gate": "CounterNumber lane plus raw64 bits/FNV/output bytes",
        },
        "python_median_ns": python_ns,
        "fingerprint": {
            "checksum": checksum,
            "output_bytes": output_bytes,
        },
        "cpp": rows,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "status": "PASS",
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
