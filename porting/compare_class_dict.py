#!/usr/bin/env python3
"""Exact differential and runtime gate for the ClassDict C++ port.

The Python oracle is loaded directly from its source file.  Importing the
``virne`` package would pull unrelated (including ML) dependencies into this
small component gate, so both the file digest and the direct-load boundary are
intentional parts of the protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import struct
import subprocess
import sys
import time
import types
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


EXPECTED_SOURCE_SHA256 = (
    "19637BBC0D4EFF9F240C5BE6B799B84AF96C7423E8C32AD97626468E3E3DE8FE"
)
MASK64 = (1 << 64) - 1
MIX_CONSTANT = 0x9E3779B97F4A7C15


@dataclass(frozen=True)
class DifferentialRecord:
    name: str
    payload: str
    facts: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    items: int
    fields: int


@dataclass(frozen=True)
class BenchmarkStats:
    median_ms: float
    mad_ms: float
    p95_ms: float
    checksum: int


BENCHMARK_CASES = (
    BenchmarkCase("string_get", 200_000, 256),
    BenchmarkCase("id_get", 200_000, 256),
    BenchmarkCase("resolved_reference_get", 200_000, 256),
    BenchmarkCase("resolved_reference_set", 200_000, 256),
    BenchmarkCase("from_dict", 256, 128),
    BenchmarkCase("to_dict", 128, 128),
    BenchmarkCase("batch_from_dict", 512, 64),
    BenchmarkCase("batch_to_dict", 512, 64),
)


def load_original(path: Path) -> types.ModuleType:
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest().upper()
    if digest != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"Python source SHA-256 {digest}, expected "
            f"{EXPECTED_SOURCE_SHA256}"
        )

    # Execute exactly this file in an isolated module.  Do not use a package
    # import: virne.__init__ has dependencies outside this component contract.
    module = types.ModuleType("_virne_class_dict_oracle")
    module.__file__ = str(path)
    module.__package__ = ""
    code = compile(source, str(path), "exec")
    exec(code, module.__dict__)
    if not hasattr(module, "ClassDict"):
        raise RuntimeError("direct-loaded source does not define ClassDict")
    return module


def utf8_size(value: str) -> int:
    return len(value.encode("utf-8"))


class CanonicalSerializer:
    """Mirror the tagged traversal format in class_dict_harness.cpp."""

    def __init__(self) -> None:
        self._identities: dict[int, int] = {}

    def value(self, item: Any) -> str:
        if item is None:
            return "n"
        if isinstance(item, bool):
            return "b1" if item else "b0"
        if isinstance(item, int):
            return "i" + str(item)
        if isinstance(item, float):
            bits = struct.unpack(">Q", struct.pack(">d", item))[0]
            return f"f{bits:016x}"
        if isinstance(item, str):
            return f"s{utf8_size(item)}:{item}"
        if isinstance(item, list):
            return self._list_pointer(item)
        if isinstance(item, Mapping):
            return self._mapping_pointer(item)
        raise TypeError(f"unsupported canonical Python type: {type(item)!r}")

    def class_dict(self, item: Any) -> str:
        return self._top_mapping(vars(item))

    def snapshot(self, item: Mapping[str, Any]) -> str:
        return self._top_mapping(item)

    def _top_mapping(self, item: Mapping[str, Any]) -> str:
        output = [f"d{len(item)}{{"]
        for key, value in item.items():
            if not isinstance(key, str):
                raise TypeError(f"ClassDict field name is not str: {key!r}")
            output.append(f"k{utf8_size(key)}:{key}={self.value(value)};")
        output.append("}")
        return "".join(output)

    def _identity(self, item: object) -> tuple[int, bool]:
        address = id(item)
        found = self._identities.get(address)
        if found is not None:
            return found, False
        identity = len(self._identities)
        self._identities[address] = identity
        return identity, True

    def _list_pointer(self, item: list[Any]) -> str:
        identity, first_visit = self._identity(item)
        if not first_visit:
            return "r" + str(identity)
        payload = "".join(self.value(value) + ";" for value in item)
        return f"L{identity}{len(item)}[{payload}]"

    def _mapping_pointer(self, item: Mapping[Any, Any]) -> str:
        identity, first_visit = self._identity(item)
        if not first_visit:
            return "r" + str(identity)
        order_tag = "o" if isinstance(item, OrderedDict) else "p"
        payload = "".join(
            self.value(key) + "=" + self.value(value) + ";"
            for key, value in item.items()
        )
        return f"M{identity}{order_tag}{len(item)}{{{payload}}}"


def canonical_class_dict(item: Any) -> str:
    return CanonicalSerializer().class_dict(item)


def canonical_snapshot(item: Mapping[str, Any]) -> str:
    return CanonicalSerializer().snapshot(item)


def canonical_value(item: Any) -> str:
    return CanonicalSerializer().value(item)


def make_mapping() -> OrderedDict[Any, Any]:
    return OrderedDict(((1, "one"), ("two", 2)))


def make_snapshot(object_index: int, field_count: int) -> dict[str, int]:
    return {
        f"field_{field}": object_index * 1000 + field
        for field in range(field_count)
    }


def python_differential_records(oracle: types.ModuleType) -> tuple[DifferentialRecord, ...]:
    ClassDict = oracle.ClassDict
    records: list[DifferentialRecord] = []

    values = ClassDict()
    fallback = 42
    records.append(
        DifferentialRecord(
            "empty",
            canonical_class_dict(values),
            (
                ("missing_is_none", "1" if values["missing"] is None else "0"),
                (
                    "default_identity",
                    "1" if values.get("missing", fallback) is fallback else "0",
                ),
            ),
        )
    )

    values = ClassDict.from_dict(
        {
            "integer": -7,
            "boolean": True,
            "float": -0.0,
            "text": "héllo\n",
            "none": None,
        }
    )
    records.append(
        DifferentialRecord(
            "from_primitives",
            canonical_class_dict(values),
            (
                ("index_0", canonical_value(values[0])),
                ("index_minus_1", canonical_value(values[-1])),
            ),
        )
    )

    values = ClassDict()
    values["first"] = 1
    first = list(vars(values)).index("first")
    values["second"] = 2
    second = list(vars(values)).index("second")
    values["first"] = 9
    first_again = list(vars(values)).index("first")
    values[""] = 3
    empty_name = list(vars(values)).index("")
    records.append(
        DifferentialRecord(
            "overwrite_order",
            canonical_class_dict(values),
            (("ids", f"{first},{second},{first_again},{empty_name}"),),
        )
    )

    values = ClassDict()
    values["existing"] = 11
    existing = list(vars(values)).index("existing")
    resolved_existing = list(vars(values)).index("existing")
    values["created"] = None
    records.append(
        DifferentialRecord(
            "resolve_or_create",
            canonical_class_dict(values),
            (
                (
                    "existing_id_same",
                    "1" if existing == resolved_existing else "0",
                ),
                ("created_none", "1" if values["created"] is None else "0"),
            ),
        )
    )

    values = ClassDict.from_dict({"a": 1, "b": 2})
    values.update({"b": 20, "c": 3})
    other = ClassDict.from_dict({"c": 30, "d": 4})
    values.update(other)
    values.update({"d": 40, "e": 5})
    records.append(
        DifferentialRecord("update_sequence", canonical_class_dict(values))
    )

    shared = [1]
    values = ClassDict.from_dict({"items": shared})
    shared.append(2)
    records.append(
        DifferentialRecord(
            "shallow_input",
            canonical_class_dict(values),
            (("same_identity", "1" if values["items"] is shared else "0"),),
        )
    )

    shared = [1, "x"]
    values = ClassDict.from_dict({"left": shared, "right": shared})
    snapshot = values.to_dict()
    left = snapshot["left"]
    right = snapshot["right"]
    alias_preserved = left is right
    detached = left is not shared
    left.append(3)
    records.append(
        DifferentialRecord(
            "deepcopy_alias",
            canonical_snapshot(snapshot),
            (
                ("alias_preserved", "1" if alias_preserved else "0"),
                ("detached", "1" if detached else "0"),
                ("source_size", str(len(shared))),
            ),
        )
    )

    shared = [10]
    nested = ClassDict.from_dict({"inner": shared})
    values = ClassDict.from_dict({"outer": shared, "nested": nested})
    snapshot = values.to_dict()
    outer = snapshot["outer"]
    inner = snapshot["nested"]["inner"]
    alias_preserved = outer is inner
    detached = outer is not shared
    outer.append(11)
    records.append(
        DifferentialRecord(
            "nested_class_dict_deepcopy",
            canonical_value(inner),
            (
                ("alias_preserved", "1" if alias_preserved else "0"),
                ("detached", "1" if detached else "0"),
                ("source_size", str(len(shared))),
            ),
        )
    )

    cycle: list[Any] = []
    cycle.append(cycle)
    values = ClassDict.from_dict({"cycle": cycle})
    snapshot = values.to_dict()
    records.append(
        DifferentialRecord("deepcopy_cycle", canonical_snapshot(snapshot))
    )

    mapping = make_mapping()
    values = ClassDict.from_dict(
        {
            "node_slots": mapping,
            "ordinary": mapping,
            "link_paths": mapping,
            "node_slots_info": mapping,
            "link_paths_info": mapping,
        }
    )
    snapshot = values.to_dict()
    records.append(
        DifferentialRecord(
            "ordered_mapping_conversion",
            canonical_snapshot(snapshot),
            (
                (
                    "special_distinct",
                    "1" if snapshot["node_slots"] is not snapshot["link_paths"] else "0",
                ),
                (
                    "ordinary_detached",
                    "1" if snapshot["ordinary"] is not mapping else "0",
                ),
            ),
        )
    )

    values = ClassDict.from_dict({"a": 1, "b": 2, "c": 3})
    positive_error = ""
    negative_error = ""
    try:
        _ = values[3]
    except IndexError:
        positive_error = "index_error"
    try:
        _ = values[-4]
    except IndexError:
        negative_error = "index_error"
    records.append(
        DifferentialRecord(
            "integer_indexing",
            canonical_class_dict(values),
            (
                ("zero", canonical_value(values[0])),
                ("minus_one", canonical_value(values[-1])),
                ("bool_true", canonical_value(values[True])),
                ("positive_error", positive_error),
                ("negative_error", negative_error),
            ),
        )
    )

    values = ClassDict()
    values["explicit_none"] = None
    fallback = 99
    explicit_found = "explicit_none" in vars(values)
    records.append(
        DifferentialRecord(
            "missing_and_explicit_none",
            canonical_class_dict(values),
            (
                ("missing_default", canonical_value(values.get("missing", fallback))),
                ("explicit_found", "1" if explicit_found else "0"),
                (
                    "explicit_value",
                    canonical_value(values["explicit_none"])
                    if explicit_found
                    else "missing",
                ),
            ),
        )
    )

    source = ClassDict.from_dict({"x": 1, "y": 2})
    target = ClassDict.from_dict({"prefix": 0, "x": -1})
    target.update(source)
    target.update(target)
    records.append(
        DifferentialRecord("update_class_dict", canonical_class_dict(target))
    )

    values = ClassDict.from_dict({"kept": 1})
    values.update(object(), 123, None)
    records.append(
        DifferentialRecord(
            "unsupported_update_ignored", canonical_class_dict(values)
        )
    )

    values = ClassDict.from_dict(
        {"": 1, "café": "☃", "line\nkey": "nul\0byte"}
    )
    records.append(
        DifferentialRecord("string_keys_and_bytes", canonical_class_dict(values))
    )

    inputs = [make_snapshot(index, 12) for index in range(32)]
    sequential = [ClassDict.from_dict(item) for item in inputs]
    parallel = [ClassDict.from_dict(item) for item in inputs]
    sequential_output = [item.to_dict() for item in sequential]
    parallel_output = [item.to_dict() for item in parallel]
    records.append(
        DifferentialRecord(
            "batch_order_workers",
            canonical_snapshot(parallel_output[0])
            + canonical_snapshot(parallel_output[-1]),
            (
                (
                    "same_first",
                    "1"
                    if canonical_snapshot(sequential_output[0])
                    == canonical_snapshot(parallel_output[0])
                    else "0",
                ),
                (
                    "same_last",
                    "1"
                    if canonical_snapshot(sequential_output[-1])
                    == canonical_snapshot(parallel_output[-1])
                    else "0",
                ),
            ),
        )
    )

    if len(records) != 16:
        raise AssertionError(f"canonical differential corpus changed: {len(records)}")
    return tuple(records)


def parse_key_values(output: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(output.splitlines(), 1):
        if not raw_line:
            continue
        key, separator, value = raw_line.partition("=")
        if not separator or not key:
            raise RuntimeError(
                f"invalid C++ protocol line {line_number}: {raw_line!r}"
            )
        if key in values:
            raise RuntimeError(f"duplicate C++ protocol key: {key!r}")
        values[key] = value
    return values


def required(values: dict[str, str], key: str) -> str:
    try:
        return values[key]
    except KeyError as error:
        raise RuntimeError(f"C++ protocol is missing {key!r}") from error


def run_cpp(arguments: list[str], timeout_seconds: float) -> str:
    completed = subprocess.run(
        arguments,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout_seconds,
    )
    stdout = completed.stdout.decode("utf-8", errors="strict")
    stderr = completed.stderr.decode("utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(
            f"C++ harness exited {completed.returncode}: {stderr.strip()}"
        )
    if stderr.strip():
        raise RuntimeError(f"C++ harness wrote to stderr: {stderr.strip()}")
    return stdout


def compare_differential(
    expected: tuple[DifferentialRecord, ...],
    output: str,
) -> None:
    values = parse_key_values(output)
    cpp_count = int(required(values, "diff_count"))
    if cpp_count != len(expected):
        raise AssertionError(
            f"C++ diff_count={cpp_count}, expected {len(expected)}"
        )

    mismatches: list[str] = []
    for index, record in enumerate(expected):
        prefix = f"diff[{index}]"
        cpp_name = required(values, prefix + ".name")
        if cpp_name != record.name:
            mismatches.append(
                f"case {index} name: C++={cpp_name!r}, Python={record.name!r}"
            )

        expected_payload = record.payload.encode("utf-8").hex()
        cpp_payload = required(values, prefix + ".payload_hex").lower()
        if cpp_payload != expected_payload:
            mismatches.append(
                f"{record.name}.payload:\n"
                f"    C++ hex={cpp_payload}\n"
                f"    Python={record.payload!r}\n"
                f"    Python hex={expected_payload}"
            )

        fact_count = int(required(values, prefix + ".fact_count"))
        if fact_count != len(record.facts):
            mismatches.append(
                f"{record.name}.fact_count: C++={fact_count}, "
                f"Python={len(record.facts)}"
            )
            continue
        for fact_index, (fact_name, fact_value) in enumerate(record.facts):
            fact_prefix = f"{prefix}.fact[{fact_index}]"
            cpp_fact_name = required(values, fact_prefix + ".name")
            cpp_fact_value = required(values, fact_prefix + ".value_hex").lower()
            expected_fact_value = fact_value.encode("utf-8").hex()
            if cpp_fact_name != fact_name:
                mismatches.append(
                    f"{record.name}.fact[{fact_index}].name: "
                    f"C++={cpp_fact_name!r}, Python={fact_name!r}"
                )
            if cpp_fact_value != expected_fact_value:
                mismatches.append(
                    f"{record.name}.{fact_name}: C++ hex={cpp_fact_value}, "
                    f"Python={fact_value!r} ({expected_fact_value})"
                )

    if mismatches:
        raise AssertionError(
            "ClassDict exact differential mismatch:\n  "
            + "\n  ".join(mismatches)
        )


def mix_checksum(checksum: int, value: int) -> int:
    mixed = (
        (value & MASK64)
        + MIX_CONSTANT
        + ((checksum << 6) & MASK64)
        + (checksum >> 2)
    ) & MASK64
    return (checksum ^ mixed) & MASK64


class PythonBenchmarkFixtures:
    def __init__(self, oracle: types.ModuleType) -> None:
        self.ClassDict = oracle.ClassDict
        self.lookup_snapshot = make_snapshot(0, 256)
        self.lookup_dict = self.ClassDict.from_dict(self.lookup_snapshot)
        self.medium_snapshot = make_snapshot(7, 128)
        self.medium_dict = self.ClassDict.from_dict(self.medium_snapshot)
        self.batch_snapshots = [make_snapshot(index, 64) for index in range(512)]
        self.batch_dicts = [
            self.ClassDict.from_dict(item) for item in self.batch_snapshots
        ]
        self.medium_dict["nested"] = list(range(32))
        self.medium_dict["node_slots"] = make_mapping()

    def operation(self, item: BenchmarkCase) -> int:
        checksum = 0
        if item.name == "string_get":
            for iteration in range(item.items):
                value = self.lookup_dict["field_127"]
                checksum = mix_checksum(checksum, value + iteration)
            return checksum

        if item.name == "id_get":
            for iteration in range(item.items):
                value = self.lookup_dict[127]
                checksum = mix_checksum(checksum, value + iteration)
            return checksum

        if item.name == "resolved_reference_get":
            value = vars(self.lookup_dict)["field_127"]
            for iteration in range(item.items):
                checksum = mix_checksum(checksum, value + iteration)
            return checksum

        if item.name == "resolved_reference_set":
            value = 127
            self.lookup_dict["field_127"] = value
            for iteration in range(item.items):
                value += int((iteration & 1) != 0)
                self.lookup_dict["field_127"] = value
                checksum = mix_checksum(checksum, value)
            return checksum

        if item.name == "from_dict":
            for _iteration in range(item.items):
                value = self.ClassDict.from_dict(self.medium_snapshot)
                checksum = mix_checksum(checksum, value[127])
            return checksum

        if item.name == "to_dict":
            for _iteration in range(item.items):
                value = self.medium_dict.to_dict()
                checksum = mix_checksum(
                    checksum, len(value) + list(value.values())[127]
                )
            return checksum

        if item.name == "batch_from_dict":
            values = [
                self.ClassDict.from_dict(snapshot)
                for snapshot in self.batch_snapshots
            ]
            for value in values:
                checksum = mix_checksum(checksum, value[63])
            return checksum

        if item.name == "batch_to_dict":
            values = [value.to_dict() for value in self.batch_dicts]
            for value in values:
                checksum = mix_checksum(checksum, list(value.values())[63])
            return checksum

        raise AssertionError(f"unknown benchmark case: {item.name}")


def percentile95(samples: list[int]) -> float:
    ordered = sorted(samples)
    return float(ordered[math.ceil(0.95 * len(ordered)) - 1])


def measure_python(
    action: Callable[[], int], warmups: int, repetitions: int
) -> BenchmarkStats:
    expected_checksum: int | None = None
    samples: list[int] = []
    for sample in range(warmups + repetitions):
        start = time.perf_counter_ns()
        checksum = action()
        stop = time.perf_counter_ns()
        if expected_checksum is None:
            expected_checksum = checksum
        elif checksum != expected_checksum:
            raise AssertionError(
                "Python benchmark checksum changed between repetitions: "
                f"{expected_checksum} != {checksum}"
            )
        if sample >= warmups:
            samples.append(stop - start)

    if expected_checksum is None or not samples:
        raise AssertionError("benchmark did not produce measured samples")
    sample_median = float(statistics.median(samples))
    deviations = [abs(sample - sample_median) for sample in samples]
    return BenchmarkStats(
        median_ms=sample_median / 1e6,
        mad_ms=float(statistics.median(deviations)) / 1e6,
        p95_ms=percentile95(samples) / 1e6,
        checksum=expected_checksum,
    )


def python_benchmark(
    oracle: types.ModuleType, warmups: int, repetitions: int
) -> dict[str, BenchmarkStats]:
    fixtures = PythonBenchmarkFixtures(oracle)
    return {
        item.name: measure_python(
            lambda item=item: fixtures.operation(item), warmups, repetitions
        )
        for item in BENCHMARK_CASES
    }


def parse_cpp_benchmark(
    output: str, workers: int, warmups: int, repetitions: int
) -> dict[str, BenchmarkStats]:
    values = parse_key_values(output)
    count = int(required(values, "bench_count"))
    if count != len(BENCHMARK_CASES):
        raise AssertionError(
            f"C++ bench_count={count}, expected {len(BENCHMARK_CASES)}"
        )
    metadata = (
        int(required(values, "bench_workers")),
        int(required(values, "bench_warmups")),
        int(required(values, "bench_repetitions")),
    )
    expected_metadata = (workers, warmups, repetitions)
    if metadata != expected_metadata:
        raise AssertionError(
            f"C++ benchmark metadata={metadata}, expected {expected_metadata}"
        )

    rows: dict[str, BenchmarkStats] = {}
    for index, item in enumerate(BENCHMARK_CASES):
        prefix = f"bench[{index}]"
        name = required(values, prefix + ".name")
        if name != item.name:
            raise AssertionError(
                f"benchmark {index} name: C++={name!r}, expected={item.name!r}"
            )
        cpp_items = int(required(values, prefix + ".items"))
        cpp_fields = int(required(values, prefix + ".fields"))
        if (cpp_items, cpp_fields) != (item.items, item.fields):
            raise AssertionError(
                f"{item.name} recipe: C++={(cpp_items, cpp_fields)}, "
                f"expected={(item.items, item.fields)}"
            )
        rows[name] = BenchmarkStats(
            median_ms=float(required(values, prefix + ".cpp_median_ms")),
            mad_ms=float(required(values, prefix + ".cpp_mad_ms")),
            p95_ms=float(required(values, prefix + ".cpp_p95_ms")),
            checksum=int(required(values, prefix + ".checksum")),
        )
    return rows


def report_benchmark(
    cpp_rows: dict[str, BenchmarkStats],
    python_rows: dict[str, BenchmarkStats],
    workers: int,
) -> tuple[dict[str, float], list[str]]:
    print(
        "operation                 workers | Python median/MAD/p95 ms | "
        "C++ median/MAD/p95 ms | speedup"
    )
    print("-" * 107)
    speedups: dict[str, float] = {}
    slower: list[str] = []
    for item in BENCHMARK_CASES:
        cpp = cpp_rows[item.name]
        python = python_rows[item.name]
        if cpp.checksum != python.checksum:
            raise AssertionError(
                f"{item.name}: checksum mismatch C++={cpp.checksum}, "
                f"Python={python.checksum}"
            )
        speedup = python.median_ms / cpp.median_ms
        speedups[item.name] = speedup
        print(
            f"{item.name:25} {workers:7d} | "
            f"{python.median_ms:9.3f}/{python.mad_ms:7.3f}/"
            f"{python.p95_ms:7.3f} | "
            f"{cpp.median_ms:9.3f}/{cpp.mad_ms:7.3f}/"
            f"{cpp.p95_ms:7.3f} | {speedup:8.2f}x"
        )
        if cpp.median_ms >= python.median_ms:
            slower.append(item.name)
    return speedups, slower


def write_json_report(
    path: Path,
    cpp_rows: dict[str, BenchmarkStats],
    python_rows: dict[str, BenchmarkStats],
    speedups: dict[str, float],
    slower: list[str],
    args: argparse.Namespace,
) -> None:
    report = {
        "status": "pass" if not slower else "measured_with_slower_rows",
        "component": "virne.utils.class_dict",
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "differential": {"status": "pass", "exact_cases": 16},
        "benchmark": {
            "workers": args.workers,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "python_runtime": platform.python_version(),
            "cpus_visible": os.cpu_count(),
            "rows": {
                item.name: {
                    "items": item.items,
                    "fields": item.fields,
                    "python": asdict(python_rows[item.name]),
                    "cpp": asdict(cpp_rows[item.name]),
                    "speedup": speedups[item.name],
                }
                for item in BENCHMARK_CASES
            },
        },
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact Python/C++ ClassDict differential and runtime gate"
    )
    parser.add_argument("--cpp", required=True, type=Path)
    parser.add_argument("--python-source", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=11)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--allow-slower",
        action="store_true",
        help="report timings without failing when a C++ median is slower",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.cpp, args.python_source):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.workers < 0:
        raise ValueError("workers must be non-negative (0 selects C++ auto policy)")
    if args.warmups < 0 or args.repetitions < 1:
        raise ValueError("warmups must be non-negative and repetitions positive")
    if args.timeout <= 0:
        raise ValueError("timeout must be positive")
    if len(BENCHMARK_CASES) != 8:
        raise AssertionError("canonical benchmark table was changed")

    oracle = load_original(args.python_source)
    expected = python_differential_records(oracle)
    compare_differential(
        expected,
        run_cpp([str(args.cpp), "--differential"], args.timeout),
    )
    print(f"source_sha256={EXPECTED_SOURCE_SHA256}")
    print("differential_status=PASS")
    print(f"differential_exact_cases={len(expected)}")

    cpp_rows = parse_cpp_benchmark(
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
            ],
            args.timeout,
        ),
        args.workers,
        args.warmups,
        args.repetitions,
    )
    python_rows = python_benchmark(oracle, args.warmups, args.repetitions)

    print(
        f"runtime=Python {platform.python_version()}, "
        f"CPUs visible {os.cpu_count()}"
    )
    print(
        f"protocol=warmups {args.warmups}, repetitions {args.repetitions}, "
        f"C++ workers {'auto' if args.workers == 0 else args.workers}; "
        "fixture construction and process startup excluded"
    )
    speedups, slower = report_benchmark(cpp_rows, python_rows, args.workers)

    for item in BENCHMARK_CASES:
        name = item.name
        print(f"benchmark.{name}.checksum={cpp_rows[name].checksum}")
        print(f"benchmark.{name}.speedup={speedups[name]:.9f}")

    if slower and not args.allow_slower:
        raise AssertionError(
            "C++ median must beat Python for every canonical benchmark: "
            + ", ".join(slower)
        )
    if args.json_output is not None:
        write_json_report(
            args.json_output,
            cpp_rows,
            python_rows,
            speedups,
            slower,
            args,
        )
        print(f"json_report={args.json_output}")
    print(
        "performance_status=PASS"
        if not slower
        else "performance_status=MEASURED_WITH_SLOWER_ROWS"
    )
    print("result=PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"compare_class_dict: FAIL: {error}", file=sys.stderr)
        raise
