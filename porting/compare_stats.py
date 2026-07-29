#!/usr/bin/env python3
"""Exact differential gate for the frozen ``virne.utils.stats`` contract."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import math
import pathlib
import subprocess
import sys
import time as standard_time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any


FIELD_NAMES = (
    "id",
    "output_hex",
    "result",
    "exception",
    "calls",
    "clocks",
    "identity",
    "lazy_constructed",
    "lazy_consumed",
)


@dataclass
class ScriptedClock:
    values: list[float]
    throw_on: int | None = None
    calls: int = 0

    def __call__(self) -> float:
        index = self.calls
        self.calls += 1
        if index == self.throw_on:
            raise RuntimeError("clock failure")
        return self.values[index]


class CaptureSink:
    encoding = "utf-8"

    def __init__(self) -> None:
        self.data = bytearray()

    def write(self, text: str) -> int:
        self.data.extend(text.encode(self.encoding))
        return len(text)

    def flush(self) -> None:
        return None


class RejectingSink:
    encoding = "utf-8"

    def write(self, _text: str) -> int:
        raise OSError("sink failure")

    def flush(self) -> None:
        return None


class CallableFailure(Exception):
    pass


def load_oracle(python_root: pathlib.Path) -> Any:
    source_path = python_root / "virne" / "utils" / "stats.py"
    if not source_path.is_file():
        raise FileNotFoundError(f"stats oracle not found: {source_path}")
    module_name = "_virne_frozen_stats_oracle"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load stats oracle: {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def patched_clock(module: Any, clock: ScriptedClock) -> Iterator[None]:
    del module
    original = standard_time.time
    standard_time.time = clock
    try:
        yield
    finally:
        standard_time.time = original


def record(
    case_id: str,
    output: bytes,
    result: str,
    exception: str,
    calls: int,
    clocks: int,
    *,
    identity: bool = False,
    lazy_constructed: int = 0,
    lazy_consumed: int = 0,
) -> dict[str, str]:
    return {
        "id": case_id,
        "output_hex": output.hex(),
        "result": result,
        "exception": exception,
        "calls": str(calls),
        "clocks": str(clocks),
        "identity": "1" if identity else "0",
        "lazy_constructed": str(lazy_constructed),
        "lazy_consumed": str(lazy_consumed),
    }


def run_success(
    module: Any,
    case_id: str,
    name: str,
    start: float,
    stop: float,
    value: int,
) -> dict[str, str]:
    calls = 0

    def callable_under_test() -> int:
        nonlocal calls
        calls += 1
        return value

    callable_under_test.__name__ = name
    timed = module.test_running_time(callable_under_test)
    clock = ScriptedClock([start, stop])
    sink = CaptureSink()
    with patched_clock(module, clock), contextlib.redirect_stdout(sink):
        observed = timed()
    return record(
        case_id,
        bytes(sink.data),
        f"int:{observed}",
        "none",
        calls,
        clock.calls,
    )


def run_void(
    module: Any,
    case_id: str,
    name: str,
    start: float,
    stop: float,
) -> dict[str, str]:
    calls = 0

    def callable_under_test() -> None:
        nonlocal calls
        calls += 1

    callable_under_test.__name__ = name
    timed = module.test_running_time(callable_under_test)
    clock = ScriptedClock([start, stop])
    sink = CaptureSink()
    with patched_clock(module, clock), contextlib.redirect_stdout(sink):
        observed = timed()
    if observed is not None:
        raise AssertionError(f"{case_id}: expected None return")
    return record(
        case_id,
        bytes(sink.data),
        "void",
        "none",
        calls,
        clock.calls,
    )


def run_argument_case(module: Any) -> dict[str, str]:
    calls = 0

    def arguments(base: int, *, scale: int) -> int:
        nonlocal calls
        calls += 1
        return base * scale + 2

    timed = module.test_running_time(arguments)
    clock = ScriptedClock([0.0, 1.0])
    sink = CaptureSink()
    with patched_clock(module, clock), contextlib.redirect_stdout(sink):
        observed = timed(5, scale=8)
    return record(
        "arguments",
        bytes(sink.data),
        f"int:{observed}",
        "none",
        calls,
        clock.calls,
    )


def run_mutation_case(module: Any) -> dict[str, str]:
    calls = 0

    def before() -> int:
        nonlocal calls
        calls += 1
        before.__name__ = "after"
        return 51

    timed = module.test_running_time(before)
    clock = ScriptedClock([0.0, 1.0])
    sink = CaptureSink()
    with patched_clock(module, clock), contextlib.redirect_stdout(sink):
        observed = timed()
    return record(
        "name_mutation",
        bytes(sink.data),
        f"int:{observed}",
        "none",
        calls,
        clock.calls,
    )


def run_nested_case(module: Any) -> dict[str, str]:
    calls = 0

    def inner() -> int:
        nonlocal calls
        calls += 1
        return 1

    timed_inner = module.test_running_time(inner)

    def outer() -> int:
        nonlocal calls
        calls += 1
        return timed_inner() + 1

    timed_outer = module.test_running_time(outer)
    clock = ScriptedClock([0.0, 10.0, 11.0, 3.0])
    sink = CaptureSink()
    with patched_clock(module, clock), contextlib.redirect_stdout(sink):
        observed = timed_outer()
    return record(
        "nested",
        bytes(sink.data),
        f"int:{observed}",
        "none",
        calls,
        clock.calls,
    )


def run_identity_case(module: Any) -> dict[str, str]:
    calls = 0
    sentinel = object()

    def identity() -> object:
        nonlocal calls
        calls += 1
        return sentinel

    timed = module.test_running_time(identity)
    clock = ScriptedClock([0.0, 1.0])
    sink = CaptureSink()
    with patched_clock(module, clock), contextlib.redirect_stdout(sink):
        observed = timed()
    return record(
        "identity",
        bytes(sink.data),
        "object",
        "none",
        calls,
        clock.calls,
        identity=observed is sentinel,
    )


def run_lazy_case(module: Any) -> dict[str, str]:
    calls = 0
    constructed = 0
    consumed = 0

    def lazy() -> Iterator[int]:
        nonlocal calls, constructed, consumed
        calls += 1
        constructed += 1

        def iterator() -> Iterator[int]:
            nonlocal consumed
            consumed += 1
            yield 1

        return iterator()

    timed = module.test_running_time(lazy)
    clock = ScriptedClock([0.0, 1.0])
    sink = CaptureSink()
    with patched_clock(module, clock), contextlib.redirect_stdout(sink):
        observed = timed()
    if not hasattr(observed, "__next__"):
        raise AssertionError("lazy case did not return an iterator")
    return record(
        "lazy",
        bytes(sink.data),
        "lazy",
        "none",
        calls,
        clock.calls,
        lazy_constructed=constructed,
        lazy_consumed=consumed,
    )


def run_exception_case(
    module: Any,
    case_id: str,
    expected_stage: str,
    clock: ScriptedClock,
    callable_factory: Callable[[list[int]], Callable[[], Any]],
    sink: CaptureSink | RejectingSink,
    expected_exception: type[BaseException],
) -> dict[str, str]:
    call_counter = [0]
    callable_under_test = callable_factory(call_counter)
    callable_under_test.__name__ = case_id
    timed = module.test_running_time(callable_under_test)
    observed_stage = "missing"
    with patched_clock(module, clock), contextlib.redirect_stdout(sink):
        try:
            timed()
        except expected_exception:
            observed_stage = expected_stage
    output = bytes(sink.data) if isinstance(sink, CaptureSink) else b""
    return record(
        case_id,
        output,
        "none",
        observed_stage,
        call_counter[0],
        clock.calls,
    )


def run_python_cases(module: Any) -> list[dict[str, str]]:
    cases = [
        run_success(module, "ascii_scalar", "ascii", 0.0, 1.25, 42),
        run_success(module, "unicode_name", "đồ_thị", 0.0, 1.0, 1),
        run_success(module, "newline_name", "line\nbreak", 0.0, 1.0, 1),
        run_success(module, "nul_name", "nul\0name", 0.0, 1.0, 1),
        run_argument_case(module),
        run_void(module, "delta_zero", "delta_zero", 0.0, 0.0),
        run_void(
            module,
            "delta_negative_zero",
            "delta_negative_zero",
            0.0,
            -0.0,
        ),
        run_void(module, "delta_negative", "delta_negative", 2.0, 1.25),
        run_void(module, "round_even_down", "round_even_down", 0.0, 0.03125),
        run_void(module, "round_even_up", "round_even_up", 0.0, 0.09375),
        run_void(module, "large_finite", "large_finite", 0.0, 1.0e20),
        run_void(module, "nan", "nan", 0.0, math.nan),
        run_void(
            module,
            "positive_infinity",
            "positive_infinity",
            0.0,
            math.inf,
        ),
        run_void(
            module,
            "negative_infinity",
            "negative_infinity",
            0.0,
            -math.inf,
        ),
        run_mutation_case(module),
        run_nested_case(module),
        run_identity_case(module),
        run_lazy_case(module),
    ]

    def succeeds(counter: list[int]) -> Callable[[], int]:
        def callable_under_test() -> int:
            counter[0] += 1
            return 1

        return callable_under_test

    def fails(counter: list[int]) -> Callable[[], int]:
        def callable_under_test() -> int:
            counter[0] += 1
            raise CallableFailure("callable failure")

        return callable_under_test

    cases.extend(
        [
            run_exception_case(
                module,
                "first_clock_exception",
                "first_clock",
                ScriptedClock([], throw_on=0),
                succeeds,
                CaptureSink(),
                RuntimeError,
            ),
            run_exception_case(
                module,
                "callable_exception",
                "callable",
                ScriptedClock([0.0, 1.0]),
                fails,
                CaptureSink(),
                CallableFailure,
            ),
            run_exception_case(
                module,
                "second_clock_exception",
                "second_clock",
                ScriptedClock([0.0], throw_on=1),
                succeeds,
                CaptureSink(),
                RuntimeError,
            ),
            run_exception_case(
                module,
                "sink_exception",
                "sink",
                ScriptedClock([0.0, 1.0]),
                succeeds,
                RejectingSink(),
                OSError,
            ),
        ]
    )
    return cases


def parse_cpp_records(harness: pathlib.Path) -> list[dict[str, str]]:
    process = subprocess.run(
        [str(harness), "differential"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    lines = process.stdout.splitlines()
    if not lines:
        raise RuntimeError("C++ harness returned no records")
    header = tuple(lines[0].split("\t"))
    if header != FIELD_NAMES:
        raise RuntimeError(f"unexpected C++ harness header: {header!r}")

    records: list[dict[str, str]] = []
    for line_number, line in enumerate(lines[1:], start=2):
        values = line.split("\t")
        if len(values) != len(FIELD_NAMES):
            raise RuntimeError(
                f"malformed C++ harness record on line {line_number}: {line!r}"
            )
        records.append(dict(zip(FIELD_NAMES, values, strict=True)))
    return records


def compare_records(
    python_records: list[dict[str, str]],
    cpp_records: list[dict[str, str]],
) -> list[str]:
    failures: list[str] = []
    python_by_id = {item["id"]: item for item in python_records}
    cpp_by_id = {item["id"]: item for item in cpp_records}
    if len(python_by_id) != len(python_records):
        failures.append("Python oracle emitted duplicate case IDs")
    if len(cpp_by_id) != len(cpp_records):
        failures.append("C++ harness emitted duplicate case IDs")

    python_ids = set(python_by_id)
    cpp_ids = set(cpp_by_id)
    if python_ids != cpp_ids:
        failures.append(
            "case set differs: "
            f"Python-only={sorted(python_ids - cpp_ids)!r}, "
            f"C++-only={sorted(cpp_ids - python_ids)!r}"
        )

    for case_id in sorted(python_ids & cpp_ids):
        python_record = python_by_id[case_id]
        cpp_record = cpp_by_id[case_id]
        for field in FIELD_NAMES[1:]:
            if python_record[field] != cpp_record[field]:
                failures.append(
                    f"{case_id}.{field}: Python={python_record[field]!r}, "
                    f"C++={cpp_record[field]!r}"
                )
    return failures


def default_python_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2] / "virne"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--harness",
        type=pathlib.Path,
        required=True,
        help="path to the strict-built stats_harness executable",
    )
    parser.add_argument(
        "--python-root",
        type=pathlib.Path,
        default=default_python_root(),
        help="root of the frozen Python checkout",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    harness = args.harness.resolve()
    if not harness.is_file():
        raise FileNotFoundError(f"C++ harness not found: {harness}")

    module = load_oracle(args.python_root.resolve())
    python_records = run_python_cases(module)
    cpp_records = parse_cpp_records(harness)
    failures = compare_records(python_records, cpp_records)
    if failures:
        print("stats differential: FAIL", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(
        "stats differential: PASS "
        f"({len(python_records)} cases; exact bytes/results/order)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
