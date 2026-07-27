#!/usr/bin/env python3
"""Differential GML round-trip and timing check against NetworkX 3.4.2."""

from __future__ import annotations

import argparse
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = ROOT / "build" / "gml_harness"
DEFAULT_INPUT = ROOT.parent / "virne" / "datasets" / "topology" / "Waxman100.gml"


def parse_values(output: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        key, separator, value = line.partition("=")
        if not separator:
            raise AssertionError(f"malformed harness output: {line!r}")
        result[key] = value
    return result


def run_harness(binary: Path, *arguments: str) -> dict[str, str]:
    completed = subprocess.run(
        [str(binary), *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip())
    return parse_values(completed.stdout)


def assert_graphs_equal(expected: nx.Graph, actual: nx.Graph) -> None:
    if expected.graph != actual.graph:
        raise AssertionError(
            f"graph metadata differs:\nexpected={expected.graph!r}\nactual={actual.graph!r}"
        )
    if dict(expected.nodes(data=True)) != dict(actual.nodes(data=True)):
        raise AssertionError("node attributes differ after C++ GML round-trip")
    expected_edges = {
        frozenset((u, v)): data for u, v, data in expected.edges(data=True)
    }
    actual_edges = {
        frozenset((u, v)): data for u, v, data in actual.edges(data=True)
    }
    if expected_edges != actual_edges:
        raise AssertionError("edge attributes differ after C++ GML round-trip")


def median_ms(function: Callable[[], object], warmups: int, reps: int) -> float:
    for _ in range(warmups):
        function()
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--parity-only", action="store_true")
    args = parser.parse_args()
    if nx.__version__ != "3.4.2":
        raise RuntimeError(f"expected NetworkX 3.4.2, found {nx.__version__}")
    if args.warmups < 0 or args.reps < 1:
        parser.error("warmups must be non-negative and reps positive")

    with tempfile.TemporaryDirectory(prefix="virne-gml-") as directory:
        cpp_output = Path(directory) / "cpp.gml"
        python_output = Path(directory) / "python.gml"
        values = run_harness(
            args.binary,
            "--roundtrip",
            str(args.input),
            str(cpp_output),
        )
        expected = nx.read_gml(args.input, label="id")
        actual = nx.read_gml(cpp_output, label="id")
        assert_graphs_equal(expected, actual)
        print(
            "GML PARITY PASS: "
            f"{values['nodes']} nodes, {values['edges']} edges, "
            f"{values['graph_attrs']} graph attributes"
        )

        if args.parity_only:
            return 0

        cpp = run_harness(
            args.binary,
            "--bench",
            str(args.input),
            str(cpp_output),
            str(max(args.warmups, 1)),
            str(args.reps),
        )
        python_graph = nx.read_gml(args.input, label="id")
        python_load = median_ms(
            lambda: nx.read_gml(args.input, label="id"),
            args.warmups,
            args.reps,
        )
        python_save = median_ms(
            lambda: nx.write_gml(python_graph, python_output),
            args.warmups,
            args.reps,
        )
        rows = [
            ("read_gml", float(cpp["load_ms"]), python_load),
            ("write_gml", float(cpp["save_ms"]), python_save),
        ]
        print("\n| API | C++ ms | NetworkX ms | Speedup | Status |")
        print("|---|---:|---:|---:|:---:|")
        failures = []
        for name, cpp_ms, python_ms in rows:
            speedup = python_ms / cpp_ms
            passed = cpp_ms < python_ms
            status = "PASS" if passed else "FAIL"
            if not passed:
                failures.append(name)
            print(
                f"| `{name}` | {cpp_ms:.6f} | {python_ms:.6f} | "
                f"{speedup:.2f}x | {status} |"
            )
        if failures:
            raise AssertionError(
                "C++ GML implementation did not beat NetworkX for: "
                + ", ".join(failures)
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
