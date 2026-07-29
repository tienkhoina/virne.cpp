#!/usr/bin/env python3
"""Checksum/GML-gated Python/C++ timing and parse-worker sweep for dataset XML."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import platform
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass

import compare_dataset_core
import compare_dataset_xml


FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211


@dataclass(frozen=True)
class ParseRow:
    name: str
    source: pathlib.Path
    documents: int
    repetitions: int = 1


def fnv1a(payload: bytes) -> int:
    checksum = FNV_OFFSET
    for value in payload:
        checksum ^= value
        checksum = (checksum * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return checksum


def percentile95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def timing_summary(samples_ns: list[int]) -> dict[str, float]:
    samples_ms = [sample / 1_000_000.0 for sample in samples_ns]
    median = statistics.median(samples_ms)
    deviations = [abs(value - median) for value in samples_ms]
    return {
        "median_ms": median,
        "mad_ms": statistics.median(deviations),
        "p95_ms": percentile95(samples_ms),
    }


def python_parse_benchmark(row: ParseRow) -> dict[str, int]:
    records = []
    start = time.perf_counter_ns()
    for _ in range(row.repetitions):
        records = [compare_dataset_xml.parse_records(row.source) for _ in range(row.documents)]
    elapsed_ns = time.perf_counter_ns() - start
    payload = compare_dataset_xml.serialize_batch(records)
    return {
        "elapsed_ns": elapsed_ns,
        "checksum": fnv1a(payload),
        "output_bytes": len(payload),
    }


def run_cpp_benchmark(harness: pathlib.Path, arguments: list[str]) -> dict[str, str]:
    process = subprocess.run(
        [str(harness), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return compare_dataset_xml.parse_response(process, benchmark=True)


def cpp_parse_benchmark(
    harness: pathlib.Path,
    row: ParseRow,
    workers: int,
) -> dict[str, int]:
    values = run_cpp_benchmark(
        harness,
        [
            "benchmark_parse",
            str(row.source),
            str(row.documents),
            str(workers),
            str(row.repetitions),
        ],
    )
    required = {
        "dataset_xml_benchmark_version",
        "command",
        "documents",
        "workers",
        "repetitions",
        "elapsed_ns",
        "checksum",
        "output_bytes",
        "status",
    }
    if (
        set(values) != required
        or values["command"] != "benchmark_parse"
        or values["status"] != "PASS"
        or int(values["documents"]) != row.documents
        or int(values["workers"]) != workers
        or int(values["repetitions"]) != row.repetitions
    ):
        raise RuntimeError(f"invalid C++ XML parse benchmark response: {values!r}")
    return {
        "elapsed_ns": int(values["elapsed_ns"]),
        "checksum": int(values["checksum"]),
        "output_bytes": int(values["output_bytes"]),
    }


def benchmark_parse_widths(
    harness: pathlib.Path,
    row: ParseRow,
    widths: list[int],
    warmups: int,
    repetitions: int,
) -> list[dict]:
    python_samples = {workers: [] for workers in widths}
    cpp_samples = {workers: [] for workers in widths}
    semantics: dict[int, tuple[int, int]] = {}
    for sample in range(warmups + repetitions):
        offset = sample % len(widths)
        ordered = widths[offset:] + widths[:offset]
        if (sample // len(widths)) % 2:
            ordered.reverse()

        # Python has no worker-width input. Sample it once per round and reuse
        # that baseline for every C++ width; this keeps 31 independent Python
        # samples while avoiding nine identical multi-document parses per
        # round. Alternate whether Python runs before or after the rotated C++
        # width set so neither implementation owns a fixed thermal position.
        python = None
        if sample % 2 == 0:
            python = python_parse_benchmark(row)
        cpp_round = {}
        for workers in ordered:
            cpp_round[workers] = cpp_parse_benchmark(harness, row, workers)
        if python is None:
            python = python_parse_benchmark(row)

        for workers in ordered:
            cpp = cpp_round[workers]
            python_semantic = (python["checksum"], python["output_bytes"])
            cpp_semantic = (cpp["checksum"], cpp["output_bytes"])
            if python_semantic != cpp_semantic:
                raise RuntimeError(
                    f"{row.name}/w{workers} semantic mismatch: "
                    f"python={python_semantic}, cpp={cpp_semantic}"
                )
            if workers in semantics and semantics[workers] != python_semantic:
                raise RuntimeError(f"{row.name}/w{workers} semantic drift")
            semantics[workers] = python_semantic
            if sample >= warmups:
                python_samples[workers].append(python["elapsed_ns"])
                cpp_samples[workers].append(cpp["elapsed_ns"])

    rows = []
    for workers in widths:
        python_stats = timing_summary(python_samples[workers])
        cpp_stats = timing_summary(cpp_samples[workers])
        rows.append(
            {
                "kind": row.name,
                "documents": row.documents,
                "inner_repetitions": row.repetitions,
                "workers": workers,
                "checksum": semantics[workers][0],
                "output_bytes": semantics[workers][1],
                "python": python_stats,
                "cpp": cpp_stats,
                "speedup": python_stats["median_ms"] / cpp_stats["median_ms"],
            }
        )
    return rows


def python_preprocess_benchmark(
    module,
    name: str,
    source: pathlib.Path,
    target: pathlib.Path,
    repetitions: int,
) -> dict[str, int | bytes]:
    target.write_bytes(b"sentinel")
    graph = None
    start = time.perf_counter_ns()
    for _ in range(repetitions):
        graph = compare_dataset_xml.oracle_preprocess(
            module, name, source, target
        )
    elapsed_ns = time.perf_counter_ns() - start
    graph_payload = compare_dataset_xml.serialize_graph(graph)
    gml = target.read_bytes()
    payload = graph_payload + b"\0" + gml
    return {
        "elapsed_ns": elapsed_ns // repetitions,
        "checksum": fnv1a(payload),
        "output_bytes": len(payload),
        "graph": graph_payload,
        "gml": gml,
    }


def cpp_preprocess_benchmark(
    harness: pathlib.Path,
    name: str,
    source: pathlib.Path,
    target: pathlib.Path,
    repetitions: int,
) -> dict[str, int | bytes]:
    target.write_bytes(b"sentinel")
    values = run_cpp_benchmark(
        harness,
        ["benchmark_preprocess", name, str(source), str(target), str(repetitions)],
    )
    required = {
        "dataset_xml_benchmark_version",
        "command",
        "documents",
        "workers",
        "repetitions",
        "elapsed_ns",
        "checksum",
        "output_bytes",
        "graph_hex",
        "gml_hex",
        "status",
    }
    if (
        set(values) != required
        or values["command"] != "benchmark_preprocess"
        or values["status"] != "PASS"
        or values["documents"] != "1"
        or values["workers"] != "1"
        or int(values["repetitions"]) != repetitions
    ):
        raise RuntimeError(f"invalid C++ XML preprocess benchmark response: {values!r}")
    graph = bytes.fromhex(values["graph_hex"])
    gml = bytes.fromhex(values["gml_hex"])
    if target.read_bytes() != gml:
        raise RuntimeError("C++ preprocess benchmark target/response byte drift")
    return {
        "elapsed_ns": int(values["elapsed_ns"]) // repetitions,
        "checksum": int(values["checksum"]),
        "output_bytes": int(values["output_bytes"]),
        "graph": graph,
        "gml": gml,
    }


def benchmark_preprocess(
    module,
    harness: pathlib.Path,
    name: str,
    source: pathlib.Path,
    python_target: pathlib.Path,
    cpp_target: pathlib.Path,
    warmups: int,
    repetitions: int,
    inner_repetitions: int,
) -> dict:
    python_samples = []
    cpp_samples = []
    semantic: tuple[int, int, str, str] | None = None
    for sample in range(warmups + repetitions):
        if sample % 2 == 0:
            python = python_preprocess_benchmark(
                module, name, source, python_target, inner_repetitions
            )
            cpp = cpp_preprocess_benchmark(
                harness, name, source, cpp_target, inner_repetitions
            )
        else:
            cpp = cpp_preprocess_benchmark(
                harness, name, source, cpp_target, inner_repetitions
            )
            python = python_preprocess_benchmark(
                module, name, source, python_target, inner_repetitions
            )
        compare_dataset_xml.assert_exact_bytes(
            "benchmark preprocess graph", python["graph"], cpp["graph"]
        )
        compare_dataset_xml.assert_exact_bytes(
            "benchmark preprocess GML", python["gml"], cpp["gml"]
        )
        python_semantic = (python["checksum"], python["output_bytes"])
        cpp_semantic = (cpp["checksum"], cpp["output_bytes"])
        if python_semantic != cpp_semantic:
            raise RuntimeError(
                f"preprocess semantic mismatch: python={python_semantic}, cpp={cpp_semantic}"
            )
        current = (
            python["checksum"],
            python["output_bytes"],
            hashlib.sha256(python["graph"]).hexdigest(),
            hashlib.sha256(python["gml"]).hexdigest(),
        )
        if semantic is not None and current != semantic:
            raise RuntimeError("preprocess semantic/GML drift between samples")
        semantic = current
        if sample >= warmups:
            python_samples.append(python["elapsed_ns"])
            cpp_samples.append(cpp["elapsed_ns"])
    python_stats = timing_summary(python_samples)
    cpp_stats = timing_summary(cpp_samples)
    return {
        "kind": "brain_preprocess",
        "documents": 1,
        "inner_repetitions": inner_repetitions,
        "workers": 1,
        "checksum": semantic[0],
        "output_bytes": semantic[1],
        "graph_sha256": semantic[2],
        "gml_sha256": semantic[3],
        "python": python_stats,
        "cpp": cpp_stats,
        "speedup": python_stats["median_ms"] / cpp_stats["median_ms"],
    }


def write_large_synthetic(path: pathlib.Path, nodes: int, directed_links: int) -> None:
    parts = ['<?xml version="1.0" encoding="UTF-8"?><network><nodes>']
    for node in range(nodes):
        parts.append(f'<node id="N{node}"><x>{node}.0</x><y>{-node}.0</y></node>')
    parts.append("</nodes><links>")
    for edge in range(directed_links):
        source = edge % nodes
        target = (source * 17 + edge // nodes + 1) % nodes
        parts.append(
            f'<link id="E{edge}"><source>N{source}</source><target>N{target}</target>'
            f"<capacity>{edge + 1}</capacity><capacity>{edge + 2}</capacity>"
            "<cost>ignored</cost></link>"
        )
    parts.append("</links></network>")
    path.write_text("".join(parts), encoding="utf-8")


def cpu_affinity() -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--python-source", type=pathlib.Path, required=True)
    parser.add_argument("--brain-xml", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 0, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=31)
    parser.add_argument("--batch-documents", type=int, default=16)
    parser.add_argument("--performance-gate", action="store_true")
    parser.add_argument("--worker-policy-gate", action="store_true")
    parser.add_argument("--json-output", type=pathlib.Path)
    args = parser.parse_args()
    if args.warmups < 0 or args.repetitions <= 0 or args.batch_documents <= 0:
        raise SystemExit("invalid benchmark counts")
    if len(set(args.workers)) != len(args.workers) or any(worker < 0 or worker > 8 for worker in args.workers):
        raise SystemExit("workers must be unique values in 0..8")
    if args.worker_policy_gate and (
        set(args.workers) != set(range(9))
        or args.warmups < 5
        or args.repetitions < 31
        or args.batch_documents != 16
    ):
        raise SystemExit(
            "worker policy gate requires workers 0..8, at least 5/31 samples, "
            "and exactly 16 batch documents"
        )

    import networkx as nx

    if nx.__version__ != compare_dataset_xml.EXPECTED_NETWORKX:
        raise RuntimeError(f"NetworkX version drift: {nx.__version__}")
    module = compare_dataset_core.load_oracle(args.python_source)
    with tempfile.TemporaryDirectory(prefix="virne_dataset_xml_benchmark_") as text_root:
        root = pathlib.Path(text_root)
        synthetic = root / "synthetic.xml"
        write_large_synthetic(synthetic, 256, 2048)
        parse_rows = [
            ParseRow("brain_parse_single", args.brain_xml, 1),
            ParseRow("brain_parse_batch", args.brain_xml, args.batch_documents),
            ParseRow("synthetic_parse_batch", synthetic, args.batch_documents),
        ]
        results = []
        for row in parse_rows:
            results.extend(
                benchmark_parse_widths(
                    args.harness,
                    row,
                    args.workers,
                    args.warmups,
                    args.repetitions,
                )
            )
        results.append(
            benchmark_preprocess(
                module,
                args.harness,
                "Brain",
                args.brain_xml,
                root / "python_brain.gml",
                root / "cpp_brain.gml",
                args.warmups,
                args.repetitions,
                5,
            )
        )

    if args.performance_gate:
        slower = [row for row in results if row["cpp"]["median_ms"] >= row["python"]["median_ms"]]
        if slower:
            raise RuntimeError(
                "C++ XML timing gate failed: "
                + ", ".join(f"{row['kind']}/w{row['workers']}" for row in slower)
            )

    if args.worker_policy_gate:
        for kind in ("brain_parse_batch", "synthetic_parse_batch"):
            family = [row for row in results if row["kind"] == kind]
            if {row["workers"] for row in family} != set(range(9)):
                raise RuntimeError(f"{kind}: incomplete worker family")
            automatic = next(row for row in family if row["workers"] == 0)
            sequential = next(row for row in family if row["workers"] == 1)
            best_explicit = min(
                (row for row in family if row["workers"] != 0),
                key=lambda row: row["cpp"]["median_ms"],
            )
            auto_ms = automatic["cpp"]["median_ms"]
            if auto_ms >= sequential["cpp"]["median_ms"]:
                raise RuntimeError(f"{kind}: automatic policy did not beat sequential")
            if auto_ms > best_explicit["cpp"]["median_ms"] * 1.15:
                raise RuntimeError(
                    f"{kind}: automatic policy is over 15% behind best explicit"
                )

    print("dataset XML benchmark: PASS")
    for row in results:
        print(
            f"{row['kind']}/w{row['workers']}: "
            f"python={row['python']['median_ms']:.6f} ms "
            f"cpp={row['cpp']['median_ms']:.6f} ms "
            f"speedup={row['speedup']:.3f}x"
        )
    artifact = {
        "affinity_cpus": cpu_affinity(),
        "batch_documents": args.batch_documents,
        "compiler_binary_sha256": hashlib.sha256(args.harness.read_bytes()).hexdigest(),
        "networkx": nx.__version__,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_source_sha256": compare_dataset_core.SOURCE_SHA256,
        "repetitions": args.repetitions,
        "results": results,
        "status": "PASS",
        "warmups": args.warmups,
        "workers": args.workers,
        "performance_gate": args.performance_gate,
        "worker_policy_gate": args.worker_policy_gate,
    }
    if args.json_output:
        args.json_output.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
