#!/usr/bin/env python3
"""C++-only calibrated worker sweep for the dataset XML parse batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import tempfile

import benchmark_dataset_xml


def affinity_cpus() -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def sweep_family(
    harness: pathlib.Path,
    corpus: str,
    source: pathlib.Path,
    documents: int,
    workers: list[int],
    warmups: int,
    repetitions: int,
) -> list[dict]:
    row = benchmark_dataset_xml.ParseRow(corpus, source, documents)
    samples = {width: [] for width in workers}
    semantic: tuple[int, int] | None = None
    for sample in range(warmups + repetitions):
        offset = sample % len(workers)
        ordered = workers[offset:] + workers[:offset]
        if (sample // len(workers)) % 2:
            ordered.reverse()
        for width in ordered:
            result = benchmark_dataset_xml.cpp_parse_benchmark(
                harness, row, width
            )
            current = (result["checksum"], result["output_bytes"])
            if semantic is not None and current != semantic:
                raise RuntimeError(
                    f"{corpus}/n{documents}/w{width}: semantic drift"
                )
            semantic = current
            if sample >= warmups:
                samples[width].append(result["elapsed_ns"])

    results = []
    for width in workers:
        results.append(
            {
                "corpus": corpus,
                "documents": documents,
                "workers": width,
                "checksum": semantic[0],
                "output_bytes": semantic[1],
                "cpp": benchmark_dataset_xml.timing_summary(samples[width]),
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", required=True, type=pathlib.Path)
    parser.add_argument("--brain-xml", required=True, type=pathlib.Path)
    parser.add_argument(
        "--documents", nargs="+", type=int, default=[2, 4, 8, 16, 32, 64]
    )
    parser.add_argument(
        "--workers", nargs="+", type=int, default=[1, 0, 2, 3, 4, 5, 6, 7, 8]
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=31)
    parser.add_argument("--policy-gate", action="store_true")
    parser.add_argument("--json-output", type=pathlib.Path)
    args = parser.parse_args()

    if (
        not args.documents
        or any(count < 2 for count in args.documents)
        or len(set(args.documents)) != len(args.documents)
    ):
        raise SystemExit("documents must be unique counts >=2")
    if (
        set(args.workers) != set(range(9))
        or len(args.workers) != 9
        or args.warmups < 0
        or args.repetitions <= 0
    ):
        raise SystemExit("workers must be exactly 0..8 with valid sample counts")
    if args.policy_gate and (
        args.documents != [2, 4, 8, 16, 32, 64]
        or args.warmups < 5
        or args.repetitions < 31
    ):
        raise SystemExit(
            "policy gate requires documents 2 4 8 16 32 64 and at least 5/31"
        )

    with tempfile.TemporaryDirectory(prefix="virne_dataset_xml_sweep_") as root_text:
        synthetic = pathlib.Path(root_text) / "synthetic.xml"
        benchmark_dataset_xml.write_large_synthetic(synthetic, 256, 2048)
        corpora = [("brain", args.brain_xml), ("synthetic", synthetic)]
        results = []
        for corpus, source in corpora:
            for documents in args.documents:
                results.extend(
                    sweep_family(
                        args.harness,
                        corpus,
                        source,
                        documents,
                        args.workers,
                        args.warmups,
                        args.repetitions,
                    )
                )

    if args.policy_gate:
        for corpus in ("brain", "synthetic"):
            for documents in args.documents:
                family = [
                    row
                    for row in results
                    if row["corpus"] == corpus and row["documents"] == documents
                ]
                automatic = next(row for row in family if row["workers"] == 0)
                sequential = next(row for row in family if row["workers"] == 1)
                best = min(
                    (row for row in family if row["workers"] != 0),
                    key=lambda row: row["cpp"]["median_ms"],
                )
                auto_ms = automatic["cpp"]["median_ms"]
                if auto_ms >= sequential["cpp"]["median_ms"]:
                    raise RuntimeError(
                        f"{corpus}/n{documents}: auto did not beat sequential"
                    )
                if auto_ms > best["cpp"]["median_ms"] * 1.25:
                    raise RuntimeError(
                        f"{corpus}/n{documents}: auto over 25% behind best explicit"
                    )

    artifact = {
        "affinity_cpus": affinity_cpus(),
        "documents": args.documents,
        "harness_sha256": hashlib.sha256(args.harness.read_bytes()).hexdigest(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "policy_gate": args.policy_gate,
        "repetitions": args.repetitions,
        "results": results,
        "status": "PASS",
        "warmups": args.warmups,
        "workers": args.workers,
    }
    print("dataset XML worker sweep: PASS")
    for corpus in ("brain", "synthetic"):
        for documents in args.documents:
            family = [
                row
                for row in results
                if row["corpus"] == corpus and row["documents"] == documents
            ]
            automatic = next(row for row in family if row["workers"] == 0)
            best = min(
                (row for row in family if row["workers"] != 0),
                key=lambda row: row["cpp"]["median_ms"],
            )
            print(
                f"{corpus}/n{documents}: auto={automatic['cpp']['median_ms']:.6f} ms "
                f"best=w{best['workers']} {best['cpp']['median_ms']:.6f} ms"
            )
    if args.json_output:
        args.json_output.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
