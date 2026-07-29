#!/usr/bin/env python3
"""AST-isolated differential for the pinned Python LinkRank leaf."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import struct
import subprocess
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


SOURCE_SHA256 = "c4d52bd389a004a91d8fcc7e0827bb9620252d9619d4f9121d11ba4e17ff0880"
WORKERS = (1, 2, 8)

BOUNDARIES = {
    "base_network_typo":
        "Pinned FFDLinkRank calls network.get_attrs('link', 'resource'), but the pinned BaseNetwork exposes get_link_attrs(...). A dedicated case preserves that AttributeError; intended FFD arithmetic is isolated with a narrow fake that supplies get_attrs.",
    "resource_selection":
        "The intended fake returns link-resource rows in already-resolved registry order. Native code resolves dynamic resource selection once during preparation and ranks by numeric IDs/direct slots in hot loops.",
    "numpy_reduction":
        "Int-only rows use NumPy int64 modular reduction before conversion to binary64; mixed rows use binary64 reduction. Every successful score is compared by its big-endian IEEE-754 bits.",
    "native_workers":
        "Python exposes no worker parameter. Repeated mixed cases compare native caller widths 1/2/8 while preserving identical ranking order and score bits.",
}


def load_link_rank(source: pathlib.Path):
    """Load only the three pinned ranking classes, without importing virne."""
    source = source.resolve()
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"LinkRank source hash drift: {actual}")

    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    classes = [
        node for node in tree.body if isinstance(node, ast.ClassDef)
    ]
    names = [node.name for node in classes]
    expected_names = ["LinkRank", "OrderLinkRank", "FFDLinkRank"]
    if names != expected_names:
        raise RuntimeError(
            f"LinkRank class inventory drift: expected {expected_names}, got {names}")

    isolated = ast.fix_missing_locations(
        ast.Module(body=classes, type_ignores=[]))
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "ABC": ABC,
        "Any": Any,
        "BaseNetwork": Any,
        "Dict": Dict,
        "abstractmethod": abstractmethod,
        "np": np,
    }
    exec(compile(isolated, str(source), "exec"), namespace)
    return namespace["OrderLinkRank"], namespace["FFDLinkRank"]


class IntendedNetwork:
    """Narrow fake exposing the API spelling used by pinned FFDLinkRank."""

    def __init__(self, links, resource_rows):
        self.links = list(links)
        self._resource_rows = resource_rows
        self._resource_token = object()
        self.get_attrs_calls = 0
        self.get_data_calls = 0

    def get_attrs(self, owner, kind):
        if owner != "link" or kind != "resource":
            raise RuntimeError(f"unexpected attribute selection: {owner}/{kind}")
        self.get_attrs_calls += 1
        return self._resource_token

    def get_link_attrs_data(self, attrs):
        if attrs is not self._resource_token:
            raise RuntimeError("FFDLinkRank did not forward get_attrs result")
        self.get_data_calls += 1
        return self._resource_rows


class CurrentBaseNetworkShape:
    """The relevant pinned BaseNetwork surface: no generic get_attrs method."""

    def __init__(self):
        self.links = [(0, 1)]

    @staticmethod
    def get_link_attrs(types=None, names=None):
        del types, names
        return []

    @staticmethod
    def get_link_attrs_data(attrs):
        del attrs
        return [[1]]


def double_bits(value) -> str:
    return struct.pack(">d", float(value)).hex()


def ranking_payload(ranking=None, error="none") -> str:
    entries = []
    if ranking is not None:
        for link, value in ranking.items():
            if (not isinstance(link, tuple) or len(link) != 2 or
                    not all(isinstance(endpoint, int) for endpoint in link)):
                raise RuntimeError(f"unsupported oracle link key: {link!r}")
            entries.append([link[0], link[1], double_bits(value)])
    return json.dumps(
        {"error": error, "ranking": entries},
        sort_keys=True,
        separators=(",", ":"),
    )


def run_success(ranker, network, *, sort):
    ranking = ranker.rank(network, sort=sort)
    if isinstance(network, IntendedNetwork):
        if network.get_attrs_calls != 1 or network.get_data_calls != 1:
            raise RuntimeError(
                "FFDLinkRank must select and fetch resource rows exactly once")
    return ranking_payload(ranking)


def run_expected_error(ranker, network, *, expected_type, error):
    try:
        ranker.rank(network, sort=True)
    except expected_type:
        if isinstance(network, IntendedNetwork):
            if network.get_attrs_calls != 1 or network.get_data_calls != 1:
                raise RuntimeError(
                    "FFDLinkRank error path changed attribute access order")
        return ranking_payload(error=error)
    except Exception as exception:
        raise RuntimeError(
            f"expected {expected_type.__name__}, got "
            f"{type(exception).__name__}") from exception
    raise RuntimeError(f"expected {expected_type.__name__}, no error raised")


def python_cases(OrderLinkRank, FFDLinkRank):
    # Graph insertion [(2, 4), (0, 3), (1, 4), (0, 1)] normalizes to
    # this frozen public undirected edge order: group by the first dense
    # vertex, retaining adjacency insertion order inside each group.
    order_links = [(0, 3), (0, 1), (1, 4), (2, 4)]
    cases = {
        "order_unsorted": ranking_payload(
            OrderLinkRank().rank(
                IntendedNetwork(order_links, []), sort=False)),
        "order_sorted": ranking_payload(
            OrderLinkRank().rank(
                IntendedNetwork(order_links, []), sort=True)),
    }

    cases["ffd_int"] = run_success(
        FFDLinkRank(),
        IntendedNetwork(
            [(0, 1), (1, 2), (2, 3)],
            [[7, -3, 10], [2, 8, -4], [1, 0, 2]]),
        sort=False,
    )

    mixed_links = [(10, 11), (11, 12), (12, 13), (13, 14)]
    mixed_rows = [
        [0.5, -2.0, 8, -0.0],
        [0.25, 4, -1.5, 0.0],
        [1, -0.5, 0.25, -2.0],
    ]
    for workers in WORKERS:
        cases[f"ffd_mixed_workers_{workers}"] = run_success(
            FFDLinkRank(),
            IntendedNetwork(mixed_links, mixed_rows),
            sort=True,
        )

    cases["ffd_stable_ties"] = run_success(
        FFDLinkRank(),
        IntendedNetwork(
            [(20, 21), (21, 22), (22, 23), (23, 24)],
            [[4, 2, 4, 3], [1, 3, 0, 2]]),
        sort=True,
    )

    cases["ffd_zero_edge"] = run_success(
        FFDLinkRank(),
        IntendedNetwork([], [[], []]),
        sort=True,
    )

    int64_max = (1 << 63) - 1
    int64_min = -(1 << 63)
    cases["ffd_int64_wrap"] = run_success(
        FFDLinkRank(),
        IntendedNetwork(
            [(30, 31), (31, 32)],
            [[int64_max, int64_min], [1, -1]]),
        sort=False,
    )

    cases["ffd_empty_resources_error"] = run_expected_error(
        FFDLinkRank(),
        IntendedNetwork([(40, 41)], []),
        expected_type=IndexError,
        error="empty_resources",
    )

    cases["ffd_ragged_error"] = run_expected_error(
        FFDLinkRank(),
        IntendedNetwork(
            [(50, 51), (51, 52), (52, 53)],
            [[1, 2, 3], [4, 5]]),
        expected_type=ValueError,
        error="ragged_resources",
    )

    cases["ffd_base_network_typo"] = run_expected_error(
        FFDLinkRank(),
        CurrentBaseNetworkShape(),
        expected_type=AttributeError,
        error="missing_get_attrs",
    )
    return cases


def cpp_cases(harness: pathlib.Path):
    process = subprocess.run(
        [str(harness), "differential"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"LinkRank harness failed: {process.stderr.strip()}")
    result = {}
    for line in process.stdout.splitlines():
        name, payload = line.split("\t", 1)
        if name in result:
            raise RuntimeError(f"duplicate native case: {name}")
        # Reject semantically equivalent but non-canonical native JSON.
        decoded = json.loads(payload)
        canonical = json.dumps(
            decoded, sort_keys=True, separators=(",", ":"))
        if payload != canonical:
            raise RuntimeError(f"non-canonical native JSON for case {name}")
        result[name] = payload
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    OrderLinkRank, FFDLinkRank = load_link_rank(args.source)
    expected = python_cases(OrderLinkRank, FFDLinkRank)
    actual = cpp_cases(args.harness)
    if actual.keys() != expected.keys():
        raise RuntimeError(
            f"case inventory drift: Python={list(expected)}, C++={list(actual)}")

    worker_payloads = [
        actual[f"ffd_mixed_workers_{workers}"] for workers in WORKERS
    ]
    if len(set(worker_payloads)) != 1:
        raise RuntimeError("native LinkRank worker output is not deterministic")

    mismatches = {
        name: {"python": expected[name], "cpp": actual[name]}
        for name in expected if expected[name] != actual[name]
    }
    if mismatches:
        raise RuntimeError(
            "LinkRank differential mismatch:\n" +
            json.dumps(mismatches, indent=2, sort_keys=True))

    canonical_cases = json.dumps(
        expected, sort_keys=True, separators=(",", ":")).encode("utf-8")
    report = {
        "component": "solver.rank.LinkRank",
        "source_sha256": SOURCE_SHA256.upper(),
        "case_payload_sha256":
            hashlib.sha256(canonical_cases).hexdigest().upper(),
        "shared_case_count": len(expected),
        "native_workers": list(WORKERS),
        "boundaries": BOUNDARIES,
        "payload_encoding":
            "canonical JSON; ranking entries are [u,v,big-endian IEEE-754 binary64 hex]",
        "status": "PASS",
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
