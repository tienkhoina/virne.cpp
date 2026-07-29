#!/usr/bin/env python3
"""AST-isolated differential for the pinned VirtualNetworkEvent dataclass."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
import pathlib
import struct
import subprocess
from typing import Any


SOURCE_SHA256 = "970e63f9dac59f60e2ed1786606dc87d3271af062ba1d8d6c67aef8d3c7478e1"
BOUNDARIES = {
    "dynamic_getitem":
        "Python getattr accepts arbitrary dynamic keys; native fixed fields use direct accessors.",
    "dynamic_setitem":
        "Python setattr can add fields and bypass validation; native setters preserve invariants.",
    "numeric_type_equality":
        "Python accepts False/True and 0.0/1.0 as event types; native uses an enum.",
    "unvalidated_python_id":
        "Python does not validate event id; native VirtualEventId is non-negative size_t.",
    "python_nan_time":
        "Python accepts NaN because NaN < 0 is false; native rejects NaN for sortable events.",
    "arbitrary_comparison_protocols":
        "User-defined equality/order side effects remain outside the fixed native scalar domain.",
}


def load_event_class(source: pathlib.Path):
    source = source.resolve()
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"VirtualNetworkRequestSimulator source hash drift: {actual}")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    matches = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "VirtualNetworkEvent"
    ]
    if len(matches) != 1:
        raise RuntimeError("VirtualNetworkEvent class inventory drift")
    isolated = ast.fix_missing_locations(ast.Module(body=matches, type_ignores=[]))
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "dataclass": dataclass,
    }
    exec(compile(isolated, str(source), "exec"), namespace)
    return namespace["VirtualNetworkEvent"]


def double_token(value: float) -> str:
    return "d:" + struct.pack(">d", float(value)).hex()


def error_payload(callable_) -> str:
    try:
        callable_()
    except ValueError as error:
        return "error=" + str(error)
    return "error=none"


def batch_inputs(Event):
    return [
        Event(
            id=index,
            type=1 if index % 2 == 0 else 0,
            v_net_id=index // 2,
            time=float((index * 37) % 11) + (0.25 if index % 3 == 0 else 0.0),
        )
        for index in range(32)
    ]


def events_payload(events) -> str:
    return "[" + ",".join(
        f"{event.id}:{event.type}:{event.v_net_id}:{double_token(event.time)}"
        for event in events
    ) + "]"


def oracle_cases(Event) -> list[tuple[str, str]]:
    ordinary = Event(id=9, type=1, v_net_id=4, time=2.5)
    cases = [(
        "ordinary",
        f"id={ordinary.id};type={ordinary.type};vnet={ordinary.v_net_id};"
        f"time={double_token(ordinary.time)};repr={ordinary!r}",
    )]
    negative_zero = Event(id=0, type=0, v_net_id=0, time=-0.0)
    cases.append((
        "negative_zero",
        f"time={double_token(negative_zero.time)};repr={negative_zero!r}",
    ))
    infinite = Event(id=1, type=1, v_net_id=2, time=float("inf"))
    cases.append((
        "infinity",
        f"time={double_token(infinite.time)};repr={infinite!r}",
    ))
    cases.extend([
        ("invalid_type", error_payload(
            lambda: Event(id=0, type=2, v_net_id=0, time=0.0))),
        ("negative_vnet", error_payload(
            lambda: Event(id=0, type=1, v_net_id=-1, time=0.0))),
        ("negative_time", error_payload(
            lambda: Event(id=0, type=1, v_net_id=0, time=-1.0))),
        ("validation_order", error_payload(
            lambda: Event(id=0, type=7, v_net_id=-1, time=-1.0))),
    ])

    changed = Event(id=5, type=1, v_net_id=9, time=4.0)
    changed["id"] = 11
    changed["type"] = 0
    changed["v_net_id"] = 3
    changed["time"] = 8.5
    cases.append((
        "typed_setters",
        f"id={changed.id};type={changed.type};vnet={changed.v_net_id};"
        f"time={double_token(changed.time)}",
    ))

    batch = batch_inputs(Event)
    payload = events_payload(batch)
    for workers in (0, 1, 2, 8):
        cases.append((f"batch_w{workers}", payload))

    ties = [
        Event(id=0, type=1, v_net_id=0, time=1.0),
        Event(id=1, type=1, v_net_id=1, time=0.0),
        Event(id=2, type=0, v_net_id=0, time=1.0),
        Event(id=3, type=0, v_net_id=1, time=0.0),
    ]
    ties = sorted(ties, key=lambda event: event.time)
    cases.append(("stable_ties", events_payload(ties)))
    cases.append((
        "native_nan_rejected",
        "error=Event time must be non-negative",
    ))
    return cases


def parse_cpp(path: pathlib.Path) -> list[tuple[str, str]]:
    process = subprocess.run(
        [str(path)], check=False, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"VirtualNetworkEvent harness failed: {process.stderr.strip()}")
    result = []
    for line in process.stdout.splitlines():
        fields = line.split("|")
        if len(fields) != 3 or not fields[0].startswith("case=") or fields[1] != "ok":
            raise RuntimeError(f"malformed VirtualNetworkEvent line: {line!r}")
        result.append((fields[0][5:], bytes.fromhex(fields[2]).decode("utf-8")))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    Event = load_event_class(args.source)
    expected = oracle_cases(Event)
    actual = parse_cpp(args.harness.resolve())
    if actual != expected:
        for got, wanted in zip(actual, expected):
            if got != wanted:
                raise RuntimeError(
                    f"VirtualNetworkEvent mismatch {wanted[0]}: C++={got!r}, Python={wanted!r}"
                )
        raise RuntimeError(
            f"VirtualNetworkEvent inventory mismatch: C++={actual!r}, Python={expected!r}"
        )

    payload = {
        "source_sha256": SOURCE_SHA256,
        "shared_cases": [name for name, _ in expected[:-1]],
        "native_extension_cases": [expected[-1][0]],
        "python_only_boundaries": BOUNDARIES,
        "case_count": len(expected) + len(BOUNDARIES),
        "result": "PASS",
    }
    if args.output:
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        f"VirtualNetworkEvent differential: PASS ({len(expected) - 1} shared + "
        f"1 native + {len(BOUNDARIES)} boundaries)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
