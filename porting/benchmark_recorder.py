#!/usr/bin/env python3
"""One checksum-gated non-ML Recorder benchmark; freeze after acceptance.

The Python path AST-isolates the exact pinned ``Recorder`` class and reuses
the frozen exact ``Counter`` loader. Narrow typed network/Solution boundaries
provide only the protocols those two original classes consume. Filesystem
setup, fixture construction, prepared bindings, checksum work, and process
startup are outside the measured prepared-arrival plus deep-history snapshot.
"""

from __future__ import annotations

import argparse
import ast
import collections
import copy
import csv
import hashlib
import json
import os
import pathlib
import platform
import statistics
import struct
import subprocess
import tempfile
import time
import types
from typing import Any, Dict

import compare_counter as counter_oracle


SOURCE_SHA256 = "41e66702f29644e6f8a374ef909a1c41f87982558863be17b6fdf72ceb971f4a"
COUNTER_SOURCE_SHA256 = (
    "574745f86e99cd656cb0165330e3196b4f5ebf4eaa0687b076b8d9602db4d637")
WORKERS = (1, 2, 8)
WARMUPS = 1
REPETITIONS = 3
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


class FakePandas:
    class DataFrame:
        pass


class FakeOmegaConf:
    @staticmethod
    def create(value):
        return value


def load_recorder(source: pathlib.Path):
    source = source.resolve()
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"Recorder source hash drift: {actual}")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    matches = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Recorder"
    ]
    if len(matches) != 1:
        raise RuntimeError("Recorder class inventory drift")
    isolated = ast.fix_missing_locations(
        ast.Module(body=matches, type_ignores=[]))
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "Any": Any,
        "Dict": Dict,
        "OmegaConf": FakeOmegaConf,
        "OrderedDict": collections.OrderedDict,
        "PhysicalNetwork": Any,
        "Solution": Any,
        "VirtualNetwork": Any,
        "copy": copy,
        "csv": csv,
        "defaultdict": collections.defaultdict,
        "np": counter_oracle.np,
        "os": os,
        "pd": FakePandas,
        "solve": lambda *args, **kwargs: None,
        "solver": types.SimpleNamespace(),
    }
    exec(compile(isolated, str(source), "exec"), namespace)
    return namespace["Recorder"]


def make_config(root: pathlib.Path, run_id: str):
    return types.SimpleNamespace(
        experiment=types.SimpleNamespace(
            save_root_dir=str(root), run_id=run_id),
        recorder=types.SimpleNamespace(
            if_temp_save_records=False, record_dir_name="records"),
        solver=types.SimpleNamespace(solver_name="recorder-benchmark"),
    )


def make_networks(item_count: int):
    physical_nodes = {
        index: {"cpu": float((index * 23) % 107) + 10.0}
        for index in range(item_count)
    }
    physical_links = {
        (index, index + 1): {
            "bandwidth": float((index * 29) % 109) + 5.0}
        for index in range(item_count - 1)
    }
    virtual_nodes = {
        index: {"cpu": float((index * 17) % 101) + 1.25}
        for index in range(item_count)
    }
    virtual_links = {
        (index, index + 1): {
            "bandwidth": float((index * 19) % 103) + 0.5}
        for index in range(item_count - 1)
    }
    return (
        counter_oracle.FakeNetwork(
            physical_nodes, physical_links, lifetime_marker=None),
        counter_oracle.FakeNetwork(
            virtual_nodes, virtual_links, lifetime_marker=3.0),
    )


def make_solution(item_count: int):
    node_slots = {index: index for index in range(item_count)}
    link_paths = {
        (index, index + 1): [(index, index + 1)]
        for index in range(item_count - 1)
    }
    link_paths_info = {
        ((index, index + 1), (index, index + 1)): {
            "bandwidth": float((index * 19) % 103) + 0.5}
        for index in range(item_count - 1)
    }
    return counter_oracle.FakeSolution({
        "v_net_id": 7,
        "v_net_lifetime": 3.0,
        "v_net_arrival_time": 1.25,
        "v_net_num_nodes": item_count,
        "v_net_num_edges": item_count - 1,
        "result": True,
        "node_slots": node_slots,
        "link_paths": link_paths,
        "node_slots_info": {},
        "link_paths_info": link_paths_info,
        "v_net_cost": 0.0,
        "v_net_revenue": 0.0,
        "v_net_demand": 0.0,
        "v_net_node_demand": 0.0,
        "v_net_link_demand": 0.0,
        "v_net_node_revenue": 0.0,
        "v_net_link_revenue": 0.0,
        "v_net_node_cost": 0.0,
        "v_net_link_cost": 0.0,
        "v_net_path_cost": 0.0,
        "v_net_r2c_ratio": 0.0,
        "v_net_time_cost": 0.0,
        "v_net_time_revenue": 0.0,
        "v_net_time_rc_ratio": 0.0,
        "description": "benchmark",
        "v_net_total_hard_constraint_violation": 0.0,
        "v_net_single_step_constraint_offset": {},
        "v_net_constraint_offsets": {},
        "v_net_constraint_violations": {},
        "v_net_single_step_violation_list": [],
        "v_net_single_step_hard_constraint_offset": float("-inf"),
        "v_net_max_single_step_hard_constraint_violation": float("-inf"),
        "place_result": True,
        "route_result": True,
        "early_rejection": False,
        "revoke_times": 0,
        "selected_actions": [],
        "num_interactions": 0,
        "v_net_reward": 0.0,
    })


def double_token(value) -> str:
    return "d:" + struct.pack(">d", float(value)).hex()


def optional_size(record: dict[str, Any], name: str) -> str:
    return str(record[name]) if name in record else "none"


def link_token(link) -> str:
    return f"({link[0]},{link[1]})"


def snapshot_payload(recorder, item_count: int) -> str:
    if len(recorder.memory) != 1:
        raise RuntimeError("Python Recorder history invariant failed")
    record = recorder.memory[0]
    if record["node_slots_info"] or record["v_net_single_step_constraint_offset"]:
        raise RuntimeError("Python Recorder empty-table invariant failed")
    if record["v_net_constraint_offsets"] or record["v_net_constraint_violations"]:
        raise RuntimeError("Python Recorder empty constraint invariant failed")

    output: list[str] = []
    output.append(
        f"state:event={record['event_id']},{record['event_type']};"
        f"counts={record['v_net_count']},{record['success_count']},"
        f"{record['inservice_count']};totals=")
    for name in (
        "total_revenue", "total_cost", "total_time_revenue",
        "total_time_cost", "long_term_r2c_ratio",
        "long_term_time_r2c_ratio",
    ):
        output.append(double_token(record[name]) + ",")
    output[-1] = output[-1][:-1]
    output.append(
        f";physical={record['num_running_p_net_nodes']},"
        f"{double_token(record['p_net_available_resource'])},"
        f"{double_token(record['p_net_node_available_resource'])},"
        f"{double_token(record['p_net_link_available_resource'])},"
        f"{double_token(record['p_net_node_resource_utilization'])},"
        f"{double_token(record['p_net_link_resource_utilization'])}")
    output.append(
        f";solution:meta={record['v_net_id']},"
        f"{double_token(record['v_net_lifetime'])},"
        f"{double_token(record['v_net_arrival_time'])},"
        f"{record['v_net_num_nodes']},{record['v_net_num_edges']};"
        f"flags={int(record['result'])},{int(record['place_result'])},"
        f"{int(record['route_result'])},{int(record['early_rejection'])}")

    output.append(f";node_slots={len(record['node_slots'])}:")
    for virtual_node, physical_node in record["node_slots"].items():
        output.append(f"{virtual_node}>{physical_node},")
    output.append(f";link_paths={len(record['link_paths'])}:")
    for virtual_link, path in record["link_paths"].items():
        output.append(link_token(virtual_link) + ">[")
        for physical_link in path:
            output.append(link_token(physical_link) + ",")
        output.append("],")
    output.append(
        f";node_slots_info=0;link_paths_info="
        f"{len(record['link_paths_info'])}:")
    for (virtual_link, physical_link), values in (
            record["link_paths_info"].items()):
        output.append(
            link_token(virtual_link) + "@" + link_token(physical_link) + "{")
        output.append("0=" + double_token(values["bandwidth"]) + ",},")

    output.append(";metrics=")
    for name in (
        "v_net_cost", "v_net_revenue", "v_net_demand",
        "v_net_node_demand", "v_net_link_demand", "v_net_node_revenue",
        "v_net_link_revenue", "v_net_node_cost", "v_net_link_cost",
        "v_net_path_cost", "v_net_r2c_ratio", "v_net_time_cost",
        "v_net_time_revenue", "v_net_time_rc_ratio",
        "v_net_total_hard_constraint_violation",
    ):
        output.append(double_token(record[name]) + ",")
    output.append(
        f";description={len(record['description'])}:{record['description']}"
        ";constraint_tables=0,0,0,0,0,0;violation_list=")
    violations = record["v_net_single_step_violation_list"]
    output.append(f"{len(violations)}:")
    for value in violations:
        output.append(double_token(value) + ",")
    output.append(
        ";hard_offsets="
        f"{double_token(record['v_net_single_step_hard_constraint_offset'])},"
        f"{double_token(record['v_net_max_single_step_hard_constraint_violation'])}"
        f";runtime={record['revoke_times']},{record['num_interactions']},"
        f"{double_token(record['v_net_reward'])};actions=")
    output.append(f"{len(record['selected_actions'])}:")
    for action in record["selected_actions"]:
        output.append(f"{action},")
    output.append(
        f";optional={optional_size(record, 'num_placed_nodes')},"
        f"{optional_size(record, 'num_routed_links')},"
        f"{optional_size(record, 'num_attempt_times')}")
    running = recorder.get_running_p_net_nodes()
    output.append(f";extra=0;history=1;running={len(running)}:")
    for node in running:
        output.append(f"{node},")
    payload = "".join(output)
    if len(running) != item_count:
        raise RuntimeError("Python Recorder running-node gate failed")
    return payload


def fingerprint(value: str) -> tuple[int, int]:
    encoded = value.encode("utf-8")
    checksum = FNV_OFFSET
    for byte in encoded:
        checksum = ((checksum ^ byte) * FNV_PRIME) & MASK64
    return checksum, len(encoded)


def make_scenario(
    Recorder,
    Counter,
    root: pathlib.Path,
    run_id: str,
    physical_network,
    virtual_network,
    item_count: int,
):
    counter = counter_oracle.make_counter(
        Counter,
        [counter_oracle.FakeAttribute("cpu")],
        [counter_oracle.FakeAttribute("bandwidth")],
    )
    recorder = Recorder(counter, make_config(root, run_id))
    recorder.count_init_p_net_info(physical_network)
    recorder.update_state({"event_id": 0, "event_type": 1})
    return recorder, make_solution(item_count), virtual_network


def time_python(
    Recorder,
    Counter,
    root: pathlib.Path,
    physical_network,
    virtual_network,
    item_count: int,
) -> tuple[int, str]:
    samples: list[int] = []
    expected_payload: str | None = None
    for index in range(WARMUPS + REPETITIONS):
        recorder, solution, network = make_scenario(
            Recorder,
            Counter,
            root,
            f"python-{index}",
            physical_network,
            virtual_network,
            item_count,
        )
        started = time.perf_counter_ns()
        record = recorder.count(network, physical_network, solution)
        recorder.add_record(record)
        stopped = time.perf_counter_ns()
        payload = snapshot_payload(recorder, item_count)
        if expected_payload is None:
            expected_payload = payload
        elif payload != expected_payload:
            raise RuntimeError("Python Recorder output changed between samples")
        if index >= WARMUPS:
            samples.append(stopped - started)
    assert expected_payload is not None
    return int(statistics.median(samples)), expected_payload


def parse_cpp(
    executable: pathlib.Path,
    item_count: int,
    workers: int,
) -> dict[str, int]:
    process = subprocess.run(
        [str(executable), str(item_count), str(workers)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"Recorder benchmark failed at workers={workers}: "
            f"{process.stderr.strip()}")
    fields: dict[str, str] = {}
    for line in process.stdout.splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in fields:
            raise RuntimeError(f"malformed Recorder benchmark line: {line!r}")
        fields[key] = value
    expected = {
        "protocol": "1",
        "kind": "recorder_prepared_arrival_history_snapshot",
        "semantics": "exact_recorder_arrival_and_deep_snapshot_v1",
        "item_count": str(item_count),
        "workers": str(workers),
        "type_tag": "recorder_fixed_fields_raw64_ordered_maps_v1",
        "status": "PASS",
    }
    for key, value in expected.items():
        if fields.get(key) != value:
            raise RuntimeError(
                f"Recorder benchmark field {key}: "
                f"{fields.get(key)!r} != {value!r}")
    numeric = {
        "elapsed_ns", "checksum", "output_bytes", "entry_count",
        "history_size", "running_node_count",
    }
    if set(fields) != set(expected) | numeric:
        raise RuntimeError(
            f"Recorder benchmark field inventory drift: {sorted(fields)}")
    return {key: int(fields[key]) for key in numeric}


def validate_native(
    result: dict[str, int],
    expected_checksum: int,
    expected_size: int,
    expected_entries: int,
    item_count: int,
    label: str,
) -> None:
    expected = {
        "checksum": expected_checksum,
        "output_bytes": expected_size,
        "entry_count": expected_entries,
        "history_size": 1,
        "running_node_count": item_count,
    }
    mismatches = {
        key: {"expected": value, "actual": result[key]}
        for key, value in expected.items()
        if result[key] != value
    }
    if mismatches:
        raise RuntimeError(f"{label} Recorder checksum mismatch: {mismatches}")


def time_cpp(
    executable: pathlib.Path,
    item_count: int,
    workers: int,
    expected_checksum: int,
    expected_size: int,
    expected_entries: int,
) -> int:
    for _ in range(WARMUPS):
        validate_native(
            parse_cpp(executable, item_count, workers),
            expected_checksum,
            expected_size,
            expected_entries,
            item_count,
            f"workers={workers} warm-up",
        )
    samples: list[int] = []
    for repetition in range(REPETITIONS):
        result = parse_cpp(executable, item_count, workers)
        validate_native(
            result,
            expected_checksum,
            expected_size,
            expected_entries,
            item_count,
            f"workers={workers} repetition={repetition}",
        )
        samples.append(result["elapsed_ns"])
    return int(statistics.median(samples))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--counter-source", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark", type=pathlib.Path, required=True)
    parser.add_argument("--item-count", type=int, default=8192)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.item_count < 128:
        raise ValueError("item-count must be at least 128")
    if counter_oracle.SOURCE_SHA256 != COUNTER_SOURCE_SHA256:
        raise RuntimeError("compare_counter oracle hash constant drift")

    Recorder = load_recorder(args.source)
    Counter = counter_oracle.load_counter(args.counter_source)
    physical_network, virtual_network = make_networks(args.item_count)
    with tempfile.TemporaryDirectory(prefix="virne_recorder_benchmark_") as root:
        python_ns, payload = time_python(
            Recorder,
            Counter,
            pathlib.Path(root),
            physical_network,
            virtual_network,
            args.item_count,
        )
    checksum, output_bytes = fingerprint(payload)
    entry_count = 4 * args.item_count - 1

    rows = []
    for workers in WORKERS:
        cpp_ns = time_cpp(
            args.benchmark,
            args.item_count,
            workers,
            checksum,
            output_bytes,
            entry_count,
        )
        speedup = python_ns / cpp_ns
        if speedup <= 1.0:
            raise RuntimeError(
                f"Recorder C++ workers={workers} did not beat Python: "
                f"{speedup:.3f}x")
        rows.append({
            "workers": workers,
            "cpp_median_ns": cpp_ns,
            "speedup_vs_python": speedup,
        })

    report = {
        "component": "core.Recorder",
        "case": "prepared_arrival_plus_deep_history_snapshot",
        "source_sha256": SOURCE_SHA256.upper(),
        "counter_source_sha256": COUNTER_SOURCE_SHA256.upper(),
        "benchmark_sha256": hashlib.sha256(
            args.benchmark.read_bytes()).hexdigest().upper(),
        "oracle": {
            "loader": "exact Recorder and Counter single-class AST isolation",
            "fake_boundary": (
                "typed network/Solution protocols only; no Recorder/Counter "
                "logic, Pandas, RL, ML, solver, or system emulation"
            ),
            "numpy_version": counter_oracle.np.__version__,
        },
        "protocol": {
            "item_count": args.item_count,
            "entry_count": entry_count,
            "workers": list(WORKERS),
            "warmups": WARMUPS,
            "repetitions": REPETITIONS,
            "fixture_prepare_and_io_excluded": True,
            "process_startup_excluded": True,
            "fingerprint_excluded": True,
            "native_uses_prepared_counter_ids": True,
            "timed_operation": "arrival count plus deep history snapshot",
            "result_gate": (
                "all fixed state/Solution fields, ordered mappings, history, "
                "membership, raw64 doubles, FNV checksum, and output bytes"
            ),
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
