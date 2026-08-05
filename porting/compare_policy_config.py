#!/usr/bin/env python3
"""AST-isolated differential gate for typed policy configuration."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import statistics
import struct
import subprocess
import time
from types import SimpleNamespace
from typing import Any, Dict


FIXTURE_VERSION = 1
BENCHMARK_CONFIGURATION_COUNT = 65536
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1
MAX_REPORTED_MISMATCHES = 48
EXPECTED_API_SHA256 = {
    "PolicyBuilder.get_feature_dim_config": (
        "da9fe38abe293fd7e0dcaba4f359527b93a830f061551c3ff497680829714aae"
    ),
    "PolicyBuilder.get_general_nn_config": (
        "9cf3c58863c14f180aaca199379dd1e56b0a3f1845319fca14875fedf4752578"
    ),
    "get_p_net_x_dim": (
        "f7f512f9dfb461f24d86309e90e94e4078169e62933db40fa3e81ff640c9de3c"
    ),
    "get_p_net_edge_dim": (
        "3e3668d9065c788f93475b2277c8e5797ce3740cf8744229eef4d2aa7450da1d"
    ),
    "get_v_node_x_dim": (
        "9777dfb9f0847b2f554a5b86963c119b86642665648f6ad721246e991e40ba82"
    ),
    "get_v_net_x_dim": (
        "042ac8fcbc1687c9e3a01e9484d4c825c7c0fcd71477941efa18bbe15820a924"
    ),
    "get_v_net_edge_dim": (
        "e38e17e131c3a160d75d00a5a437eeedf3a0f74d41609b11070d60eab1670ff9"
    ),
}
HELPER_NAMES = (
    "get_p_net_x_dim",
    "get_p_net_edge_dim",
    "get_v_node_x_dim",
    "get_v_net_x_dim",
    "get_v_net_edge_dim",
)


def _source_path() -> pathlib.Path:
    return (
        pathlib.Path(__file__).resolve().parents[2]
        / "virne"
        / "virne"
        / "solver"
        / "learning"
        / "rl_core"
        / "policy_builder.py"
    )


def _segment(
    source_text: str,
    node: ast.AST,
    *,
    label: str,
) -> tuple[str, str]:
    value = ast.get_source_segment(source_text, node)
    if value is None:
        raise RuntimeError(f"cannot isolate {label} source")
    return value, hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_oracle() -> tuple[dict[str, Any], dict[str, Any]]:
    source_path = _source_path()
    source_text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source_text, source_path)
    policy_classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PolicyBuilder"
    ]
    if len(policy_classes) != 1:
        raise RuntimeError("expected exactly one top-level PolicyBuilder")

    class_methods = {
        node.name: node
        for node in policy_classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in ("get_feature_dim_config", "get_general_nn_config")
    }
    if set(class_methods) != {
        "get_feature_dim_config", "get_general_nn_config"
    }:
        raise RuntimeError("PolicyBuilder configuration API inventory changed")

    top_level = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in HELPER_NAMES
    }
    if set(top_level) != set(HELPER_NAMES):
        raise RuntimeError("policy dimension helper inventory changed")

    selected_methods = [
        class_methods["get_feature_dim_config"],
        class_methods["get_general_nn_config"],
    ]
    selected_helpers = [top_level[name] for name in HELPER_NAMES]
    api_hashes: dict[str, str] = {}
    segments: list[str] = []
    for method in selected_methods:
        label = f"PolicyBuilder.{method.name}"
        source, digest = _segment(source_text, method, label=label)
        expected = EXPECTED_API_SHA256[label]
        if digest != expected:
            raise RuntimeError(
                f"{label} oracle SHA256 changed: expected {expected}, got {digest}"
            )
        api_hashes[label] = digest
        segments.append(source)
    for function in selected_helpers:
        label = function.name
        source, digest = _segment(source_text, function, label=label)
        expected = EXPECTED_API_SHA256[label]
        if digest != expected:
            raise RuntimeError(
                f"{label} oracle SHA256 changed: expected {expected}, got {digest}"
            )
        api_hashes[label] = digest
        segments.append(source)

    class_arguments: dict[str, Any] = {
        "name": "PolicyBuilder",
        "bases": [],
        "keywords": [],
        "body": selected_methods,
        "decorator_list": [],
    }
    if "type_params" in ast.ClassDef._fields:
        class_arguments["type_params"] = []
    isolated_class = ast.ClassDef(**class_arguments)
    isolated = ast.Module(
        body=[*selected_helpers, isolated_class],
        type_ignores=[],
    )
    ast.fix_missing_locations(isolated)
    namespace: dict[str, Any] = {"Any": Any, "Dict": Dict}
    exec(compile(isolated, str(source_path), "exec"), namespace)
    combined_hash = hashlib.sha256(
        "\n\n".join(segments).encode("utf-8")
    ).hexdigest()
    metadata = {
        "path": str(source_path),
        "file_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "api_sha256": api_hashes,
        "combined_api_sha256": combined_hash,
    }
    return namespace, metadata


ORACLE, SOURCE_METADATA = load_oracle()
POLICY_BUILDER = ORACLE["PolicyBuilder"]
ORACLE_BACKEND = "ast_isolated_policy_builder_seven_apis"


def make_config(
    p_node_attrs: int,
    p_link_attrs: int,
    v_node_attrs: int,
    v_link_attrs: int,
    p_nodes: int,
    flag_mask: int = 15,
    embedding_dim: int = 128,
    dropout_prob: float = 0.0,
    batch_norm: bool = False,
) -> SimpleNamespace:
    feature_constructor = SimpleNamespace(
        num_extracted_p_node_attrs=p_node_attrs,
        num_extracted_p_link_attrs=p_link_attrs,
        num_extracted_v_node_attrs=v_node_attrs,
        num_extracted_v_link_attrs=v_link_attrs,
        if_use_node_status_flags=bool(flag_mask & 1),
        if_use_aggregated_link_attrs=bool(flag_mask & 2),
        if_use_degree_metric=bool(flag_mask & 4),
        if_use_more_topological_metrics=bool(flag_mask & 8),
    )
    return SimpleNamespace(
        rl=SimpleNamespace(feature_constructor=feature_constructor),
        simulation=SimpleNamespace(p_net_setting_num_nodes=p_nodes),
        nn=SimpleNamespace(
            embedding_dim=embedding_dim,
            dropout_prob=dropout_prob,
            batch_norm=batch_norm,
        ),
    )


def _double_bits(value: float) -> str:
    return struct.pack(">d", float(value)).hex()


def _canonical_general(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "embedding_dim": int(value["embedding_dim"]),
        "dropout_prob_bits": _double_bits(value["dropout_prob"]),
        "batch_norm": bool(value["batch_norm"]),
    }


def oracle_case(name: str, config: SimpleNamespace) -> dict[str, Any]:
    helpers = {
        helper_name: int(ORACLE[helper_name](config))
        for helper_name in HELPER_NAMES
    }
    feature_config = {
        key: int(value)
        for key, value in POLICY_BUILDER.get_feature_dim_config(config).items()
    }
    general = _canonical_general(
        POLICY_BUILDER.get_general_nn_config(config)
    )
    return {
        "name": name,
        "helpers": helpers,
        "feature_config": feature_config,
        "general_nn": general,
    }


def exact_cases() -> list[dict[str, Any]]:
    result = [oracle_case("defaults", make_config(3, 1, 3, 1, 500))]
    for flag_mask in range(16):
        result.append(oracle_case(
            f"flags_{flag_mask:02d}",
            make_config(2, 3, 5, 2, 17, flag_mask=flag_mask),
        ))
    result.append(oracle_case(
        "zero_attrs", make_config(0, 0, 0, 0, 1, flag_mask=0)
    ))
    result.append(oracle_case(
        "multi_attrs", make_config(7, 4, 11, 5, 1024, flag_mask=5)
    ))
    result.append(oracle_case(
        "general_nn_forwarding",
        make_config(
            1, 2, 3, 4, 33,
            embedding_dim=257,
            dropout_prob=0.375,
            batch_norm=True,
        ),
    ))
    return result


def _mix(checksum: int, value: int) -> int:
    return ((checksum ^ (value & MASK64)) * FNV_PRIME) & MASK64


def _mix_configuration(
    checksum: int,
    feature: dict[str, Any],
    general: dict[str, Any],
) -> int:
    for name in (
        "p_net_x_dim",
        "p_net_edge_dim",
        "v_net_x_dim",
        "v_net_edge_dim",
        "v_node_x_dim",
        "p_net_num_nodes",
    ):
        checksum = _mix(checksum, int(feature[name]))
    checksum = _mix(checksum, int(general["embedding_dim"]))
    dropout_bits = int.from_bytes(
        struct.pack("<d", float(general["dropout_prob"])),
        "little",
        signed=False,
    )
    checksum = _mix(checksum, dropout_bits)
    checksum = _mix(checksum, int(bool(general["batch_norm"])))
    return checksum


def prepare_benchmark_configurations() -> list[SimpleNamespace]:
    result: list[SimpleNamespace] = []
    append = result.append
    for index in range(BENCHMARK_CONFIGURATION_COUNT):
        append(make_config(
            index % 8,
            (index // 8) % 6,
            (index // 48) % 9,
            (index // 432) % 7,
            1 + index % 2048,
            flag_mask=(index // 3) & 15,
            embedding_dim=32 + (index % 8) * 32,
            dropout_prob=(index % 9) * 0.125,
            batch_norm=bool(index & 1),
        ))
    return result


def derive_prepared_batch(inputs: list[SimpleNamespace]) -> int:
    checksum = FNV_OFFSET
    for config in inputs:
        feature = POLICY_BUILDER.get_feature_dim_config(config)
        general = POLICY_BUILDER.get_general_nn_config(config)
        checksum = _mix_configuration(checksum, feature, general)
    return checksum


def benchmark_python(warmups: int, repetitions: int) -> dict[str, Any]:
    # Namespace construction is deliberately outside the measured boundary.
    # The timed loop starts at the seven pinned policy-builder APIs.
    inputs = prepare_benchmark_configurations()
    expected_checksum: int | None = None
    for _ in range(warmups):
        checksum = derive_prepared_batch(inputs)
        if expected_checksum is None:
            expected_checksum = checksum
        elif checksum != expected_checksum:
            raise RuntimeError("Python policy configuration checksum drift")
    if expected_checksum is None:
        raise RuntimeError("policy configuration warmup did not run")

    samples_ns: list[int] = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        checksum = derive_prepared_batch(inputs)
        samples_ns.append(time.perf_counter_ns() - started)
        if checksum != expected_checksum:
            raise RuntimeError("Python policy configuration checksum drift")
    return {
        "name": "prepared_65536",
        "warmups": warmups,
        "repetitions": repetitions,
        "configurations": BENCHMARK_CONFIGURATION_COUNT,
        "samples_ns": samples_ns,
        "checksum": str(expected_checksum),
    }


def compare_exact(
    expected: Any,
    actual: Any,
    path: str = "",
) -> list[dict[str, Any]]:
    if type(expected) is not type(actual):
        return [{
            "path": path,
            "python": expected,
            "cpp": actual,
            "reason": "type",
        }]
    if isinstance(expected, dict):
        result: list[dict[str, Any]] = []
        for key in sorted(set(expected) | set(actual)):
            if key not in expected or key not in actual:
                result.append({"path": f"{path}.{key}", "reason": "missing"})
            else:
                result.extend(
                    compare_exact(expected[key], actual[key], f"{path}.{key}")
                )
            if len(result) >= MAX_REPORTED_MISMATCHES:
                return result[:MAX_REPORTED_MISMATCHES]
        return result
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [{
                "path": path,
                "python_length": len(expected),
                "cpp_length": len(actual),
                "reason": "length",
            }]
        result: list[dict[str, Any]] = []
        for index, (left, right) in enumerate(zip(expected, actual)):
            result.extend(
                compare_exact(left, right, f"{path}[{index}]")
            )
            if len(result) >= MAX_REPORTED_MISMATCHES:
                return result[:MAX_REPORTED_MISMATCHES]
        return result
    return [] if expected == actual else [{
        "path": path,
        "python": expected,
        "cpp": actual,
        "reason": "value",
    }]


def run_native(args: argparse.Namespace, *, benchmark: bool) -> dict[str, Any]:
    command = [
        str(args.native),
        "--warmups", str(args.warmups),
        "--repetitions", str(args.repetitions),
    ]
    if benchmark:
        command.append("--benchmark")
    process = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"native policy-config harness failed: {process.stderr.strip()}"
        )
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"native emitted {len(lines)} JSON lines")
    return json.loads(lines[0])


def _write_report(args: argparse.Namespace, report: dict[str, Any]) -> None:
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({
        "component": report["component"],
        "status": report["status"],
        "exact_cases": report["exact_cases"],
        "oracle_backend": report["oracle_backend"],
        "benchmark": report["benchmark"],
    }, sort_keys=True, separators=(",", ":")))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", "--harness", required=True, type=pathlib.Path)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--json-output", "--output", type=pathlib.Path)
    args = parser.parse_args()
    for name in ("warmups", "repetitions"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} requires a positive integer")

    expected = {
        "component": "policy_config",
        "cases": exact_cases(),
    }
    # The semantic gate is a separate native process. Neither implementation
    # enters its benchmark path if this comparison fails.
    native_exact = run_native(args, benchmark=False)
    mismatches = compare_exact(expected, native_exact)
    report: dict[str, Any] = {
        "component": "policy_config",
        "fixture_version": FIXTURE_VERSION,
        "python_source": SOURCE_METADATA,
        "oracle_backend": ORACLE_BACKEND,
        "status": "FAIL" if mismatches else "PASS",
        "exact_cases": (
            "FAIL" if mismatches
            else f"{len(expected['cases'])}/{len(expected['cases'])}"
        ),
        "mismatches": mismatches,
        "benchmark_mismatches": [],
        "benchmark": "NOT_REQUESTED",
    }
    if mismatches:
        report["benchmark"] = "DEFERRED_UNTIL_EXACT"
        _write_report(args, report)
        return 1

    if args.benchmark:
        native_with_benchmark = run_native(args, benchmark=True)
        repeated_exact = {
            key: value
            for key, value in native_with_benchmark.items()
            if key != "benchmark"
        }
        repeated_mismatches = compare_exact(
            expected, repeated_exact, "repeat_exact"
        )
        if repeated_mismatches:
            report["status"] = "FAIL"
            report["mismatches"] = repeated_mismatches
            report["benchmark"] = "DEFERRED_AFTER_REPEAT_EXACT"
            _write_report(args, report)
            return 1

        native_benchmark = native_with_benchmark.get("benchmark")
        if not isinstance(native_benchmark, dict):
            raise RuntimeError("native benchmark report is missing")
        python_benchmark = benchmark_python(
            args.warmups, args.repetitions
        )
        benchmark_mismatches = compare_exact(
            {
                key: value
                for key, value in python_benchmark.items()
                if key != "samples_ns"
            },
            {
                key: value
                for key, value in native_benchmark.items()
                if key != "samples_ns"
            },
            "benchmark",
        )
        native_samples = [int(value) for value in native_benchmark["samples_ns"]]
        python_samples = [int(value) for value in python_benchmark["samples_ns"]]
        native_median_batch = statistics.median(native_samples)
        python_median_batch = statistics.median(python_samples)
        report["benchmark_mismatches"] = benchmark_mismatches
        report["benchmark"] = {
            "name": python_benchmark["name"],
            "native_scope": "derive_policy_configuration(prepared typed batch)",
            "python_scope": "seven AST policy-builder APIs(prepared config batch)",
            "configurations_per_sample": BENCHMARK_CONFIGURATION_COUNT,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "native_samples_ns_per_batch": native_samples,
            "python_samples_ns_per_batch": python_samples,
            "native_median_ns_per_configuration": (
                native_median_batch / BENCHMARK_CONFIGURATION_COUNT
            ),
            "python_median_ns_per_configuration": (
                python_median_batch / BENCHMARK_CONFIGURATION_COUNT
            ),
            "python_over_native_speedup": (
                python_median_batch / native_median_batch
            ),
            "output_checksum": python_benchmark["checksum"],
        }
        if benchmark_mismatches:
            report["status"] = "FAIL"

    _write_report(args, report)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
