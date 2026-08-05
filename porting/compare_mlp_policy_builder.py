#!/usr/bin/env python3
"""Exact source gate and one cold timing for the typed MLP policy adapter."""

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
from typing import Any

import compare_policy_config as policy_config_oracle


BUILD_METHOD_SHA256 = (
    "6838dd4480c24caa08ded302a68c1b91e171ade21b194c8eae08f6ecfead9db7"
)
COUNT = 65536
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


class CapturedPolicy:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.device: Any = None

    def to(self, device: Any) -> "CapturedPolicy":
        self.device = device
        return self


class CapturedOptimizerBuilder:
    @staticmethod
    def build_optimizer(config: Any, policy: CapturedPolicy) -> tuple[Any, Any]:
        return config, policy


def load_build_method() -> tuple[Any, dict[str, Any]]:
    path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "virne"
        / "virne"
        / "solver"
        / "learning"
        / "rl_core"
        / "policy_builder.py"
    )
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    policy_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PolicyBuilder"
    )
    method = next(
        node for node in policy_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_mlp_policy"
    )
    segment = ast.get_source_segment(source, method)
    if segment is None:
        raise RuntimeError("cannot isolate PolicyBuilder.build_mlp_policy")
    digest = hashlib.sha256(segment.encode("utf-8")).hexdigest()
    if digest != BUILD_METHOD_SHA256:
        raise RuntimeError(
            "PolicyBuilder.build_mlp_policy source drift: "
            f"expected {BUILD_METHOD_SHA256}, got {digest}"
        )

    class_arguments: dict[str, Any] = {
        "name": "PolicyBuilder",
        "bases": [],
        "keywords": [],
        "body": [method],
        "decorator_list": [],
    }
    if "type_params" in ast.ClassDef._fields:
        class_arguments["type_params"] = []
    module = ast.Module(
        body=[ast.ClassDef(**class_arguments)], type_ignores=[]
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {
        "Any": Any,
        "Tuple": tuple,
        "get_p_net_x_dim": policy_config_oracle.ORACLE["get_p_net_x_dim"],
        "get_v_node_x_dim": policy_config_oracle.ORACLE["get_v_node_x_dim"],
        "MlpActorCritic": CapturedPolicy,
        "OptimizerBuilder": CapturedOptimizerBuilder,
    }
    exec(compile(module, str(path), "exec"), namespace)
    isolated_builder = namespace["PolicyBuilder"]
    isolated_builder.get_general_nn_config = staticmethod(
        policy_config_oracle.POLICY_BUILDER.get_general_nn_config
    )
    return isolated_builder, {
        "path": str(path),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "build_mlp_policy_sha256": digest,
        "policy_config_sources": policy_config_oracle.SOURCE_METADATA,
    }


POLICY_BUILDER, SOURCE_METADATA = load_build_method()


def bits(value: float) -> str:
    return str(struct.unpack("<Q", struct.pack("<d", float(value)))[0])


def derived_options(config: SimpleNamespace) -> dict[str, Any]:
    general = policy_config_oracle.POLICY_BUILDER.get_general_nn_config(config)
    return {
        "feature_dim": int(
            policy_config_oracle.ORACLE["get_p_net_x_dim"](config)
            + policy_config_oracle.ORACLE["get_v_node_x_dim"](config)
        ),
        "action_dim": int(config.simulation.p_net_setting_num_nodes),
        "num_layers": 3,
        "embedding_dim": int(general["embedding_dim"]),
        "dropout_bits": bits(general["dropout_prob"]),
        "batch_norm": bool(general["batch_norm"]),
    }


def oracle_case(name: str, config: SimpleNamespace) -> dict[str, Any]:
    agent = SimpleNamespace(config=config, device="cpu")
    policy, _optimizer = POLICY_BUILDER.build_mlp_policy(agent)
    expected = derived_options(config)
    captured = {
        "feature_dim": int(policy.kwargs["feature_dim"]),
        "action_dim": int(policy.kwargs["action_dim"]),
        "num_layers": 3,
        "embedding_dim": int(policy.kwargs["embedding_dim"]),
        "dropout_bits": bits(policy.kwargs["dropout_prob"]),
        "batch_norm": bool(policy.kwargs["batch_norm"]),
    }
    if captured != expected or policy.device != "cpu":
        raise RuntimeError("isolated Python build_mlp_policy mapping drift")
    return {"name": name, **captured}


def exact_cases() -> list[dict[str, Any]]:
    make = policy_config_oracle.make_config
    return [
        oracle_case(
            "multi_resource_default",
            make(3, 1, 3, 1, 500, flag_mask=15),
        ),
        oracle_case(
            "custom_network",
            make(
                6, 0, 4, 0, 17, flag_mask=0,
                embedding_dim=256, dropout_prob=0.25, batch_norm=True,
            ),
        ),
        oracle_case(
            "minimal",
            make(
                0, 0, 0, 0, 1, flag_mask=0,
                embedding_dim=1, dropout_prob=1.0, batch_norm=False,
            ),
        ),
    ]


def prepared_configs() -> list[SimpleNamespace]:
    make = policy_config_oracle.make_config
    return [
        make(
            index % 24,
            0,
            (index // 3) % 20,
            0,
            1 + index % 2048,
            flag_mask=0,
            embedding_dim=32 + (index % 8) * 32,
            dropout_prob=(index % 9) * 0.125,
            batch_norm=bool(index & 1),
        )
        for index in range(COUNT)
    ]


def mix(checksum: int, value: int) -> int:
    for byte in range(8):
        checksum ^= (value >> (byte * 8)) & 0xFF
        checksum = (checksum * FNV_PRIME) & MASK64
    return checksum


def derive_batch(inputs: list[SimpleNamespace]) -> int:
    checksum = FNV_OFFSET
    for config in inputs:
        value = derived_options(config)
        checksum = mix(checksum, value["feature_dim"])
        checksum = mix(checksum, value["action_dim"])
        checksum = mix(checksum, value["num_layers"])
        checksum = mix(checksum, value["embedding_dim"])
        checksum = mix(checksum, int(value["dropout_bits"]))
        checksum = mix(checksum, int(value["batch_norm"]))
    return checksum


def benchmark_python(warmups: int, repetitions: int) -> dict[str, Any]:
    inputs = prepared_configs()
    checksum: int | None = None
    for _ in range(warmups):
        current = derive_batch(inputs)
        checksum = current if checksum is None else checksum
        if current != checksum:
            raise RuntimeError("Python MLP option checksum drift")
    samples: list[int] = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        current = derive_batch(inputs)
        samples.append(time.perf_counter_ns() - started)
        if current != checksum:
            raise RuntimeError("Python MLP option checksum drift")
    return {"samples_ns": samples, "checksum": str(checksum)}


def run_native(
    executable: pathlib.Path,
    benchmark: bool,
    warmups: int,
    repetitions: int,
) -> dict[str, Any]:
    command = [
        str(executable.resolve()),
        "--warmups", str(warmups),
        "--repetitions", str(repetitions),
    ]
    if benchmark:
        command.append("--benchmark")
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def write_report(path: pathlib.Path | None, report: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", "--harness", required=True, type=pathlib.Path)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--json-output", "--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.warmups <= 0 or args.repetitions <= 0:
        parser.error("warmups and repetitions must be positive")

    expected = {"component": "mlp_policy_builder", "cases": exact_cases()}
    native = run_native(
        args.native, False, args.warmups, args.repetitions
    )
    mismatches = [] if native == expected else [{
        "path": "exact", "python": expected, "cpp": native,
    }]
    report: dict[str, Any] = {
        "component": "mlp_policy_builder",
        "fixture_version": 1,
        "oracle_backend": "ast_isolated_build_mlp_policy_and_helpers",
        "python_source": SOURCE_METADATA,
        "exact_cases": "3/3" if not mismatches else "FAIL",
        "mismatches": mismatches,
        "benchmark_mismatches": [],
        "benchmark": "NOT_REQUESTED",
        "status": "PASS" if not mismatches else "FAIL",
    }
    if mismatches:
        write_report(args.json_output, report)
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 1

    if args.benchmark:
        native_timed = run_native(
            args.native, True, args.warmups, args.repetitions
        )["benchmark"]
        python_timed = benchmark_python(args.warmups, args.repetitions)
        if native_timed["checksum"] != python_timed["checksum"]:
            report["benchmark_mismatches"] = [{
                "path": "checksum",
                "python": python_timed["checksum"],
                "cpp": native_timed["checksum"],
            }]
            report["status"] = "FAIL"
        native_median = statistics.median(native_timed["samples_ns"]) / COUNT
        python_median = statistics.median(python_timed["samples_ns"]) / COUNT
        report["benchmark"] = {
            "name": native_timed["name"],
            "configurations_per_sample": COUNT,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "native_scope": "make_mlp_actor_critic_options(prepared typed config)",
            "python_scope": "three AST policy-builder option APIs(prepared config)",
            "native_samples_ns_per_batch": native_timed["samples_ns"],
            "python_samples_ns_per_batch": python_timed["samples_ns"],
            "native_median_ns_per_configuration": native_median,
            "python_median_ns_per_configuration": python_median,
            "python_over_native_speedup": python_median / native_median,
            "output_checksum": native_timed["checksum"],
        }

    write_report(args.json_output, report)
    compact = {
        "component": report["component"],
        "exact_cases": report["exact_cases"],
        "oracle_backend": report["oracle_backend"],
        "benchmark": report["benchmark"],
        "status": report["status"],
    }
    print(json.dumps(compact, sort_keys=True, separators=(",", ":")))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
