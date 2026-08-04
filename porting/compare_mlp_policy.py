#!/usr/bin/env python3
"""Bit-exact Python/LibTorch parity and timing gate for MLP actor/critic."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import pathlib
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable

import torch
import torch.nn as nn


POLICY_SOURCE_SIZE = 1936
POLICY_SOURCE_SHA256 = (
    "c8998d8bbfb14477c55fbcf9240bf4e0d82ff1e5c51da7b4aa82387432a2e407"
)
MLP_SOURCE_SIZE = 1759
MLP_SOURCE_SHA256 = (
    "4f6d24dde669ec310fd2961bb1a6c0945f5a2d93f2042402025acf094eba89cb"
)
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1
MAX_REPORTED_MISMATCHES = 32


def verify_source(
    source: pathlib.Path,
    expected_size: int,
    expected_sha256: str,
    label: str,
) -> bytes:
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != expected_size or digest != expected_sha256:
        raise RuntimeError(
            f"{label} source drift: size={len(payload)}, sha256={digest}; "
            f"expected size={expected_size}, sha256={expected_sha256}"
        )
    return payload


def load_mlp(mlp_source: pathlib.Path) -> type[nn.Module]:
    verify_source(
        mlp_source, MLP_SOURCE_SIZE, MLP_SOURCE_SHA256, "MLPNet")
    spec = importlib.util.spec_from_file_location(
        "_virne_mlp_policy_mlp_oracle", mlp_source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load MLPNet source: {mlp_source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mlp_type = getattr(module, "MLPNet", None)
    if not isinstance(mlp_type, type) or not issubclass(mlp_type, nn.Module):
        raise RuntimeError("pinned MLPNet source did not define an nn.Module")
    return mlp_type


class OracleBaseActorCritic(nn.Module):
    """Dependency-free base sufficient for the pinned policy leaf."""


class OracleActorCriticRegistry:
    registrations: list[str] = []

    @classmethod
    def register(cls, name: str):
        cls.registrations.append(name)

        def decorate(value: type[nn.Module]) -> type[nn.Module]:
            return value

        return decorate


def load_policy_types(
    policy_source: pathlib.Path,
    mlp_source: pathlib.Path,
) -> tuple[type[nn.Module], type[nn.Module], type[nn.Module]]:
    before = {
        name for name in sys.modules
        if name == "virne" or name.startswith("virne.")
    }
    mlp_type = load_mlp(mlp_source)
    payload = verify_source(
        policy_source,
        POLICY_SOURCE_SIZE,
        POLICY_SOURCE_SHA256,
        "MLP policy",
    )
    tree = ast.parse(payload.decode("utf-8"), filename=str(policy_source))
    expected_names = ["MlpActorCritic", "Actor", "Critic"]
    selected = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in expected_names
    ]
    if [node.name for node in selected] != expected_names:
        raise RuntimeError(
            "pinned MLP policy did not contain the expected class sequence"
        )

    OracleActorCriticRegistry.registrations.clear()
    namespace: dict[str, Any] = {
        "__name__": "_virne_mlp_policy_oracle",
        "nn": nn,
        "MLPNet": mlp_type,
        "BaseActorCritic": OracleBaseActorCritic,
        "ActorCriticRegistry": OracleActorCriticRegistry,
    }
    isolated = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(isolated)
    exec(compile(isolated, str(policy_source), "exec"), namespace)
    if OracleActorCriticRegistry.registrations != ["mlp"]:
        raise RuntimeError(
            "isolated MLP policy registered an unexpected policy name"
        )

    after = {
        name for name in sys.modules
        if name == "virne" or name.startswith("virne.")
    }
    imported = sorted(after - before)
    if imported:
        raise RuntimeError(
            f"isolated MLP policy unexpectedly imported Virne: {imported}"
        )

    result = tuple(namespace[name] for name in expected_names)
    if any(
        not isinstance(value, type) or not issubclass(value, nn.Module)
        for value in result
    ):
        raise RuntimeError("isolated MLP policy classes are not nn.Module types")
    return result  # type: ignore[return-value]


def raw_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().to("cpu").contiguous().reshape(-1)
    return bytes(value.view(torch.uint8).tolist())


def fnv1a(payload: bytes) -> int:
    checksum = FNV_OFFSET
    for byte in payload:
        checksum ^= byte
        checksum = (checksum * FNV_PRIME) & MASK64
    return checksum


def dtype_name(dtype: torch.dtype) -> str:
    names = {
        torch.float32: "float32",
        torch.float64: "float64",
        torch.int64: "int64",
    }
    try:
        return names[dtype]
    except KeyError as error:
        raise RuntimeError(
            f"unsupported tensor dtype in MLP policy comparator: {dtype}"
        ) from error


def tensor_summary(tensor: torch.Tensor, include_bits: bool) -> dict[str, Any]:
    value = tensor.detach().to("cpu").contiguous()
    payload = raw_bytes(value)
    dtype = dtype_name(value.dtype)
    width = {"float32": 4, "float64": 8, "int64": 8}[dtype]
    bits = []
    if include_bits:
        bits = [
            format(
                int.from_bytes(
                    payload[offset:offset + width], "little", signed=False),
                f"0{width * 2}x",
            )
            for offset in range(0, len(payload), width)
        ]
    return {
        "shape": list(value.shape),
        "dtype": dtype,
        "checksum": str(fnv1a(payload)),
        "bytes": len(payload),
        "bits": bits,
    }


def fixture_values(
    count: int,
    tensor_id: int,
    *,
    running_variance: bool,
) -> list[float]:
    if running_variance:
        return [
            0.5 + ((tensor_id + index) % 16) / 16.0
            for index in range(count)
        ]
    return [
        (((tensor_id * 37 + index * 17) % 61) - 30) / 32.0
        for index in range(count)
    ]


def fill_state(model: nn.Module) -> None:
    tensor_id = 0
    with torch.no_grad():
        for _, tensor in model.named_parameters(recurse=True):
            values = fixture_values(
                tensor.numel(), tensor_id, running_variance=False)
            tensor.copy_(
                torch.tensor(values, dtype=tensor.dtype).reshape(tensor.shape)
            )
            tensor_id += 1
        for name, tensor in model.named_buffers(recurse=True):
            if tensor.is_floating_point():
                values = fixture_values(
                    tensor.numel(),
                    tensor_id,
                    running_variance=name.endswith("running_var"),
                )
                tensor.copy_(
                    torch.tensor(values, dtype=tensor.dtype).reshape(tensor.shape)
                )
            elif tensor.dtype == torch.int64:
                tensor.fill_(tensor_id + 3)
            else:
                raise RuntimeError(
                    f"unexpected MLP policy state dtype: {tensor.dtype}"
                )
            tensor_id += 1


def summarize_state(model: nn.Module) -> list[dict[str, Any]]:
    result = []
    for name, tensor in model.named_parameters(recurse=True):
        result.append({"name": name, "tensor": tensor_summary(tensor, False)})
    for name, tensor in model.named_buffers(recurse=True):
        result.append({"name": name, "tensor": tensor_summary(tensor, False)})
    return result


def make_input(shape: Iterable[int]) -> torch.Tensor:
    dimensions = tuple(shape)
    count = 1
    for dimension in dimensions:
        count *= dimension
    values = [
        (((index * 13) % 47) - 23) / 16.0
        for index in range(count)
    ]
    return torch.tensor(values, dtype=torch.float32).reshape(dimensions)


def make_composite(
    composite_type: type[nn.Module],
    feature_dim: int,
    action_dim: int,
    num_layers: int,
    embedding_dim: int,
    dropout_prob: float,
    batch_norm: bool,
) -> nn.Module:
    return composite_type(
        feature_dim,
        action_dim,
        num_layers=num_layers,
        embedding_dim=embedding_dim,
        dropout_prob=dropout_prob,
        batch_norm=batch_norm,
    )


def make_standalone(
    module_type: type[nn.Module],
    feature_dim: int,
    action_dim: int,
    num_layers: int,
    embedding_dim: int,
    dropout_prob: float,
    batch_norm: bool,
) -> nn.Module:
    return module_type(
        feature_dim,
        action_dim,
        num_layers=num_layers,
        embedding_dim=embedding_dim,
        dropout_prob=dropout_prob,
        batch_norm=batch_norm,
    )


def initialization_case(composite_type: type[nn.Module]) -> dict[str, Any]:
    torch.manual_seed(0)
    model = make_composite(composite_type, 4, 9, 3, 6, 0.25, False)
    state = summarize_state(model)
    next_rng = tensor_summary(torch.rand(8, dtype=torch.float32), True)
    return {
        "name": "init_composite",
        "kind": "composite",
        "mode": "init",
        "state": state,
        "next_rng": next_rng,
    }


def composite_forward_case(
    composite_type: type[nn.Module],
    name: str,
    feature_dim: int,
    action_dim: int,
    num_layers: int,
    embedding_dim: int,
    dropout_prob: float,
    input_shape: tuple[int, ...],
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = make_composite(
        composite_type,
        feature_dim,
        action_dim,
        num_layers,
        embedding_dim,
        dropout_prob,
        False,
    )
    fill_state(model)
    model.eval()
    value = make_input(input_shape)
    observation = {"p_net_x": value}
    with torch.no_grad():
        actor_output = model.actor(observation).contiguous()
        critic_output = model.critic(observation).contiguous()
    return {
        "name": name,
        "kind": "composite",
        "mode": "eval",
        "input": tensor_summary(value, True),
        "actor_output": tensor_summary(actor_output, True),
        "critic_output": tensor_summary(critic_output, True),
        "state": summarize_state(model),
    }


def standalone_forward_case(
    module_type: type[nn.Module],
    name: str,
    kind: str,
) -> dict[str, Any]:
    torch.manual_seed(13)
    model = make_standalone(module_type, 3, 17, 4, 5, 0.4, True)
    fill_state(model)
    model.eval()
    value = make_input((8, 3))
    observation = {"p_net_x": value}
    with torch.no_grad():
        output = model(observation).contiguous()
    result = {
        "name": name,
        "kind": kind,
        "mode": "eval",
        "input": tensor_summary(value, True),
        "state": summarize_state(model),
    }
    result[f"{kind}_output"] = tensor_summary(output, True)
    return result


def run_cases(
    composite_type: type[nn.Module],
    actor_type: type[nn.Module],
    critic_type: type[nn.Module],
) -> list[dict[str, Any]]:
    return [
        initialization_case(composite_type),
        composite_forward_case(
            composite_type,
            "forward_rank3",
            5, 7, 3, 7, 0.0, (2, 4, 5), 7,
        ),
        composite_forward_case(
            composite_type,
            "production_multi_resource",
            25, 500, 3, 128, 0.0, (8, 500, 25), 7,
        ),
        composite_forward_case(
            composite_type,
            "action_dim_one",
            4, 1, 3, 6, 0.75, (7, 4), 11,
        ),
        composite_forward_case(
            composite_type,
            "action_dim_997",
            4, 997, 3, 6, 0.75, (7, 4), 11,
        ),
        standalone_forward_case(
            actor_type, "actor_batch_norm_eval", "actor"),
        standalone_forward_case(
            critic_type, "critic_batch_norm_eval", "critic"),
    ]


def run_benchmark(
    composite_type: type[nn.Module],
    warmups: int,
    repetitions: int,
    iterations: int,
) -> dict[str, Any]:
    torch.manual_seed(19)
    model = make_composite(composite_type, 25, 500, 3, 128, 0.0, False)
    fill_state(model)
    model.eval()
    value = make_input((8, 500, 25))
    observation = {"p_net_x": value}
    actor_output: torch.Tensor | None = None
    critic_output: torch.Tensor | None = None
    samples = []
    with torch.inference_mode():
        for _ in range(warmups):
            for _ in range(iterations):
                actor_output = model.actor(observation)
                critic_output = model.critic(observation)
        for _ in range(repetitions):
            started = time.perf_counter_ns()
            for _ in range(iterations):
                actor_output = model.actor(observation)
                critic_output = model.critic(observation)
            samples.append(time.perf_counter_ns() - started)
    if actor_output is None or critic_output is None:
        raise RuntimeError("MLP policy benchmark did not produce both outputs")
    return {
        "warmups": warmups,
        "repetitions": repetitions,
        "iterations": iterations,
        "samples_ns": samples,
        "actor_output": tensor_summary(actor_output, False),
        "critic_output": tensor_summary(critic_output, False),
    }


def run_native(
    harness: pathlib.Path,
    threads: int,
    warmups: int,
    repetitions: int,
    iterations: int,
) -> dict[str, Any]:
    process = subprocess.run(
        [
            str(harness),
            "--threads", str(threads),
            "--warmups", str(warmups),
            "--repetitions", str(repetitions),
            "--iterations", str(iterations),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"MLP policy harness failed ({process.returncode}): "
            f"{process.stderr.strip()}"
        )
    lines = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"expected one native JSON line, got {lines!r}")
    report = json.loads(lines[0])
    if not isinstance(report, dict):
        raise RuntimeError("native MLP policy report must be a JSON object")
    return report


def short_value(value: Any) -> Any:
    if isinstance(value, list) and len(value) > 12:
        return value[:6] + [f"... {len(value) - 12} omitted ..."] + value[-6:]
    return value


def compare_exact(
    expected: Any,
    actual: Any,
    path: str,
    mismatches: list[dict[str, Any]],
) -> None:
    if len(mismatches) >= MAX_REPORTED_MISMATCHES:
        return
    if type(expected) is not type(actual):
        mismatches.append({
            "path": path,
            "python": short_value(expected),
            "cpp": short_value(actual),
            "reason": "type",
        })
        return
    if isinstance(expected, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            mismatches.append({
                "path": path,
                "python_keys": sorted(expected_keys),
                "cpp_keys": sorted(actual_keys),
                "reason": "keys",
            })
            if len(mismatches) >= MAX_REPORTED_MISMATCHES:
                return
        for key in sorted(expected_keys & actual_keys):
            compare_exact(expected[key], actual[key], f"{path}.{key}", mismatches)
        return
    if isinstance(expected, list):
        if len(expected) != len(actual):
            mismatches.append({
                "path": path,
                "python_length": len(expected),
                "cpp_length": len(actual),
                "reason": "length",
            })
            if len(mismatches) >= MAX_REPORTED_MISMATCHES:
                return
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare_exact(left, right, f"{path}[{index}]", mismatches)
        return
    if expected != actual:
        mismatches.append({
            "path": path,
            "python": expected,
            "cpp": actual,
            "reason": "value",
        })


def action_dim_projection(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "input": case["input"],
        "actor_output": case["actor_output"],
        "critic_output": case["critic_output"],
        "state": case["state"],
    }


def validate_samples(value: Any, repetitions: int, label: str) -> list[int]:
    if not isinstance(value, list) or len(value) != repetitions:
        raise RuntimeError(f"{label} benchmark returned invalid sample count")
    if any(type(sample) is not int or sample <= 0 for sample in value):
        raise RuntimeError(f"{label} benchmark returned a non-positive timing")
    return value


def version_core(value: str) -> str:
    return value.split("+", maxsplit=1)[0]


def positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--harness", "--native", dest="harness",
        type=pathlib.Path, required=True)
    parser.add_argument("--python-source", type=pathlib.Path, required=True)
    parser.add_argument("--mlp-source", type=pathlib.Path, required=True)
    parser.add_argument("--threads", type=positive, default=1)
    parser.add_argument("--warmups", type=positive, default=3)
    parser.add_argument("--repetitions", type=positive, default=5)
    parser.add_argument("--iterations", type=positive, default=20)
    parser.add_argument(
        "--json-output", "--output", dest="json_output", type=pathlib.Path)
    args = parser.parse_args()

    if sys.byteorder != "little":
        raise RuntimeError("MLP policy parity format requires a little-endian host")
    if not args.harness.is_file():
        raise RuntimeError(f"MLP policy harness does not exist: {args.harness}")

    composite_type, actor_type, critic_type = load_policy_types(
        args.python_source, args.mlp_source)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    native = run_native(
        args.harness,
        args.threads,
        args.warmups,
        args.repetitions,
        args.iterations,
    )
    expected_cases = run_cases(composite_type, actor_type, critic_type)
    python_benchmark = run_benchmark(
        composite_type,
        args.warmups,
        args.repetitions,
        args.iterations,
    )

    global_mismatches: list[dict[str, Any]] = []
    if native.get("component") != "mlp_policy":
        global_mismatches.append({
            "path": "component",
            "python": "mlp_policy",
            "cpp": native.get("component"),
            "reason": "value",
        })
    if native.get("threads") != args.threads:
        global_mismatches.append({
            "path": "threads",
            "python": args.threads,
            "cpp": native.get("threads"),
            "reason": "value",
        })
    cpp_version = str(native.get("libtorch_version", "unknown"))
    python_version = str(torch.__version__)
    if version_core(cpp_version) != version_core(python_version):
        global_mismatches.append({
            "path": "torch_version_core",
            "python": version_core(python_version),
            "cpp": version_core(cpp_version),
            "reason": "value",
        })

    actual_cases = native.get("cases")
    if not isinstance(actual_cases, list):
        raise RuntimeError("native MLP policy report has no cases array")
    if len(actual_cases) != len(expected_cases):
        global_mismatches.append({
            "path": "cases",
            "python_length": len(expected_cases),
            "cpp_length": len(actual_cases),
            "reason": "length",
        })
    case_results = []
    for index, expected in enumerate(expected_cases):
        mismatches: list[dict[str, Any]] = []
        if index >= len(actual_cases):
            mismatches.append({
                "path": f"cases[{index}]",
                "python": expected["name"],
                "cpp": None,
                "reason": "missing",
            })
        else:
            compare_exact(
                expected, actual_cases[index], f"cases[{index}]", mismatches)
        case_results.append({
            "name": expected["name"],
            "status": "PASS" if not mismatches else "FAIL",
            "mismatches": mismatches,
        })

    invariant_mismatches: list[dict[str, Any]] = []
    compare_exact(
        action_dim_projection(expected_cases[3]),
        action_dim_projection(expected_cases[4]),
        "python.action_dim_inert",
        invariant_mismatches,
    )
    if len(actual_cases) >= 5:
        compare_exact(
            action_dim_projection(actual_cases[3]),
            action_dim_projection(actual_cases[4]),
            "cpp.action_dim_inert",
            invariant_mismatches,
        )

    native_benchmark = native.get("benchmark")
    if not isinstance(native_benchmark, dict):
        raise RuntimeError("native MLP policy report has no benchmark object")
    benchmark_mismatches: list[dict[str, Any]] = []
    for field in (
        "warmups",
        "repetitions",
        "iterations",
        "actor_output",
        "critic_output",
    ):
        compare_exact(
            python_benchmark[field],
            native_benchmark.get(field),
            f"benchmark.{field}",
            benchmark_mismatches,
        )
    python_samples = validate_samples(
        python_benchmark["samples_ns"], args.repetitions, "Python")
    cpp_samples = validate_samples(
        native_benchmark.get("samples_ns"), args.repetitions, "C++")
    python_median = float(statistics.median(python_samples))
    cpp_median = float(statistics.median(cpp_samples))

    cases_passed = sum(
        item["status"] == "PASS" for item in case_results)
    status = "PASS"
    if (
        global_mismatches
        or invariant_mismatches
        or benchmark_mismatches
        or cases_passed != len(expected_cases)
    ):
        status = "FAIL"

    report = {
        "component": "mlp_policy",
        "status": status,
        "policy_source_size": POLICY_SOURCE_SIZE,
        "policy_source_sha256": POLICY_SOURCE_SHA256,
        "mlp_source_size": MLP_SOURCE_SIZE,
        "mlp_source_sha256": MLP_SOURCE_SHA256,
        "python_torch": python_version,
        "cpp_libtorch": cpp_version,
        "threads": args.threads,
        "exact_cases": {
            "passed": cases_passed,
            "total": len(expected_cases),
            "results": case_results,
        },
        "action_dim_inert": {
            "status": "PASS" if not invariant_mismatches else "FAIL",
            "mismatches": invariant_mismatches,
        },
        "global_mismatches": global_mismatches,
        "benchmark": {
            "status": "PASS" if not benchmark_mismatches else "FAIL",
            "output_mismatches": benchmark_mismatches,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "iterations": args.iterations,
            "python_samples_ns": python_samples,
            "cpp_samples_ns": cpp_samples,
            "python_median_ns": python_median,
            "cpp_median_ns": cpp_median,
            "python_median_ns_per_actor_critic": (
                python_median / args.iterations),
            "cpp_median_ns_per_actor_critic": (
                cpp_median / args.iterations),
            "cpp_speedup": python_median / cpp_median,
            "actor_output": python_benchmark["actor_output"],
            "critic_output": python_benchmark["critic_output"],
        },
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output is not None:
        args.json_output.write_text(encoded, encoding="utf-8")

    compact: dict[str, Any] = {
        "component": "mlp_policy",
        "status": status,
        "exact_cases": f"{cases_passed}/{len(expected_cases)}",
        "action_dim_inert": "PASS" if not invariant_mismatches else "FAIL",
        "python_torch": python_version,
        "cpp_libtorch": cpp_version,
        "threads": args.threads,
        "python_median_ns_per_actor_critic": python_median / args.iterations,
        "cpp_median_ns_per_actor_critic": cpp_median / args.iterations,
        "cpp_speedup": python_median / cpp_median,
        "benchmark_output": "PASS" if not benchmark_mismatches else "FAIL",
    }
    if status != "PASS":
        compact["global_mismatches"] = global_mismatches
        compact["failed_cases"] = [
            item for item in case_results if item["status"] != "PASS"]
        compact["invariant_mismatches"] = invariant_mismatches
        compact["benchmark_mismatches"] = benchmark_mismatches
    print(json.dumps(compact, sort_keys=True, separators=(",", ":")))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
