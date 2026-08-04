#!/usr/bin/env python3
"""Exact Python/LibTorch parity and compact timing gate for MLPNet."""

from __future__ import annotations

import argparse
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


SOURCE_SIZE = 1759
SOURCE_SHA256 = "4f6d24dde669ec310fd2961bb1a6c0945f5a2d93f2042402025acf094eba89cb"
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1
MAX_REPORTED_MISMATCHES = 32


def verify_and_load(source: pathlib.Path) -> type[torch.nn.Module]:
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != SOURCE_SIZE or digest != SOURCE_SHA256:
        raise RuntimeError(
            "MLPNet Python source drift: "
            f"size={len(payload)}, sha256={digest}; expected "
            f"size={SOURCE_SIZE}, sha256={SOURCE_SHA256}"
        )
    spec = importlib.util.spec_from_file_location("_virne_mlp_net_oracle", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load MLPNet source: {source}")
    module = importlib.util.module_from_spec(spec)
    before = {name for name in sys.modules if name == "virne" or name.startswith("virne.")}
    spec.loader.exec_module(module)
    after = {name for name in sys.modules if name == "virne" or name.startswith("virne.")}
    imported = sorted(after - before)
    if imported:
        raise RuntimeError(f"direct MLPNet load unexpectedly imported Virne: {imported}")
    mlp_type = getattr(module, "MLPNet", None)
    if not isinstance(mlp_type, type) or not issubclass(mlp_type, torch.nn.Module):
        raise RuntimeError("pinned MLPNet source did not define an nn.Module")
    return mlp_type


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
        raise RuntimeError(f"unsupported tensor dtype in MLP comparator: {dtype}") from error


def tensor_summary(tensor: torch.Tensor, include_bits: bool) -> dict[str, Any]:
    value = tensor.detach().to("cpu").contiguous()
    payload = raw_bytes(value)
    dtype = dtype_name(value.dtype)
    width = {"float32": 4, "float64": 8, "int64": 8}[dtype]
    bits = []
    if include_bits:
        bits = [
            format(
                int.from_bytes(payload[offset:offset + width], "little", signed=False),
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
        return [0.5 + ((tensor_id + index) % 16) / 16.0 for index in range(count)]
    return [
        (((tensor_id * 37 + index * 17) % 61) - 30) / 32.0
        for index in range(count)
    ]


def fill_state(model: torch.nn.Module) -> None:
    tensor_id = 0
    with torch.no_grad():
        for _, tensor in model.named_parameters(recurse=True):
            values = fixture_values(
                tensor.numel(), tensor_id, running_variance=False)
            tensor.copy_(torch.tensor(values, dtype=tensor.dtype).reshape(tensor.shape))
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
                raise RuntimeError(f"unexpected MLP state dtype: {tensor.dtype}")
            tensor_id += 1


def summarize_state(model: torch.nn.Module) -> list[dict[str, Any]]:
    result = []
    for name, tensor in model.named_parameters(recurse=True):
        result.append({"name": name, "tensor": tensor_summary(tensor, False)})
    for name, tensor in model.named_buffers(recurse=True):
        result.append({"name": name, "tensor": tensor_summary(tensor, False)})
    return result


def make_input(shape: Iterable[int], dtype: torch.dtype) -> torch.Tensor:
    dimensions = tuple(shape)
    count = 1
    for dimension in dimensions:
        count *= dimension
    values = [(((index * 13) % 47) - 23) / 16.0 for index in range(count)]
    return torch.tensor(values, dtype=dtype).reshape(dimensions)


def make_model(
    mlp_type: type[torch.nn.Module],
    input_dim: int,
    output_dim: int,
    num_layers: int,
    embedding_dims: None | int | list[int],
    batch_norm: bool,
    dropout_prob: float,
) -> torch.nn.Module:
    return mlp_type(
        input_dim,
        output_dim,
        num_layers=num_layers,
        embedding_dims=embedding_dims,
        batch_norm=batch_norm,
        dropout_prob=dropout_prob,
    )


def initialization_case(mlp_type: type[torch.nn.Module]) -> dict[str, Any]:
    torch.manual_seed(0)
    model = make_model(mlp_type, 4, 3, 2, None, False, 1.0)
    return {
        "name": "init_default",
        "mode": "init",
        "next_rng": tensor_summary(torch.rand(8, dtype=torch.float32), True),
        "state": summarize_state(model),
    }


def forward_case(
    mlp_type: type[torch.nn.Module],
    name: str,
    dimensions: tuple[int, int, int, None | int | list[int], bool, float],
    input_shape: tuple[int, ...],
    dtype: torch.dtype,
    training: bool,
) -> dict[str, Any]:
    torch.manual_seed(7)
    model = make_model(mlp_type, *dimensions)
    if dtype == torch.float64:
        model.to(dtype=torch.float64)
    fill_state(model)
    value = make_input(input_shape, dtype)
    model.train(training)
    with torch.no_grad():
        output = model(value).contiguous()
    return {
        "name": name,
        "mode": "train" if training else "eval",
        "input": tensor_summary(value, True),
        "output": tensor_summary(output, True),
        "state": summarize_state(model),
    }


def run_cases(mlp_type: type[torch.nn.Module]) -> list[dict[str, Any]]:
    explicit = [5, 4, 6]
    return [
        initialization_case(mlp_type),
        forward_case(
            mlp_type, "forward_default", (4, 3, 2, None, False, 1.0),
            (7, 4), torch.float32, False),
        forward_case(
            mlp_type, "forward_scalar_rank3", (5, 1, 3, 7, False, 1.0),
            (2, 4, 5), torch.float32, False),
        forward_case(
            mlp_type, "forward_explicit", (3, 2, 4, explicit, False, 1.0),
            (8, 3), torch.float32, False),
        forward_case(
            mlp_type, "batch_norm_eval", (3, 2, 4, explicit, True, 1.0),
            (8, 3), torch.float32, False),
        forward_case(
            mlp_type, "batch_norm_train", (3, 2, 4, explicit, True, 1.0),
            (8, 3), torch.float32, True),
        forward_case(
            mlp_type, "dropout_zero", (4, 3, 2, None, False, 0.0),
            (7, 4), torch.float32, False),
        forward_case(
            mlp_type, "dropout_one", (4, 3, 2, None, False, 1.0),
            (7, 4), torch.float32, False),
        forward_case(
            mlp_type, "forward_float64", (4, 2, 3, 6, False, 1.0),
            (5, 4), torch.float64, False),
    ]


def run_benchmark(
    mlp_type: type[torch.nn.Module],
    warmups: int,
    repetitions: int,
    iterations: int,
) -> dict[str, Any]:
    torch.manual_seed(19)
    model = make_model(mlp_type, 25, 1, 3, 128, False, 0.0)
    fill_state(model)
    model.eval()
    value = make_input((8, 500, 25), torch.float32)
    output: torch.Tensor | None = None
    samples = []
    with torch.inference_mode():
        for _ in range(warmups):
            for _ in range(iterations):
                output = model(value)
        for _ in range(repetitions):
            started = time.perf_counter_ns()
            for _ in range(iterations):
                output = model(value)
            samples.append(time.perf_counter_ns() - started)
    if output is None:
        raise RuntimeError("MLP benchmark did not produce output")
    return {
        "warmups": warmups,
        "repetitions": repetitions,
        "iterations": iterations,
        "samples_ns": samples,
        "output": tensor_summary(output, False),
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
            f"MLPNet harness failed ({process.returncode}): "
            f"{process.stderr.strip()}"
        )
    lines = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"expected one native JSON line, got {lines!r}")
    report = json.loads(lines[0])
    if not isinstance(report, dict):
        raise RuntimeError("native MLPNet report must be a JSON object")
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
    parser.add_argument("--harness", "--native", dest="harness", type=pathlib.Path,
                        required=True)
    parser.add_argument("--python-source", type=pathlib.Path, required=True)
    parser.add_argument("--threads", type=positive, default=1)
    parser.add_argument("--warmups", type=positive, default=3)
    parser.add_argument("--repetitions", type=positive, default=5)
    parser.add_argument("--iterations", type=positive, default=20)
    parser.add_argument("--json-output", "--output", dest="json_output",
                        type=pathlib.Path)
    args = parser.parse_args()

    if sys.byteorder != "little":
        raise RuntimeError("MLPNet parity format requires a little-endian host")
    if not args.harness.is_file():
        raise RuntimeError(f"MLPNet harness does not exist: {args.harness}")

    mlp_type = verify_and_load(args.python_source)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    native = run_native(
        args.harness, args.threads, args.warmups,
        args.repetitions, args.iterations)
    expected_cases = run_cases(mlp_type)
    python_benchmark = run_benchmark(
        mlp_type, args.warmups, args.repetitions, args.iterations)

    global_mismatches: list[dict[str, Any]] = []
    if native.get("component") != "mlp_net":
        global_mismatches.append({
            "path": "component", "python": "mlp_net",
            "cpp": native.get("component"), "reason": "value"})
    if native.get("threads") != args.threads:
        global_mismatches.append({
            "path": "threads", "python": args.threads,
            "cpp": native.get("threads"), "reason": "value"})

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
        raise RuntimeError("native MLPNet report has no cases array")
    case_results = []
    if len(actual_cases) != len(expected_cases):
        global_mismatches.append({
            "path": "cases", "python_length": len(expected_cases),
            "cpp_length": len(actual_cases), "reason": "length"})
    for index, expected in enumerate(expected_cases):
        mismatches: list[dict[str, Any]] = []
        if index >= len(actual_cases):
            mismatches.append({
                "path": f"cases[{index}]", "reason": "missing",
                "python": expected["name"], "cpp": None})
        else:
            compare_exact(expected, actual_cases[index], f"cases[{index}]", mismatches)
        case_results.append({
            "name": expected["name"],
            "status": "PASS" if not mismatches else "FAIL",
            "mismatches": mismatches,
        })

    native_benchmark = native.get("benchmark")
    if not isinstance(native_benchmark, dict):
        raise RuntimeError("native MLPNet report has no benchmark object")
    benchmark_mismatches: list[dict[str, Any]] = []
    for field in ("warmups", "repetitions", "iterations", "output"):
        compare_exact(
            python_benchmark[field], native_benchmark.get(field),
            f"benchmark.{field}", benchmark_mismatches)
    python_samples = validate_samples(
        python_benchmark["samples_ns"], args.repetitions, "Python")
    cpp_samples = validate_samples(
        native_benchmark.get("samples_ns"), args.repetitions, "C++")
    python_median = float(statistics.median(python_samples))
    cpp_median = float(statistics.median(cpp_samples))

    cases_passed = sum(item["status"] == "PASS" for item in case_results)
    status = "PASS"
    if (global_mismatches or benchmark_mismatches or
            cases_passed != len(expected_cases)):
        status = "FAIL"
    report = {
        "component": "mlp_net",
        "status": status,
        "source_size": SOURCE_SIZE,
        "source_sha256": SOURCE_SHA256,
        "python_torch": python_version,
        "cpp_libtorch": cpp_version,
        "threads": args.threads,
        "exact_cases": {
            "passed": cases_passed,
            "total": len(expected_cases),
            "results": case_results,
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
            "python_median_ns_per_forward": python_median / args.iterations,
            "cpp_median_ns_per_forward": cpp_median / args.iterations,
            "cpp_speedup": python_median / cpp_median,
            "output": python_benchmark["output"],
        },
    }

    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output is not None:
        args.json_output.write_text(encoded, encoding="utf-8")
    compact: dict[str, Any] = {
        "component": "mlp_net",
        "status": status,
        "exact_cases": f"{cases_passed}/{len(expected_cases)}",
        "python_torch": python_version,
        "cpp_libtorch": cpp_version,
        "threads": args.threads,
        "python_median_ns_per_forward": python_median / args.iterations,
        "cpp_median_ns_per_forward": cpp_median / args.iterations,
        "cpp_speedup": python_median / cpp_median,
        "benchmark_output": "PASS" if not benchmark_mismatches else "FAIL",
    }
    if status != "PASS":
        compact["global_mismatches"] = global_mismatches
        compact["failed_cases"] = [
            item for item in case_results if item["status"] != "PASS"]
        compact["benchmark_mismatches"] = benchmark_mismatches
    print(json.dumps(compact, sort_keys=True, separators=(",", ":")))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
