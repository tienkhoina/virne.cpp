#!/usr/bin/env python3
"""Exact parity and compact timing gate for dense TensorConvertor paths."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import statistics
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np
import torch


SOURCE_SIZE = 23806
SOURCE_SHA256 = "75a13356834688d6dff77f7e5fc78ca64cf49171f183aedce92116ad31035f2b"
NUMPY_VERSION = "1.26.4"
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1
MAX_MISMATCHES = 32


def load_convertor(source: pathlib.Path) -> type:
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != SOURCE_SIZE or digest != SOURCE_SHA256:
        raise RuntimeError(
            f"TensorConvertor source drift: size={len(payload)}, sha256={digest}"
        )
    tree = ast.parse(payload.decode("utf-8"), filename=str(source))
    selected = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TensorConvertor"
    ]
    if len(selected) != 1:
        raise RuntimeError("pinned source did not define exactly one TensorConvertor")
    before = {
        name for name in sys.modules
        if name == "virne" or name.startswith("virne.")
    }
    namespace: dict[str, Any] = {
        "__name__": "_virne_tensor_convertor_oracle",
        "np": np,
        "torch": torch,
    }
    isolated = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(isolated)
    exec(compile(isolated, str(source), "exec"), namespace)
    after = {
        name for name in sys.modules
        if name == "virne" or name.startswith("virne.")
    }
    imported = sorted(after - before)
    if imported:
        raise RuntimeError(f"isolated TensorConvertor imported Virne: {imported}")
    result = namespace["TensorConvertor"]
    if not isinstance(result, type):
        raise RuntimeError("isolated TensorConvertor is not a class")
    return result


def raw_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().to("cpu").contiguous().reshape(-1)
    return bytes(value.view(torch.uint8).tolist())


def tensor_summary(tensor: torch.Tensor, include_bits: bool) -> dict[str, Any]:
    value = tensor.detach().to("cpu").contiguous()
    if value.dtype == torch.float32:
        dtype = "float32"
        width = 4
    elif value.dtype == torch.int64:
        dtype = "int64"
        width = 8
    else:
        raise RuntimeError(f"unexpected dense-convertor dtype: {value.dtype}")
    payload = raw_bytes(value)
    checksum = FNV_OFFSET
    for byte in payload:
        checksum ^= byte
        checksum = (checksum * FNV_PRIME) & MASK64
    bits = []
    if include_bits:
        bits = [
            format(
                int.from_bytes(payload[index:index + width], "little"),
                f"0{width * 2}x",
            )
            for index in range(0, len(payload), width)
        ]
    return {
        "shape": list(value.shape),
        "dtype": dtype,
        "checksum": str(checksum),
        "bytes": len(payload),
        "bits": bits,
    }


def patterned(
    shape: tuple[int, ...],
    multiplier: int,
    modulus: int,
    offset: int,
    denominator: float,
) -> np.ndarray:
    count = int(np.prod(shape))
    indices = np.arange(count, dtype=np.int64)
    values = ((indices * multiplier) % modulus - offset) / denominator
    return values.astype(np.float64).reshape(shape)


def make_single() -> dict[str, Any]:
    return {
        "p_net_x": patterned((3, 2), 7, 29, 14, 8.0),
        "v_node_x": patterned((2,), 5, 13, 6, 4.0),
        "curr_v_node_id": 2,
        "action_mask": np.array([1, 0, 1], dtype=np.int64),
        "v_net_size": 3,
    }


def make_batch(include_general: bool) -> list[dict[str, Any]]:
    p_net_x = patterned((2, 3, 2), 7, 29, 14, 8.0)
    v_node_x = patterned((2, 2), 5, 13, 6, 4.0)
    result = []
    action_masks = np.array(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float64)
    for batch_id in range(2):
        observation: dict[str, Any] = {
            "p_net_x": p_net_x[batch_id],
            "v_node_x": v_node_x[batch_id],
        }
        if include_general:
            observation.update({
                "curr_v_node_id": batch_id + 1,
                "action_mask": action_masks[batch_id],
                "v_net_size": batch_id + 3,
            })
        result.append(observation)
    return result


def make_production_batch() -> list[dict[str, Any]]:
    p_net_x = patterned((8, 500, 14), 13, 47, 23, 16.0)
    v_node_x = patterned((8, 11), 17, 31, 15, 8.0)
    result = []
    for batch_id in range(8):
        result.append({
            "p_net_x": p_net_x[batch_id],
            "v_node_x": v_node_x[batch_id],
            "curr_v_node_id": batch_id,
            "action_mask": np.ones(500, dtype=np.float64),
            "v_net_size": batch_id + 2,
        })
    return result


def summarize_fields(
    output: dict[str, torch.Tensor],
    names: tuple[str, ...],
) -> list[dict[str, Any]]:
    return [
        {"name": name, "tensor": tensor_summary(output[name], True)}
        for name in names if name in output
    ]


def run_cases(convertor: type) -> list[dict[str, Any]]:
    single = make_single()
    batch = make_batch(True)
    batch_without_general = make_batch(False)
    general = ("curr_v_node_id", "action_mask", "v_net_size")

    specifications: list[
        tuple[str, Callable[[Any, str], dict[str, torch.Tensor]], Any,
              tuple[str, ...]]
    ] = [
        ("single_mlp", convertor.obs_as_tensor_for_mlp, single,
         ("p_net_x",) + general),
        ("batch_mlp", convertor.obs_as_tensor_for_mlp, batch,
         ("p_net_x",) + general),
        ("batch_mlp_no_general", convertor.obs_as_tensor_for_mlp,
         batch_without_general, ("p_net_x",)),
        ("single_cnn", convertor.obs_as_tensor_for_cnn, single,
         ("p_net_x",) + general),
        ("batch_cnn", convertor.obs_as_tensor_for_cnn, batch,
         ("p_net_x",) + general),
        ("single_mlp_fc", convertor.obs_as_tensor_for_mlp_fc, single,
         ("pv_nodes_x",) + general),
        ("batch_mlp_fc", convertor.obs_as_tensor_for_mlp_fc, batch,
         ("pv_nodes_x",) + general),
        ("single_att", convertor.obs_as_tensor_for_att, single,
         ("p_net_x", "v_node_x") + general),
        ("batch_att", convertor.obs_as_tensor_for_att, batch,
         ("p_net_x", "v_node_x") + general),
    ]
    result = []
    for name, operation, observation, fields in specifications:
        output = operation(observation, "cpu")
        result.append({"name": name, "fields": summarize_fields(output, fields)})
    return result


def run_benchmark(
    convertor: type,
    warmups: int,
    repetitions: int,
    iterations: int,
) -> dict[str, Any]:
    observation = make_production_batch()
    output: dict[str, torch.Tensor] | None = None
    for _ in range(warmups):
        for _ in range(iterations):
            output = convertor.obs_as_tensor_for_mlp(observation, "cpu")
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        for _ in range(iterations):
            output = convertor.obs_as_tensor_for_mlp(observation, "cpu")
        samples.append(time.perf_counter_ns() - started)
    if output is None:
        raise RuntimeError("dense TensorConvertor benchmark produced no output")
    return {
        "warmups": warmups,
        "repetitions": repetitions,
        "iterations": iterations,
        "samples_ns": samples,
        "output": tensor_summary(output["p_net_x"], False),
    }


def run_native(
    harness: pathlib.Path,
    threads: int,
    warmups: int,
    repetitions: int,
    iterations: int,
) -> dict[str, Any]:
    process = subprocess.run(
        [str(harness), "--threads", str(threads), "--warmups", str(warmups),
         "--repetitions", str(repetitions), "--iterations", str(iterations)],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if process.returncode != 0:
        raise RuntimeError(
            f"TensorConvertor harness failed ({process.returncode}): "
            f"{process.stderr.strip()}"
        )
    lines = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"expected one native JSON line, got {len(lines)}")
    result = json.loads(lines[0])
    if not isinstance(result, dict):
        raise RuntimeError("native TensorConvertor report is not an object")
    return result


def compact_value(value: Any) -> Any:
    if isinstance(value, list) and len(value) > 12:
        return value[:6] + [f"... {len(value) - 12} omitted ..."] + value[-6:]
    return value


def compare_exact(
    expected: Any,
    actual: Any,
    path: str,
    mismatches: list[dict[str, Any]],
) -> None:
    if len(mismatches) >= MAX_MISMATCHES:
        return
    if type(expected) is not type(actual):
        mismatches.append({
            "path": path, "python": compact_value(expected),
            "cpp": compact_value(actual), "reason": "type"})
        return
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            mismatches.append({
                "path": path, "python_keys": sorted(expected),
                "cpp_keys": sorted(actual), "reason": "keys"})
        for key in sorted(set(expected) & set(actual)):
            compare_exact(expected[key], actual[key], f"{path}.{key}", mismatches)
        return
    if isinstance(expected, list):
        if len(expected) != len(actual):
            mismatches.append({
                "path": path, "python_length": len(expected),
                "cpp_length": len(actual), "reason": "length"})
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare_exact(left, right, f"{path}[{index}]", mismatches)
        return
    if expected != actual:
        mismatches.append({
            "path": path, "python": expected, "cpp": actual,
            "reason": "value"})


def positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def validate_samples(value: Any, count: int, label: str) -> list[int]:
    if not isinstance(value, list) or len(value) != count:
        raise RuntimeError(f"{label} returned an invalid timing sample count")
    if any(type(item) is not int or item <= 0 for item in value):
        raise RuntimeError(f"{label} returned an invalid timing sample")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", "--native", dest="harness",
                        type=pathlib.Path, required=True)
    parser.add_argument("--python-source", type=pathlib.Path, required=True)
    parser.add_argument("--threads", type=positive, default=1)
    parser.add_argument("--warmups", type=positive, default=3)
    parser.add_argument("--repetitions", type=positive, default=5)
    parser.add_argument("--iterations", type=positive, default=20)
    parser.add_argument("--json-output", "--output", dest="json_output",
                        type=pathlib.Path)
    args = parser.parse_args()

    if sys.byteorder != "little":
        raise RuntimeError("TensorConvertor parity requires little-endian bytes")
    if np.__version__ != NUMPY_VERSION:
        raise RuntimeError(
            f"NumPy drift: expected {NUMPY_VERSION}, got {np.__version__}")
    convertor = load_convertor(args.python_source)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    expected_cases = run_cases(convertor)
    native = run_native(
        args.harness, args.threads, args.warmups,
        args.repetitions, args.iterations)
    python_benchmark = run_benchmark(
        convertor, args.warmups, args.repetitions, args.iterations)

    global_mismatches: list[dict[str, Any]] = []
    if native.get("component") != "tensor_convertor_dense":
        global_mismatches.append({
            "path": "component", "python": "tensor_convertor_dense",
            "cpp": native.get("component"), "reason": "value"})
    if native.get("threads") != args.threads:
        global_mismatches.append({
            "path": "threads", "python": args.threads,
            "cpp": native.get("threads"), "reason": "value"})
    cpp_version = str(native.get("libtorch_version", "unknown"))
    python_version = str(torch.__version__)
    if cpp_version.split("+", 1)[0] != python_version.split("+", 1)[0]:
        global_mismatches.append({
            "path": "torch_version", "python": python_version,
            "cpp": cpp_version, "reason": "value"})

    actual_cases = native.get("cases")
    if not isinstance(actual_cases, list):
        raise RuntimeError("native report has no cases array")
    case_results = []
    if len(actual_cases) != len(expected_cases):
        global_mismatches.append({
            "path": "cases", "python_length": len(expected_cases),
            "cpp_length": len(actual_cases), "reason": "length"})
    for index, expected in enumerate(expected_cases):
        mismatches: list[dict[str, Any]] = []
        if index < len(actual_cases):
            compare_exact(expected, actual_cases[index], f"cases[{index}]", mismatches)
        else:
            mismatches.append({"path": f"cases[{index}]", "reason": "missing"})
        case_results.append({
            "name": expected["name"],
            "status": "PASS" if not mismatches else "FAIL",
            "mismatches": mismatches,
        })

    native_benchmark = native.get("benchmark")
    if not isinstance(native_benchmark, dict):
        raise RuntimeError("native report has no benchmark object")
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
    passed = sum(item["status"] == "PASS" for item in case_results)
    status = "PASS"
    if global_mismatches or benchmark_mismatches or passed != len(expected_cases):
        status = "FAIL"

    report = {
        "component": "tensor_convertor_dense",
        "status": status,
        "source_size": SOURCE_SIZE,
        "source_sha256": SOURCE_SHA256,
        "numpy": np.__version__,
        "python_torch": python_version,
        "cpp_libtorch": cpp_version,
        "threads": args.threads,
        "exact_cases": {
            "passed": passed, "total": len(expected_cases),
            "results": case_results},
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
            "python_median_ns_per_conversion": python_median / args.iterations,
            "cpp_median_ns_per_conversion": cpp_median / args.iterations,
            "cpp_speedup": python_median / cpp_median,
            "output": python_benchmark["output"],
        },
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output is not None:
        args.json_output.write_text(encoded, encoding="utf-8")
    compact: dict[str, Any] = {
        "component": "tensor_convertor_dense",
        "status": status,
        "exact_cases": f"{passed}/{len(expected_cases)}",
        "benchmark_output": "PASS" if not benchmark_mismatches else "FAIL",
        "python_median_ns_per_conversion": python_median / args.iterations,
        "cpp_median_ns_per_conversion": cpp_median / args.iterations,
        "cpp_speedup": python_median / cpp_median,
        "threads": args.threads,
    }
    if status != "PASS":
        compact["global_mismatches"] = global_mismatches
        compact["failed_cases"] = [
            item for item in case_results if item["status"] == "FAIL"]
        compact["benchmark_mismatches"] = benchmark_mismatches
    print(json.dumps(compact, sort_keys=True, separators=(",", ":")))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
