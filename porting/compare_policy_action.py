#!/usr/bin/env python3
"""AST-isolated parity gate and one timing case for policy action helpers."""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
import pathlib
import statistics
import subprocess
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical


FIXTURE_VERSION = 1
EXPECTED_RL_METHOD_SHA256 = {
    "get_action_prob_dist": (
        "26fbf85787c899f9f22af5a100ce612b881b34e6fbf5729f63789f455872facf"
    ),
    "select_action": (
        "a5428e4f299954e211c3bf8b9475e1f6904097601923695f63f170aecc58f9f4"
    ),
    "evaluate_actions": (
        "4a0951519b8e114945d308e5a614f25dba155a245c10e3618c17637b5c13504f"
    ),
}
EXPECTED_MASK_METHOD_SHA256 = (
    "558964667b0ba69bcbf819542a0cd46d7db39d009998b50ecdacc76fe192e9e2"
)
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1
MAX_REPORTED_MISMATCHES = 48


def _source_paths() -> tuple[pathlib.Path, pathlib.Path]:
    learning = (
        pathlib.Path(__file__).resolve().parents[2]
        / "virne"
        / "virne"
        / "solver"
        / "learning"
    )
    return learning / "rl_core" / "rl_solver.py", learning / "utils.py"


def _source_segment(
    source_text: str,
    node: ast.AST,
    *,
    label: str,
) -> tuple[str, str]:
    segment = ast.get_source_segment(source_text, node)
    if segment is None:
        raise RuntimeError(f"cannot isolate {label} source")
    digest = hashlib.sha256(segment.encode("utf-8")).hexdigest()
    return segment, digest


def _load_mask_oracle(source_path: pathlib.Path) -> tuple[Any, dict[str, Any]]:
    source_text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source_text, source_path)
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "apply_mask_to_logit"
    ]
    if len(matches) != 1:
        raise RuntimeError("expected exactly one top-level apply_mask_to_logit")
    function = matches[0]
    _, digest = _source_segment(
        source_text, function, label="apply_mask_to_logit"
    )
    if digest != EXPECTED_MASK_METHOD_SHA256:
        raise RuntimeError(
            "apply_mask_to_logit oracle SHA256 changed: "
            f"expected {EXPECTED_MASK_METHOD_SHA256}, got {digest}"
        )
    isolated = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace: dict[str, Any] = {
        "torch": torch,
        "NEG_TENSER": torch.tensor(-1e8).float(),
    }
    exec(compile(isolated, str(source_path), "exec"), namespace)
    return namespace["apply_mask_to_logit"], {
        "path": str(source_path),
        "file_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "method_sha256": digest,
    }


def _load_rl_solver_oracle(
    source_path: pathlib.Path,
    apply_mask_to_logit: Any,
) -> tuple[type[Any], dict[str, Any]]:
    source_text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source_text, source_path)
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RLSolver"
    ]
    if len(classes) != 1:
        raise RuntimeError("expected exactly one top-level RLSolver class")
    methods_by_name = {
        node.name: node
        for node in classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in EXPECTED_RL_METHOD_SHA256
    }
    if set(methods_by_name) != set(EXPECTED_RL_METHOD_SHA256):
        raise RuntimeError(
            "RLSolver does not contain the required policy-action methods"
        )

    methods: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    method_hashes: dict[str, str] = {}
    method_segments: list[str] = []
    for name in EXPECTED_RL_METHOD_SHA256:
        method = methods_by_name[name]
        segment, digest = _source_segment(
            source_text, method, label=f"RLSolver.{name}"
        )
        expected = EXPECTED_RL_METHOD_SHA256[name]
        if digest != expected:
            raise RuntimeError(
                f"RLSolver.{name} oracle SHA256 changed: "
                f"expected {expected}, got {digest}"
            )
        methods.append(method)
        method_hashes[name] = digest
        method_segments.append(segment)

    class_arguments: dict[str, Any] = {
        "name": "_OracleRLSolver",
        "bases": [],
        "keywords": [],
        "body": methods,
        "decorator_list": [],
    }
    if "type_params" in ast.ClassDef._fields:
        class_arguments["type_params"] = []
    isolated_class = ast.ClassDef(**class_arguments)
    isolated = ast.Module(body=[isolated_class], type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace: dict[str, Any] = {
        "torch": torch,
        "F": F,
        "Categorical": Categorical,
        "apply_mask_to_logit": apply_mask_to_logit,
    }
    exec(compile(isolated, str(source_path), "exec"), namespace)
    combined = hashlib.sha256(
        "\n\n".join(method_segments).encode("utf-8")
    ).hexdigest()
    return namespace["_OracleRLSolver"], {
        "path": str(source_path),
        "file_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "method_sha256": method_hashes,
        "combined_method_sha256": combined,
    }


RL_SOURCE, UTILS_SOURCE = _source_paths()
ORACLE_APPLY_MASK_TO_LOGIT, MASK_SOURCE_METADATA = _load_mask_oracle(
    UTILS_SOURCE
)
ORACLE_RL_SOLVER, RL_SOURCE_METADATA = _load_rl_solver_oracle(
    RL_SOURCE, ORACLE_APPLY_MASK_TO_LOGIT
)
ORACLE_BACKEND = "ast_isolated_torch_2_6_categorical"


class _ActionPolicy:
    def act(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        return observation["logits"]


class _ValuePredictPolicy(_ActionPolicy):
    predictor = True

    def evaluate(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        return observation["values"]

    def predict(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        return observation["predicted_mask"]


def _make_solver(
    *,
    temperature: float = 1.0,
    mask_actions: bool = True,
    maskable_policy: bool = True,
    values_and_predictor: bool = False,
) -> Any:
    result = ORACLE_RL_SOLVER()
    result.policy = _ValuePredictPolicy() if values_and_predictor else _ActionPolicy()
    result.softmax_temp = temperature
    result.config = SimpleNamespace(
        rl=SimpleNamespace(
            mask_actions=mask_actions,
            maskable_policy=maskable_policy,
        )
    )
    return result


def _raw_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().to("cpu").contiguous().reshape(-1)
    return bytes(value.view(torch.uint8).tolist())


def _fnv1a(payload: bytes) -> int:
    checksum = FNV_OFFSET
    for byte in payload:
        checksum ^= byte
        checksum = (checksum * FNV_PRIME) & MASK64
    return checksum


def _dtype_name(dtype: torch.dtype) -> tuple[str, int]:
    names = {
        torch.float32: ("float32", 4),
        torch.float64: ("float64", 8),
        torch.int64: ("int64", 8),
    }
    try:
        return names[dtype]
    except KeyError as error:
        raise RuntimeError(
            f"unsupported policy-action summary dtype: {dtype}"
        ) from error


def tensor_summary(
    value: torch.Tensor,
    include_bits: bool = True,
) -> dict[str, Any]:
    tensor = value.detach().to("cpu").contiguous()
    payload = _raw_bytes(tensor)
    dtype, width = _dtype_name(tensor.dtype)
    bits: list[str] = []
    if include_bits:
        bits = [
            format(
                int.from_bytes(
                    payload[offset:offset + width], "little", signed=False
                ),
                f"0{width * 2}x",
            )
            for offset in range(0, len(payload), width)
        ]
    return {
        "shape": list(tensor.shape),
        "dtype": dtype,
        "device": str(value.device),
        "checksum": str(_fnv1a(payload)),
        "bytes": len(payload),
        "bits": bits,
    }


def _append_case(
    output: list[dict[str, Any]],
    name: str,
    value: torch.Tensor,
    include_bits: bool = True,
) -> None:
    output.append({
        "name": name,
        "output": tensor_summary(value, include_bits=include_bits),
    })


def _logits_fixture(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, -0.5, 0.25, 2.0],
            [-1.0, 0.5, 1.5, 0.0],
            [0.1, 0.2, 0.3, 0.4],
        ],
        dtype=torch.float32,
        device=device,
    )


def _mask_fixture(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[1, 0, 1, 1], [0, 1, 1, 0], [0, 0, 0, 0]],
        dtype=torch.bool,
        device=device,
    )


def _numpy_tensor(
    value: Any,
    *,
    device: torch.device,
    scalar_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy()).to(device)
    if scalar_dtype is None:
        return torch.as_tensor(value, device=device)
    return torch.tensor(value, dtype=scalar_dtype, device=device)


def exact_cases(device: torch.device) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    logits = _logits_fixture(device)
    mask = _mask_fixture(device)
    observation = {"logits": logits, "action_mask": mask}

    solver = _make_solver(temperature=0.75)
    probabilities, candidate_logits = solver.get_action_prob_dist(observation)
    _append_case(output, "distribution.probabilities", probabilities)
    _append_case(output, "distribution.candidate_logits", candidate_logits)

    action, log_probability = solver.select_action(observation, sample=False)
    _append_case(
        output,
        "greedy.actions",
        _numpy_tensor(action, device=device, scalar_dtype=torch.int64),
    )
    _append_case(
        output,
        "greedy.log_probabilities",
        _numpy_tensor(log_probability, device=device),
    )

    raw_solver = _make_solver(temperature=0.75, maskable_policy=False)
    action, log_probability = raw_solver.select_action(observation, sample=False)
    _append_case(
        output,
        "greedy_raw.actions",
        _numpy_tensor(action, device=device, scalar_dtype=torch.int64),
    )
    _append_case(
        output,
        "greedy_raw.log_probabilities",
        _numpy_tensor(log_probability, device=device),
    )

    torch.manual_seed(90210)
    action, log_probability = solver.select_action(observation, sample=True)
    _append_case(
        output,
        "sample.actions",
        _numpy_tensor(action, device=device, scalar_dtype=torch.int64),
    )
    _append_case(
        output,
        "sample.log_probabilities",
        _numpy_tensor(log_probability, device=device),
    )
    _append_case(
        output,
        "sample.rng_continuation",
        torch.rand((4,), device=device),
    )

    grad_logits = logits.clone().requires_grad_(True)
    values = torch.tensor(
        [[0.5], [1.5], [-0.25]], dtype=torch.float32, device=device
    ).requires_grad_(True)
    predicted = torch.tensor(
        [
            [0.8, 0.2, 0.7, 0.9],
            [0.1, 0.6, 0.8, 0.2],
            [0.4, 0.3, 0.2, 0.1],
        ],
        dtype=torch.float32,
        device=device,
    ).requires_grad_(True)
    actions = torch.tensor([3, 2, 1], dtype=torch.int64, device=device)
    evaluation_observation = {
        "logits": grad_logits,
        "action_mask": mask,
        "values": values,
        "predicted_mask": predicted,
    }
    evaluation_solver = _make_solver(
        temperature=0.5, values_and_predictor=True
    )
    with contextlib.redirect_stdout(io.StringIO()):
        (
            evaluated_values,
            evaluated_log_probabilities,
            entropy,
            other,
        ) = evaluation_solver.evaluate_actions(
            evaluation_observation, actions, return_others=True
        )
    _append_case(output, "evaluation.values", evaluated_values)
    _append_case(
        output, "evaluation.log_probabilities", evaluated_log_probabilities
    )
    _append_case(output, "evaluation.entropy", entropy)
    _append_case(
        output,
        "evaluation.masked_action_probability",
        other["mask_actions_probs"],
    )
    _append_case(output, "evaluation.prediction_loss", other["prediction_loss"])
    loss = (
        evaluated_log_probabilities.mean()
        + entropy.mean()
        + evaluated_values.mean()
        + other["mask_actions_probs"]
        + other["prediction_loss"]
    )
    loss.backward()
    _append_case(output, "evaluation.logits_grad", grad_logits.grad)
    _append_case(output, "evaluation.values_grad", values.grad)
    _append_case(output, "evaluation.predicted_mask_grad", predicted.grad)

    raw_evaluation_solver = _make_solver(
        temperature=0.5, mask_actions=False
    )
    (
        raw_values,
        raw_log_probabilities,
        raw_entropy,
        raw_other,
    ) = raw_evaluation_solver.evaluate_actions(
        observation, actions, return_others=True
    )
    if raw_values is not None:
        raise RuntimeError("raw policy unexpectedly produced values")
    _append_case(
        output,
        "evaluation_raw.log_probabilities",
        raw_log_probabilities,
    )
    _append_case(output, "evaluation_raw.entropy", raw_entropy)
    _append_case(
        output,
        "evaluation_raw.masked_action_probability",
        raw_other["mask_actions_probs"],
    )

    scalar_logits = torch.tensor(
        [0.2, -0.3, 1.1], dtype=torch.float64, device=device
    )
    scalar_solver = _make_solver()
    scalar_action, scalar_log_probability = scalar_solver.select_action(
        {"logits": scalar_logits}, sample=False
    )
    _append_case(
        output,
        "scalar.actions",
        _numpy_tensor(
            scalar_action, device=device, scalar_dtype=torch.int64
        ),
    )
    _append_case(
        output,
        "scalar.log_probabilities",
        _numpy_tensor(scalar_log_probability, device=device),
    )
    _, _, scalar_entropy = scalar_solver.evaluate_actions(
        {"logits": scalar_logits},
        torch.tensor(2, dtype=torch.int64, device=device),
    )
    _append_case(output, "scalar.entropy", scalar_entropy)
    return output


def _benchmark_fixture(
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    rows = 256
    action_count = 256
    logits = (
        torch.arange(
            rows * action_count, dtype=torch.float32, device=device
        )
        .reshape(rows, action_count)
        .remainder(97)
        .mul_(0.03125)
        .sub_(1.5)
    )
    mask = (
        torch.arange(
            rows * action_count, dtype=torch.int64, device=device
        )
        .reshape(rows, action_count)
        .remainder(5)
        .ne(0)
    )
    actions = (
        torch.arange(rows, dtype=torch.int64, device=device)
        .mul(37)
        .remainder(action_count)
    )
    return {"logits": logits, "action_mask": mask}, actions


def benchmark_python(
    device: torch.device,
    warmups: int,
    repetitions: int,
    iterations: int,
) -> dict[str, Any]:
    observation, actions = _benchmark_fixture(device)
    solver = _make_solver(temperature=0.75)
    sampled: tuple[Any, ...] | None = None
    with torch.inference_mode():
        for _ in range(warmups):
            for _ in range(iterations):
                sampled = solver.evaluate_actions(
                    observation, actions, return_others=True
                )
        samples_ns: list[int] = []
        for _ in range(repetitions):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter_ns()
            for _ in range(iterations):
                sampled = solver.evaluate_actions(
                    observation, actions, return_others=True
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples_ns.append(time.perf_counter_ns() - started)
    if sampled is None:
        raise RuntimeError("Python policy-action benchmark produced no output")
    values, log_probabilities, entropy, other = sampled
    if values is not None:
        raise RuntimeError("benchmark policy unexpectedly produced values")
    output: list[dict[str, Any]] = []
    _append_case(
        output,
        "benchmark.log_probabilities",
        log_probabilities,
        include_bits=False,
    )
    _append_case(
        output, "benchmark.entropy", entropy, include_bits=False
    )
    _append_case(
        output,
        "benchmark.masked_action_probability",
        other["mask_actions_probs"],
        include_bits=False,
    )
    return {
        "name": "evaluate_256x256",
        "warmups": warmups,
        "repetitions": repetitions,
        "iterations": iterations,
        "samples_ns": samples_ns,
        "output": output,
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


def run_native(
    args: argparse.Namespace,
    device: torch.device,
    *,
    benchmark: bool,
) -> dict[str, Any]:
    command = [
        str(args.native),
        "--device", device.type,
        "--threads", str(args.threads),
        "--warmups", str(args.warmups),
        "--repetitions", str(args.repetitions),
        "--iterations", str(args.iterations),
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
            "native policy-action harness failed: "
            f"{process.stderr.strip()}"
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


def _require_torch_2_6() -> None:
    release = torch.__version__.split("+", maxsplit=1)[0]
    if not release.startswith("2.6."):
        raise RuntimeError(
            "policy-action comparator requires PyTorch 2.6.x to match "
            f"LibTorch 2.6; found {torch.__version__}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", "--harness", required=True, type=pathlib.Path)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--json-output", "--output", type=pathlib.Path)
    args = parser.parse_args()
    for name in ("threads", "warmups", "repetitions", "iterations"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} requires a positive integer")
    _require_torch_2_6()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    expected = {
        "component": "policy_action",
        "device": device.type,
        "threads": args.threads,
        "cases": exact_cases(device),
    }
    # Exact parity is a separate native execution. Timing is never reached
    # when the semantic gate fails.
    native_exact = run_native(args, device, benchmark=False)
    mismatches = compare_exact(expected, native_exact)
    report: dict[str, Any] = {
        "component": "policy_action",
        "fixture_version": FIXTURE_VERSION,
        "torch_version": torch.__version__,
        "python_sources": {
            "rl_solver": RL_SOURCE_METADATA,
            "action_mask": MASK_SOURCE_METADATA,
        },
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
        native_with_benchmark = run_native(args, device, benchmark=True)
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
            device,
            args.warmups,
            args.repetitions,
            args.iterations,
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
        native_samples = [
            int(value) / args.iterations
            for value in native_benchmark["samples_ns"]
        ]
        python_samples = [
            int(value) / args.iterations
            for value in python_benchmark["samples_ns"]
        ]
        native_median = statistics.median(native_samples)
        python_median = statistics.median(python_samples)
        report["benchmark_mismatches"] = benchmark_mismatches
        report["benchmark"] = {
            "name": python_benchmark["name"],
            "native_scope": "evaluate_policy_actions(256x256)",
            "python_scope": "AST RLSolver.evaluate_actions(256x256)",
            "device": device.type,
            "threads": args.threads,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "iterations": args.iterations,
            "native_samples_ns_per_call": native_samples,
            "python_samples_ns_per_call": python_samples,
            "native_median_ns_per_call": native_median,
            "python_median_ns_per_call": python_median,
            "python_over_native_speedup": python_median / native_median,
            "output_checksums": {
                item["name"]: item["output"]["checksum"]
                for item in python_benchmark["output"]
            },
        }
        if benchmark_mismatches:
            report["status"] = "FAIL"

    _write_report(args, report)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
