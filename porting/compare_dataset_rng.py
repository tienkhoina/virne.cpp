#!/usr/bin/env python3
"""Exact NumPy-2.2.6 differential for the dataset RNG wrapper."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import struct
import subprocess
import warnings
from collections import OrderedDict
from dataclasses import dataclass

import compare_dataset_core


FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211


@dataclass(frozen=True)
class Case:
    name: str
    seed: int
    size: int
    distribution: str
    dtype: str
    kwargs: dict


def build_cases() -> list[Case]:
    cases = [
        Case(f"normal_default_float_n{size}", 17, size, "normal", "float", {})
        for size in (0, 1, 2, 7, 257)
    ]
    cases.extend(
        [
            Case(
                "normal_custom_int",
                123,
                257,
                "normal",
                "int",
                {"loc": -1.25, "scale": 4.5},
            ),
            Case(
                "normal_custom_bool",
                123,
                257,
                "normal",
                "bool",
                {"loc": -1.25, "scale": 4.5},
            ),
            Case(
                "normal_nan_int",
                123,
                7,
                "normal",
                "int",
                {"loc": float("nan"), "scale": 0.0},
            ),
            Case(
                "normal_positive_inf_int",
                123,
                7,
                "normal",
                "int",
                {"loc": float("inf"), "scale": 0.0},
            ),
            Case(
                "normal_huge_int",
                123,
                7,
                "normal",
                "int",
                {"loc": 1.0e300, "scale": 0.0},
            ),
            Case(
                "normal_zero_scale_float",
                123,
                7,
                "normal",
                "float",
                {"loc": 1.5, "scale": 0.0},
            ),
            Case(
                "normal_int_upper_limit",
                123,
                3,
                "normal",
                "int",
                {"loc": float(2**63), "scale": 0.0},
            ),
            Case(
                "normal_int_upper_inside",
                123,
                3,
                "normal",
                "int",
                {"loc": math.nextafter(float(2**63), 0.0), "scale": 0.0},
            ),
            Case(
                "normal_int_lower_limit",
                123,
                3,
                "normal",
                "int",
                {"loc": -float(2**63), "scale": 0.0},
            ),
            Case(
                "normal_int_lower_outside",
                123,
                3,
                "normal",
                "int",
                {"loc": math.nextafter(-float(2**63), -math.inf), "scale": 0.0},
            ),
            Case(
                "normal_nan_bool",
                123,
                3,
                "normal",
                "bool",
                {"loc": float("nan"), "scale": 0.0},
            ),
            Case(
                "normal_positive_inf_bool",
                123,
                3,
                "normal",
                "bool",
                {"loc": float("inf"), "scale": 0.0},
            ),
            Case(
                "normal_negative_zero_bool",
                123,
                3,
                "normal",
                "bool",
                {"loc": -0.0, "scale": 0.0},
            ),
            Case(
                "uniform_int_inclusive",
                91,
                513,
                "uniform",
                "int",
                {"low": -4, "high": 7},
            ),
            Case(
                "uniform_float",
                3,
                513,
                "uniform",
                "float",
                {"low": -2, "high": 7.5},
            ),
            Case(
                "uniform_int64_max_from_zero",
                99,
                9,
                "uniform",
                "int",
                {"low": 0, "high": 2**63 - 1},
            ),
            Case(
                "uniform_int64_max_near",
                99,
                17,
                "uniform",
                "int",
                {"low": 2**63 - 6, "high": 2**63 - 1},
            ),
            Case(
                "uniform_int64_max_signed",
                99,
                17,
                "uniform",
                "int",
                {"low": -4, "high": 2**63 - 1},
            ),
            Case(
                "uniform_full_int64",
                99,
                17,
                "uniform",
                "int",
                {"low": -(2**63), "high": 2**63 - 1},
            ),
            Case(
                "uniform_int64_max_singleton",
                99,
                4,
                "uniform",
                "int",
                {"low": 2**63 - 1, "high": 2**63 - 1},
            ),
        ]
    )
    for dtype in ("float", "int", "bool"):
        cases.append(
            Case(
                f"exponential_{dtype}",
                42,
                513,
                "exponential",
                dtype,
                {"scale": 0.5},
            )
        )
        cases.append(
            Case(
                f"poisson_{dtype}",
                12,
                777,
                "poisson",
                dtype,
                {"lam": 20.0},
            )
        )
    cases.extend(
        [
            Case(
                "exponential_zero_float",
                42,
                7,
                "exponential",
                "float",
                {"scale": 0.0},
            ),
            Case(
                "exponential_zero_int",
                42,
                7,
                "exponential",
                "int",
                {"scale": 0.0},
            ),
            Case(
                "exponential_nan_int",
                42,
                7,
                "exponential",
                "int",
                {"scale": float("nan")},
            ),
            Case(
                "exponential_positive_inf_int",
                42,
                7,
                "exponential",
                "int",
                {"scale": float("inf")},
            ),
        ]
    )
    for dtype in ("float", "int", "bool"):
        cases.append(
            Case(
                f"poisson_zero_{dtype}",
                12,
                7,
                "poisson",
                dtype,
                {"lam": 0.0},
            )
        )
    cases.append(
        Case(
            "poisson_large_float_rounding",
            12,
            5,
            "poisson",
            "float",
            {"lam": 1.0e16},
        )
    )
    cases.extend(
        [
            Case(
                "poisson_reciprocal",
                12,
                777,
                "poisson",
                "int",
                {"lam": 0.04, "reciprocal": True},
            ),
            Case("error_invalid_distribution", 99, 1, "customized", "float", {}),
            Case("error_invalid_value_kind", 99, 1, "normal", "bad", {}),
            Case(
                "error_uniform_bool",
                99,
                10,
                "uniform",
                "bool",
                {"low": 0, "high": 1},
            ),
            Case("error_uniform_bool_missing", 99, 1, "uniform", "bool", {}),
            Case(
                "error_uniform_bool_low_only",
                99,
                1,
                "uniform",
                "bool",
                {"low": 0},
            ),
            Case(
                "error_uniform_bool_high_only",
                99,
                1,
                "uniform",
                "bool",
                {"high": 1},
            ),
            Case(
                "error_uniform_bool_strings",
                99,
                1,
                "uniform",
                "bool",
                {"low": "x", "high": "y"},
            ),
            Case("error_uniform_missing_bounds", 99, 1, "uniform", "int", {}),
            Case(
                "error_uniform_missing_high",
                99,
                1,
                "uniform",
                "int",
                {"low": 0},
            ),
            Case(
                "error_uniform_explicit_none",
                99,
                1,
                "uniform",
                "int",
                {"low": None, "high": 1},
            ),
            Case("error_uniform_missing_bounds_n0", 99, 0, "uniform", "int", {}),
            Case("error_missing_scale", 99, 1, "exponential", "float", {}),
            Case(
                "error_explicit_none_scale",
                99,
                1,
                "exponential",
                "float",
                {"scale": None},
            ),
            Case("error_missing_scale_n0", 99, 0, "exponential", "float", {}),
            Case("error_missing_lambda", 99, 1, "poisson", "int", {}),
            Case(
                "error_explicit_none_lambda",
                99,
                1,
                "poisson",
                "int",
                {"lam": None},
            ),
            Case("error_missing_lambda_n0", 99, 0, "poisson", "int", {}),
            Case(
                "error_reciprocal_zero",
                99,
                1,
                "poisson",
                "int",
                {"lam": 0.0, "reciprocal": True},
            ),
            Case(
                "error_negative_scale",
                99,
                10,
                "normal",
                "float",
                {"scale": -1.0},
            ),
            Case(
                "error_negative_scale_n0",
                99,
                0,
                "normal",
                "float",
                {"scale": -1.0},
            ),
            Case(
                "error_negative_exponential",
                99,
                1,
                "exponential",
                "float",
                {"scale": -1.0},
            ),
            Case(
                "error_negative_exponential_n0",
                99,
                0,
                "exponential",
                "float",
                {"scale": -1.0},
            ),
            Case(
                "error_negative_poisson",
                99,
                1,
                "poisson",
                "int",
                {"lam": -1.0},
            ),
            Case(
                "error_negative_poisson_n0",
                99,
                0,
                "poisson",
                "int",
                {"lam": -1.0},
            ),
            Case(
                "error_string_location",
                99,
                1,
                "normal",
                "float",
                {"loc": "x"},
            ),
            Case(
                "error_explicit_none_normal",
                99,
                1,
                "normal",
                "float",
                {"loc": None},
            ),
        ]
    )
    large_int = Case(
        "",
        123,
        300_000,
        "normal",
        "int",
        {"loc": -1.25, "scale": 4.5},
    )
    large_bool = Case(
        "",
        123,
        300_000,
        "normal",
        "bool",
        {"loc": -1.25, "scale": 4.5},
    )
    for workers in ("1", "2", "4", "8", "auto"):
        cases.append(
            Case(
                f"large_normal_int_w{workers}",
                large_int.seed,
                large_int.size,
                large_int.distribution,
                large_int.dtype,
                large_int.kwargs,
            )
        )
        cases.append(
            Case(
                f"large_normal_bool_w{workers}",
                large_bool.seed,
                large_bool.size,
                large_bool.distribution,
                large_bool.dtype,
                large_bool.kwargs,
            )
        )
    for workers in ("1", "auto"):
        cases.append(
            Case(
                f"medium_exponential_int_w{workers}",
                42,
                192_000,
                "exponential",
                "int",
                {"scale": 0.5},
            )
        )
        cases.append(
            Case(
                f"medium_exponential_bool_w{workers}",
                42,
                192_000,
                "exponential",
                "bool",
                {"scale": 0.5},
            )
        )
    for workers in ("1", "2", "3", "4", "8", "auto"):
        cases.append(
            Case(
                f"large_exponential_int_w{workers}",
                42,
                600_000,
                "exponential",
                "int",
                {"scale": 0.5},
            )
        )
        cases.append(
            Case(
                f"large_exponential_bool_w{workers}",
                42,
                600_000,
                "exponential",
                "bool",
                {"scale": 0.5},
            )
        )
    return cases


def float_bits(value: float) -> str:
    return f"{struct.unpack('=Q', struct.pack('=d', value))[0]:016x}"


def fnv(values: list, dtype: str) -> tuple[int, int]:
    checksum = FNV_OFFSET
    width = 8 if dtype != "bool" else 1
    for value in values:
        if dtype == "float":
            encoded = struct.pack("<d", value)
        elif dtype == "int":
            encoded = int(value).to_bytes(8, "little", signed=True)
        else:
            encoded = bytes((1 if value else 0,))
        for byte in encoded:
            checksum ^= byte
            checksum = (checksum * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return checksum, len(values) * width


def serialize_data(values: list, dtype: str) -> str:
    prefix = {"float": "f", "int": "i", "bool": "b"}[dtype]
    if len(values) <= 1024:
        if dtype == "float":
            payload = ",".join(float_bits(value) for value in values)
        elif dtype == "int":
            payload = ",".join(str(value) for value in values)
        else:
            payload = ",".join("1" if value else "0" for value in values)
        return f"{prefix}:{payload}"
    checksum, output_bytes = fnv(values, dtype)
    return f"summary:{prefix}:{len(values)}:{checksum}:{output_bytes}"


def continuation(np) -> str:
    return (
        f"random={float_bits(float(np.random.random_sample()))};"
        f"normal={float_bits(float(np.random.normal()))}"
    )


ERROR_MAP = {
    "error_invalid_distribution": (
        AssertionError,
        "invalid_distribution",
        "resolve_distribution",
    ),
    "error_invalid_value_kind": (
        AssertionError,
        "invalid_value_kind",
        "cast_values",
    ),
    "error_uniform_bool": (
        UnboundLocalError,
        "uniform_boolean_uninitialized",
        "generate_values",
    ),
    "error_uniform_bool_missing": (
        UnboundLocalError,
        "uniform_boolean_uninitialized",
        "generate_values",
    ),
    "error_uniform_bool_low_only": (
        UnboundLocalError,
        "uniform_boolean_uninitialized",
        "generate_values",
    ),
    "error_uniform_bool_high_only": (
        UnboundLocalError,
        "uniform_boolean_uninitialized",
        "generate_values",
    ),
    "error_uniform_bool_strings": (
        UnboundLocalError,
        "uniform_boolean_uninitialized",
        "generate_values",
    ),
    "error_uniform_missing_bounds": (
        TypeError,
        "missing_parameter",
        "generate_values",
    ),
    "error_uniform_missing_high": (
        TypeError,
        "missing_parameter",
        "generate_values",
    ),
    "error_uniform_explicit_none": (
        TypeError,
        "missing_parameter",
        "generate_values",
    ),
    "error_uniform_missing_bounds_n0": (
        TypeError,
        "missing_parameter",
        "generate_values",
    ),
    "error_missing_scale": (
        TypeError,
        "missing_parameter",
        "generate_values",
    ),
    "error_explicit_none_scale": (
        TypeError,
        "missing_parameter",
        "generate_values",
    ),
    "error_missing_scale_n0": (
        TypeError,
        "missing_parameter",
        "generate_values",
    ),
    "error_missing_lambda": (
        TypeError,
        "missing_parameter",
        "generate_values",
    ),
    "error_explicit_none_lambda": (
        TypeError,
        "missing_parameter",
        "generate_values",
    ),
    "error_missing_lambda_n0": (
        TypeError,
        "missing_parameter",
        "generate_values",
    ),
    "error_reciprocal_zero": (
        ZeroDivisionError,
        "invalid_parameter",
        "generate_values",
    ),
    "error_negative_scale": (
        ValueError,
        "rng_backend_failure",
        "generate_values",
    ),
    "error_negative_scale_n0": (
        ValueError,
        "rng_backend_failure",
        "generate_values",
    ),
    "error_negative_exponential": (
        ValueError,
        "rng_backend_failure",
        "generate_values",
    ),
    "error_negative_exponential_n0": (
        ValueError,
        "rng_backend_failure",
        "generate_values",
    ),
    "error_negative_poisson": (
        ValueError,
        "rng_backend_failure",
        "generate_values",
    ),
    "error_negative_poisson_n0": (
        ValueError,
        "rng_backend_failure",
        "generate_values",
    ),
    "error_string_location": (
        ValueError,
        "invalid_parameter",
        "generate_values",
    ),
    "error_explicit_none_normal": (
        TypeError,
        "invalid_parameter",
        "generate_values",
    ),
}

WARNING_MAP = {
    "normal_nan_int": [("RuntimeWarning", "invalid value encountered in cast")],
    "normal_positive_inf_int": [
        ("RuntimeWarning", "invalid value encountered in cast")
    ],
    "normal_huge_int": [("RuntimeWarning", "invalid value encountered in cast")],
    "normal_int_upper_limit": [
        ("RuntimeWarning", "invalid value encountered in cast")
    ],
    "normal_int_lower_outside": [
        ("RuntimeWarning", "invalid value encountered in cast")
    ],
    "exponential_nan_int": [
        ("RuntimeWarning", "invalid value encountered in cast")
    ],
    "exponential_positive_inf_int": [
        ("RuntimeWarning", "invalid value encountered in cast")
    ],
}


def python_cases(module) -> OrderedDict[str, tuple[str, ...]]:
    np = module.np
    results: OrderedDict[str, tuple[str, ...]] = OrderedDict()
    for case in build_cases():
        np.random.seed(case.seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                values = module.generate_data_with_distribution(
                    case.size,
                    case.distribution,
                    case.dtype,
                    **case.kwargs,
                )
                payload = serialize_data(values, case.dtype) + "|" + continuation(np)
                results[case.name] = ("ok", payload)
            except Exception as error:
                if case.name not in ERROR_MAP:
                    raise
                expected_type, code, operation = ERROR_MAP[case.name]
                if not isinstance(error, expected_type):
                    raise RuntimeError(
                        f"{case.name}: unexpected Python error "
                        f"{type(error).__name__}: {error}"
                    ) from error
                results[case.name] = ("error", code, operation, continuation(np))
        observed_warnings = [
            (item.category.__name__, str(item.message)) for item in caught
        ]
        if observed_warnings != WARNING_MAP.get(case.name, []):
            raise RuntimeError(
                f"Python warning drift for {case.name}: {observed_warnings!r}"
            )
    return results


def cpp_cases(harness: pathlib.Path) -> OrderedDict[str, tuple[str, ...]]:
    process = subprocess.run(
        [str(harness), "cases"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"C++ dataset RNG harness failed: {process.stderr}")
    lines = process.stdout.splitlines()
    if not lines or lines[0] != "dataset_rng_harness_version=1" or lines[-1] != "status=PASS":
        raise RuntimeError("invalid dataset RNG harness protocol")
    results: OrderedDict[str, tuple[str, ...]] = OrderedDict()
    for line in lines[1:-1]:
        fields = line.removeprefix("case=").split("|")
        if len(fields) == 3 and fields[1] == "ok":
            results[fields[0]] = (
                "ok",
                bytes.fromhex(fields[2]).decode("ascii"),
            )
        elif len(fields) == 5 and fields[1] == "error":
            results[fields[0]] = (
                "error",
                fields[2],
                fields[3],
                bytes.fromhex(fields[4]).decode("ascii"),
            )
        else:
            raise RuntimeError(f"malformed dataset RNG case: {line!r}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--python-source", type=pathlib.Path, required=True)
    parser.add_argument("--json-output", type=pathlib.Path)
    args = parser.parse_args()

    module = compare_dataset_core.load_oracle(args.python_source)
    expected = python_cases(module)
    actual = cpp_cases(args.harness)
    if list(expected) != list(actual):
        raise RuntimeError("dataset RNG case inventory/order mismatch")
    mismatches = [
        (name, expected[name], actual[name])
        for name in expected
        if expected[name] != actual[name]
    ]
    if mismatches:
        def compact(value: tuple[str, ...]) -> str:
            rendered = repr(value)
            if len(rendered) <= 320:
                return rendered
            digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16]
            return f"{rendered[:280]}...<len={len(rendered)},sha256={digest}>"

        details = "\n".join(
            f"{name}: python={compact(python)} cpp={compact(cpp)}"
            for name, python, cpp in mismatches[:12]
        )
        raise RuntimeError(f"dataset RNG differential mismatch:\n{details}")

    payload = {
        "cases": len(expected),
        "harness_sha256": hashlib.sha256(args.harness.read_bytes()).hexdigest(),
        "numpy_version": module.np.__version__,
        "python_warning_cases": WARNING_MAP,
        "source_sha256": compare_dataset_core.SOURCE_SHA256,
        "status": "PASS",
        "torch_backend": "controlled-fake-only",
    }
    if args.json_output:
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"dataset RNG differential: PASS ({len(expected)}/{len(expected)} cases)")
    print(f"numpy={module.np.__version__}")
    print("continuation=random+normal exact")
    print(f"python_cast_warning_cases={len(WARNING_MAP)} locked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
