#!/usr/bin/env python3
"""Exact differential for the typed attribute-method policy leaf."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pathlib
import struct
import subprocess
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable


SOURCE_SHA256 = "e17499af8e6ffbdb12f2100dd58abeab48dd06d92feb442ef192f8b9310b6b4f"
SOURCE_SIZE = 5_543
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


@dataclass(frozen=True)
class UpdateCase:
    name: str
    virtual_value: bool | int | float
    physical_value: bool | int | float
    method: str
    safe: bool = True
    diagnostic_name: str = "cpu"
    range_boundary: bool = False


@dataclass(frozen=True)
class CalculationCase:
    name: str
    virtual_value: bool | int | float
    physical_value: bool | int | float
    method: str
    restriction: str
    range_boundary: bool = False


def default_source() -> pathlib.Path:
    workspace = pathlib.Path(__file__).resolve().parents[2]
    return workspace / "virne" / "virne" / "network" / "attribute" / "attribute_method.py"


def load_original(source: pathlib.Path) -> ModuleType:
    payload = source.read_bytes()
    actual_hash = hashlib.sha256(payload).hexdigest()
    if len(payload) != SOURCE_SIZE or actual_hash != SOURCE_SHA256:
        raise RuntimeError(
            "attribute_method.py identity mismatch: "
            f"size={len(payload)}, sha256={actual_hash}"
        )
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location(
        "_virne_pinned_attribute_method_oracle", source
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to create private attribute-method module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    imported = set(sys.modules) - before
    forbidden = sorted(
        name
        for name in imported
        if name == "virne"
        or name.startswith("virne.")
        or name == "torch"
        or name.startswith("torch.")
        or name == "omegaconf"
        or name.startswith("omegaconf.")
    )
    if forbidden:
        raise RuntimeError(f"oracle imported forbidden modules: {forbidden}")
    return module


def float_bits(value: float) -> str:
    return f"{struct.unpack('>Q', struct.pack('>d', value))[0]:016x}"


def float_from_bits(value: str) -> float:
    if len(value) != 16:
        raise ValueError(f"double bit string must have 16 digits: {value!r}")
    return struct.unpack(">d", struct.pack(">Q", int(value, 16)))[0]


def number_token(value: bool | int | float) -> str:
    if type(value) is bool:
        return f"b:{int(value)}"
    if type(value) is int:
        if value < INT64_MIN or value > INT64_MAX:
            raise OverflowError(f"integer is outside int64: {value}")
        return f"i:{value}"
    if type(value) is float:
        return f"f:{float_bits(value)}"
    raise TypeError(f"unsupported typed number: {type(value).__name__}")


def exact_python_number(value: Any) -> str:
    if type(value) is bool:
        return f"b:{int(value)}"
    if type(value) is int:
        return f"i:{value}"
    if type(value) is float:
        return f"f:{float_bits(value)}"
    raise TypeError(f"unexpected Python scalar result: {type(value).__name__}")


def encode_text(value: str) -> str:
    return value.encode("utf-8").hex()


def run_harness(harness: pathlib.Path, *arguments: str) -> dict[str, str]:
    process = subprocess.run(
        [str(harness), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"C++ harness failed for {arguments!r}: {process.stderr.strip()}"
        )
    result: dict[str, str] = {}
    for line in process.stdout.splitlines():
        if "=" not in line:
            raise RuntimeError(f"malformed harness line: {line!r}")
        key, value = line.split("=", 1)
        if key in result:
            raise RuntimeError(f"duplicate harness key: {key}")
        result[key] = value
    if result.get("version") != "1" or result.get("status") not in {"ok", "error"}:
        raise RuntimeError(f"invalid harness response: {result!r}")
    return result


def expect_cpp_error(
    response: dict[str, str], code: str, operation: str, context: str
) -> None:
    if (
        response.get("status") != "error"
        or response.get("error_code") != code
        or response.get("operation") != operation
        or not response.get("message_hex")
    ):
        raise AssertionError(
            f"{context}: expected {code}/{operation}, got {response!r}"
        )


def expect_exception(
    callable_: Callable[[], Any], exception_type: type[BaseException], message: str
) -> BaseException:
    try:
        callable_()
    except exception_type as error:
        if str(error) != message:
            raise AssertionError(
                f"exception message mismatch: expected {message!r}, got {str(error)!r}"
            ) from error
        return error
    except BaseException as error:
        raise AssertionError(
            f"exception type mismatch: expected {exception_type.__name__}, "
            f"got {type(error).__name__}: {error}"
        ) from error
    raise AssertionError(f"expected {exception_type.__name__}: {message}")


def resource_instance(module: ModuleType):
    class Resource(module.ResourceAttributeMethod):
        pass

    result = Resource()
    result.type = "resource"
    result.name = "cpu"
    return result


def run_update_cases(module: ModuleType, harness: pathlib.Path) -> list[dict]:
    negative_zero = struct.unpack(">d", bytes.fromhex("8000000000000000"))[0]
    quiet_nan = struct.unpack(">d", bytes.fromhex("7ff8000000000042"))[0]
    cases = [
        UpdateCase("add_int_plus", 3, 10, "+"),
        UpdateCase("add_int_alias", -4, 10, "add", False),
        UpdateCase("add_bool_bool", True, True, "+"),
        UpdateCase("add_bool_int", True, 7, "+"),
        UpdateCase("add_int_bool", 2, True, "+"),
        UpdateCase("add_double_int", 0.5, 2, "+"),
        UpdateCase("add_int_double", 2, 0.5, "add"),
        UpdateCase("add_negative_zeros", negative_zero, negative_zero, "+"),
        UpdateCase("subtract_int", 3, 10, "-"),
        UpdateCase("subtract_alias", 3, 10, "sub"),
        UpdateCase("subtract_equal_safe", 5, 5, "-"),
        UpdateCase("subtract_negative_virtual", -2, 5, "-"),
        UpdateCase("subtract_unsafe_overcommit", 6, 5, "-", False),
        UpdateCase("subtract_guard_failure", 6, 5, "-", True),
        UpdateCase("subtract_nan_guard", quiet_nan, 5.0, "-", True),
        UpdateCase("subtract_infinity", 1.0, math.inf, "-", True),
        UpdateCase("add_int64_overflow", 1, INT64_MAX, "+", True, "cpu", True),
        UpdateCase("sub_int64_overflow", 1, INT64_MIN, "-", False, "cpu", True),
    ]
    records = []
    for case in cases:
        instance = resource_instance(module)
        virtual = {case.diagnostic_name: case.virtual_value}
        physical = {case.diagnostic_name: case.physical_value}
        try:
            returned = instance.update(virtual, physical, case.method, case.safe)
            python_record = {
                "status": "ok",
                "return_type": type(returned).__name__,
                "return_value": returned,
                "physical": exact_python_number(physical[case.diagnostic_name]),
            }
        except BaseException as error:
            python_record = {
                "status": "error",
                "type": type(error).__name__,
                "message": str(error),
                "physical": exact_python_number(physical[case.diagnostic_name]),
            }

        response = run_harness(
            harness,
            "update",
            number_token(case.virtual_value),
            number_token(case.physical_value),
            encode_text(case.method),
            "1" if case.safe else "0",
            encode_text(case.diagnostic_name),
        )
        if case.range_boundary:
            if python_record["status"] != "ok":
                raise AssertionError(f"{case.name}: Python arbitrary-int boundary failed")
            expect_cpp_error(
                response, "numeric_range", "update_resource", case.name
            )
            records.append(
                {
                    "name": case.name,
                    "boundary": "python_arbitrary_int_vs_cpp_int64",
                    "python": python_record,
                    "cpp": response,
                }
            )
            continue

        if python_record["status"] == "error":
            expected_message = (
                f"{case.diagnostic_name}: (v = {case.virtual_value}) > "
                f"(p = {case.physical_value})"
            )
            if (
                python_record["type"] != "ValueError"
                or python_record["message"] != expected_message
                or python_record["physical"] != number_token(case.physical_value)
            ):
                raise AssertionError(f"{case.name}: unexpected Python guard {python_record}")
            expect_cpp_error(
                response, "insufficient_resource", "update_resource", case.name
            )
        else:
            if python_record["return_type"] != "bool" or returned is not True:
                raise AssertionError(f"{case.name}: Python did not return literal True")
            if (
                response.get("status") != "ok"
                or response.get("return_type") != "bool"
                or response.get("return_value") != "true"
                or response.get("physical") != python_record["physical"]
            ):
                raise AssertionError(
                    f"{case.name}: typed update mismatch: "
                    f"python={python_record}, cpp={response}"
                )
        records.append({"name": case.name, "python": python_record, "cpp": response})
    return records


def run_resolver_cases(module: ModuleType, harness: pathlib.Path) -> list[dict]:
    records = []
    update_values = {
        "+": "add",
        "add": "add",
        "-": "subtract",
        "sub": "subtract",
    }
    for value in [*update_values, "", "Add", "subtract", "++", "==", "更新"]:
        response = run_harness(harness, "resolve_update", encode_text(value))
        instance = resource_instance(module)
        if value in update_values:
            if response.get("status") != "ok" or response.get("value") != update_values[value]:
                raise AssertionError(f"update resolver mismatch for {value!r}: {response}")
        else:
            expected = f"Update method '{value}' is not supported."
            expect_exception(
                lambda value=value: instance.update({"cpu": 1}, {"cpu": 2}, value),
                NotImplementedError,
                expected,
            )
            expect_cpp_error(
                response,
                "unsupported_update_operation",
                "resolve_update",
                f"update resolver {value!r}",
            )
        records.append({"family": "update", "input": value, "cpp": response})

    comparison_values = {
        ">=": "greater_equal",
        "ge": "greater_equal",
        "<=": "less_equal",
        "le": "less_equal",
        "eq": "equal",
    }
    calculator = object.__new__(module.ConstraintAttributeMethod)
    calculator.constraint_restrictions = "hard"
    for value in [*comparison_values, "==", "", "GE", "gt", "比較"]:
        response = run_harness(harness, "resolve_comparison", encode_text(value))
        if value in comparison_values:
            if (
                response.get("status") != "ok"
                or response.get("value") != comparison_values[value]
            ):
                raise AssertionError(
                    f"comparison resolver mismatch for {value!r}: {response}"
                )
        else:
            expect_exception(
                lambda value=value: calculator._calculate_satisfiability_values(
                    1, 2, value
                ),
                NotImplementedError,
                f"Used method {value}",
            )
            expect_cpp_error(
                response,
                "unsupported_comparison",
                "resolve_comparison",
                f"comparison resolver {value!r}",
            )
        records.append({"family": "comparison", "input": value, "cpp": response})

    for value, accepted in [
        ("hard", True),
        ("soft", True),
        ("", False),
        ("Hard", False),
        ("SOFT", False),
        (" hard", False),
        ("柔軟", False),
    ]:
        response = run_harness(harness, "resolve_restriction", encode_text(value))
        if accepted:
            if response.get("status") != "ok" or response.get("value") != value:
                raise AssertionError(
                    f"restriction resolver mismatch for {value!r}: {response}"
                )
        else:
            expect_cpp_error(
                response,
                "invalid_restriction",
                "resolve_restriction",
                f"restriction resolver {value!r}",
            )
        records.append({"family": "restriction", "input": value, "cpp": response})
    return records


def run_calculation_cases(module: ModuleType, harness: pathlib.Path) -> list[dict]:
    negative_zero = struct.unpack(">d", bytes.fromhex("8000000000000000"))[0]
    quiet_nan = struct.unpack(">d", bytes.fromhex("7ff8000000000042"))[0]
    cases = [
        CalculationCase("hard_ge_true_symbol", 5, 3, ">=", "hard"),
        CalculationCase("hard_ge_false_alias", 2, 3, "ge", "hard"),
        CalculationCase("hard_le_true_symbol", -4, 1, "<=", "hard"),
        CalculationCase("hard_le_false_alias", 4, 1, "le", "hard"),
        CalculationCase("hard_eq_true", 7, 7, "eq", "hard"),
        CalculationCase("hard_eq_false", -4, 1, "eq", "hard"),
        CalculationCase("soft_masks_ge", 2, 3, "ge", "soft"),
        CalculationCase("soft_preserves_offset", 4, 1, "le", "soft"),
        CalculationCase("bool_ge", True, False, "ge", "hard"),
        CalculationCase("bool_le", False, True, "le", "hard"),
        CalculationCase("mixed_int_double", 2, 2.5, "le", "hard"),
        CalculationCase("mixed_double_int", 2.5, 2, "ge", "hard"),
        CalculationCase(
            "mixed_eq_int_beyond_2p53",
            9007199254740993,
            9007199254740992.0,
            "eq",
            "hard",
        ),
        CalculationCase(
            "mixed_eq_double_below_int_beyond_2p53",
            9007199254740992.0,
            9007199254740993,
            "eq",
            "hard",
        ),
        CalculationCase(
            "mixed_eq_int64_max_vs_2p63",
            INT64_MAX,
            float(2**63),
            "eq",
            "hard",
        ),
        CalculationCase(
            "mixed_eq_int64_min_vs_negative_2p63",
            INT64_MIN,
            -float(2**63),
            "eq",
            "hard",
        ),
        CalculationCase("signed_zero_eq", negative_zero, 0.0, "eq", "hard"),
        CalculationCase("nan_le_hard", quiet_nan, 1.0, "le", "hard"),
        CalculationCase("nan_le_soft", quiet_nan, 1.0, "le", "soft"),
        CalculationCase("infinity_eq", math.inf, math.inf, "eq", "hard"),
        CalculationCase(
            "ge_int64_offset_overflow", INT64_MIN, INT64_MAX, "ge", "hard", True
        ),
        CalculationCase(
            "le_int64_offset_overflow", INT64_MAX, INT64_MIN, "le", "hard", True
        ),
        CalculationCase(
            "eq_int64_offset_overflow", INT64_MIN, INT64_MAX, "eq", "hard", True
        ),
    ]
    records = []
    for case in cases:
        calculator = object.__new__(module.ConstraintAttributeMethod)
        calculator.constraint_restrictions = case.restriction
        python_flag, python_offset = calculator._calculate_satisfiability_values(
            case.virtual_value, case.physical_value, case.method
        )
        python_record = {
            "flag_type": type(python_flag).__name__,
            "flag": python_flag,
            "offset": exact_python_number(python_offset),
        }
        response = run_harness(
            harness,
            "calculate",
            number_token(case.virtual_value),
            number_token(case.physical_value),
            encode_text(case.method),
            encode_text(case.restriction),
        )
        if case.range_boundary:
            expect_cpp_error(
                response,
                "numeric_range",
                "calculate_satisfiability",
                case.name,
            )
            records.append(
                {
                    "name": case.name,
                    "boundary": "python_arbitrary_int_vs_cpp_int64",
                    "python": python_record,
                    "cpp": response,
                }
            )
            continue
        expected_flag = "true" if bool(python_flag) else "false"
        if (
            type(python_flag) is not bool
            or response.get("status") != "ok"
            or response.get("flag_type") != "bool"
            or response.get("flag") != expected_flag
            or response.get("offset") != python_record["offset"]
        ):
            raise AssertionError(
                f"{case.name}: calculation mismatch: "
                f"python={python_record}, cpp={response}"
            )
        records.append({"name": case.name, "python": python_record, "cpp": response})
    return records


def python_batch_oracle(
    module: ModuleType,
    virtual_bits: list[str],
    physical_bits: list[str],
    method: str,
    restriction: str,
) -> tuple[str, str]:
    if len(virtual_bits) != len(physical_bits):
        raise ValueError("Python batch oracle shape mismatch")
    calculator = object.__new__(module.ConstraintAttributeMethod)
    calculator.constraint_restrictions = restriction
    flags = []
    offsets = []
    for index, (virtual_raw, physical_raw) in enumerate(
        zip(virtual_bits, physical_bits, strict=True)
    ):
        flag, offset = calculator._calculate_satisfiability_values(
            float_from_bits(virtual_raw),
            float_from_bits(physical_raw),
            method,
        )
        if type(flag) is not bool or type(offset) is not float:
            raise AssertionError(
                f"Python batch scalar representation drift at {index}: "
                f"flag={type(flag).__name__}, offset={type(offset).__name__}"
            )
        flags.append("1" if flag else "0")
        offsets.append(float_bits(offset))
    return ",".join(flags), ",".join(offsets)


def stable_signaling_nan_oracle(module: ModuleType) -> bool:
    virtual_bits = ["7ff0000000000001", "3ff0000000000000"]
    physical_bits = ["3ff0000000000000", "fff0000000000042"]
    try:
        for method in ("ge", "le", "eq"):
            for restriction in ("hard", "soft"):
                first = python_batch_oracle(
                    module,
                    virtual_bits,
                    physical_bits,
                    method,
                    restriction,
                )
                second = python_batch_oracle(
                    module,
                    virtual_bits,
                    physical_bits,
                    method,
                    restriction,
                )
                if first != second:
                    return False
    except (ArithmeticError, FloatingPointError, ValueError):
        return False
    return True


def run_batch_cases(module: ModuleType, harness: pathlib.Path) -> dict[str, Any]:
    pairs = [
        ("8000000000000000", "0000000000000000"),
        ("0000000000000000", "8000000000000000"),
        ("0000000000000001", "8000000000000001"),
        ("8000000000000001", "0000000000000001"),
        ("3ff0000000000000", "4000000000000000"),
        ("c00c000000000000", "c010000000000000"),
        ("7ff0000000000000", "7ff0000000000000"),
        ("fff0000000000000", "fff0000000000000"),
        ("7ff0000000000000", "fff0000000000000"),
        ("fff0000000000000", "7ff0000000000000"),
        ("7ff8000000000001", "3ff0000000000000"),
        ("3ff0000000000000", "7ff8000000000042"),
        ("fff80000000000a5", "bff0000000000000"),
        ("7ff9000000001234", "fff8000000004321"),
        ("7fefffffffffffff", "ffefffffffffffff"),
    ]
    signaling_nan_enabled = stable_signaling_nan_oracle(module)
    if signaling_nan_enabled:
        pairs.extend(
            [
                ("7ff0000000000001", "3ff0000000000000"),
                ("3ff0000000000000", "fff0000000000042"),
            ]
        )
    virtual_bits = [pair[0] for pair in pairs]
    physical_bits = [pair[1] for pair in pairs]
    virtual_argument = ",".join(virtual_bits)
    physical_argument = ",".join(physical_bits)
    workers_to_test = (0, 1, 2, 3, 8)
    records = []
    for method in ("ge", "le", "eq"):
        for restriction in ("hard", "soft"):
            expected_flags, expected_offsets = python_batch_oracle(
                module,
                virtual_bits,
                physical_bits,
                method,
                restriction,
            )
            expected_hash = hashlib.sha256(
                f"{expected_flags}|{expected_offsets}".encode("ascii")
            ).hexdigest()
            for workers in workers_to_test:
                response = run_harness(
                    harness,
                    "batch",
                    method,
                    restriction,
                    str(workers),
                    virtual_argument,
                    physical_argument,
                )
                required = {
                    "version",
                    "status",
                    "kind",
                    "operation",
                    "restriction",
                    "workers",
                    "count",
                    "flags",
                    "offset_bits",
                }
                if (
                    set(response) != required
                    or response.get("status") != "ok"
                    or response.get("kind") != "batch"
                    or response.get("operation") != method
                    or response.get("restriction") != restriction
                    or int(response.get("workers", "-1")) != workers
                    or int(response.get("count", "-1")) != len(pairs)
                    or response.get("flags") != expected_flags
                    or response.get("offset_bits") != expected_offsets
                ):
                    raise AssertionError(
                        f"batch mismatch for {method}/{restriction}/w{workers}: "
                        f"expected flags={expected_flags}, offsets={expected_offsets}; "
                        f"cpp={response}"
                    )
                records.append(
                    {
                        "operation": method,
                        "restriction": restriction,
                        "workers": workers,
                        "count": len(pairs),
                        "output_sha256": expected_hash,
                    }
                )
    return {
        "status": "PASS",
        "cases": records,
        "case_count": len(records),
        "values_per_case": len(pairs),
        "workers": list(workers_to_test),
        "signaling_nan_enabled": signaling_nan_enabled,
        "qnan_payload_count": 5,
    }


def run_dynamic_characterization(module: ModuleType) -> list[dict]:
    records: list[dict] = []

    class AccessProbe(module.ResourceAttributeMethod):
        def __init__(self, type_value: Any, name_value: Any):
            object.__setattr__(self, "trace", [])
            object.__setattr__(self, "type_value", type_value)
            object.__setattr__(self, "name_value", name_value)

        def __getattribute__(self, name: str):
            if name in {"type", "name"}:
                trace = object.__getattribute__(self, "trace")
                trace.append(name)
                value = object.__getattribute__(self, f"{name}_value")
                if value is _MISSING:
                    raise AttributeError(name)
                return value
            return object.__getattribute__(self, name)

    _MISSING = object()
    probe = AccessProbe("wrong", _MISSING)
    expect_exception(
        lambda: probe.update({}, {}, "+"),
        TypeError,
        "ResourceAttributeMethod requires 'type' == 'resource' and 'name' "
        "attribute in the main class.",
    )
    if probe.trace != ["type", "name"]:
        raise AssertionError(f"resource state access order drift: {probe.trace}")
    records.append({"name": "resource_state_access_order", "trace": probe.trace})

    empty_name = resource_instance(module)
    empty_name.name = ""
    empty_physical = {"": 4}
    if empty_name.update({"": 2}, empty_physical, "+") is not True or empty_physical[""] != 6:
        raise AssertionError("empty resource name boundary drift")
    records.append({"name": "resource_empty_name_is_valid", "value": 6})

    class TraceMap:
        def __init__(self, label: str, value: Any, trace: list[str]):
            self.label = label
            self.value = value
            self.trace = trace

        def __getitem__(self, key: Any):
            self.trace.append(f"{self.label}:get:{key}")
            return self.value

        def __setitem__(self, key: Any, value: Any):
            self.trace.append(f"{self.label}:set:{key}")
            self.value = value

    instance = resource_instance(module)
    trace: list[str] = []
    virtual = TraceMap("v", 3, trace)
    physical = TraceMap("p", 10, trace)
    if instance.update(virtual, physical, "+", object()) is not True:
        raise AssertionError("resource add did not return literal True")
    expected = ["p:get:cpu", "v:get:cpu", "p:set:cpu"]
    if trace != expected or physical.value != 13:
        raise AssertionError(f"resource add mapping order drift: {trace}")
    records.append({"name": "resource_add_mapping_order", "trace": trace})

    trace = []
    virtual = TraceMap("v", 3, trace)
    physical = TraceMap("p", 10, trace)
    instance.update(virtual, physical, "-", True)
    expected = [
        "v:get:cpu",
        "p:get:cpu",
        "p:get:cpu",
        "v:get:cpu",
        "p:set:cpu",
    ]
    if trace != expected or physical.value != 7:
        raise AssertionError(f"resource subtract mapping order drift: {trace}")
    records.append({"name": "resource_subtract_mapping_order", "trace": trace})

    trace = []
    virtual = TraceMap("v", 11, trace)
    physical = TraceMap("p", 10, trace)
    expect_exception(
        lambda: instance.update(virtual, physical, "-", True),
        ValueError,
        "cpu: (v = 11) > (p = 10)",
    )
    expected = ["v:get:cpu", "p:get:cpu", "v:get:cpu", "p:get:cpu"]
    if trace != expected or physical.value != 10:
        raise AssertionError(f"resource guard mapping order drift: {trace}")
    records.append({"name": "resource_guard_mapping_order", "trace": trace})

    class ExplosiveTruth:
        def __bool__(self):
            raise AssertionError("safe was evaluated during addition")

    physical_dict = {"cpu": 2}
    instance.update({"cpu": 1}, physical_dict, "+", ExplosiveTruth())
    if physical_dict != {"cpu": 3}:
        raise AssertionError("add-safe ignored boundary drift")
    records.append({"name": "resource_add_ignores_safe", "value": 3})

    class GenerateProbe(module.ResourceAttributeMethod):
        def __init__(self, generative: Any, delegate: Any = _MISSING):
            object.__setattr__(self, "trace", [])
            object.__setattr__(self, "generative_value", generative)
            object.__setattr__(self, "delegate_value", delegate)

        def __getattribute__(self, name: str):
            if name == "generative":
                object.__getattribute__(self, "trace").append("generative")
                return object.__getattribute__(self, "generative_value")
            if name == "_generate_data":
                object.__getattribute__(self, "trace").append("_generate_data")
                value = object.__getattribute__(self, "delegate_value")
                if value is _MISSING:
                    raise AttributeError(name)
                return value
            return object.__getattribute__(self, name)

    generate_probe = GenerateProbe(False)
    expect_exception(
        lambda: generate_probe.generate_data(object()),
        NotImplementedError,
        "Non-generative resource attribute must implement generate_data.",
    )
    if generate_probe.trace != ["generative"]:
        raise AssertionError(f"false generator inspected delegate: {generate_probe.trace}")
    records.append({"name": "false_generator_no_delegate_read", "trace": generate_probe.trace})

    network = object()
    marker = object()
    delegate_trace: list[str] = []

    def delegate(value: Any):
        delegate_trace.append("same" if value is network else "different")
        return marker

    generate_probe = GenerateProbe(True, delegate)
    returned = generate_probe.generate_data(network)
    if returned is not marker or delegate_trace != ["same"]:
        raise AssertionError("resource generation identity drift")
    records.append(
        {
            "name": "generator_argument_and_return_identity",
            "trace": generate_probe.trace + delegate_trace,
        }
    )

    generator_message = (
        "ResourceAttributeMethod requires '_generate_data' method in the main "
        "class for generative attributes."
    )
    noncallable_traces = []
    for noncallable in (_MISSING, None, 7):
        generate_probe = GenerateProbe(True, noncallable)
        expect_exception(
            lambda generate_probe=generate_probe: generate_probe.generate_data(network),
            NotImplementedError,
            generator_message,
        )
        if generate_probe.trace != ["generative", "_generate_data"]:
            raise AssertionError(
                f"non-callable generator trace drift: {generate_probe.trace}"
            )
        noncallable_traces.append(list(generate_probe.trace))
    records.append(
        {"name": "missing_and_noncallable_generators", "traces": noncallable_traces}
    )

    def failing_delegate(value: Any):
        if value is not network:
            raise AssertionError("delegate received a copied network")
        raise RuntimeError("delegate failure")

    generate_probe = GenerateProbe(True, failing_delegate)
    expect_exception(
        lambda: generate_probe.generate_data(network),
        RuntimeError,
        "delegate failure",
    )
    records.append({"name": "generator_exception_propagation", "trace": generate_probe.trace})

    class Explosive:
        def __getattribute__(self, name: str):
            raise AssertionError(f"no-op accessed {name}")

    explosive = Explosive()
    if module.ExtremaAttributeMethod.update(explosive, explosive, explosive) is not True:
        raise AssertionError("extrema update no-op drift")
    if (
        module.ExtremaAttributeMethod.check(
            explosive, explosive, explosive, explosive, explosive
        )
        is not True
    ):
        raise AssertionError("extrema check no-op drift")
    records.append({"name": "extrema_no_ops", "return_type": "bool"})

    class Origin:
        def __init__(self, label: str, returned: Any):
            self.label = label
            self.returned = returned
            self.expected_network: Any = None
            self.calls: list[str] = []

        def get_data(self, value: Any):
            self.calls.append("same" if value is self.expected_network else "different")
            return self.returned

    node_marker = object()
    link_marker = object()
    node_origin = Origin("node", node_marker)
    link_origin = Origin("link", link_marker)

    class Network:
        node_attrs = {"origin": node_origin}
        link_attrs = {"origin": link_origin}

    extrema_network = Network()
    node_origin.expected_network = extrema_network
    link_origin.expected_network = extrema_network
    extrema = object.__new__(module.ExtremaAttributeMethod)
    extrema.owner = "node"
    extrema.originator = "origin"
    if extrema.generate_data(extrema_network) is not node_marker:
        raise AssertionError("node extrema registry selection drift")
    extrema.owner = "graph"
    if extrema.generate_data(extrema_network) is not link_marker:
        raise AssertionError("graph extrema did not fall back to link registry")
    for owner in ("link", "", object()):
        extrema.owner = owner
        if extrema.generate_data(extrema_network) is not link_marker:
            raise AssertionError(f"non-node extrema fallback drift for {owner!r}")
    if node_origin.calls != ["same"] or link_origin.calls != ["same"] * 4:
        raise AssertionError("extrema delegate network identity drift")
    records.append(
        {
            "name": "extrema_non_node_link_fallback",
            "node_calls": node_origin.calls,
            "link_calls": link_origin.calls,
        }
    )

    missing_extrema = object.__new__(module.ExtremaAttributeMethod)
    missing_extrema.owner = None
    missing_extrema.originator = "origin"
    expect_exception(
        lambda: missing_extrema.generate_data(Network()),
        AttributeError,
        "ExtremaAttributeMethod requires 'owner' and 'originator' attributes "
        "in the main class.",
    )
    records.append({"name": "extrema_missing_field_error", "status": "locked"})

    extrema.owner = "link"
    extrema.originator = "missing"
    expect_exception(
        lambda: extrema.generate_data(extrema_network), KeyError, "'missing'"
    )
    extrema_network.link_attrs["no_get_data"] = object()
    extrema.originator = "no_get_data"
    expect_exception(
        lambda: extrema.generate_data(extrema_network),
        AttributeError,
        "'object' object has no attribute 'get_data'",
    )

    class FailingOrigin:
        def get_data(self, value: Any):
            if value is not extrema_network:
                raise AssertionError("extrema delegate received a copied network")
            raise RuntimeError("extrema delegate failure")

    extrema_network.link_attrs["failing"] = FailingOrigin()
    extrema.originator = "failing"
    expect_exception(
        lambda: extrema.generate_data(extrema_network),
        RuntimeError,
        "extrema delegate failure",
    )
    records.append(
        {"name": "extrema_lookup_and_delegate_failures", "status": "locked"}
    )

    class NextInitializer:
        def __init__(self, *args: Any, **kwargs: Any):
            self.trace.append(["next", list(args), sorted(kwargs.items())])
            self.is_constraint = kwargs.get("is_constraint", "next")

    class InformationProbe(module.InformationAttributeMethod, NextInitializer):
        def __init__(self, *args: Any, **kwargs: Any):
            object.__setattr__(self, "trace", [])
            super().__init__(*args, **kwargs)

        def __setattr__(self, name: str, value: Any):
            if name == "is_constraint":
                object.__getattribute__(self, "trace").append(
                    ["set", name, value]
                )
            object.__setattr__(self, name, value)

    information = InformationProbe("arg", is_constraint="kwarg")
    if information.is_constraint is not False:
        raise AssertionError("information constraint flag drift")
    if [entry[0] for entry in information.trace] != ["next", "set", "set"]:
        raise AssertionError(f"information cooperative order drift: {information.trace}")
    records.append({"name": "information_cooperative_mro", "trace": information.trace})

    class FailingNextInitializer:
        def __init__(self, *args: Any, **kwargs: Any):
            self.trace.append(["failing_next", list(args), sorted(kwargs.items())])
            raise RuntimeError("cooperative super failure")

    class FailingInformation(
        module.InformationAttributeMethod, FailingNextInitializer
    ):
        def __init__(self, *args: Any, **kwargs: Any):
            self.trace: list[Any] = []
            super().__init__(*args, **kwargs)

    failing_information = object.__new__(FailingInformation)
    failing_information.trace = []
    expect_exception(
        lambda: module.InformationAttributeMethod.__init__(
            failing_information, "arg", key="value"
        ),
        RuntimeError,
        "cooperative super failure",
    )
    if hasattr(failing_information, "is_constraint"):
        raise AssertionError("information assigned after cooperative super failure")
    records.append(
        {"name": "information_super_failure_before_assignment", "trace": failing_information.trace}
    )

    class ConstraintProbe(module.ConstraintAttributeMethod, NextInitializer):
        def __init__(self, *args: Any, **kwargs: Any):
            object.__setattr__(self, "trace", [])
            super().__init__(*args, **kwargs)

        def __setattr__(self, name: str, value: Any):
            if name in {"is_constraint", "constraint_restrictions"}:
                object.__getattribute__(self, "trace").append(
                    ["set", name, value]
                )
            object.__setattr__(self, name, value)

    constraint = ConstraintProbe(restriction="soft", is_constraint=False)
    if constraint.is_constraint is not True or constraint.constraint_restrictions != "soft":
        raise AssertionError("constraint initialization drift")
    if [entry[0] for entry in constraint.trace] != ["next", "set", "set", "set"]:
        raise AssertionError(f"constraint cooperative order drift: {constraint.trace}")
    records.append({"name": "constraint_cooperative_mro", "trace": constraint.trace})

    direct_keyword = ConstraintProbe(constraint_restrictions="soft")
    if direct_keyword.constraint_restrictions != "hard":
        raise AssertionError("direct constraint_restrictions keyword was consumed")
    records.append(
        {
            "name": "direct_constraint_restrictions_ignored",
            "stored": direct_keyword.constraint_restrictions,
        }
    )

    invalid = object.__new__(ConstraintProbe)
    object.__setattr__(invalid, "trace", [])
    expect_exception(
        lambda: module.ConstraintAttributeMethod.__init__(invalid, restriction="invalid"),
        ValueError,
        "constraint_restrictions must be 'hard' or 'soft', got invalid",
    )
    if getattr(invalid, "is_constraint", None) is not True or getattr(
        invalid, "constraint_restrictions", None
    ) != "invalid":
        raise AssertionError("invalid restriction did not preserve assignments")
    records.append({"name": "invalid_restriction_after_assignment", "trace": invalid.trace})

    invalid_none = object.__new__(ConstraintProbe)
    object.__setattr__(invalid_none, "trace", [])
    expect_exception(
        lambda: module.ConstraintAttributeMethod.__init__(
            invalid_none, restriction=None
        ),
        ValueError,
        "constraint_restrictions must be 'hard' or 'soft', got None",
    )
    if invalid_none.is_constraint is not True or invalid_none.constraint_restrictions is not None:
        raise AssertionError("None restriction assignment timing drift")
    records.append({"name": "none_restriction_after_assignment", "trace": invalid_none.trace})

    class Combined(
        module.InformationAttributeMethod,
        module.ConstraintAttributeMethod,
        NextInitializer,
    ):
        def __init__(self, **kwargs: Any):
            self.trace = []
            super().__init__(**kwargs)

    combined = Combined(restriction="hard")
    if combined.is_constraint is not False or combined.constraint_restrictions != "hard":
        raise AssertionError("combined cooperative MRO final state drift")
    records.append(
        {
            "name": "combined_information_constraint_mro",
            "is_constraint": combined.is_constraint,
            "restriction": combined.constraint_restrictions,
        }
    )

    argument = Explosive()
    checker = object.__new__(module.ConstraintAttributeMethod)
    expect_exception(
        lambda: checker.check_constraint_satisfiability(argument, argument, argument),
        NotImplementedError,
        "The attribute has not implemented the check_constraint_satisfiability method",
    )
    records.append({"name": "base_check_no_argument_access", "status": "locked"})

    op_trace: list[str] = []

    class Difference:
        def __abs__(self):
            op_trace.append("abs")
            return 9

    class OperatorProbe:
        def __init__(self, label: str):
            self.label = label

        def __ge__(self, other: Any):
            op_trace.append(f"{self.label}:ge:{other.label}")
            return False

        def __eq__(self, other: Any):
            op_trace.append(f"{self.label}:eq:{other.label}")
            return False

        def __sub__(self, other: Any):
            op_trace.append(f"{self.label}:sub:{other.label}")
            return Difference()

    calculator = object.__new__(module.ConstraintAttributeMethod)
    calculator.constraint_restrictions = "hard"
    left = OperatorProbe("v")
    right = OperatorProbe("p")
    flag, offset = calculator._calculate_satisfiability_values(left, right, "eq")
    if flag is not False or offset != 9 or op_trace != ["v:eq:p", "v:sub:p", "abs"]:
        raise AssertionError(f"calculation operator order drift: {op_trace}")
    records.append({"name": "calculation_operator_order", "trace": op_trace})

    np = module.np
    calculator.constraint_restrictions = "hard"
    array_flag, array_offset = calculator._calculate_satisfiability_values(
        np.array([1, 3]), np.array([2, 2]), "le"
    )
    if (
        type(array_flag).__module__.split(".")[0] != "numpy"
        or type(array_offset).__module__.split(".")[0] != "numpy"
        or array_flag.tolist() != [True, False]
        or array_offset.tolist() != [-1, 1]
    ):
        raise AssertionError("NumPy dynamic calculation boundary drift")
    records.append(
        {
            "name": "numpy_array_dynamic_boundary",
            "flag_dtype": str(array_flag.dtype),
            "offset_dtype": str(array_offset.dtype),
            "flag": array_flag.tolist(),
            "offset": array_offset.tolist(),
        }
    )

    for dynamic_method in [None, 1, True]:
        expect_exception(
            lambda dynamic_method=dynamic_method: instance.update(
                {"cpu": 1}, {"cpu": 2}, dynamic_method
            ),
            NotImplementedError,
            f"Update method '{dynamic_method}' is not supported.",
        )
    records.append(
        {
            "name": "python_non_string_method_boundary",
            "types": ["NoneType", "int", "bool"],
        }
    )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, default=default_source())
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    arguments = parser.parse_args()

    source = arguments.source.resolve()
    harness = arguments.harness.resolve()
    module = load_original(source)
    update_records = run_update_cases(module, harness)
    resolver_records = run_resolver_cases(module, harness)
    calculation_records = run_calculation_cases(module, harness)
    dynamic_records = run_dynamic_characterization(module)
    batch_result = run_batch_cases(module, harness)
    scalar_total = (
        len(update_records)
        + len(resolver_records)
        + len(calculation_records)
        + len(dynamic_records)
    )
    if scalar_total != 93:
        raise AssertionError(f"scalar differential count drift: {scalar_total} != 93")

    result = {
        "status": "PASS",
        "python_source_sha256": SOURCE_SHA256,
        "python_source_size": SOURCE_SIZE,
        "harness_sha256": hashlib.sha256(harness.read_bytes()).hexdigest(),
        "python": sys.version.split()[0],
        "numpy": module.np.__version__,
        "networkx": module.nx.__version__,
        "typed_update_cases": len(update_records),
        "resolver_cases": len(resolver_records),
        "typed_calculation_cases": len(calculation_records),
        "python_dynamic_mro_cases": len(dynamic_records),
        "total_cases": scalar_total,
        "scalar_total_cases": scalar_total,
        "batch_cases": batch_result["case_count"],
        "combined_total_cases": scalar_total + batch_result["case_count"],
        "batch": batch_result,
        "int64_boundary_cases": [
            record["name"]
            for record in [*update_records, *calculation_records]
            if "boundary" in record
        ],
        "dynamic_mro": dynamic_records,
    }
    if arguments.output is not None:
        arguments.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
