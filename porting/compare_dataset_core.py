#!/usr/bin/env python3
"""Exact Python/C++ differential for the Torch-free dataset core leaf."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import pathlib
import random
import struct
import subprocess
import sys
import tempfile
import types
from collections import OrderedDict
from typing import Callable


SOURCE_SHA256 = "269650ebcc373d7bdf79fa17346bd6f847973f17e60c1ac9bcae7cfd97bf936f"
SOURCE_SIZE = 9635
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211


class RejectingTorch(types.ModuleType):
    def __getattr__(self, name: str):
        raise AssertionError(f"non-Torch dataset core touched torch.{name}")


def load_oracle(source: pathlib.Path):
    source_bytes = source.read_bytes()
    if len(source_bytes) != SOURCE_SIZE:
        raise RuntimeError(f"dataset.py size drift: {len(source_bytes)}")
    digest = hashlib.sha256(source_bytes).hexdigest()
    if digest != SOURCE_SHA256:
        raise RuntimeError(f"dataset.py SHA-256 drift: {digest}")

    fake_torch = RejectingTorch("torch")
    fake_omegaconf = types.ModuleType("omegaconf")
    fake_omegaconf.DictConfig = type("FakeDictConfig", (), {})
    fake_omegaconf.OmegaConf = type("FakeOmegaConf", (), {})
    previous = {
        name: sys.modules.get(name)
        for name in ("torch", "omegaconf")
    }
    sys.modules["torch"] = fake_torch
    sys.modules["omegaconf"] = fake_omegaconf
    try:
        spec = importlib.util.spec_from_file_location(
            "_virne_dataset_core_oracle", source
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("unable to create dataset oracle spec")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    if module.torch is not fake_torch:
        raise RuntimeError("dataset oracle did not retain the controlled Torch fake")
    return module


def physical_setting(root: pathlib.Path) -> dict:
    return {
        "output": {"save_dir": str(root / "p_out")},
        "topology": {
            "num_nodes": 100,
            "type": "waxman",
            "wm_alpha": 0.5,
            "wm_beta": 0.2,
        },
        "node_attrs_setting": [
            {
                "name": "cpu",
                "distribution": "uniform",
                "low": 50,
                "high": 100,
            },
            {"name": "max_cpu"},
        ],
        "link_attrs_setting": [
            {
                "name": "bw",
                "distribution": "uniform",
                "low": 50,
                "high": 100,
            },
            {"name": "max_bw"},
        ],
    }


def virtual_setting(root: pathlib.Path) -> dict:
    return {
        "output": {"save_dir": str(root / "v_out")},
        "num_v_nets": 1000,
        "v_net_size": {"low": 2, "high": 10},
        "topology": {"type": "random"},
        "lifetime": {"distribution": "exponential", "scale": 500},
        "arrival_rate": {"lam": 0.04},
        "node_attrs_setting": [
            {
                "name": "cpu",
                "distribution": "uniform",
                "low": 0,
                "high": 20,
            }
        ],
        "link_attrs_setting": [
            {
                "name": "bw",
                "distribution": "uniform",
                "low": 0,
                "high": 50,
            }
        ],
    }


def fnv_summary(values: list[str]) -> str:
    checksum = FNV_OFFSET
    output_bytes = 0
    for value in values:
        encoded = os.fsencode(value)
        output_bytes += len(encoded)
        for byte in encoded:
            checksum ^= byte
            checksum = (checksum * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
        checksum ^= 0xFF
        checksum = (checksum * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return f"{checksum}:{output_bytes}"


CaseResult = tuple[str, ...]


def ok(value: object) -> CaseResult:
    return ("ok", str(value))


def mapped_call(
    function: Callable[[], object],
    error_code: str | None = None,
    operation: str = "resolve_distribution",
) -> CaseResult:
    try:
        return ok(function())
    except Exception as error:  # exact Python class is checked before mapping
        if error_code is None:
            raise
        expected_type = (
            UnboundLocalError
            if error_code == "unsupported_parameter_distribution"
            else KeyError
        )
        if not isinstance(error, expected_type):
            raise RuntimeError(
                f"unexpected Python error {type(error).__name__}: {error}"
            ) from error
        return ("error", error_code, operation)


def build_python_cases(module, root: pathlib.Path) -> OrderedDict[str, CaseResult]:
    cases: OrderedDict[str, CaseResult] = OrderedDict()
    scalar_values = [
        ("scalar_none", None),
        ("scalar_true", True),
        ("scalar_false", False),
        ("scalar_int_min", -(2**63)),
        ("scalar_string", "a-b=c\n"),
        ("float_one", 1.0),
        ("float_negative_zero", -0.0),
        ("float_million", 1.0e6),
        ("float_e15", 1.0e15),
        ("float_e16", 1.0e16),
        ("float_em4", 1.0e-4),
        ("float_em5", 1.0e-5),
        ("float_inf", float("inf")),
        ("float_negative_inf", float("-inf")),
        ("float_nan", float("nan")),
    ]
    for name, value in scalar_values:
        cases[name] = ok(value)

    parameter_inputs = [
        ("parameters_none", {}),
        ("parameters_exponential", {"distribution": "exponential", "scale": 500}),
        ("parameters_poisson", {"distribution": "poisson", "lam": 0.04}),
        ("parameters_uniform", {"distribution": "uniform", "low": -2, "high": 7}),
        (
            "parameters_customized",
            {"distribution": "customized", "min": "a", "max": False},
        ),
    ]
    for name, value in parameter_inputs:
        cases[name] = mapped_call(
            lambda value=value: module.get_parameters_string(
                module.get_distribution_parameters(value)
            )
        )
    cases["parameters_normal_error"] = mapped_call(
        lambda: module.get_parameters_string(
            module.get_distribution_parameters({"distribution": "normal"})
        ),
        "unsupported_parameter_distribution",
    )
    cases["parameters_missing_error"] = mapped_call(
        lambda: module.get_parameters_string(
            module.get_distribution_parameters(
                {"distribution": "uniform", "low": 1}
            )
        ),
        "missing_parameter",
    )
    cases["average_stub"] = ok(
        module.get_distribution_average(object(), "ignored", "ignored")
    )
    cases["filename_empty"] = ok(module.generate_file_name({"solver_name": "solver"}))
    cases["filename_ordered"] = ok(
        module.generate_file_name(
            {"solver_name": "solver"},
            epoch_id=-3,
            alpha=1,
            flag=True,
            **{"raw-key": "x=y/z", "none": None},
        )
    )
    cases["filename_special"] = ok(
        module.generate_file_name(
            {"solver_name": "nghiệm"},
            epoch_id=42,
            **{"line\nkey": "x\0y", "khóa": "giá-trị"},
        )
    )

    physical = physical_setting(root)
    cases["physical_default"] = ok(
        module.get_p_net_dataset_dir_from_setting(physical)
    )
    cases["physical_seed_zero"] = ok(
        module.get_p_net_dataset_dir_from_setting(physical, 0)
    )
    cases["physical_seed_false"] = ok(
        module.get_p_net_dataset_dir_from_setting(physical, False)
    )
    cases["physical_seed_negative"] = ok(
        module.get_p_net_dataset_dir_from_setting(physical, -7)
    )
    empty_physical = copy.deepcopy(physical)
    empty_physical["node_attrs_setting"] = []
    empty_physical["link_attrs_setting"] = []
    cases["physical_empty_attributes"] = ok(
        module.get_p_net_dataset_dir_from_setting(empty_physical)
    )
    for name, file_path in [
        ("physical_existing_file", root / "Topo.multi.part.gml"),
        ("physical_hidden_file", root / ".hidden.gml"),
        ("physical_missing_file", root / "missing.gml"),
        ("physical_none_text_file", "None"),
    ]:
        setting = copy.deepcopy(physical)
        setting["topology"]["file_path"] = str(file_path)
        cases[name] = ok(module.get_p_net_dataset_dir_from_setting(setting))
    empty_text = copy.deepcopy(physical)
    empty_text["topology"]["file_path"] = ""
    cases["physical_empty_text_file"] = ok(
        module.get_p_net_dataset_dir_from_setting(empty_text)
    )
    path_topology = copy.deepcopy(physical)
    path_topology["topology"].update(
        {"num_nodes": 10, "type": "path", "wm_alpha": 1, "wm_beta": 2}
    )
    cases["physical_path_topology"] = ok(
        module.get_p_net_dataset_dir_from_setting(path_topology)
    )
    unicode_physical = copy.deepcopy(physical)
    unicode_physical["node_attrs_setting"] = [
        {
            "name": "cpu-động",
            "distribution": "uniform",
            "low": 1,
            "high": 2,
        }
    ]
    cases["physical_unicode_attribute"] = ok(
        module.get_p_net_dataset_dir_from_setting(unicode_physical)
    )

    virtual_value = virtual_setting(root)
    cases["virtual_default"] = ok(
        module.get_v_nets_dataset_dir_from_setting(virtual_value)
    )
    cases["virtual_seed_false"] = ok(
        module.get_v_nets_dataset_dir_from_setting(virtual_value, False)
    )
    cases["virtual_seed_zero"] = ok(
        module.get_v_nets_dataset_dir_from_setting(virtual_value, 0)
    )
    cases["virtual_seed_negative"] = ok(
        module.get_v_nets_dataset_dir_from_setting(virtual_value, -9)
    )
    empty_virtual = copy.deepcopy(virtual_value)
    empty_virtual["node_attrs_setting"] = []
    empty_virtual["link_attrs_setting"] = []
    cases["virtual_empty_attributes"] = ok(
        module.get_v_nets_dataset_dir_from_setting(empty_virtual)
    )
    invalid_virtual = copy.deepcopy(virtual_value)
    invalid_virtual["lifetime"] = {"distribution": "normal"}
    cases["virtual_normal_error"] = mapped_call(
        lambda: module.get_v_nets_dataset_dir_from_setting(invalid_virtual),
        "unsupported_parameter_distribution",
    )
    customized_virtual = copy.deepcopy(virtual_value)
    customized_virtual["lifetime"] = {
        "distribution": "customized",
        "min": "a",
        "max": False,
    }
    cases["virtual_customized_lifetime"] = ok(
        module.get_v_nets_dataset_dir_from_setting(customized_virtual)
    )

    batch_size = 4096
    file_names = [
        module.generate_file_name(
            {"solver_name": "solver" if index % 2 == 0 else "solver-x"},
            epoch_id=index,
            index=index,
            even=index % 2 == 0,
            raw="x=y/z",
        )
        for index in range(batch_size)
    ]
    physical_paths = [
        module.get_p_net_dataset_dir_from_setting(physical, index)
        for index in range(batch_size)
    ]
    virtual_paths = [
        module.get_v_nets_dataset_dir_from_setting(virtual_value, index % 2 == 0)
        for index in range(batch_size)
    ]
    summaries = {
        "filename": fnv_summary(file_names),
        "physical": fnv_summary(physical_paths),
        "virtual": fnv_summary(virtual_paths),
    }
    for workers in ("1", "2", "4", "8", "auto"):
        for kind in ("filename", "physical", "virtual"):
            cases[f"batch_{kind}_w{workers}"] = ok(summaries[kind])
    return cases


def run_cpp_cases(harness: pathlib.Path, root: pathlib.Path) -> OrderedDict[str, CaseResult]:
    process = subprocess.run(
        [str(harness), "cases", str(root)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"C++ harness failed: {process.stderr.strip()}")
    lines = process.stdout.splitlines()
    if not lines or lines[0] != "dataset_core_harness_version=1":
        raise RuntimeError("invalid dataset core harness version")
    if lines[-1] != "status=PASS":
        raise RuntimeError("dataset core harness did not report PASS")
    cases: OrderedDict[str, CaseResult] = OrderedDict()
    for line in lines[1:-1]:
        if not line.startswith("case="):
            raise RuntimeError(f"unexpected harness line: {line!r}")
        fields = line[5:].split("|")
        if len(fields) == 3 and fields[1] == "ok":
            value = bytes.fromhex(fields[2]).decode("utf-8")
            cases[fields[0]] = ("ok", value)
        elif len(fields) == 4 and fields[1] == "error":
            cases[fields[0]] = ("error", fields[2], fields[3])
        else:
            raise RuntimeError(f"malformed harness case: {line!r}")
    return cases


def compare_float_corpus(harness: pathlib.Path) -> int:
    deterministic = [
        0x0000000000000000,
        0x8000000000000000,
        0x0000000000000001,
        0x000FFFFFFFFFFFFF,
        0x0010000000000000,
        0x3FF0000000000000,
        0x7FEFFFFFFFFFFFFF,
        0x7FF0000000000000,
        0xFFF0000000000000,
        0x7FF8000000000000,
        0xFFF8000000000001,
    ]
    generator = random.Random(0xDADA5E7)
    bits = deterministic + [generator.getrandbits(64) for _ in range(16_384)]
    process = subprocess.run(
        [str(harness), "float_bits"],
        input="".join(f"{value:016x}\n" for value in bits),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"C++ float corpus failed: {process.stderr.strip()}")
    lines = process.stdout.splitlines()
    if not lines or lines[0] != "dataset_core_float_version=1" or lines[-1] != "status=PASS":
        raise RuntimeError("invalid C++ float corpus protocol")
    actual = [bytes.fromhex(value).decode("ascii") for value in lines[1:-1]]
    expected = [
        str(struct.unpack("=d", struct.pack("=Q", value))[0])
        for value in bits
    ]
    if len(actual) != len(expected):
        raise RuntimeError("float corpus length mismatch")
    mismatches = [
        (index, bits[index], expected[index], actual[index])
        for index in range(len(bits))
        if expected[index] != actual[index]
    ]
    if mismatches:
        details = "\n".join(
            f"index={index} bits={value:016x} python={python!r} cpp={cpp!r}"
            for index, value, python, cpp in mismatches[:20]
        )
        raise RuntimeError(f"binary64 formatting mismatch:\n{details}")
    return len(bits)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--python-source", type=pathlib.Path, required=True)
    parser.add_argument("--json-output", type=pathlib.Path)
    args = parser.parse_args()

    module = load_oracle(args.python_source)
    with tempfile.TemporaryDirectory(prefix="virne_dataset_core_diff_") as text_root:
        root = pathlib.Path(text_root)
        (root / "Topo.multi.part.gml").write_bytes(b"fixture")
        (root / ".hidden.gml").write_bytes(b"fixture")
        expected = build_python_cases(module, root)
        actual = run_cpp_cases(args.harness, root)
    float_cases = compare_float_corpus(args.harness)
    if list(actual) != list(expected):
        missing = [name for name in expected if name not in actual]
        extra = [name for name in actual if name not in expected]
        raise RuntimeError(
            f"case order/inventory mismatch: missing={missing}, extra={extra}"
        )
    mismatches = [
        (name, expected[name], actual[name])
        for name in expected
        if expected[name] != actual[name]
    ]
    if mismatches:
        details = "\n".join(
            f"{name}: python={python!r} cpp={cpp!r}"
            for name, python, cpp in mismatches[:20]
        )
        raise RuntimeError(f"dataset core differential mismatch:\n{details}")
    print(f"dataset core differential: PASS ({len(expected)}/{len(expected)} cases)")
    print(f"binary64_str_cases={float_cases}")
    print(f"source_sha256={SOURCE_SHA256}")
    print("torch_backend=controlled-fake-only")
    if args.json_output:
        args.json_output.write_text(
            json.dumps(
                {
                    "binary64_str_cases": float_cases,
                    "compatibility_cases": len(expected),
                    "harness_sha256": hashlib.sha256(
                        args.harness.read_bytes()
                    ).hexdigest(),
                    "source_sha256": SOURCE_SHA256,
                    "source_size": SOURCE_SIZE,
                    "status": "PASS",
                    "torch_backend": "controlled-fake-only",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
