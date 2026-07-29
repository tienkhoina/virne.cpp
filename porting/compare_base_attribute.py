#!/usr/bin/env python3
"""Exact direct-source oracle for the typed BaseAttribute leaf."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import struct
import subprocess
import sys
import types
from dataclasses import dataclass, field
from typing import Any

import compare_dataset_core


SOURCE_SHA256 = "103c5c16126ca76e782c2191ff0811b95ed88a0c3637f3a61e84c0b22df42e8e"


@dataclass(frozen=True)
class Case:
    name: str
    owner: str = "node"
    generative: bool = True
    distribution: str | None = None
    dtype: str | None = None
    nodes: int = 17
    links: int = 13
    seed: int = 0
    workers: int = 1
    kwargs: dict[str, Any] = field(default_factory=dict)
    expected_error: tuple[str, str] | None = None


CASES = (
    Case("not_generative", generative=False, distribution="uniform", seed=1,
         kwargs={"low": 0.0, "high": 1.0},
         expected_error=("base", "not_generative")),
    Case("unsupported_none", seed=2,
         expected_error=("base", "unsupported_distribution")),
    Case("unsupported_unknown", distribution="bogus", seed=3, workers=2,
         expected_error=("base", "unsupported_distribution")),
    Case("uniform_float_node", distribution="uniform", dtype="float", seed=11,
         kwargs={"low": -2.25, "high": 8.5}),
    Case("uniform_int_link", owner="link", distribution="uniform", dtype="int",
         seed=12, workers=8, kwargs={"low": -7, "high": 13}),
    Case("normal_float_default_dtype", owner="graph", distribution="normal",
         seed=13, workers=0, kwargs={"loc": -1.25, "scale": 2.5}),
    Case("normal_int", distribution="normal", dtype="int", seed=14, workers=2,
         kwargs={"loc": -1.25, "scale": 2.5}),
    Case("normal_bool", owner="link", distribution="normal", dtype="bool",
         seed=15, workers=8, kwargs={"loc": -1.25, "scale": 2.5}),
    Case("exponential_float", distribution="exponential", dtype="float", seed=16,
         kwargs={"scale": 0.5}),
    Case("exponential_int", owner="link", distribution="exponential", dtype="int",
         seed=17, workers=2, kwargs={"scale": 2.75}),
    Case("exponential_bool", owner="graph", distribution="exponential",
         dtype="bool", seed=18, workers=8, kwargs={"scale": 2.75}),
    Case("poisson_int", distribution="poisson", dtype="int", seed=19,
         kwargs={"lam": 4.25}),
    Case("poisson_float", owner="link", distribution="poisson", dtype="float",
         seed=20, workers=2, kwargs={"lam": 4.25}),
    Case("poisson_bool", owner="graph", distribution="poisson", dtype="bool",
         seed=21, workers=8, kwargs={"lam": 4.25}),
    Case("poisson_reciprocal_ignored", distribution="poisson", dtype="int",
         seed=22, workers=8, kwargs={"lam": 0.0, "reciprocal": True}),
    Case("custom_bool_bounds", distribution="customized", dtype="bool", seed=31,
         workers=0, kwargs={"min": False, "max": True}),
    Case("custom_mixed_graph", owner="graph", distribution="customized",
         dtype="int", seed=32, workers=2, kwargs={"min": -7, "max": 4.5}),
    Case("custom_integral_large_span", distribution="customized", nodes=31,
         links=0, seed=7, workers=8,
         kwargs={"min": 2**62, "max": 2**62 + 1025}),
    Case("custom_zero", distribution="customized", nodes=0, links=0, seed=33,
         workers=8, kwargs={"min": -1.0, "max": 1.0}),
    Case("custom_missing", distribution="customized", seed=34,
         kwargs={"min": None, "max": 1.0},
         expected_error=("base", "invalid_custom_range")),
    Case("custom_string", distribution="customized", seed=35, workers=2,
         kwargs={"min": "bad", "max": 1.0},
         expected_error=("base", "invalid_custom_range")),
    Case("custom_equal", distribution="customized", seed=36, workers=8,
         kwargs={"min": 1.0, "max": 1.0},
         expected_error=("base", "invalid_custom_range")),
    Case("custom_reversed", distribution="customized", seed=37,
         kwargs={"min": 2.0, "max": 1.0},
         expected_error=("base", "invalid_custom_range")),
    Case("normal_missing_location", distribution="normal", dtype="float", seed=41,
         workers=8, kwargs={"scale": 1.0},
         expected_error=("dataset", "invalid_parameter")),
    Case("normal_missing_scale", distribution="normal", dtype="float", seed=42,
         workers=8, kwargs={"loc": 0.0},
         expected_error=("dataset", "invalid_parameter")),
    Case("uniform_bool_bug", distribution="uniform", dtype="bool", seed=43,
         workers=8, kwargs={"low": False, "high": True},
         expected_error=("dataset", "uniform_boolean_uninitialized")),
    Case("exponential_negative", distribution="exponential", dtype="float", seed=44,
         kwargs={"scale": -1.0},
         expected_error=("dataset", "rng_backend_failure")),
    Case("poisson_negative", distribution="poisson", dtype="int", seed=45,
         workers=2, kwargs={"lam": -1.0},
         expected_error=("dataset", "rng_backend_failure")),
)


def float_bits(value: float) -> str:
    return f"{struct.unpack('>Q', struct.pack('>d', float(value)))[0]:016x}"


def hex_text(value: str) -> str:
    return value.encode("utf-8").hex()


def load_base_oracle(source: pathlib.Path, dataset_source: pathlib.Path):
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SOURCE_SHA256:
        raise RuntimeError(f"BaseAttribute source hash drift: {digest}")
    dataset = compare_dataset_core.load_oracle(dataset_source)

    virne_module = types.ModuleType("virne")
    virne_module.__path__ = []
    utils_module = types.ModuleType("virne.utils")
    utils_module.path_to_links = lambda path: path
    utils_module.generate_data_with_distribution = (
        dataset.generate_data_with_distribution
    )
    network_module = types.ModuleType("virne.network")
    network_module.__path__ = []
    base_network_module = types.ModuleType("virne.network.base_network")

    class BaseNetwork:
        def __init__(self, num_nodes: int, num_links: int):
            self.num_nodes = num_nodes
            self.num_links = num_links

    base_network_module.BaseNetwork = BaseNetwork
    replacements = {
        "virne": virne_module,
        "virne.utils": utils_module,
        "virne.network": network_module,
        "virne.network.base_network": base_network_module,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    module_name = "virne.network.attribute_base_oracle"
    module_spec = importlib.util.spec_from_file_location(module_name, source)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("cannot construct BaseAttribute oracle loader")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    try:
        module_spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
        raise
    return module, dataset, BaseNetwork, previous, module_name


def restore_oracle(previous: dict[str, types.ModuleType | None], module_name: str):
    sys.modules.pop(module_name, None)
    for name, old in previous.items():
        if old is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old


def parse_harness(harness: pathlib.Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    process = subprocess.run(
        [str(harness)], check=False, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"BaseAttribute harness failed: {process.stderr.strip()}")
    static: dict[str, str] = {}
    cases: dict[str, list[str]] = {}
    for line in process.stdout.splitlines():
        if line == "status=PASS":
            static["status"] = "PASS"
        elif line.startswith("case="):
            fields = line.split("|")
            name = fields[0][5:]
            if name in cases:
                raise RuntimeError(f"duplicate C++ case: {name}")
            cases[name] = fields[1:]
        elif "=" in line:
            key, value = line.split("=", 1)
            static[key] = value
        else:
            raise RuntimeError(f"malformed harness line: {line!r}")
    if static.get("status") != "PASS":
        raise RuntimeError("harness did not report PASS")
    return static, cases


def python_static(module) -> dict[str, str]:
    class Probe(module.BaseAttribute):
        def set_data(self, network, attribute_data):
            return None

        def get_data(self, network):
            return []

    value = Probe(
        "cpu", "link", "resource", True,
        distribution="uniform", dtype="int", low=-3, high=7.5,
        originator="capacity", is_constraint=True,
    )
    snapshot = "\n".join(f"{key}={item}" for key, item in value.to_dict().items())
    plain = Probe("plain", "node", "status")
    errors = []
    for call in (
        lambda: plain.generate_data(None),
        lambda: plain.update_data(None, None),
    ):
        try:
            call()
        except NotImplementedError as error:
            errors.append(str(error))
        else:
            raise RuntimeError("Python default method unexpectedly succeeded")
    return {
        "snapshot": hex_text(snapshot),
        "repr": hex_text(repr(value)),
        "default_generate": "not_implemented|" + hex_text(errors[0]),
        "default_update": "not_implemented|" + hex_text(errors[1]),
    }


def serialize_python(values, dtype: str) -> tuple[str, str]:
    if dtype == "float":
        return dtype, ",".join(float_bits(value) for value in values)
    if dtype == "int":
        return dtype, ",".join(str(int(value)) for value in values)
    if dtype == "bool":
        return dtype, ",".join("1" if bool(value) else "0" for value in values)
    raise RuntimeError(f"unknown output dtype: {dtype}")


def run_python_case(module, BaseNetwork, item: Case) -> dict[str, Any]:
    class Probe(module.BaseAttribute):
        def set_data(self, network, attribute_data):
            return None

        def get_data(self, network):
            return []

    constructor_kwargs = dict(item.kwargs)
    if item.distribution is not None:
        constructor_kwargs = {"distribution": item.distribution, **constructor_kwargs}
    if item.dtype is not None:
        constructor_kwargs = {"dtype": item.dtype, **constructor_kwargs}
    value = Probe(
        "x", item.owner, "resource", item.generative, **constructor_kwargs
    )
    module.np.random.seed(item.seed)
    caught: Exception | None = None
    output = None
    try:
        output = value._generate_data(BaseNetwork(item.nodes, item.links))
    except Exception as error:  # exact class is checked against case intent below
        caught = error
    next_value = float_bits(module.np.random.random_sample())
    if item.expected_error is not None:
        if caught is None:
            raise RuntimeError(f"Python case {item.name} unexpectedly succeeded")
        return {"status": "error", "next": next_value, "error": caught}
    if caught is not None:
        raise RuntimeError(
            f"Python case {item.name} unexpectedly failed: "
            f"{type(caught).__name__}: {caught}"
        )
    output_dtype = "float" if item.distribution == "customized" else (item.dtype or "float")
    lane, payload = serialize_python(output, output_dtype)
    return {"status": "ok", "lane": lane, "values": payload, "next": next_value}


def compare_case(item: Case, cpp: list[str], python: dict[str, Any]):
    if python["status"] == "ok":
        expected = ["ok", python["lane"], python["values"], python["next"]]
        if cpp != expected:
            raise RuntimeError(
                f"case {item.name} output/state mismatch:\n"
                f"python={expected!r}\ncpp={cpp!r}"
            )
        return
    if len(cpp) != 5 or cpp[0] != "error":
        raise RuntimeError(f"case {item.name} expected C++ error, got {cpp!r}")
    expected_domain, expected_code = item.expected_error
    if cpp[1] != expected_domain or cpp[2] != expected_code:
        raise RuntimeError(
            f"case {item.name} typed error mismatch: "
            f"expected {(expected_domain, expected_code)!r}, got {cpp[1:3]!r}"
        )
    if cpp[4] != python["next"]:
        raise RuntimeError(
            f"case {item.name} RNG continuation mismatch: "
            f"python={python['next']}, cpp={cpp[4]}"
        )
    if expected_domain == "base" and item.name != "unsupported_unknown":
        python_message = str(python["error"])
        if cpp[3] != hex_text(python_message):
            raise RuntimeError(
                f"case {item.name} base error message mismatch: "
                f"python={python_message!r}, cpp_hex={cpp[3]!r}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--dataset-source", type=pathlib.Path, required=True)
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    module, dataset, BaseNetwork, previous, module_name = load_base_oracle(
        args.source, args.dataset_source
    )
    try:
        cpp_static, cpp_cases = parse_harness(args.harness)
        expected_static = python_static(module)
        for key, expected in expected_static.items():
            if cpp_static.get(key) != expected:
                raise RuntimeError(
                    f"static case {key} mismatch: "
                    f"python={expected!r}, cpp={cpp_static.get(key)!r}"
                )
        expected_names = {item.name for item in CASES}
        if set(cpp_cases) != expected_names:
            raise RuntimeError(
                f"case inventory mismatch: missing={sorted(expected_names - set(cpp_cases))}, "
                f"extra={sorted(set(cpp_cases) - expected_names)}"
            )
        for item in CASES:
            compare_case(
                item,
                cpp_cases[item.name],
                run_python_case(module, BaseNetwork, item),
            )
    finally:
        restore_oracle(previous, module_name)

    payload = {
        "source_sha256": SOURCE_SHA256,
        "dataset_source_sha256": compare_dataset_core.SOURCE_SHA256,
        "harness_sha256": hashlib.sha256(args.harness.read_bytes()).hexdigest(),
        "numpy_version": dataset.np.__version__,
        "static_cases": len(expected_static),
        "generation_cases": len(CASES),
        "total_cases": len(expected_static) + len(CASES),
        "result": "PASS",
    }
    if args.output:
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"base_attribute differential: PASS ({payload['total_cases']} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
