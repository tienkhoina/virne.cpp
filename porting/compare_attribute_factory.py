#!/usr/bin/env python3
"""Pinned direct-source differential for the non-ML AttributeFactory leaf."""

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
from typing import Any, Callable


SOURCE_SHA256 = "11ac0c2235ba02f6538427f3774c93766820ea640531c97c94f4b4daf2d7b010"
BASE_SHA256 = "103c5c16126ca76e782c2191ff0811b95ed88a0c3637f3a61e84c0b22df42e8e"
METHOD_SHA256 = "e17499af8e6ffbdb12f2100dd58abeab48dd06d92feb442ef192f8b9310b6b4f"
NODE_SHA256 = "e90e286320e59ebbba9957b701f6f12acdc1785e4821de80ccc5a0ab0f3ee56a"
LINK_SHA256 = "a95cfd2b8e2b46d4de23f70934ca942de502e674e2a7eaa139482df640e8646f"
GRAPH_SHA256 = "dfa858918068a792cb0a673b400ee6f3f93107f1d17baad28f72546f50fbcede"

PYTHON_BOUNDARY_DESCRIPTIONS = {
    "integer_generative_coercion":
        "Python retains integer 1; the typed C++ setting boundary requires bool.",
    "non_string_name":
        "Python retains a non-string name; the typed C++ registry owns strings.",
    "arbitrary_extra_attribute":
        "Python reflects arbitrary kwargs; C++ stores only documented fixed fields.",
    "nested_python_config":
        "Python expands a dynamic nested config; repository settings use flat fields.",
    "duplicate_restriction_call_binding":
        "Flat restriction plus constraint_restrictions reaches Python as duplicate "
        "restriction kwargs and raises TypeError; C++ resolves precedence once.",
    "duplicate_checking_level_call_binding":
        "A flat checking_level is re-forwarded by Python as a duplicate kwarg; C++ "
        "validates and stores the checking enum directly.",
    "dynamic_key_stringified_once":
        "Python stringifies an arbitrary dynamic key once; C++ accepts fixed setting keys.",
}

_RESTORE_STATE: tuple[dict[str, Any], list[str]] | None = None


def verify(path: pathlib.Path, expected: str, label: str) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"{label} source hash drift: {actual}")


def execute_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot direct-load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_oracle(source: pathlib.Path):
    """Load the exact factory package without importing Virne's ML stack."""
    global _RESTORE_STATE
    if _RESTORE_STATE is not None:
        raise RuntimeError("attribute factory oracle is already loaded")

    source = source.resolve()
    root = source.parent
    files = {
        "factory": (source, SOURCE_SHA256),
        "base": (root / "base_attribute.py", BASE_SHA256),
        "method": (root / "attribute_method.py", METHOD_SHA256),
        "node": (root / "node_attribute.py", NODE_SHA256),
        "link": (root / "link_attribute.py", LINK_SHA256),
        "graph": (root / "graph_attribute.py", GRAPH_SHA256),
    }
    for label, (path, expected) in files.items():
        verify(path, expected, label)

    # Construction only needs these two virne.utils names. Any accidental
    # generation call fails loudly instead of silently importing more code.
    virne = types.ModuleType("virne")
    virne.__path__ = []
    utils = types.ModuleType("virne.utils")
    utils.path_to_links = lambda path: list(zip(path[:-1], path[1:]))

    def generation_is_out_of_scope(*_args, **_kwargs):
        raise RuntimeError("factory oracle unexpectedly entered generation")

    utils.generate_data_with_distribution = generation_is_out_of_scope
    network = types.ModuleType("virne.network")
    network.__path__ = []
    package = types.ModuleType("virne.network.attribute")
    package.__path__ = [str(root)]

    omegaconf = types.ModuleType("omegaconf")

    class DictConfig(dict):
        pass

    class OmegaConf:
        pass

    omegaconf.DictConfig = DictConfig
    omegaconf.OmegaConf = OmegaConf

    benchmark_stub = types.ModuleType(
        "virne.network.attribute.attribute_benchmark_manager"
    )

    class AttributeBenchmarkManager:
        pass

    class AttributeBenchmarks(dict):
        pass

    benchmark_stub.AttributeBenchmarkManager = AttributeBenchmarkManager
    benchmark_stub.AttributeBenchmarks = AttributeBenchmarks

    module_names = [
        "virne",
        "virne.utils",
        "virne.network",
        "virne.network.attribute",
        "omegaconf",
        "virne.network.attribute.attribute_benchmark_manager",
        "virne.network.attribute.attribute_method",
        "virne.network.attribute.base_attribute",
        "virne.network.attribute.node_attribute",
        "virne.network.attribute.link_attribute",
        "virne.network.attribute.graph_attribute",
    ]
    previous = {name: sys.modules.get(name) for name in module_names}
    sys.modules.update(
        {
            "virne": virne,
            "virne.utils": utils,
            "virne.network": network,
            "virne.network.attribute": package,
            "omegaconf": omegaconf,
            "virne.network.attribute.attribute_benchmark_manager": benchmark_stub,
        }
    )
    loaded_children: list[str] = []
    try:
        execute_module(
            "virne.network.attribute.attribute_method", files["method"][0]
        )
        loaded_children.append("virne.network.attribute.attribute_method")
        execute_module("virne.network.attribute.base_attribute", files["base"][0])
        loaded_children.append("virne.network.attribute.base_attribute")
        execute_module("virne.network.attribute.node_attribute", files["node"][0])
        loaded_children.append("virne.network.attribute.node_attribute")
        execute_module("virne.network.attribute.link_attribute", files["link"][0])
        loaded_children.append("virne.network.attribute.link_attribute")
        execute_module(
            "virne.network.attribute.graph_attribute", files["graph"][0]
        )
        loaded_children.append("virne.network.attribute.graph_attribute")

        spec = importlib.util.spec_from_file_location(
            "virne.network.attribute",
            source,
            submodule_search_locations=[str(root)],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot direct-load {source}")
        factory = importlib.util.module_from_spec(spec)
        sys.modules["virne.network.attribute"] = factory
        spec.loader.exec_module(factory)
    except Exception:
        for name in module_names:
            sys.modules.pop(name, None)
        for name, old in previous.items():
            if old is not None:
                sys.modules[name] = old
        raise

    _RESTORE_STATE = previous, module_names
    return factory


def unload_oracle() -> None:
    global _RESTORE_STATE
    if _RESTORE_STATE is None:
        return
    previous, names = _RESTORE_STATE
    for name in names:
        sys.modules.pop(name, None)
    for name, old in previous.items():
        if old is not None:
            sys.modules[name] = old
    _RESTORE_STATE = None


def hex_text(value: str) -> str:
    return value.encode("utf-8").hex()


def double_token(value: float) -> str:
    return "d:" + struct.pack(">d", float(value)).hex()


def scalar_token(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "b:1" if value else "b:0"
    if isinstance(value, int):
        return f"i:{value}"
    if isinstance(value, float) or type(value).__module__.startswith("numpy"):
        return double_token(value)
    if isinstance(value, str):
        return "s:" + hex_text(value)
    raise RuntimeError(f"unsupported canonical scalar: {type(value)!r}")


def normalized_dtype(value: Any) -> str:
    if value is None:
        return "none"
    aliases = {
        "int": "integer",
        "integer": "integer",
        "int64": "integer",
        "float": "floating",
        "floating": "floating",
        "float64": "floating",
        "bool": "boolean",
        "boolean": "boolean",
    }
    return aliases.get(str(value), str(value))


def canonical_attribute(value: Any) -> str:
    class_name = type(value).__name__
    distribution = getattr(value, "distribution", None)
    distribution_name = "none" if distribution is None else str(distribution)
    restriction = "none"
    checking = "none"
    minimum_radius = "none"
    maximum_radius = "none"
    latency_generation = "none"
    minimum = "none"
    maximum = "none"

    if class_name in ("NodeResourceAttribute", "LinkResourceAttribute"):
        restriction = str(getattr(value, "constraint_restrictions"))
        checking = str(getattr(value, "checking_level"))
    elif class_name == "NodePositionAttribute":
        restriction = str(getattr(value, "constraint_restrictions"))
        minimum_radius = double_token(getattr(value, "min_r"))
        maximum_radius = double_token(getattr(value, "max_r"))
    elif class_name == "LinkLatencyAttribute":
        restriction = str(getattr(value, "constraint_restrictions"))
        checking = str(getattr(value, "checking_level"))
        latency_generation = (
            "position" if distribution_name == "position" else "configured"
        )
        raw_minimum = getattr(value, "min", None)
        raw_maximum = getattr(value, "max", None)
        minimum = scalar_token(0.0 if raw_minimum is None else raw_minimum)
        maximum = scalar_token(1.0 if raw_maximum is None else raw_maximum)

    originator = getattr(value, "originator", None)
    constraint = "1" if bool(getattr(value, "is_constraint", False)) else "0"
    parts = [
        "name=" + hex_text(str(getattr(value, "name"))),
        "owner=" + str(getattr(value, "owner")),
        "kind=" + str(getattr(value, "type")),
        "class=" + class_name,
        "generative=" + ("1" if bool(getattr(value, "generative", False)) else "0"),
        "distribution=" + distribution_name,
        "dtype=" + normalized_dtype(getattr(value, "dtype", None)),
        "low=" + scalar_token(getattr(value, "low", None)),
        "high=" + scalar_token(getattr(value, "high", None)),
        "loc=" + scalar_token(getattr(value, "loc", None)),
        "scale=" + scalar_token(getattr(value, "scale", None)),
        "lam=" + scalar_token(getattr(value, "lam", None)),
        "originator=" + ("none" if originator is None else hex_text(str(originator))),
        "constraint=" + constraint,
        "restriction=" + restriction,
        "checking=" + checking,
        "min_r=" + minimum_radius,
        "max_r=" + maximum_radius,
        "latency_generation=" + latency_generation,
        "minimum=" + minimum,
        "maximum=" + maximum,
    ]
    return ";".join(parts)


def canonical_collection(values) -> str:
    return "\n".join(canonical_attribute(value) for value in values.values())


def ok_attribute(value: Any) -> list[str]:
    return ["ok", hex_text(canonical_attribute(value))]


def ok_collection(values) -> list[str]:
    return ["ok", hex_text(canonical_collection(values))]


def expect_python_error(
    callable_: Callable[[], Any], expected: type[BaseException] | tuple[type[BaseException], ...]
) -> None:
    try:
        callable_()
    except expected:
        return
    except Exception as error:
        raise RuntimeError(
            f"Python error precedence drift: got {type(error).__name__}"
        ) from error
    raise RuntimeError("Python factory unexpectedly accepted an error case")


def differential_cases(module) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    registered = {
        "node_status_default": {
            "name": "node-status", "owner": "node", "type": "status"
        },
        "link_status_default": {
            "name": "link-status", "owner": "link", "type": "status"
        },
        "node_extrema_default": {
            "name": "node-peak", "owner": "node", "type": "extrema",
            "originator": "cpu",
        },
        "link_extrema_default": {
            "name": "link-peak", "owner": "link", "type": "extrema",
            "originator": "bandwidth",
        },
        "node_resource_default": {
            "name": "cpu", "owner": "node", "type": "resource"
        },
        "link_resource_default": {
            "name": "bandwidth", "owner": "link", "type": "resource"
        },
        "node_position_default": {
            "name": "pos", "owner": "node", "type": "position"
        },
        "link_latency_default": {
            "name": "latency", "owner": "link", "type": "latency"
        },
    }
    for name, setting in registered.items():
        result[name] = ok_attribute(module.create_attr_from_dict(setting))

    result["raw_resource_fields"] = ok_attribute(
        module.create_attr_from_dict(
            {
                "name": "cpu-rich",
                "owner": "node",
                "type": "resource",
                "generative": True,
                "distribution": "uniform",
                "dtype": "int",
                "low": 5,
                "high": 12.5,
                "constraint_restrictions": "soft",
            }
        )
    )
    result["raw_latency_position"] = ok_attribute(
        module.create_attr_from_dict(
            {
                "name": "delay",
                "owner": "link",
                "type": "latency",
                "generative": True,
                "distribution": "position",
                "min": 2,
                "max": 7.0,
            }
        )
    )
    result["raw_distribution_null"] = ok_attribute(
        module.create_attr_from_dict(
            {
                "name": "nullable",
                "owner": "node",
                "type": "status",
                "distribution": None,
                "dtype": None,
            }
        )
    )
    result["empty_general"] = ok_collection(module.create_attrs_from_setting([]))

    duplicate_settings = [
        {"name": "a", "owner": "node", "type": "status"},
        {"name": "b", "owner": "node", "type": "resource"},
        {"name": "a", "owner": "node", "type": "position"},
    ]
    result["duplicate_order_setting"] = ok_collection(
        module.create_node_attrs_from_setting(duplicate_settings)
    )

    worker_settings = [
        {"name": "cpu", "owner": "node", "type": "resource"},
        {
            "name": "peak", "owner": "node", "type": "extrema",
            "originator": "cpu",
        },
        {"name": "state", "owner": "node", "type": "status"},
        {
            "name": "cpu", "owner": "node", "type": "resource",
            "constraint_restrictions": "soft",
        },
        {"name": "pos", "owner": "node", "type": "position"},
    ]
    worker_expected = ok_collection(
        module.create_node_attrs_from_setting(worker_settings)
    )
    for workers in (0, 1, 2, 8):
        result[f"workers_{workers}"] = worker_expected

    error_inputs = {
        "missing_owner": (
            lambda: module.create_attr_from_dict(
                {"name": "x", "type": "status"}
            ),
            AssertionError,
            ["error", "missing_owner", "decode_owner", "-"],
        ),
        "false_owner": (
            lambda: module.create_attr_from_dict(
                {"name": "x", "owner": "", "type": "status"}
            ),
            AssertionError,
            ["error", "missing_owner", "decode_owner", "-"],
        ),
        "missing_kind": (
            lambda: module.create_attr_from_dict(
                {"name": "x", "owner": "node"}
            ),
            AssertionError,
            ["error", "missing_kind", "decode_kind", "-"],
        ),
        "unsupported_graph_pair": (
            lambda: module.create_attr_from_dict(
                {"name": "x", "owner": "graph", "type": "status"}
            ),
            ValueError,
            ["error", "unsupported_pair", "validate_pair", "-"],
        ),
        "unsupported_node_latency": (
            lambda: module.create_attr_from_dict(
                {"name": "x", "owner": "node", "type": "latency"}
            ),
            ValueError,
            ["error", "unsupported_pair", "validate_pair", "-"],
        ),
        "node_helper_mismatch": (
            lambda: module.create_node_attrs_from_dict(
                {"name": "x", "owner": "link", "type": "status"}
            ),
            TypeError,
            ["error", "family_mismatch", "validate_family", "-"],
        ),
        "link_helper_mismatch": (
            lambda: module.create_link_attrs_from_dict(
                {"name": "x", "owner": "node", "type": "status"}
            ),
            TypeError,
            ["error", "family_mismatch", "validate_family", "-"],
        ),
        "graph_helper_mismatch": (
            lambda: module.create_graph_attrs_from_dict(
                {"name": "x", "owner": "node", "type": "status"}
            ),
            TypeError,
            ["error", "family_mismatch", "validate_family", "-"],
        ),
        "graph_helper_unsupported": (
            lambda: module.create_graph_attrs_from_dict(
                {"name": "x", "owner": "graph", "type": "status"}
            ),
            ValueError,
            ["error", "unsupported_pair", "validate_pair", "-"],
        ),
        "null_restriction": (
            lambda: module.create_attr_from_dict(
                {
                    "name": "x", "owner": "node", "type": "resource",
                    "constraint_restrictions": None,
                }
            ),
            ValueError,
            ["error", "invalid_restriction", "decode_fields", "-"],
        ),
        "invalid_setting_item": (
            lambda: module.create_node_attrs_from_setting(
                [
                    {"name": "ok", "owner": "node", "type": "status"},
                    3,
                ]
            ),
            AssertionError,
            ["error", "invalid_setting_item", "decode_setting_list", "1"],
        ),
        "lowest_worker_error": (
            lambda: module.create_node_attrs_from_setting(
                [
                    {"name": "good", "owner": "node", "type": "status"},
                    {"name": "family", "owner": "link", "type": "status"},
                    {"name": "pair", "owner": "node", "type": "latency"},
                ]
            ),
            TypeError,
            ["error", "family_mismatch", "validate_family", "-"],
        ),
    }
    for name, (callable_, exception, expected) in error_inputs.items():
        expect_python_error(callable_, exception)
        result[name] = expected
    return result


def native_extension_cases(module) -> dict[str, list[str]]:
    # These typed boundaries intentionally reject Python coercions or bypass a
    # Python duplicate-key call-binding bug. They are still gated explicitly.
    representative = module.create_attr_from_dict(
        {
            "name": "precedence",
            "owner": "link",
            "type": "resource",
            "constraint_restrictions": "soft",
        }
    )
    return {
        "raw_restriction_precedence": ok_attribute(representative),
        "invalid_bool_field": [
            "error", "invalid_setting_value", "decode_fields", "-"
        ],
        "invalid_checking_level": [
            "error", "invalid_checking_level", "decode_fields", "-"
        ],
    }


def characterization_cases(module) -> list[str]:
    names: list[str] = []

    integer_bool = module.create_attr_from_dict(
        {
            "name": "coerced", "owner": "node", "type": "resource",
            "generative": 1,
        }
    )
    if integer_bool.generative != 1:
        raise RuntimeError("integer generative boundary drift")
    names.append("integer_generative_coercion")

    non_string = module.create_attr_from_dict(
        {"name": 17, "owner": "node", "type": "status"}
    )
    if non_string.name != 17:
        raise RuntimeError("non-string name boundary drift")
    names.append("non_string_name")

    extra = module.create_attr_from_dict(
        {
            "name": "extra", "owner": "node", "type": "status",
            "arbitrary": {"nested": True},
        }
    )
    if extra.arbitrary != {"nested": True}:
        raise RuntimeError("arbitrary kwargs boundary drift")
    names.append("arbitrary_extra_attribute")

    nested = module.create_attr_from_dict(
        {
            "name": "nested", "owner": "node", "type": "status",
            "config": {"generative": True, "distribution": "uniform"},
        }
    )
    if not nested.generative or nested.distribution != "uniform":
        raise RuntimeError("nested config boundary drift")
    names.append("nested_python_config")

    expect_python_error(
        lambda: module.create_attr_from_dict(
            {
                "name": "precedence", "owner": "link", "type": "resource",
                "constraint_restrictions": "soft", "restriction": "hard",
            }
        ),
        TypeError,
    )
    names.append("duplicate_restriction_call_binding")

    expect_python_error(
        lambda: module.create_attr_from_dict(
            {
                "name": "checking", "owner": "node", "type": "resource",
                "checking_level": "planet",
            }
        ),
        TypeError,
    )
    names.append("duplicate_checking_level_call_binding")

    class DynamicKey:
        def __init__(self):
            self.calls = 0

        def __hash__(self):
            return id(self)

        def __str__(self):
            self.calls += 1
            return "dynamic_field"

    key = DynamicKey()
    dynamic = module.create_attr_from_dict(
        {"name": "dynamic", "owner": "node", "type": "status", key: 9}
    )
    if key.calls != 1 or dynamic.dynamic_field != 9:
        raise RuntimeError("dynamic key stringification boundary drift")
    names.append("dynamic_key_stringified_once")
    return names


def parse_harness(path: pathlib.Path) -> dict[str, list[str]]:
    process = subprocess.run(
        [str(path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"attribute factory harness failed: {process.stderr.strip()}")
    result: dict[str, list[str]] = {}
    passed = False
    for line in process.stdout.splitlines():
        if line == "status=PASS":
            passed = True
            continue
        if not line.startswith("case="):
            raise RuntimeError(f"malformed harness line: {line!r}")
        fields = line.split("|")
        name = fields[0][5:]
        if name in result:
            raise RuntimeError(f"duplicate harness case: {name}")
        result[name] = fields[1:]
    if not passed:
        raise RuntimeError("attribute factory harness did not report PASS")
    return result


def compare_cases(cpp: dict[str, list[str]], expected: dict[str, list[str]]) -> None:
    if set(cpp) != set(expected):
        raise RuntimeError(
            f"case inventory mismatch: missing={sorted(set(expected) - set(cpp))}, "
            f"extra={sorted(set(cpp) - set(expected))}"
        )
    for name, wanted in expected.items():
        if cpp[name] != wanted:
            raise RuntimeError(
                f"{name} mismatch:\npython/contract={wanted!r}\ncpp={cpp[name]!r}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    module = load_oracle(args.source)
    try:
        differential = differential_cases(module)
        native = native_extension_cases(module)
        boundaries = characterization_cases(module)
        expected = dict(differential)
        expected.update(native)
        cpp = parse_harness(args.harness)
        compare_cases(cpp, expected)
        numpy_version = module.np.__version__
    finally:
        unload_oracle()

    payload = {
        "source_sha256": SOURCE_SHA256,
        "dependency_sha256": {
            "base_attribute.py": BASE_SHA256,
            "attribute_method.py": METHOD_SHA256,
            "node_attribute.py": NODE_SHA256,
            "link_attribute.py": LINK_SHA256,
            "graph_attribute.py": GRAPH_SHA256,
        },
        "harness_sha256": hashlib.sha256(args.harness.read_bytes()).hexdigest(),
        "numpy_version": numpy_version,
        "differential_cases": len(differential),
        "native_extension_cases": len(native),
        "python_boundary_cases": len(boundaries),
        "python_boundary_names": boundaries,
        "python_boundary_descriptions": {
            name: PYTHON_BOUNDARY_DESCRIPTIONS[name] for name in boundaries
        },
        "total_cases": len(differential) + len(native) + len(boundaries),
        "result": "PASS",
    }
    if args.output:
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"attribute_factory differential: PASS ({payload['total_cases']} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
