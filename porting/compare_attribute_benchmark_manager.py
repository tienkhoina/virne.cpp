#!/usr/bin/env python3
"""Direct-source Python/C++ differential for AttributeBenchmarkManager."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import struct
import subprocess
import sys
from typing import Any


SOURCE_SHA256 = "8ad9bab52a40342eaf331e3ac71b3d5f6db0326742d3d3c7c6aad7d34ad52d9e"
_MODULE_NAME = "_virne_attribute_benchmark_manager_oracle"
_MISSING = object()


def bits(value: float) -> str:
    return f"{struct.unpack('>Q', struct.pack('>d', float(value)))[0]:016x}"


def hex_text(value: str) -> str:
    return value.encode("utf-8").hex()


def encode_mapping(values) -> str:
    return ",".join(
        f"{hex_text(key)}={bits(value)}" for key, value in values.items()
    )


def encode_groups(values) -> str:
    def encode_optional(group) -> str:
        return "-" if group is None else encode_mapping(group)

    return (
        f"node={encode_optional(values.node_attr_benchmarks)};"
        f"link={encode_optional(values.link_attr_benchmarks)};"
        f"link_sum={encode_optional(values.link_sum_attr_benchmarks)}"
    )


def load_oracle(source: pathlib.Path):
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"attribute_benchmark_manager source hash drift: {actual}")
    sys.modules.pop(_MODULE_NAME, None)
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot direct-load {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_MODULE_NAME, None)
        raise
    return module


def unload_oracle() -> None:
    sys.modules.pop(_MODULE_NAME, None)


def parse_harness(path: pathlib.Path) -> dict[str, list[str]]:
    process = subprocess.run(
        [str(path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"attribute benchmark harness failed: {process.stderr.strip()}"
        )
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
        raise RuntimeError("attribute benchmark harness did not report PASS")
    return result


class OracleAttribute:
    def __init__(
        self,
        attr_type: str,
        name: str,
        originator: Any = _MISSING,
    ) -> None:
        self.type = attr_type
        self.name = name
        if originator is not _MISSING:
            self.originator = originator

    def __str__(self) -> str:
        return f"oracle:{self.name}"


def uint32_matrix(np, rows: list[list[int]]):
    if not rows:
        return np.empty((0, 0), dtype=np.float32)
    columns = len(rows[0])
    if any(len(row) != columns for row in rows):
        raise RuntimeError("oracle bit matrix is ragged")
    return np.asarray(rows, dtype=np.uint32).view(np.float32)


def reduce(module, attr_types, attrs, data) -> list[str]:
    with module.np.errstate(all="ignore"):
        values = module.get_attr_benchmarks(attr_types, attrs, data)
    return ["ok", encode_mapping(values)]


def worker_fixture(module):
    attrs = [OracleAttribute("status", f"worker_{row}") for row in range(12)]
    data = module.np.empty((12, 9), dtype=module.np.float32)
    for row in range(12):
        for column in range(9):
            residue = (row * 31 + column * 7) % 101
            data[row, column] = module.np.float32((residue - 50) * 0.25)
    return attrs, module.np.concatenate([data, data], axis=1)


def reduction_cases(module) -> dict[str, list[str]]:
    np = module.np
    result = {
        "empty": reduce(
            module, ["status"], [], np.empty((0, 0), dtype=np.float32)
        ),
        "truncated_rows": reduce(
            module,
            ["status"],
            [
                OracleAttribute("status", "alpha"),
                OracleAttribute("status", "beta"),
                OracleAttribute("status", "omitted"),
            ],
            np.asarray(
                [[1.0, 3.0, 2.0], [-4.0, -1.0, -2.0]],
                dtype=np.float32,
            ),
        ),
        "extra_rows": reduce(
            module,
            ["status"],
            [OracleAttribute("status", "only")],
            np.asarray([[1.0, 2.0], [9.0, 10.0], [11.0, 12.0]], dtype=np.float32),
        ),
        "extrema_resource_originator_duplicate": reduce(
            module,
            ["resource", "extrema"],
            [
                OracleAttribute("resource", "capacity"),
                OracleAttribute("extrema", "latency_max", "latency"),
                OracleAttribute("status", "fallback"),
                OracleAttribute("extrema", "latency_late", "latency"),
            ],
            np.asarray(
                [[99.0, 100.0], [-2.0, -1.0], [4.0, 5.0], [7.0, 6.0]],
                dtype=np.float32,
            ),
        ),
        "non_extrema_names_duplicate": reduce(
            module,
            ["status"],
            [
                OracleAttribute("resource", "same"),
                OracleAttribute("extrema", "other", "ignored_originator"),
                OracleAttribute("status", "same"),
            ],
            np.asarray([[1.0, 2.0], [8.0, 9.0], [4.0, 6.0]], dtype=np.float32),
        ),
        "numeric_basics": reduce(
            module,
            ["status"],
            [
                OracleAttribute("status", "rounding"),
                OracleAttribute("status", "positive_inf"),
                OracleAttribute("status", "negative_inf"),
                OracleAttribute("status", "subnormal"),
            ],
            uint32_matrix(
                np,
                [
                    [0x4B800000, 0x4B7FFFFF],
                    [0xFF800000, 0x7F800000],
                    [0xFF800000, 0xC0A00000],
                    [0x80000001, 0x00000001],
                ],
            ),
        ),
        "signed_zero_pairs": reduce(
            module,
            ["status"],
            [
                OracleAttribute("status", "negative_positive"),
                OracleAttribute("status", "positive_negative"),
            ],
            uint32_matrix(np, [[0x80000000, 0x00000000], [0x00000000, 0x80000000]]),
        ),
        "nan_singletons": reduce(
            module,
            ["status"],
            [OracleAttribute("status", "qnan"), OracleAttribute("status", "snan")],
            uint32_matrix(np, [[0x7FC12345], [0xFF812346]]),
        ),
        "nan_pairs": reduce(
            module,
            ["status"],
            [
                OracleAttribute("status", "qnan_first"),
                OracleAttribute("status", "qnan_later"),
                OracleAttribute("status", "snan_first"),
                OracleAttribute("status", "snan_later"),
            ],
            uint32_matrix(
                np,
                [
                    [0x7FC12345, 0x3F800000],
                    [0x3F800000, 0xFFC23456],
                    [0x7F812345, 0x3F800000],
                    [0x3F800000, 0xFF812346],
                ],
            ),
        ),
        "nan_repetition": reduce(
            module,
            ["status"],
            [
                OracleAttribute("status", "qnan_repeat"),
                OracleAttribute("status", "snan_repeat"),
            ],
            np.concatenate(
                [
                    uint32_matrix(np, [[0x7FC12345], [0xFF812346]]),
                    uint32_matrix(np, [[0x7FC12345], [0xFF812346]]),
                ],
                axis=1,
            ),
        ),
    }
    simd_bits = [[0xC0800000] * 17 for _ in range(3)]
    simd_bits[0][0] = 0x7FC12345
    simd_bits[1][8] = 0xFFC23456
    simd_bits[2][16] = 0x7F812345
    result["nan_simd_17"] = reduce(
        module,
        ["status"],
        [
            OracleAttribute("status", "nan_first"),
            OracleAttribute("status", "nan_middle"),
            OracleAttribute("status", "nan_last"),
        ],
        uint32_matrix(np, simd_bits),
    )
    attrs, repeated = worker_fixture(module)
    worker_value = reduce(module, ["status"], attrs, repeated)
    for workers in (0, 1, 2, 8):
        result[f"workers_{workers}"] = worker_value
    return result


class FakeNetwork:
    def __init__(self) -> None:
        self.node = OracleAttribute("extrema", "node_max", "node")
        self.link = OracleAttribute("extrema", "link_max", "link")

    def get_node_attrs(self, attr_types=None):
        del attr_types
        return [self.node]

    def get_node_attrs_data(self, attrs):
        del attrs
        return [[1.0, 3.0, 2.0]]

    def get_link_attrs(self, attr_types=None):
        del attr_types
        return [self.link]

    def get_link_attrs_data(self, attrs):
        del attrs
        return [[-1.0, 4.0, 2.0]]

    def get_aggregation_attrs_data(self, attrs, aggr):
        del attrs
        if aggr != "sum":
            raise RuntimeError("unexpected aggregation")
        return [[7.0, 6.0]]


def manager_cache_cases(module) -> dict[str, list[str]]:
    network = FakeNetwork()
    all_groups = module.AttributeBenchmarkManager.get_benchmarks(network)
    node_only = module.AttributeBenchmarkManager.get_benchmarks(
        network, link_attrs=False, link_sum_attrs=False
    )
    manager = module.AttributeBenchmarkManager(network)
    manager_groups = module.AttributeBenchmarks(
        node_attr_benchmarks=manager.node_attr_benchmarks,
        link_attr_benchmarks=manager.link_attr_benchmarks,
        link_sum_attr_benchmarks=manager.link_sum_attr_benchmarks,
    )

    module.AttributeBenchmarkManager.clear_cache()
    initially_missing = module.AttributeBenchmarkManager.get_from_cache("same") is None
    first = module.AttributeBenchmarks()
    second = module.AttributeBenchmarks()
    module.AttributeBenchmarkManager.add_to_cache("same", first)
    first_identity = module.AttributeBenchmarkManager.get_from_cache("same") is first
    module.AttributeBenchmarkManager.add_to_cache("same", second)
    second_identity = module.AttributeBenchmarkManager.get_from_cache("same") is second
    overwritten_size = len(module.AttributeBenchmarkManager._cache)
    module.AttributeBenchmarkManager.add_to_cache("other", first)
    two_size = len(module.AttributeBenchmarkManager._cache)
    module.AttributeBenchmarkManager.clear_cache()
    cleared_size = len(module.AttributeBenchmarkManager._cache)
    finally_missing = module.AttributeBenchmarkManager.get_from_cache("same") is None
    cache_payload = ",".join(
        str(int(value)) if isinstance(value, bool) else str(value)
        for value in (
            initially_missing,
            first_identity,
            second_identity,
            overwritten_size,
            two_size,
            cleared_size,
            finally_missing,
        )
    )
    return {
        "optional_all_groups": ["ok", encode_groups(all_groups)],
        "optional_node_only": ["ok", encode_groups(node_only)],
        "manager_constructor": ["ok", encode_groups(manager_groups)],
        "cache_semantics": ["ok", cache_payload],
    }


def native_extension_cases() -> dict[str, list[str]]:
    alpha = bits(3.0)
    copied = f"{hex_text('alpha')}={bits(3.0)},{hex_text('beta')}={bits(5.0)}"
    return {
        "map_bind_access_copy": [
            "ok", f"0,1,1,{alpha},{alpha},{copied}"
        ],
        "concurrent_independent": ["ok", "1"],
        "invalid_matrix_shape": ["error", "invalid_matrix_shape"],
        "invalid_matrix_overflow": ["error", "invalid_matrix_shape"],
        "zero_column_repetitions": [
            "error", "invalid_column_repetitions"
        ],
        "empty_retained_row": ["error", "empty_retained_row"],
        "invalid_benchmark_id": ["error", "invalid_benchmark_id"],
    }


def characterization_cases(module) -> list[str]:
    np = module.np
    names: list[str] = []

    class DynamicString:
        type = "status"

        def __init__(self) -> None:
            self.calls = 0

        def __str__(self) -> str:
            self.calls += 1
            return "dynamic-key"

    dynamic = DynamicString()
    dynamic_result = module.get_attr_benchmarks(
        ["status"], [dynamic], np.asarray([[1.0]], dtype=np.float32)
    )
    if list(dynamic_result) != ["dynamic-key"] or dynamic.calls != 1:
        raise RuntimeError("dynamic str fallback boundary drift")
    names.append("dynamic_str_fallback")

    non_string = OracleAttribute("extrema", "maximum", 17)
    non_string_result = module.get_attr_benchmarks(
        ["extrema"], [non_string], np.asarray([[2.0]], dtype=np.float32)
    )
    if list(non_string_result) != [17]:
        raise RuntimeError("non-string originator boundary drift")
    names.append("non_string_key")

    precise = np.nextafter(np.float64(1.0), np.float64(2.0))
    arbitrary_dtype = module.get_attr_benchmarks(
        ["status"],
        [OracleAttribute("status", "precise")],
        np.asarray([[precise]], dtype=np.float64),
    )
    if arbitrary_dtype["precise"] != precise:
        raise RuntimeError("arbitrary dtype boundary drift")
    names.append("arbitrary_numpy_dtype")

    try:
        np.asarray([[1.0], [2.0, 3.0]], dtype=np.float32)
    except ValueError:
        names.append("ragged_array")
    else:
        raise RuntimeError("ragged float32 array unexpectedly succeeded")

    with np.errstate(over="ignore"):
        unbounded = np.asarray([[1 << 200]], dtype=np.float32)
    if not np.isposinf(unbounded[0, 0]):
        raise RuntimeError("unbounded integer conversion boundary drift")
    names.append("unbounded_integer_to_infinity")

    class EmptyAdapter(FakeNetwork):
        def get_node_attrs(self, attr_types=None):
            del attr_types
            return []

        def get_node_attrs_data(self, attrs):
            del attrs
            raise IndexError("BaseNetwork empty-filter characterization")

    try:
        module.AttributeBenchmarkManager.get_node_attr_benchmarks(EmptyAdapter())
    except IndexError:
        names.append("base_network_empty_filter")
    else:
        raise RuntimeError("BaseNetwork empty-filter boundary unexpectedly succeeded")
    return names


def compare_cases(cpp, expected) -> None:
    if set(cpp) != set(expected):
        raise RuntimeError(
            f"case inventory mismatch: missing={sorted(set(expected) - set(cpp))}, "
            f"extra={sorted(set(cpp) - set(expected))}"
        )
    for name, wanted in expected.items():
        actual = cpp[name]
        if wanted[0] == "ok":
            if actual != wanted:
                raise RuntimeError(
                    f"{name} output mismatch:\npython={wanted!r}\ncpp={actual!r}"
                )
        elif actual[0] != "error" or actual[1] != wanted[1]:
            raise RuntimeError(
                f"{name} typed error mismatch: python={wanted!r}, cpp={actual!r}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    module = load_oracle(args.source)
    try:
        differential = reduction_cases(module)
        differential.update(manager_cache_cases(module))
        native = native_extension_cases()
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
        "harness_sha256": hashlib.sha256(args.harness.read_bytes()).hexdigest(),
        "numpy_version": numpy_version,
        "differential_cases": len(differential),
        "native_extension_cases": len(native),
        "python_boundary_cases": len(boundaries),
        "python_boundary_names": boundaries,
        "total_cases": len(differential) + len(native) + len(boundaries),
        "result": "PASS",
    }
    if args.output:
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        "attribute_benchmark_manager differential: "
        f"PASS ({payload['total_cases']} cases)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
