#!/usr/bin/env python3
"""Exact differential and paired runtime gate for ``virne.utils.setting``.

The script direct-loads the pinned Python leaf instead of importing ``virne``.
It expects every physical fixture root separately, so duplicate files in the
original ``settings`` and ``tests/settings`` trees remain distinct test rows.
Run ``--help`` for the complete, reusable command-line contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml


EXPECTED_SOURCE_SHA256 = (
    "89FB0E0D6E40FB10703ADB30AD588731DACD0475F34E2E86F1EBFBB4E2031C8E"
)
EXPECTED_PYTHON = "3.10.20"
EXPECTED_PYYAML = "6.0.1"
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class Fixture:
    name: str
    path: Path
    source: bytes


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    format: str
    source: bytes
    policy: str = "compatible"


@dataclass(frozen=True)
class TimingStats:
    median_ms: float
    mad_ms: float
    p95_ms: float


@dataclass(frozen=True)
class TimingRow:
    python: TimingStats
    cpp: TimingStats
    speedup: float


@dataclass(frozen=True)
class JsonFloatToken:
    name: str
    token: str
    expected_bits: int


def load_original(path: Path) -> types.ModuleType:
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest().upper()
    if digest != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"Python source SHA-256 {digest}, expected {EXPECTED_SOURCE_SHA256}"
        )
    module = types.ModuleType("_virne_setting_oracle")
    module.__file__ = str(path)
    module.__package__ = ""
    torch_modules_before = {
        name
        for name in sys.modules
        if name == "torch" or name.startswith("torch.")
    }
    exec(compile(source, str(path), "exec"), module.__dict__)
    torch_modules_after = {
        name
        for name in sys.modules
        if name == "torch" or name.startswith("torch.")
    }
    if torch_modules_after != torch_modules_before:
        raise RuntimeError("direct-loading setting.py imported an ML dependency")
    for name in ("read_setting", "write_setting", "conver_format"):
        if not callable(getattr(module, name, None)):
            raise RuntimeError(f"direct-loaded source does not define {name}()")
    return module


def verify_runtime() -> None:
    python_version = platform.python_version()
    if python_version != EXPECTED_PYTHON:
        raise RuntimeError(
            f"oracle requires CPython {EXPECTED_PYTHON}, found {python_version}"
        )
    if yaml.__version__ != EXPECTED_PYYAML:
        raise RuntimeError(
            f"oracle requires PyYAML {EXPECTED_PYYAML}, found {yaml.__version__}"
        )


def collect_fixtures(roots: Iterable[Path], expected_count: int) -> list[Fixture]:
    fixtures: list[Fixture] = []
    resolved_roots: set[Path] = set()
    for raw_root in roots:
        root = raw_root.resolve()
        if root in resolved_roots:
            raise ValueError(f"duplicate fixture root: {root}")
        resolved_roots.add(root)
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in sorted(root.rglob("*.yaml")):
            relative = path.relative_to(root).as_posix()
            fixtures.append(Fixture(f"{root.name}/{relative}", path, path.read_bytes()))
    if len(fixtures) != expected_count:
        raise AssertionError(
            f"fixture corpus has {len(fixtures)} YAML files, expected {expected_count}"
        )
    return fixtures


def synthetic_cases() -> tuple[SyntheticCase, ...]:
    return (
        SyntheticCase("yaml_empty", "yaml", b""),
        SyntheticCase("yaml_null", "yaml", b"null\n"),
        SyntheticCase(
            "yaml_bools",
            "yaml",
            b"'yes-key': yes\n'no-key': NO\n'false-key': false\n",
        ),
        SyntheticCase("yaml_octal", "yaml", b"octal: 012\ndecimal: -123\n"),
        SyntheticCase(
            "yaml_float_bits",
            "yaml",
            b"zero: -0.0\nsmall: !!float 5e-324\nnormal: 0.1\nlarge: 1.7976931348623157e308\n",
        ),
        SyntheticCase("yaml_nonfinite", "yaml", b"nan: .nan\npos: .inf\nneg: -.inf\n"),
        SyntheticCase(
            "yaml_order_duplicates",
            "yaml",
            b"z: 1\na: 2\nz: 3\nm: [null, true, false]\n",
        ),
        SyntheticCase(
            "yaml_unicode_strings",
            "yaml",
            "plain: hello\nquoted: 'yes'\nunicode: héllo 世界 😀\nescaped: \"a\\tb\\n\"\n".encode(),
        ),
        SyntheticCase(
            "yaml_nested_empty",
            "yaml",
            b"object: {inner: {}}\nlist: [[], {}, [1, 2]]\n",
        ),
        SyntheticCase(
            "yaml_alias",
            "yaml",
            b"shared: &node\n  x: 1\nleft: *node\nright: *node\n",
        ),
        SyntheticCase("yaml_cycle", "yaml", b"cycle: &items [*items]\n"),
        SyntheticCase("yaml_huge_integer", "yaml", b"huge: 123456789012345678901234567890\n"),
        SyntheticCase("json_null", "json", b"null"),
        SyntheticCase("json_scalars", "json", b"[true, false, -0.0, 0.1, 5e-324]"),
        SyntheticCase(
            "json_order_duplicates",
            "json",
            b'{"z": 1, "a": 2, "z": 3, "nested": {"b": 1, "a": 2}}',
        ),
        SyntheticCase(
            "json_unicode",
            "json",
            '{"text": "héllo 世界 😀", "escaped": "a\\tb\\n"}'.encode(),
        ),
        SyntheticCase(
            "json_float_bits",
            "json",
            b'{"negative_zero": -0.0, "small": 5e-324, "large": 1.7976931348623157e308}',
        ),
        SyntheticCase(
            "json_huge_integer",
            "json",
            b'{"huge": 123456789012345678901234567890}',
        ),
        SyntheticCase("json_nan", "json", b'{"value": NaN}'),
        SyntheticCase("json_infinity", "json", b'{"p": Infinity, "n": -Infinity}'),
        SyntheticCase("yaml_python_tag", "yaml", b"!!python/tuple [1, 2]\n", "security_reject"),
        SyntheticCase("yaml_non_string_key", "yaml", b"1: value\n", "security_reject"),
        SyntheticCase("yaml_timestamp", "yaml", b"when: 2001-01-01\n", "security_reject"),
        SyntheticCase("yaml_binary", "yaml", b"blob: !!binary SGVsbG8=\n", "security_reject"),
        SyntheticCase(
            "yaml_merge_key",
            "yaml",
            b"base: &base {a: 1}\nmerged: {<<: *base, b: 2}\n",
            "security_reject",
        ),
        SyntheticCase("yaml_multiple_documents", "yaml", b"a: 1\n---\nb: 2\n", "invalid"),
        SyntheticCase("yaml_malformed", "yaml", b"a: [1, 2\n", "invalid"),
        SyntheticCase("json_empty", "json", b"", "invalid"),
        SyntheticCase("json_trailing_comma", "json", b'{"a": 1,}', "invalid"),
        SyntheticCase("json_comment", "json", b'{"a": /*x*/ 1}', "invalid"),
        SyntheticCase("json_multiple_values", "json", b"1 2", "invalid"),
        SyntheticCase("json_bom", "json", b"\xef\xbb\xbf{}", "invalid"),
    )


def float64_bits(value: float) -> int:
    return struct.unpack(">Q", struct.pack(">d", value))[0]


def splitmix64(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, (value ^ (value >> 31)) & MASK64


def json_float_tokens(random_count: int) -> tuple[list[JsonFloatToken], int]:
    raw_tokens: list[tuple[str, str]] = [
        ("positive_zero", "0.0"),
        ("negative_zero", "-0.0"),
        ("one_tenth", "0.1"),
        ("next_after_one", "1.0000000000000002"),
        ("min_subnormal_short", "5e-324"),
        ("min_subnormal_long", "4.9406564584124654e-324"),
        ("min_normal", "2.2250738585072014e-308"),
        ("near_min_normal", "2.225073858507201e-308"),
        ("max_finite", "1.7976931348623157e308"),
        ("positive_overflow", "1.7976931348623159e308"),
        ("negative_overflow", "-1e400"),
        ("positive_underflow", "1e-400"),
        ("negative_underflow", "-1e-400"),
    ]
    exponents = (
        -400,
        -325,
        -324,
        -323,
        -309,
        -308,
        -307,
        -100,
        -10,
        -1,
        0,
        1,
        10,
        100,
        307,
        308,
        309,
        400,
    )
    digit_source = "12345678987654321"
    for digits_count in range(1, 18):
        digits = digit_source[:digits_count]
        dotted = digits[0] + "." + (digits[1:] or "0")
        for exponent in exponents:
            for sign_name, sign in (("positive", ""), ("negative", "-")):
                raw_tokens.append(
                    (
                        f"digits_{digits_count}_{sign_name}_e{exponent}",
                        f"{sign}{dotted}e{exponent}",
                    )
                )
                # Exponent-only spelling exercises the short significant-digit
                # fast path separately from a token containing a decimal point.
                if digits_count <= 14:
                    raw_tokens.append(
                        (
                            f"integer_digits_{digits_count}_{sign_name}_e{exponent}",
                            f"{sign}{digits}e{exponent}",
                        )
                    )

    tokens: list[JsonFloatToken] = []
    for name, token in raw_tokens:
        parsed = json.loads(token)
        if not isinstance(parsed, float):
            raise AssertionError(f"generated JSON float token parsed as {type(parsed).__name__}")
        tokens.append(JsonFloatToken(name, token, float64_bits(parsed)))
    deterministic_count = len(tokens)

    state = 0xD1B54A32D192ED03
    generated = 0
    while generated < random_count:
        state, bits = splitmix64(state)
        if (bits >> 52) & 0x7FF == 0x7FF:
            continue
        value = struct.unpack(">d", struct.pack(">Q", bits))[0]
        token = repr(value)
        parsed = json.loads(token)
        parsed_bits = float64_bits(parsed)
        if parsed_bits != bits:
            raise AssertionError(
                f"CPython repr/json round trip changed bits {bits:016x} -> {parsed_bits:016x}"
            )
        tokens.append(
            JsonFloatToken(f"random_{generated:05d}_{bits:016x}", token, bits)
        )
        generated += 1
    return tokens, deterministic_count


def suffix(format_name: str) -> str:
    return ".json" if format_name == "json" else ".yaml"


def python_read(module: types.ModuleType, source: bytes, format_name: str) -> Any:
    with tempfile.TemporaryDirectory(prefix="vne-setting-read-") as directory:
        path = Path(directory) / ("input" + suffix(format_name))
        path.write_bytes(source)
        return module.read_setting(str(path))


def python_write(module: types.ModuleType, value: Any, format_name: str) -> bytes:
    with tempfile.TemporaryDirectory(prefix="vne-setting-write-") as directory:
        path = Path(directory) / ("output" + suffix(format_name))
        returned = module.write_setting(value, str(path))
        if returned is not value:
            raise AssertionError("write_setting did not return the input object identity")
        return path.read_bytes()


def python_transform(
    module: types.ModuleType,
    source: bytes,
    input_format: str,
    output_format: str,
) -> tuple[Any, bytes]:
    value = python_read(module, source, input_format)
    return value, python_write(module, value, output_format)


def canonical(value: Any, memo: dict[int, int] | None = None) -> Any:
    if memo is None:
        memo = {}
    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", str(value))
    if isinstance(value, float):
        bits = struct.unpack(">Q", struct.pack(">d", value))[0]
        return ("float64", f"{bits:016x}")
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, (list, dict)):
        address = id(value)
        if address in memo:
            return ("ref", memo[address])
        identity = len(memo)
        memo[address] = identity
        if isinstance(value, list):
            return ("list", identity, tuple(canonical(item, memo) for item in value))
        return (
            "dict",
            identity,
            tuple((canonical(key, memo), canonical(item, memo)) for key, item in value.items()),
        )
    raise TypeError(f"unsupported canonical type: {type(value).__name__}")


def run_harness(
    executable: Path,
    arguments: list[str],
    stdin: bytes,
    timeout: float,
) -> dict[str, Any]:
    completed = subprocess.run(
        [str(executable), *arguments],
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    try:
        payload = json.loads(completed.stdout)
    except Exception as error:
        raise RuntimeError(
            f"harness emitted invalid JSON for {arguments!r}: "
            f"stdout={completed.stdout!r}, stderr={completed.stderr!r}"
        ) from error
    if not isinstance(payload, dict) or not isinstance(payload.get("ok"), bool):
        raise RuntimeError(f"malformed harness response: {payload!r}")
    if payload["ok"] != (completed.returncode == 0):
        raise RuntimeError(
            f"harness exit/status disagreement: exit={completed.returncode}, payload={payload!r}"
        )
    return payload


def cpp_transform(
    executable: Path,
    source: bytes,
    input_format: str,
    output_format: str,
    timeout: float,
) -> bytes:
    response = run_harness(
        executable, ["transform", input_format, output_format], source, timeout
    )
    if not response["ok"]:
        message = bytes.fromhex(str(response.get("message_hex", ""))).decode(
            "utf-8", errors="replace"
        )
        raise AssertionError(
            f"C++ transform {input_format}->{output_format} rejected input: "
            f"code={response.get('error_code')}, message={message!r}"
        )
    output = bytes.fromhex(str(response["output_hex"]))
    if response.get("output_size") != len(output):
        raise AssertionError("harness output_size differs from decoded output_hex")
    return output


def require_cpp_rejection(
    executable: Path, case: SyntheticCase, timeout: float
) -> dict[str, Any]:
    response = run_harness(
        executable,
        ["transform", case.format, case.format],
        case.source,
        timeout,
    )
    if response["ok"]:
        raise AssertionError(f"{case.name}: C++ accepted a required rejection boundary")
    if not isinstance(response.get("error_code"), int):
        raise AssertionError(f"{case.name}: rejection has no integer error_code")
    return response


def parse_output_with_oracle(
    module: types.ModuleType, output: bytes, format_name: str
) -> Any:
    return python_read(module, output, format_name)


def exact_transform_gate(
    module: types.ModuleType,
    executable: Path,
    name: str,
    source: bytes,
    input_format: str,
    output_format: str,
    timeout: float,
) -> None:
    expected_value, expected_bytes = python_transform(
        module, source, input_format, output_format
    )
    actual_bytes = cpp_transform(
        executable, source, input_format, output_format, timeout
    )
    if actual_bytes != expected_bytes:
        mismatch = next(
            (
                index
                for index, pair in enumerate(zip(actual_bytes, expected_bytes))
                if pair[0] != pair[1]
            ),
            min(len(actual_bytes), len(expected_bytes)),
        )
        raise AssertionError(
            f"{name} {input_format}->{output_format}: byte mismatch at {mismatch}; "
            f"expected={expected_bytes!r}, actual={actual_bytes!r}"
        )
    actual_value = parse_output_with_oracle(module, actual_bytes, output_format)
    # Compare the observable output AST.  PyYAML's default Dumper sorts every
    # mapping, so a YAML round trip intentionally changes the insertion order
    # seen by Loader.  The YAML->JSON row separately checks the original Loader
    # insertion order because json.dump does not sort keys.
    expected_output_value = parse_output_with_oracle(
        module, expected_bytes, output_format
    )
    expected_canonical = canonical(expected_output_value)
    actual_canonical = canonical(actual_value)
    if actual_canonical != expected_canonical:
        raise AssertionError(
            f"{name} {input_format}->{output_format}: AST/type/order/alias mismatch; "
            f"expected={expected_canonical!r}, actual={actual_canonical!r}"
        )


def json_float_bit_gate(
    module: types.ModuleType,
    executable: Path,
    random_count: int,
    chunk_size: int,
    timeout: float,
) -> tuple[int, int]:
    tokens, deterministic_count = json_float_tokens(random_count)
    for begin in range(0, len(tokens), chunk_size):
        chunk = tokens[begin : begin + chunk_size]
        source = ("[" + ",".join(item.token for item in chunk) + "]").encode(
            "ascii"
        )
        try:
            exact_transform_gate(
                module,
                executable,
                f"json_float_chunk_{begin // chunk_size:04d}",
                source,
                "json",
                "json",
                timeout,
            )
        except Exception as chunk_error:
            for item in chunk:
                single = ("[" + item.token + "]").encode("ascii")
                try:
                    exact_transform_gate(
                        module,
                        executable,
                        "json_float_" + item.name,
                        single,
                        "json",
                        "json",
                        timeout,
                    )
                except Exception as item_error:
                    raise AssertionError(
                        f"JSON binary64 mismatch at {item.name}: token={item.token!r}, "
                        f"expected_bits={item.expected_bits:016x}: {item_error}"
                    ) from item_error
            raise AssertionError(
                f"JSON binary64 chunk {begin // chunk_size} failed although all "
                f"individual tokens passed: {chunk_error}"
            ) from chunk_error
    return len(tokens), deterministic_count


def fnv1a(data: bytes, state: int = FNV_OFFSET) -> int:
    for byte in data:
        state ^= byte
        state = (state * FNV_PRIME) & MASK64
    return state


def batch_checksum(outputs: list[bytes]) -> int:
    state = FNV_OFFSET
    for output in outputs:
        state = fnv1a(output, state)
        state ^= len(output)
        state = (state * FNV_PRIME) & MASK64
    return state


def exact_batch_gate(
    executable: Path,
    sources: list[bytes],
    expected: list[bytes],
    format_name: str,
    workers: list[int],
    timeout: float,
) -> None:
    if not sources or len(sources) != len(expected):
        raise ValueError("batch gate requires equally-sized, non-empty input/output lists")
    expected_checksum = batch_checksum(expected)
    stdin = b"\0".join(sources)
    for worker in workers:
        response = run_harness(
            executable,
            ["batch", format_name, format_name, str(worker)],
            stdin,
            timeout,
        )
        if not response["ok"]:
            raise AssertionError(f"batch workers={worker} failed: {response!r}")
        outputs = [bytes.fromhex(item) for item in response.get("outputs_hex", [])]
        if outputs != expected:
            index = next(
                (index for index, pair in enumerate(zip(outputs, expected)) if pair[0] != pair[1]),
                min(len(outputs), len(expected)),
            )
            raise AssertionError(f"batch workers={worker}: output mismatch at item {index}")
        if response.get("count") != len(expected):
            raise AssertionError(f"batch workers={worker}: count mismatch")
        if response.get("checksum") != expected_checksum:
            raise AssertionError(
                f"batch workers={worker}: checksum {response.get('checksum')} "
                f"!= independently computed {expected_checksum}"
            )


def python_file_contract(module: types.ModuleType) -> int:
    checks = 0
    with tempfile.TemporaryDirectory(prefix="vne-setting-file-contract-") as directory:
        root = Path(directory)
        odd_json = root / "acceptedjson"
        odd_json.write_text('{"a": 1}', encoding="utf-8")
        if module.read_setting(str(odd_json)) != {"a": 1}:
            raise AssertionError("last-four-character JSON suffix contract changed")
        checks += 1

        odd_yaml = root / "acceptedyaml"
        odd_yaml.write_text("a: 1\n", encoding="utf-8")
        if module.read_setting(str(odd_yaml)) != {"a": 1}:
            raise AssertionError("last-four-character YAML suffix contract changed")
        checks += 1

        existing_bad = root / "existing.txt"
        existing_bad.write_text("payload", encoding="utf-8")
        try:
            module.read_setting(str(existing_bad))
        except ValueError:
            pass
        else:
            raise AssertionError("read_setting wrong suffix no longer raises ValueError")
        checks += 1

        try:
            module.read_setting(str(root / "missing.txt"))
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("read_setting no longer opens before suffix dispatch")
        checks += 1

        bad_output = root / "output.txt"
        bad_output.write_bytes(b"must be truncated")
        returned = module.write_setting({"a": 1}, str(bad_output))
        if not isinstance(returned, ValueError) or bad_output.read_bytes() != b"":
            raise AssertionError("write_setting wrong-suffix return/truncation contract changed")
        checks += 1

        uppercase = root / "output.JSON"
        returned = module.write_setting({"a": 1}, str(uppercase))
        if not isinstance(returned, ValueError) or uppercase.read_bytes() != b"":
            raise AssertionError("suffix dispatch unexpectedly became case-insensitive")
        checks += 1

        converted = root / "converted.txt"
        result = module.conver_format(str(odd_json), str(converted))
        if result is not None or converted.read_bytes() != b"":
            raise AssertionError("conver_format wrong-output-suffix behavior changed")
        checks += 1

        pathlib_input = root / "pathlib.yaml"
        pathlib_input.write_text("a: 1\n", encoding="utf-8")
        try:
            module.read_setting(pathlib_input)
        except TypeError:
            pass
        else:
            raise AssertionError("Path input no longer opens before failing suffix slicing")
        checks += 1

        pathlib_output = root / "pathlib-output.yaml"
        try:
            module.write_setting({"a": 1}, pathlib_output)
        except TypeError:
            pass
        else:
            raise AssertionError("Path output no longer fails suffix slicing")
        if pathlib_output.read_bytes() != b"":
            raise AssertionError("Path output must be created before suffix slicing fails")
        checks += 1

        try:
            module.read_setting(str(odd_json), mode="rb")
        except ValueError:
            pass
        else:
            raise AssertionError("binary mode plus text encoding no longer fails at open")
        checks += 1
    return checks


def serializer(format_name: str) -> tuple[Callable[[str], Any], Callable[[Any], str]]:
    if format_name == "json":
        return json.loads, json.dumps
    return (
        lambda text: yaml.load(text, Loader=yaml.Loader),
        lambda value: yaml.dump(value),
    )


def first_mapping_value(value: Any) -> tuple[Any, Any] | None:
    if isinstance(value, dict) and value:
        key = next(iter(value))
        return value, key
    return None


def timed_python_sample(
    format_name: str,
    source: bytes,
    scalar_operations: int,
    batch_operations: int,
    batch_size: int,
    id_iterations: int,
) -> dict[str, tuple[int, int]]:
    loads, dumps = serializer(format_name)
    text = source.decode("utf-8")
    seed = loads(text)
    expected_dump = dumps(seed)
    parsed: Any = None

    begin = time.perf_counter_ns()
    for _ in range(scalar_operations):
        parsed = loads(text)
    parse_ns = time.perf_counter_ns() - begin
    if canonical(parsed) != canonical(seed):
        raise AssertionError("Python timed parse changed the AST")

    output = ""
    begin = time.perf_counter_ns()
    for _ in range(scalar_operations):
        output = dumps(seed)
    dump_ns = time.perf_counter_ns() - begin
    if output != expected_dump:
        raise AssertionError("Python timed dump changed output")

    parsed_batch: list[Any] = []
    begin = time.perf_counter_ns()
    for _ in range(batch_operations):
        parsed_batch = [loads(text) for _ in range(batch_size)]
    batch_parse_ns = time.perf_counter_ns() - begin
    if len(parsed_batch) != batch_size:
        raise AssertionError("Python timed batch parse changed result count")

    dumped_batch: list[str] = []
    begin = time.perf_counter_ns()
    for _ in range(batch_operations):
        dumped_batch = [dumps(seed) for _ in range(batch_size)]
    batch_dump_ns = time.perf_counter_ns() - begin
    if dumped_batch != [expected_dump] * batch_size:
        raise AssertionError("Python timed batch dump changed output")

    access_ns = 0
    access_count = 0
    mapping = first_mapping_value(seed)
    if mapping is not None and id_iterations:
        object_value, key = mapping
        sink = 0
        begin = time.perf_counter_ns()
        for index in range(id_iterations):
            sink ^= hash(type(object_value[key])) + index
        access_ns = time.perf_counter_ns() - begin
        access_count = id_iterations
        if sink == -1:
            raise AssertionError("unreachable benchmark sink")

    return {
        "parse": (parse_ns, scalar_operations),
        "dump": (dump_ns, scalar_operations),
        "batch_parse": (batch_parse_ns, batch_operations * batch_size),
        "batch_dump": (batch_dump_ns, batch_operations * batch_size),
        "id_access": (access_ns, access_count),
    }


def timed_cpp_sample(
    executable: Path,
    format_name: str,
    source: bytes,
    workers: int,
    operations: int,
    batch_size: int,
    id_iterations: int,
    timeout: float,
) -> tuple[dict[str, tuple[int, int]], int]:
    response = run_harness(
        executable,
        [
            "benchmark",
            format_name,
            str(workers),
            str(operations),
            str(batch_size),
            str(id_iterations),
        ],
        source,
        timeout,
    )
    if not response["ok"]:
        raise AssertionError(f"C++ benchmark rejected {format_name}: {response!r}")
    if response.get("workers") != workers:
        raise AssertionError("C++ benchmark worker metadata mismatch")
    rows: dict[str, tuple[int, int]] = {}
    for name in ("parse", "dump", "batch_parse", "batch_dump", "id_access"):
        raw = response.get(name)
        if not isinstance(raw, dict):
            raise AssertionError(f"C++ benchmark omitted row {name}")
        total_ns = raw.get("total_ns")
        count = raw.get("operations")
        if not isinstance(total_ns, int) or not isinstance(count, int):
            raise AssertionError(f"C++ benchmark row {name} has invalid timing metadata")
        if total_ns < 0 or count < 0 or (count and total_ns == 0):
            raise AssertionError(f"C++ benchmark row {name} has invalid timing values")
        rows[name] = (total_ns, count)
    checksum = response.get("checksum")
    if not isinstance(checksum, int):
        raise AssertionError("C++ benchmark omitted checksum")
    return rows, checksum


def percentile95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def stats(values: list[float]) -> TimingStats:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("timing samples must be finite and positive")
    median = float(statistics.median(values))
    deviations = [abs(value - median) for value in values]
    return TimingStats(median, float(statistics.median(deviations)), percentile95(values))


def benchmark_gate(
    executable: Path,
    yaml_source: bytes,
    workers: int,
    warmups: int,
    repetitions: int,
    scalar_operations: int,
    batch_operations: int,
    batch_size: int,
    id_iterations: int,
    timeout: float,
) -> tuple[dict[str, TimingRow], list[str]]:
    yaml_value = yaml.load(yaml_source.decode("utf-8"), Loader=yaml.Loader)
    inputs = {
        "yaml": yaml_source,
        "json": json.dumps(yaml_value).encode("utf-8"),
    }
    samples: dict[str, dict[str, list[float]]] = {
        f"{format_name}_{row}_{runtime}": []
        for format_name in inputs
        for row in ("parse", "dump", "batch_parse", "batch_dump", "id_access")
        for runtime in ("python", "cpp")
    }
    checksums: dict[tuple[str, str], int] = {}
    for format_index, (format_name, source) in enumerate(inputs.items()):
        for index in range(warmups + repetitions):
            python_rows: dict[str, tuple[int, int]]
            cpp_rows: dict[str, tuple[int, int]]

            def cpp_profiles() -> tuple[dict[str, tuple[int, int]], tuple[int, int]]:
                profile_arguments = (
                    ("scalar", scalar_operations, 1, id_iterations),
                    ("batch", batch_operations, batch_size, 1),
                )
                if (index + format_index) % 2:
                    profile_arguments = tuple(reversed(profile_arguments))
                profile_rows: dict[str, dict[str, tuple[int, int]]] = {}
                profile_checksums: dict[str, int] = {}
                for profile, operations, items, accesses in profile_arguments:
                    rows, checksum = timed_cpp_sample(
                        executable,
                        format_name,
                        source,
                        workers,
                        operations,
                        items,
                        accesses,
                        timeout,
                    )
                    profile_rows[profile] = rows
                    profile_checksums[profile] = checksum
                combined = {
                    "parse": profile_rows["scalar"]["parse"],
                    "dump": profile_rows["scalar"]["dump"],
                    "id_access": profile_rows["scalar"]["id_access"],
                    "batch_parse": profile_rows["batch"]["batch_parse"],
                    "batch_dump": profile_rows["batch"]["batch_dump"],
                }
                return combined, (
                    profile_checksums["scalar"],
                    profile_checksums["batch"],
                )

            if (index + format_index) % 2 == 0:
                python_rows = timed_python_sample(
                    format_name,
                    source,
                    scalar_operations,
                    batch_operations,
                    batch_size,
                    id_iterations,
                )
                cpp_rows, profile_checksums = cpp_profiles()
            else:
                cpp_rows, profile_checksums = cpp_profiles()
                python_rows = timed_python_sample(
                    format_name,
                    source,
                    scalar_operations,
                    batch_operations,
                    batch_size,
                    id_iterations,
                )
            for profile, checksum in zip(("scalar", "batch"), profile_checksums):
                key = (format_name, profile)
                previous = checksums.setdefault(key, checksum)
                if checksum != previous:
                    raise AssertionError(
                        f"C++ {format_name}/{profile} benchmark checksum changed: "
                        f"{checksum} != {previous}"
                    )
            if index < warmups:
                continue
            for row in ("parse", "dump", "batch_parse", "batch_dump", "id_access"):
                python_total, python_count = python_rows[row]
                cpp_total, cpp_count = cpp_rows[row]
                if python_count != cpp_count:
                    raise AssertionError(
                        f"{format_name}_{row}: Python/C++ operation count mismatch "
                        f"{python_count} != {cpp_count}"
                    )
                if python_count == 0:
                    continue
                samples[f"{format_name}_{row}_python"].append(
                    python_total / python_count / 1_000_000.0
                )
                samples[f"{format_name}_{row}_cpp"].append(
                    cpp_total / cpp_count / 1_000_000.0
                )

    report: dict[str, TimingRow] = {}
    slower: list[str] = []
    for format_name in inputs:
        for row in ("parse", "dump", "batch_parse", "batch_dump", "id_access"):
            name = f"{format_name}_{row}"
            python_stats = stats(samples[name + "_python"])
            cpp_stats = stats(samples[name + "_cpp"])
            speedup = python_stats.median_ms / cpp_stats.median_ms
            report[name] = TimingRow(python_stats, cpp_stats, speedup)
            if speedup <= 1.0:
                slower.append(name)
    return report, slower


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exact Python/C++ setting differential plus paired parse/dump/batch timing gate"
        )
    )
    parser.add_argument("--cpp", required=True, type=Path, help="setting harness executable")
    parser.add_argument(
        "--python-source", required=True, type=Path, help="original virne/utils/setting.py"
    )
    parser.add_argument(
        "--fixture-root",
        required=True,
        action="append",
        type=Path,
        help="fixture tree (repeat; every physical *.yaml is a distinct case)",
    )
    parser.add_argument("--expected-fixtures", type=int, default=41)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=31)
    parser.add_argument(
        "--operations",
        dest="scalar_operations",
        type=int,
        default=256,
        help="inner operations for single-document parse/dump timing",
    )
    parser.add_argument(
        "--batch-operations",
        type=int,
        default=2,
        help="inner rounds for batch parse/dump timing",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--id-iterations", type=int, default=200_000)
    parser.add_argument(
        "--random-float-cases",
        type=int,
        default=16_384,
        help="deterministic finite binary64 bit patterns in the JSON differential",
    )
    parser.add_argument(
        "--float-chunk-size",
        type=int,
        default=256,
        help="JSON binary64 values per harness invocation",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--allow-slower",
        action="store_true",
        help="report timing rows without failing when any C++ median is slower",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    verify_runtime()
    executable = args.cpp.resolve()
    source_path = args.python_source.resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if args.expected_fixtures < 1 or args.workers < 0:
        raise ValueError("expected-fixtures must be positive and workers non-negative")
    if args.warmups < 0 or args.repetitions < 1:
        raise ValueError("warmups must be non-negative and repetitions positive")
    if (
        args.scalar_operations < 1
        or args.batch_operations < 1
        or args.batch_size < 1
        or args.id_iterations < 1
    ):
        raise ValueError(
            "operations, batch-operations, batch-size, and id-iterations must be positive"
        )
    if args.random_float_cases < 0 or args.float_chunk_size < 1:
        raise ValueError("random-float-cases must be non-negative and chunk size positive")
    if args.timeout <= 0:
        raise ValueError("timeout must be positive")

    module = load_original(source_path)
    fixtures = collect_fixtures(args.fixture_root, args.expected_fixtures)
    fixture_expected: list[bytes] = []
    for index, fixture in enumerate(fixtures, 1):
        exact_transform_gate(
            module,
            executable,
            fixture.name,
            fixture.source,
            "yaml",
            "yaml",
            args.timeout,
        )
        exact_transform_gate(
            module,
            executable,
            fixture.name,
            fixture.source,
            "yaml",
            "json",
            args.timeout,
        )
        _, output = python_transform(module, fixture.source, "yaml", "yaml")
        fixture_expected.append(output)
        print(f"fixture[{index:02d}/{len(fixtures)}]={fixture.name}: PASS")

    batch_modes = list(dict.fromkeys((1, args.workers, 0)))
    exact_batch_gate(
        executable,
        [fixture.source for fixture in fixtures],
        fixture_expected,
        "yaml",
        batch_modes,
        args.timeout,
    )

    compatible = 0
    rejected = 0
    synthetic_errors: list[str] = []
    compatible_batches: dict[str, list[tuple[bytes, bytes]]] = {"json": [], "yaml": []}
    for case in synthetic_cases():
        try:
            if case.policy == "compatible":
                exact_transform_gate(
                    module,
                    executable,
                    case.name,
                    case.source,
                    case.format,
                    case.format,
                    args.timeout,
                )
                _, output = python_transform(module, case.source, case.format, case.format)
                compatible_batches[case.format].append((case.source, output))
                if case.name not in {"yaml_cycle"}:
                    other = "json" if case.format == "yaml" else "yaml"
                    try:
                        exact_transform_gate(
                            module,
                            executable,
                            case.name + "_cross",
                            case.source,
                            case.format,
                            other,
                            args.timeout,
                        )
                    except (ValueError, TypeError):
                        # Some valid YAML values (notably aliases/cycles) are
                        # not representable by stdlib json.dump.
                        if other != "json":
                            raise
                compatible += 1
            else:
                python_failed = False
                try:
                    python_read(module, case.source, case.format)
                except Exception:
                    python_failed = True
                if case.policy == "invalid" and not python_failed:
                    raise AssertionError(
                        f"{case.name}: Python unexpectedly accepted invalid input"
                    )
                if case.policy == "security_reject" and python_failed:
                    raise AssertionError(
                        f"{case.name}: security-boundary case must demonstrate "
                        "Python Loader acceptance"
                    )
                require_cpp_rejection(executable, case, args.timeout)
                rejected += 1
        except Exception as error:
            synthetic_errors.append(f"{case.name}: {error}")
            print(f"synthetic.{case.name}: FAIL ({case.policy}): {error}")
        else:
            print(f"synthetic.{case.name}: PASS ({case.policy})")

    if synthetic_errors:
        raise AssertionError(
            f"{len(synthetic_errors)} synthetic mismatch(es):\n  - "
            + "\n  - ".join(synthetic_errors)
        )

    float_cases, deterministic_float_cases = json_float_bit_gate(
        module,
        executable,
        args.random_float_cases,
        args.float_chunk_size,
        args.timeout,
    )

    for format_name, items in compatible_batches.items():
        exact_batch_gate(
            executable,
            [item[0] for item in items],
            [item[1] for item in items],
            format_name,
            batch_modes,
            args.timeout,
        )

    file_checks = python_file_contract(module)
    benchmark_fixture = max(fixtures, key=lambda item: len(item.source))
    timing, slower = benchmark_gate(
        executable,
        benchmark_fixture.source,
        args.workers,
        args.warmups,
        args.repetitions,
        args.scalar_operations,
        args.batch_operations,
        args.batch_size,
        args.id_iterations,
        args.timeout,
    )

    print(f"source_sha256={EXPECTED_SOURCE_SHA256}")
    print(f"runtime=CPython {platform.python_version()}, PyYAML {yaml.__version__}")
    print(f"fixture_exact_cases={len(fixtures)}")
    print(f"fixture_transforms={len(fixtures) * 2}")
    print(f"synthetic_compatible_cases={compatible}")
    print(f"synthetic_required_rejections={rejected}")
    print(f"json_float_bit_cases={float_cases}")
    print(f"json_float_deterministic_decimal_cases={deterministic_float_cases}")
    print(f"json_float_random_binary64_cases={args.random_float_cases}")
    print(f"python_file_contract_cases={file_checks}")
    print("cpp_file_contract=NOT_EXPOSED_BY_SETTING_HARNESS_V1")
    print(
        f"benchmark_protocol=warmups:{args.warmups},samples:{args.repetitions},"
        f"scalar_operations:{args.scalar_operations},"
        f"batch_operations:{args.batch_operations},batch_size:{args.batch_size},"
        f"workers:{'auto' if args.workers == 0 else args.workers},"
        "paired_alternating_order:true,process_startup_excluded:true"
    )
    for name, row in timing.items():
        print(
            f"{name}: python={row.python.median_ms:.9f} ms/op "
            f"(MAD {row.python.mad_ms:.9f}, p95 {row.python.p95_ms:.9f}), "
            f"cpp={row.cpp.median_ms:.9f} ms/op "
            f"(MAD {row.cpp.mad_ms:.9f}, p95 {row.cpp.p95_ms:.9f}), "
            f"speedup={row.speedup:.3f}x"
        )
    if args.json_output is not None:
        report = {
            "status": "pass" if not slower else "measured_with_slower_rows",
            "component": "virne.utils.setting",
            "source_sha256": EXPECTED_SOURCE_SHA256,
            "runtime": {
                "python": platform.python_version(),
                "pyyaml": yaml.__version__,
                "cpus_visible": os.cpu_count(),
            },
            "differential": {
                "fixtures": len(fixtures),
                "fixture_transforms": len(fixtures) * 2,
                "synthetic_compatible": compatible,
                "required_rejections": rejected,
                "json_float_bit_cases": float_cases,
                "json_float_deterministic_decimal_cases": deterministic_float_cases,
                "json_float_random_binary64_cases": args.random_float_cases,
                "python_file_contract_cases": file_checks,
            },
            "benchmark": {
                "workers": args.workers,
                "warmups": args.warmups,
                "repetitions": args.repetitions,
                "scalar_operations": args.scalar_operations,
                "batch_operations": args.batch_operations,
                "batch_size": args.batch_size,
                "rows": {name: asdict(row) for name, row in timing.items()},
            },
        }
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"json_report={args.json_output}")
    if slower and not args.allow_slower:
        raise AssertionError(
            "C++ median must beat Python for every canonical timing row: "
            + ", ".join(slower)
        )
    print("differential_status=PASS")
    print("performance_status=PASS" if not slower else "performance_status=MEASURED")
    print("result=PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"compare_setting: FAIL: {error}", file=sys.stderr)
        raise
