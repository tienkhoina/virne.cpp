#!/usr/bin/env python3
"""Exact filesystem differential for the frozen ``virne.utils.manager`` leaf."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Callable


SOURCE_SHA256 = "77afbdab961d3183d877449bfa76a68ba876929d4e822b0d61eec65f8441fa3b"

LEGACY_TEMP = 0
ENUMERATION_FAILED = 1
NOT_DIRECTORY = 2
UNSAFE_PATH_ESCAPE = 3
REMOVE_FAILED = 4

LIST_TEMP_ROOT = 0
LIST_SAVE_ROOT = 1
LIST_ALGORITHM = 2
LIST_RECORDS = 3
REMOVE_RUN_TREE = 4
REMOVE_EMPTY_DIRECTORY = 5


@dataclass(frozen=True)
class Case:
    name: str
    operation: str
    setup: Callable[[pathlib.Path], list[pathlib.Path]]
    expected_python: str
    expected_cpp_code: int | None
    expected_cpp_operation: int | None
    intentional_escape_deviation: bool = False


@dataclass(frozen=True)
class Result:
    status: str
    category: str
    output: str
    tree: tuple[tuple[str, str, str], ...]
    error_path: str
    cpp_code: int | None = None
    cpp_operation: int | None = None


def load_oracle(source: pathlib.Path) -> Any:
    source_bytes = source.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    if digest != SOURCE_SHA256:
        raise RuntimeError(f"manager source hash mismatch: {digest}")
    spec = importlib.util.spec_from_file_location("virne_manager_oracle", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to create manager oracle spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_text(text: str, root: pathlib.Path) -> str:
    return text.replace(str(root), "<ROOT>")


def tree_snapshot(root: pathlib.Path) -> tuple[tuple[str, str, str], ...]:
    records: list[tuple[str, str, str]] = []

    def visit(directory: pathlib.Path) -> None:
        with os.scandir(directory) as entries:
            ordered = sorted(entries, key=lambda entry: os.fsencode(entry.name))
        for entry in ordered:
            path = pathlib.Path(entry.path)
            relative = path.relative_to(root).as_posix()
            if entry.is_symlink():
                target = normalize_text(os.readlink(path), root)
                records.append((relative, "symlink", target))
            elif entry.is_dir(follow_symlinks=False):
                records.append((relative, "dir", ""))
                visit(path)
            elif entry.is_file(follow_symlinks=False):
                records.append((relative, "file", path.read_bytes().hex()))
            else:
                records.append((relative, "other", ""))

    visit(root)
    return tuple(records)


def python_category(error: BaseException) -> str:
    if isinstance(error, TypeError):
        return "legacy_temp_join_type_error"
    if isinstance(error, FileNotFoundError):
        return "enumeration_failed"
    if isinstance(error, NotADirectoryError):
        return "not_directory"
    if isinstance(error, OSError):
        return "remove_failed"
    return type(error).__name__


def invoke_python(
    module: Any,
    case: Case,
    root: pathlib.Path,
    arguments: list[pathlib.Path],
) -> Result:
    output = io.StringIO()
    status = "ok"
    category = "ok"
    error_path = ""
    try:
        with contextlib.redirect_stdout(output):
            if case.operation == "delete_temp":
                module.delete_temp_files(str(arguments[0]))
            elif case.operation == "clean_save":
                module.clean_save_dir(str(arguments[0]))
            elif case.operation == "delete_empty":
                config = type(
                    "Config",
                    (),
                    {
                        "record_dir": str(arguments[0]),
                        "log_dir": str(arguments[1]),
                        "save_dir": str(arguments[2]),
                    },
                )()
                module.delete_empty_dir(config)
            else:
                raise AssertionError(case.operation)
    except BaseException as error:  # exact oracle outcome is data
        status = "error"
        category = python_category(error)
        filename = getattr(error, "filename", None)
        if filename is not None:
            error_path = normalize_text(os.fspath(filename), root)
    return Result(
        status=status,
        category=category,
        output=normalize_text(output.getvalue(), root),
        tree=tree_snapshot(root),
        error_path=error_path,
    )


def parse_harness_output(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in values:
            raise RuntimeError(f"malformed manager harness line: {line!r}")
        values[key] = value
    required = {
        "status",
        "code",
        "operation",
        "path_hex",
        "system_value",
        "what_hex",
        "output_hex",
    }
    if set(values) != required:
        raise RuntimeError(
            f"manager harness fields mismatch: {sorted(set(values) ^ required)}"
        )
    return values


def decode_hex(value: str) -> str:
    return bytes.fromhex(value).decode("utf-8", errors="surrogateescape")


def invoke_cpp(
    harness: pathlib.Path,
    sandbox: pathlib.Path,
    case: Case,
    root: pathlib.Path,
    arguments: list[pathlib.Path],
) -> Result:
    process = subprocess.run(
        [
            str(harness),
            str(sandbox),
            case.operation,
            *(str(argument) for argument in arguments),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="surrogateescape",
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"manager harness failed ({process.returncode}): {process.stderr!r}"
        )
    values = parse_harness_output(process.stdout)
    code = int(values["code"])
    operation = int(values["operation"])
    status = values["status"]
    return Result(
        status=status,
        category="ok" if status == "ok" else "manager_error",
        output=normalize_text(decode_hex(values["output_hex"]), root),
        tree=tree_snapshot(root),
        error_path=normalize_text(decode_hex(values["path_hex"]), root),
        cpp_code=None if code < 0 else code,
        cpp_operation=None if operation < 0 else operation,
    )


def make_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=False)


def write(path: pathlib.Path, data: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def one_path(relative: str, builder: Callable[[pathlib.Path], None]) -> Callable[[pathlib.Path], list[pathlib.Path]]:
    def setup(root: pathlib.Path) -> list[pathlib.Path]:
        target = root / relative
        builder(target)
        return [target]

    return setup


def delete_empty_setup(
    builder: Callable[[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path], None]
) -> Callable[[pathlib.Path], list[pathlib.Path]]:
    def setup(root: pathlib.Path) -> list[pathlib.Path]:
        record = root / "record"
        log = root / "log"
        save = root / "save"
        builder(root, record, log, save)
        return [record, log, save]

    return setup


def cases() -> list[Case]:
    def nothing(_: pathlib.Path) -> None:
        return None

    def directory(path: pathlib.Path) -> None:
        make_dir(path)

    def regular(path: pathlib.Path) -> None:
        write(path)

    def nonempty_temp(path: pathlib.Path) -> None:
        make_dir(path)
        write(path / "file_temp.bin", b"payload")

    def nonempty_other(path: pathlib.Path) -> None:
        make_dir(path)
        write(path / "ordinary.bin", b"payload")

    def clean_regular_only(path: pathlib.Path) -> None:
        make_dir(path)
        write(path / "ignored.txt")

    def clean_empty_algorithm(path: pathlib.Path) -> None:
        make_dir(path / "algorithm")

    def clean_missing_records(path: pathlib.Path) -> None:
        write(path / "algorithm" / "run" / "models" / "weights", b"w")

    def clean_empty_records(path: pathlib.Path) -> None:
        make_dir(path / "algorithm" / "run" / "records")

    def clean_nonempty_records(path: pathlib.Path) -> None:
        write(path / "algorithm" / "run" / "records" / ".keep")

    def clean_records_directory_entry(path: pathlib.Path) -> None:
        make_dir(path / "algorithm" / "run" / "records" / "nested")

    def clean_records_file(path: pathlib.Path) -> None:
        write(path / "algorithm" / "run" / "records", b"not a directory")

    def clean_run_file(path: pathlib.Path) -> None:
        write(path / "algorithm" / "run_file", b"not a run directory")

    def clean_mixed(path: pathlib.Path) -> None:
        make_dir(path / "algorithm" / "a_delete")
        write(path / "algorithm" / "b_keep" / "records" / "result", b"r")

    def clean_partial(path: pathlib.Path) -> None:
        make_dir(path / "algorithm" / "a_delete")
        write(path / "algorithm" / "b_fail" / "records", b"file")

    def final_symlink(path: pathlib.Path) -> None:
        make_dir(path / "algorithm")
        outside = path.parent / "outside"
        make_dir(outside)
        os.symlink(outside, path / "algorithm" / "run_link", target_is_directory=True)

    def escaped_algorithm(path: pathlib.Path) -> None:
        make_dir(path)
        outside_run = path.parent / "outside" / "run"
        make_dir(outside_run)
        os.symlink(
            outside_run.parent,
            path / "algorithm_link",
            target_is_directory=True,
        )

    def all_missing(_: pathlib.Path, __: pathlib.Path, ___: pathlib.Path, ____: pathlib.Path) -> None:
        return None

    def all_empty(_: pathlib.Path, record: pathlib.Path, log: pathlib.Path, save: pathlib.Path) -> None:
        make_dir(record)
        make_dir(log)
        make_dir(save)

    def all_nonempty(_: pathlib.Path, record: pathlib.Path, log: pathlib.Path, save: pathlib.Path) -> None:
        write(record / "keep")
        write(log / "keep")
        write(save / "keep")

    def file_second(_: pathlib.Path, record: pathlib.Path, log: pathlib.Path, save: pathlib.Path) -> None:
        make_dir(record)
        write(log)
        make_dir(save)

    def duplicate(root: pathlib.Path, record: pathlib.Path, _: pathlib.Path, __: pathlib.Path) -> None:
        make_dir(record)
        # All fixed fields deliberately point at the same dynamic path.
        del root

    def duplicate_setup(root: pathlib.Path) -> list[pathlib.Path]:
        path = root / "duplicate"
        make_dir(path)
        return [path, path, path]

    def nested(_: pathlib.Path, record: pathlib.Path, log: pathlib.Path, save: pathlib.Path) -> None:
        make_dir(record)
        make_dir(log)
        # record/log are children of save in the custom setup below.

    def nested_setup(root: pathlib.Path) -> list[pathlib.Path]:
        save = root / "save"
        record = save / "record"
        log = save / "log"
        make_dir(record)
        make_dir(log)
        return [record, log, save]

    def dangling_setup(root: pathlib.Path) -> list[pathlib.Path]:
        record = root / "dangling"
        os.symlink(root / "missing_target", record, target_is_directory=True)
        return [record, root / "missing_log", root / "missing_save"]

    def directory_symlink_setup(root: pathlib.Path) -> list[pathlib.Path]:
        target = root / "target"
        link = root / "link"
        make_dir(target)
        os.symlink(target, link, target_is_directory=True)
        return [link, root / "missing_log", root / "missing_save"]

    del duplicate, nested
    return [
        Case("temp_empty", "delete_temp", one_path("target", directory), "ok", None, None),
        Case("temp_nonempty_temp", "delete_temp", one_path("target", nonempty_temp), "legacy_temp_join_type_error", LEGACY_TEMP, LIST_TEMP_ROOT),
        Case("temp_nonempty_other", "delete_temp", one_path("target", nonempty_other), "legacy_temp_join_type_error", LEGACY_TEMP, LIST_TEMP_ROOT),
        Case("temp_missing", "delete_temp", one_path("target", nothing), "enumeration_failed", ENUMERATION_FAILED, LIST_TEMP_ROOT),
        Case("temp_regular", "delete_temp", one_path("target", regular), "not_directory", NOT_DIRECTORY, LIST_TEMP_ROOT),
        Case("clean_regular_only", "clean_save", one_path("save", clean_regular_only), "ok", None, None),
        Case("clean_empty_algorithm", "clean_save", one_path("save", clean_empty_algorithm), "ok", None, None),
        Case("clean_missing_records", "clean_save", one_path("save", clean_missing_records), "ok", None, None),
        Case("clean_empty_records", "clean_save", one_path("save", clean_empty_records), "ok", None, None),
        Case("clean_nonempty_records", "clean_save", one_path("save", clean_nonempty_records), "ok", None, None),
        Case("clean_records_subdir", "clean_save", one_path("save", clean_records_directory_entry), "ok", None, None),
        Case("clean_records_file", "clean_save", one_path("save", clean_records_file), "not_directory", NOT_DIRECTORY, LIST_RECORDS),
        Case("clean_run_file", "clean_save", one_path("save", clean_run_file), "not_directory", REMOVE_FAILED, REMOVE_RUN_TREE),
        Case("clean_mixed", "clean_save", one_path("save", clean_mixed), "ok", None, None),
        Case("clean_partial", "clean_save", one_path("save", clean_partial), "not_directory", NOT_DIRECTORY, LIST_RECORDS),
        Case("clean_final_run_symlink", "clean_save", one_path("save", final_symlink), "remove_failed", REMOVE_FAILED, REMOVE_RUN_TREE),
        Case("clean_algorithm_escape", "clean_save", one_path("save", escaped_algorithm), "ok", UNSAFE_PATH_ESCAPE, REMOVE_RUN_TREE, True),
        Case("empty_all_missing", "delete_empty", delete_empty_setup(all_missing), "ok", None, None),
        Case("empty_all_empty", "delete_empty", delete_empty_setup(all_empty), "ok", None, None),
        Case("empty_all_nonempty", "delete_empty", delete_empty_setup(all_nonempty), "ok", None, None),
        Case("empty_file_second", "delete_empty", delete_empty_setup(file_second), "not_directory", NOT_DIRECTORY, REMOVE_EMPTY_DIRECTORY),
        Case("empty_duplicate", "delete_empty", duplicate_setup, "ok", None, None),
        Case("empty_nested_cascade", "delete_empty", nested_setup, "ok", None, None),
        Case("empty_dangling_symlink", "delete_empty", dangling_setup, "ok", None, None),
        Case("empty_directory_symlink", "delete_empty", directory_symlink_setup, "not_directory", REMOVE_FAILED, REMOVE_EMPTY_DIRECTORY),
    ]


def validate_case(
    case: Case,
    python_result: Result,
    cpp_result: Result,
) -> None:
    if case.intentional_escape_deviation:
        if python_result.status != "ok":
            raise AssertionError(f"{case.name}: Python escape fixture did not reproduce")
        if cpp_result.status != "error":
            raise AssertionError(f"{case.name}: C++ did not reject escape")
        if cpp_result.cpp_code != UNSAFE_PATH_ESCAPE or cpp_result.cpp_operation != REMOVE_RUN_TREE:
            raise AssertionError(f"{case.name}: C++ escape classification mismatch")
        if any(record[0] == "outside/run" for record in python_result.tree):
            raise AssertionError(f"{case.name}: Python did not demonstrate unsafe deletion")
        if not any(record[0] == "outside/run" for record in cpp_result.tree):
            raise AssertionError(f"{case.name}: C++ changed outside target")
        if cpp_result.output:
            raise AssertionError(f"{case.name}: C++ emitted output before rejecting escape")
        return

    expected_status = "ok" if case.expected_python == "ok" else "error"
    if python_result.status != expected_status:
        raise AssertionError(
            f"{case.name}: Python status/category {python_result.status}/{python_result.category}"
        )
    if python_result.category != case.expected_python:
        raise AssertionError(
            f"{case.name}: Python category {python_result.category!r} != {case.expected_python!r}"
        )
    if cpp_result.status != expected_status:
        raise AssertionError(f"{case.name}: C++ status {cpp_result.status!r}")
    if cpp_result.cpp_code != case.expected_cpp_code:
        raise AssertionError(
            f"{case.name}: C++ code {cpp_result.cpp_code!r} != {case.expected_cpp_code!r}"
        )
    if cpp_result.cpp_operation != case.expected_cpp_operation:
        raise AssertionError(
            f"{case.name}: C++ operation {cpp_result.cpp_operation!r} != {case.expected_cpp_operation!r}"
        )
    if python_result.output != cpp_result.output:
        raise AssertionError(
            f"{case.name}: stdout mismatch {python_result.output!r} != {cpp_result.output!r}"
        )
    if python_result.tree != cpp_result.tree:
        raise AssertionError(
            f"{case.name}: tree mismatch\nPython={python_result.tree!r}\nC++={cpp_result.tree!r}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--python-source", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    harness = args.harness.resolve()
    if not harness.is_file():
        raise FileNotFoundError(f"manager harness not found: {harness}")
    source = args.python_source.resolve()
    module = load_oracle(source)

    sandbox = pathlib.Path(tempfile.mkdtemp(prefix="virne_manager_diff_")).resolve()
    temporary_root = pathlib.Path(tempfile.gettempdir()).resolve()
    if sandbox.parent != temporary_root or not sandbox.name.startswith("virne_manager_diff_"):
        raise RuntimeError(f"unsafe manager differential sandbox: {sandbox}")
    try:
        selected_cases = cases()
        for index, case in enumerate(selected_cases, 1):
            python_root = sandbox / f"case_{index:02d}" / "python"
            cpp_root = sandbox / f"case_{index:02d}" / "cpp"
            python_root.mkdir(parents=True)
            cpp_root.mkdir(parents=True)
            python_arguments = case.setup(python_root)
            cpp_arguments = case.setup(cpp_root)
            if tree_snapshot(python_root) != tree_snapshot(cpp_root):
                raise AssertionError(f"{case.name}: initial fixture mismatch")

            python_result = invoke_python(
                module, case, python_root, python_arguments
            )
            cpp_result = invoke_cpp(
                harness, sandbox, case, cpp_root, cpp_arguments
            )
            validate_case(case, python_result, cpp_result)
            label = "intentional-safe-deviation" if case.intentional_escape_deviation else "exact"
            print(f"case[{index:02d}/{len(selected_cases):02d}]={case.name}: PASS ({label})")
    finally:
        resolved = sandbox.resolve()
        if resolved.parent != temporary_root or not resolved.name.startswith("virne_manager_diff_"):
            raise RuntimeError(f"refusing unsafe manager differential cleanup: {resolved}")
        shutil.rmtree(resolved)

    print(f"source_sha256={SOURCE_SHA256.upper()}")
    print(f"exact_cases={len(cases()) - 1}")
    print("intentional_safe_deviations=1")
    print("manager differential: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
