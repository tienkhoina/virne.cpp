#!/usr/bin/env python3
"""AST-isolated differential for the pinned non-ML Python Logger leaf."""

from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import hashlib
import io
import json
import logging
import os
import pathlib
import subprocess
import tempfile
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union


SOURCE_SHA256 = "274189af7211aba134c4696b2da8190f743b2f59d4287cf56843510f0b2c7083"

BOUNDARIES = {
    "backend_shim":
        "The class is AST-isolated and receives Python stdlib logging, a narrowly equivalent ColoredFormatter, and an LF-pinned FileHandler matching the Docker oracle because colorlog is an eager import and host Windows otherwise translates log newlines.",
    "native_schema_lock":
        "Native Logger rejects metric-schema growth after the first CSV row; Python can silently misalign later dict keys. The typed safety deviation belongs to the unit gate.",
    "dynamic_protocols":
        "Python accepts arbitrary levels, mappings, values, handlers, and filesystem objects; native Logger uses fixed level/backend fields and pre-registered dense metric IDs.",
    "optional_sinks":
        "WandB, TensorBoard, Torch, RL/training integration, and network sinks are deliberately excluded; the native cold LoggerSink seam prepares those future modules.",
    "native_workers":
        "Python exposes scalar log calls only. Native log_batch workers 1/2/8 are compared with the same ordered Python scalar sequence.",
}


class OracleColoredFormatter(logging.Formatter):
    """The fixed colorlog surface used by the pinned Logger class."""

    COLORS = {
        "DEBUG": "\x1b[36m",
        "INFO": "\x1b[32m",
        "WARNING": "\x1b[33m",
        "ERROR": "\x1b[31m",
        "CRITICAL": "\x1b[1;31m",
    }

    def __init__(self, _format: str, *, log_colors: dict[str, str]):
        super().__init__("%(message)s")
        expected = {
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        }
        if log_colors != expected:
            raise RuntimeError("pinned color schema drift")

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS[record.levelname]
        return (
            f"{color}{record.levelname:<8}\x1b[0m "
            f"{record.getMessage()}\x1b[0m"
        )


class OracleFileHandler(logging.FileHandler):
    """FileHandler with the pinned Docker/Linux LF stream contract."""

    def _open(self):
        return open(
            self.baseFilename,
            self.mode,
            encoding=self.encoding,
            errors=self.errors,
            newline="\n",
        )


FAKE_COLORLOG = SimpleNamespace(ColoredFormatter=OracleColoredFormatter)
FAKE_LOGGING = SimpleNamespace(
    FileHandler=OracleFileHandler,
    StreamHandler=logging.StreamHandler,
    getLogger=logging.getLogger,
)


def load_logger(source: pathlib.Path):
    source = source.resolve()
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"Logger source hash drift: {actual}")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    matches = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Logger"
    ]
    if len(matches) != 1:
        raise RuntimeError("Logger class inventory drift")
    isolated = ast.fix_missing_locations(
        ast.Module(body=matches, type_ignores=[]))
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "Any": Any,
        "Dict": Dict,
        "DictConfig": Any,
        "List": List,
        "Optional": Optional,
        "Union": Union,
        "colorlog": FAKE_COLORLOG,
        "csv": csv,
        "logging": FAKE_LOGGING,
        "os": os,
    }
    exec(compile(isolated, str(source), "exec"), namespace)
    return namespace["Logger"]


def make_config(
        root: pathlib.Path,
        case_name: str,
        *,
        backends=("console", "file"),
        level="DEBUG",
        interval=1):
    return SimpleNamespace(
        logger=SimpleNamespace(
            backends=list(backends),
            level=level,
            log_file_name="run.log",
            project_name="",
            experiment_name="",
            log_dir_name="logs",
            log_show_interval=interval,
        ),
        experiment=SimpleNamespace(save_root_dir=str(root), run_id="run"),
        solver=SimpleNamespace(solver_name=case_name),
    )


def read_binary_if_present(path: pathlib.Path) -> bytes:
    return path.read_bytes() if path.exists() else b""


def payload(stderr: bytes, log_file: bytes, metric_file: bytes, error: str):
    return (
        f"stderr={stderr.hex()};"
        f"file={log_file.hex()};"
        f"csv={metric_file.hex()};"
        f"error={error}"
    )


def observe(Logger, root, case_name, config, action, expected_error="none"):
    stream = io.StringIO()
    logger = None
    error = "none"
    with contextlib.redirect_stderr(stream):
        try:
            logger = Logger(config)
            action(logger)
        except TypeError:
            if expected_error != "invalid_step":
                raise
            error = "invalid_step"
        except ZeroDivisionError:
            if expected_error != "invalid_interval":
                raise
            error = "invalid_interval"
        finally:
            if logger is not None:
                logger.close()
    if error != expected_error:
        raise RuntimeError(
            f"Python case {case_name!r}: expected error "
            f"{expected_error!r}, got {error!r}")

    directory = root / case_name / "run" / "logs"
    return payload(
        stream.getvalue().encode("utf-8"),
        read_binary_if_present(directory / "run.log"),
        read_binary_if_present(directory / "training_info.csv"),
        error,
    )


def batch_action(logger):
    entries = [
        ("batch-a", "INFO", {
            "loss_total": 1.0, "accuracy": 0.25, "value_head": -2.0}, 1),
        ("", "WARNING", {
            "loss_total": 1.5, "accuracy": 0.5, "value_head": -1.0}, 2),
        ("filtered-debug", "DEBUG", {
            "loss_total": 2.0, "accuracy": 0.75, "value_head": 0.0}, 0),
        ("batch-d", "ERROR", None, None),
    ]
    for message, level, data, step in entries:
        logger.log(message, level=level, data=data, step=step)


def python_cases(Logger, root: pathlib.Path):
    cases = {}

    name = "ansi_console_file"
    cases[name] = observe(
        Logger, root, name, make_config(root, name),
        lambda logger: (
            logger.debug("debug"),
            logger.info("info"),
            logger.warning("warning"),
            logger.error("error"),
            logger.critical("critical"),
        ))

    name = "level_filter"
    cases[name] = observe(
        Logger, root, name, make_config(root, name, level="WARNING"),
        lambda logger: (
            logger.debug("drop-debug"),
            logger.info("drop-info"),
            logger.warning("keep-warning"),
            logger.error("keep-error"),
            logger.critical("keep-critical"),
        ))

    name = "message_csv_progress"
    def message_csv_progress(logger):
        logger.log(
            'epoch "one", ok',
            level="INFO",
            data={
                'loss,"quoted"': 1.25,
                "accuracy": None,
                "return_value": -0.5,
            },
            step=2,
        )
        logger.log(
            "",
            level="INFO",
            data={
                'loss,"quoted"': 2.0,
                "accuracy": 0.75,
                "return_value": 4.25,
            },
            step=3,
        )
    cases[name] = observe(
        Logger, root, name, make_config(root, name, interval=2),
        message_csv_progress)

    name = "partial_missing_step"
    cases[name] = observe(
        Logger, root, name, make_config(root, name, interval=2),
        lambda logger: logger.log(
            "before missing step",
            level="INFO",
            data={"loss": 0.5},
            step=None,
        ),
        expected_error="invalid_step",
    )

    name = "partial_zero_interval"
    cases[name] = observe(
        Logger, root, name, make_config(root, name, interval=0),
        lambda logger: logger.log(
            "before zero interval",
            level="WARNING",
            data={"loss": -0.25},
            step=1,
        ),
        expected_error="invalid_interval",
    )

    name = "step_zero_with_zero_interval"
    cases[name] = observe(
        Logger, root, name, make_config(root, name, interval=0),
        lambda logger: logger.log(
            "zero step",
            level="INFO",
            data={"loss": 3.0},
            step=0,
        ),
    )

    for workers in (1, 2, 8):
        name = f"batch_workers_{workers}"
        cases[name] = observe(
            Logger,
            root,
            name,
            make_config(
                root,
                name,
                backends=("file",),
                level="INFO",
                interval=2,
            ),
            batch_action,
        )
    return cases


def cpp_cases(harness: pathlib.Path, root: pathlib.Path):
    process = subprocess.run(
        [str(harness), "differential", str(root)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"Logger harness failed: {process.stderr.strip()}")
    result = {}
    for line in process.stdout.splitlines():
        name, value = line.split("\t", 1)
        if name in result:
            raise RuntimeError(f"duplicate native case: {name}")
        result[name] = value
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    Logger = load_logger(args.source)
    with tempfile.TemporaryDirectory(prefix="virne_logger_diff_") as temporary:
        temporary_root = pathlib.Path(temporary)
        expected = python_cases(Logger, temporary_root / "python")
        actual = cpp_cases(args.harness, temporary_root / "cpp")

    if actual.keys() != expected.keys():
        raise RuntimeError(
            f"case inventory drift: Python={list(expected)}, C++={list(actual)}")

    worker_cases = [actual[f"batch_workers_{value}"] for value in (1, 2, 8)]
    if len(set(worker_cases)) != 1:
        raise RuntimeError("native Logger worker output is not deterministic")

    mismatches = {
        name: {"python": expected[name], "cpp": actual[name]}
        for name in expected
        if expected[name] != actual[name]
    }
    if mismatches:
        raise RuntimeError(
            "Logger differential mismatch:\n" +
            json.dumps(mismatches, indent=2, sort_keys=True))

    report = {
        "component": "core.Logger",
        "source_sha256": SOURCE_SHA256.upper(),
        "shared_case_count": len(expected),
        "native_workers": [1, 2, 8],
        "python_only_boundaries": BOUNDARIES,
        "status": "PASS",
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
