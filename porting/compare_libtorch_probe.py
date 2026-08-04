#!/usr/bin/env python3
"""Bit-exact CPU output gate for the vendored LibTorch probe."""

from __future__ import annotations

import argparse
import json
import pathlib
import struct
import subprocess
from typing import Any

import torch


FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
CHECK_FIELDS = ("shape", "dtype", "checksum", "output_bytes", "sum_bits")


class Digest:
    def __init__(self) -> None:
        self.value = FNV_OFFSET
        self.bytes = 0

    def append_byte(self, value: int) -> None:
        self.value ^= value & 0xFF
        self.value = (self.value * FNV_PRIME) & ((1 << 64) - 1)
        self.bytes += 1

    def append_double(self, value: float) -> None:
        for byte in struct.pack("<d", float(value)):
            self.append_byte(byte)


def python_output(threads: int, device_name: str) -> dict[str, Any]:
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(threads)
    torch.manual_seed(0)
    cuda_available = bool(torch.cuda.is_available())
    if device_name == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but is unavailable")
    selected = (
        "cuda" if device_name == "cuda" or
        (device_name == "auto" and cuda_available) else "cpu"
    )
    device = torch.device(selected)
    values = torch.arange(
        0.0, 12.0, dtype=torch.float64, device=device).reshape(3, 4)
    output = (values @ values.transpose(0, 1)).to("cpu").contiguous()
    digest = Digest()
    for value in output.reshape(-1).tolist():
        digest.append_double(value)
    sum_bits = struct.unpack("<Q", struct.pack("<d", float(output.sum())))[0]
    return {
        "libtorch_version": torch.__version__,
        "device": selected,
        "cuda_available": cuda_available,
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "shape": list(output.shape),
        "dtype": str(output.dtype).removeprefix("torch."),
        "checksum": str(digest.value),
        "output_bytes": digest.bytes,
        "sum_bits": format(sum_bits, "x"),
    }


def native_output(executable: pathlib.Path, threads: int, device: str) -> dict:
    process = subprocess.run(
        [str(executable), "--threads", str(threads), "--device", device],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"LibTorch probe failed ({process.returncode}): "
            f"{process.stderr.strip()}"
        )
    lines = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"expected one JSON line, got {lines!r}")
    return json.loads(lines[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=pathlib.Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")

    expected = python_output(args.threads, args.device)
    actual = native_output(args.native, args.threads, args.device)
    mismatches = {
        field: {"python": expected[field], "cpp": actual.get(field)}
        for field in CHECK_FIELDS
        if expected[field] != actual.get(field)
    }
    report = {
        "component": "libtorch_probe",
        "python_torch": torch.__version__,
        "cpp_libtorch": actual.get("libtorch_version"),
        "threads": args.threads,
        "device": args.device,
        "checked_fields": list(CHECK_FIELDS),
        "python": expected,
        "cpp": actual,
        "status": "PASS" if not mismatches else "FAIL",
    }
    if mismatches:
        report["mismatches"] = mismatches
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
