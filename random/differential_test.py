#!/usr/bin/env python3
"""Differential test against local CPython and NumPy 1.26.4 oracles."""

from __future__ import annotations

import argparse
import random
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np


PYTHON_SEEDS = [0, 1, 2, 42, 999, 123456789, 2**32 - 1, 2**63, 2**64 - 1]
NUMPY_SEEDS = [0, 1, 2, 42, 999, 123456789, 2**31, 2**32 - 1]


def float_bits(value: float) -> str:
    return f"{struct.unpack('<Q', struct.pack('<d', float(value)))[0]:016x}"


def values(items) -> str:
    return ",".join(str(int(item)) for item in items)


def doubles(items) -> str:
    return ",".join(float_bits(item) for item in items)


def parse_cpp(executable: Path, mode: str, seed: int) -> dict[str, str]:
    completed = subprocess.run(
        [str(executable), mode, str(seed)],
        check=True,
        capture_output=True,
        text=True,
    )
    result: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, encoded = line.split("\t", 1)
        result[key] = encoded
    return result


def python_oracle(seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    result = {"random": doubles(rng.random() for _ in range(8))}
    result["randint"] = values(rng.randint(-1000, 1000) for _ in range(20))
    result["randrange_positive"] = values(
        rng.randrange(-500, 701, 13) for _ in range(15)
    )
    result["randrange_negative"] = values(
        rng.randrange(700, -501, -17) for _ in range(15)
    )
    population = list(range(11))
    result["choices_uniform"] = values(rng.choices(population, k=20))
    result["choices_weighted"] = values(
        rng.choices(
            population,
            weights=[1, 3, 2, 7, 4, 9, 5, 6, 8, 10, 11],
            k=20,
        )
    )
    result["choices_cum_weights"] = values(
        rng.choices(
            population,
            cum_weights=[1, 4, 6, 13, 17, 26, 31, 37, 45, 55, 66],
            k=20,
        )
    )
    shuffled = list(range(20))
    rng.shuffle(shuffled)
    result["shuffle"] = values(shuffled)
    try:
        rng.choices([], k=5)
    except IndexError:
        result["choices_empty_error"] = "1"
    else:
        result["choices_empty_error"] = "0"
    result["next"] = doubles([rng.random()])
    return result


def numpy_oracle(seed: int) -> dict[str, str]:
    rng = np.random.RandomState(seed)
    result = {"random": doubles(rng.random_sample(8))}
    result["uniform"] = doubles(rng.uniform(-17.25, 93.5, 10))
    result["randint"] = values(rng.randint(-200, 301, 20, dtype=np.int64))

    random_shape = rng.random_sample((2, 3, 2))
    result["random_shape_shape"] = values(random_shape.shape)
    result["random_shape"] = doubles(random_shape.ravel(order="C"))

    result["rand_scalar"] = doubles([rng.rand()])
    rand_dimensions = rng.rand(2, 3, 2)
    result["rand_dimensions_shape"] = values(rand_dimensions.shape)
    result["rand_dimensions"] = doubles(rand_dimensions.ravel(order="C"))

    result["uniform_shape"] = doubles(
        rng.uniform(-4.5, 7.25, (2, 2, 3)).ravel(order="C")
    )
    result["randint_shape"] = values(
        rng.randint(-37, 52, (2, 3, 2), dtype=np.int64).ravel(order="C")
    )
    result["normal_shape"] = doubles(
        rng.normal(1.25, 0.75, (2, 3)).ravel(order="C")
    )
    result["exponential_shape"] = doubles(
        rng.exponential(2.5, (2, 2, 2)).ravel(order="C")
    )
    result["poisson_shape"] = values(
        rng.poisson(12.5, (2, 3)).ravel(order="C")
    )

    population = np.arange(12)
    result["choice_uniform"] = values(rng.choice(population, 15))
    probabilities = np.array(
        [0.02, 0.03, 0.05, 0.07, 0.08, 0.10,
         0.11, 0.12, 0.10, 0.09, 0.13, 0.10],
        dtype=np.float64,
    )
    result["choice_weighted"] = values(
        rng.choice(population, 15, p=probabilities)
    )
    result["choice_uniform_no_replace"] = values(
        rng.choice(population, 6, replace=False)
    )
    no_replace_probabilities = np.arange(1, 13, dtype=np.float64) / 78.0
    result["choice_weighted_no_replace"] = values(
        rng.choice(
            population,
            6,
            replace=False,
            p=no_replace_probabilities,
        )
    )
    result["choice_integer_scalar"] = values([rng.choice(12)])
    result["choice_integer_uniform"] = values(rng.choice(12, 15))
    result["choice_integer_weighted"] = values(
        rng.choice(12, 15, p=probabilities)
    )
    result["choice_integer_uniform_no_replace"] = values(
        rng.choice(12, 6, replace=False)
    )
    result["choice_integer_weighted_no_replace"] = values(
        rng.choice(
            12,
            6,
            replace=False,
            p=no_replace_probabilities,
        )
    )
    result["normal"] = doubles(rng.normal(-3.0, 2.25, 11))
    result["exponential"] = doubles(rng.exponential(1.75, 9))
    result["poisson_small"] = values(rng.poisson(0.75, 10))
    result["poisson_threshold"] = values(rng.poisson(10.0, 10))
    result["poisson_large"] = values(rng.poisson(250.0, 10))
    shuffled = np.arange(20)
    rng.shuffle(shuffled)
    result["shuffle"] = values(shuffled)

    matrix = np.arange(21, dtype=np.int64).reshape(7, 3)
    matrix_permutation = rng.permutation(matrix)
    result["matrix_permutation_shape"] = values(matrix_permutation.shape)
    result["matrix_permutation"] = values(
        matrix_permutation.ravel(order="C")
    )

    empty_matrix = np.empty((3, 0), dtype=np.int64)
    rng.shuffle(empty_matrix)
    result["empty_shuffle_shape"] = values(empty_matrix.shape)
    result["empty_shuffle"] = values(empty_matrix.ravel(order="C"))
    empty_permutation = rng.permutation(empty_matrix)
    result["empty_permutation_shape"] = values(empty_permutation.shape)
    result["empty_permutation"] = values(
        empty_permutation.ravel(order="C")
    )
    result["next"] = doubles([rng.random_sample()])
    return result


def numpy_large_oracle(seed: int) -> dict[str, str]:
    rng = np.random.RandomState(seed)
    return {
        "randint_large": values(
            rng.randint(-17, 1_000_003, 262_144, dtype=np.int64)
        ),
        "next": doubles([rng.random_sample()]),
    }


def check_mode(executable: Path, mode: str, seeds: list[int], oracle) -> int:
    checked = 0
    for seed in seeds:
        expected = oracle(seed)
        actual = parse_cpp(executable, mode, seed)
        if actual != expected:
            keys = sorted(set(actual) | set(expected))
            mismatches = [key for key in keys if actual.get(key) != expected.get(key)]
            details = "\n".join(
                f"  {key}:\n    C++={actual.get(key)}\n    oracle={expected.get(key)}"
                for key in mismatches
            )
            raise AssertionError(f"{mode} seed {seed} mismatch:\n{details}")
        checked += sum(0 if not encoded else encoded.count(",") + 1 for encoded in actual.values())
    return checked


def main() -> None:
    parser = argparse.ArgumentParser()
    default_cpp = (
        Path(__file__).resolve().parent.parent
        / "build"
        / "random"
        / "random_differential_harness"
    )
    parser.add_argument("--cpp", type=Path, default=default_cpp)
    args = parser.parse_args()

    if np.__version__ != "1.26.4":
        raise RuntimeError(
            f"differential oracle requires NumPy 1.26.4, found {np.__version__}"
        )
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(
            "differential oracle requires CPython 3.10, found "
            f"{sys.version_info.major}.{sys.version_info.minor}"
        )
    if not args.cpp.is_file():
        raise FileNotFoundError(
            f"build random_differential_harness first; not found: {args.cpp}"
        )

    python_values = check_mode(args.cpp, "python", PYTHON_SEEDS, python_oracle)
    numpy_values = check_mode(args.cpp, "numpy", NUMPY_SEEDS, numpy_oracle)
    numpy_large_values = check_mode(
        args.cpp,
        "numpy-large",
        [42],
        numpy_large_oracle,
    )
    print(
        "differential_test: PASS "
        f"({len(PYTHON_SEEDS)} CPython seeds/{python_values} values, "
        f"{len(NUMPY_SEEDS)} NumPy seeds/{numpy_values} values, "
        f"large randint/{numpy_large_values} values)"
    )


if __name__ == "__main__":
    main()
