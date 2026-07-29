#!/usr/bin/env python3
"""Lock Python-only ``-O`` assertion boundaries for dataset RNG."""

from __future__ import annotations

import argparse
import json
import pathlib

import compare_dataset_core
import compare_dataset_rng


def same_state(left, right) -> bool:
    return (
        left[0] == right[0]
        and (left[1] == right[1]).all()
        and left[2:] == right[2:]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-source", type=pathlib.Path, required=True)
    parser.add_argument("--json-output", type=pathlib.Path)
    args = parser.parse_args()
    if __debug__:
        raise RuntimeError("this boundary check must run with python -O")

    module = compare_dataset_core.load_oracle(args.python_source)
    np = module.np

    np.random.seed(99)
    untouched = np.random.get_state()
    try:
        module.generate_data_with_distribution(1, "customized", "float")
    except NotImplementedError as error:
        expected = (
            "Generating float data following the customized distribution "
            "is unsupporrted!"
        )
        if str(error) != expected:
            raise RuntimeError(f"optimized discriminator error drift: {error}") from error
    else:
        raise RuntimeError("optimized discriminator boundary did not fail")
    if not same_state(untouched, np.random.get_state()):
        raise RuntimeError("optimized discriminator error consumed RNG state")

    np.random.seed(99)
    try:
        module.generate_data_with_distribution(1, "normal", "bad")
    except TypeError as error:
        if "data type 'bad' not understood" not in str(error):
            raise RuntimeError(f"optimized final-cast error drift: {error}") from error
    else:
        raise RuntimeError("optimized final-cast boundary did not fail")
    random_value = float(np.random.random_sample())
    normal_value = float(np.random.normal())

    reference = np.random.RandomState(99)
    reference.normal(0.0, 1.0, 1)
    reference_random = float(reference.random_sample())
    reference_normal = float(reference.normal())
    continuation = {
        "random_bits": compare_dataset_rng.float_bits(random_value),
        "normal_bits": compare_dataset_rng.float_bits(normal_value),
    }
    expected_continuation = {
        "random_bits": compare_dataset_rng.float_bits(reference_random),
        "normal_bits": compare_dataset_rng.float_bits(reference_normal),
    }
    if continuation != expected_continuation:
        raise RuntimeError(
            f"optimized final-cast continuation drift: {continuation!r}"
        )

    payload = {
        "cases": 2,
        "final_cast_continuation": continuation,
        "numpy_version": np.__version__,
        "python_optimized": True,
        "source_sha256": compare_dataset_core.SOURCE_SHA256,
        "status": "PASS",
    }
    if args.json_output:
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print("dataset RNG python -O boundary: PASS (2/2 cases)")
    print("customized rejects before draw; invalid dtype rejects after draw")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
