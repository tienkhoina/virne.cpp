#!/usr/bin/env python3
"""Exact AST-isolated differential for the non-ML virne.utils.config leaf."""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
import os
import pathlib
import random as random_module
import subprocess
import sys
from types import ModuleType
from typing import Any


CONFIG_SOURCE_SHA256 = (
    "ad1d32e6b2db842c958240301535d356919dbd822e403d6a767ce2ed2bbfe787"
)
DATASET_SOURCE_SHA256 = (
    "269650ebcc373d7bdf79fa17346bd6f847973f17e60c1ac9bcae7cfd97bf936f"
)
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1

PYTHON_ONLY_BOUNDARIES = {
    "generic_mutation": (
        "Python mutates DictConfig fields incrementally and retains earlier writes "
        "on failure; native code returns fixed typed state for direct ownership."
    ),
    "dynamic_mapping_protocols": (
        "Python accepts DictConfig/dict subclasses and arbitrary key/value objects; "
        "native YAML is a cold compatibility boundary only."
    ),
    "unknown_attribute_types": (
        "Python treats unknown type strings as non-status and may match custom "
        "membership containers; native decoding rejects values outside AttributeKind."
    ),
    "plain_dict_identity": (
        "Python returns the exact plain dict object; YAML::Node preserves shared node "
        "state but does not claim Python object identity."
    ),
    "config_scalar_coercion": (
        "Python os.path.join rejects non-path components; the native Config adapter "
        "requires callers to supply its documented typed string schema."
    ),
}


class DotDict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class DictConfig(DotDict):
    pass


def deep_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: deep_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [deep_plain(item) for item in value]
    return value


class OmegaConf:
    @staticmethod
    def create() -> DictConfig:
        return DictConfig()

    @staticmethod
    def to_container(config: DictConfig, resolve: bool = False) -> dict:
        if resolve is not True:
            raise RuntimeError("resolve_config_to_dict did not request resolution")
        return deep_plain(config)


@contextmanager
def open_dict(config: DictConfig):
    yield config


def source_functions(
    source: pathlib.Path,
    expected_sha256: str,
    names: set[str],
    namespace: dict[str, Any],
) -> dict[str, Any]:
    raw = source.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"source hash drift for {source}: {actual_sha256}"
        )
    tree = ast.parse(raw.decode("utf-8"), filename=str(source))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    inventory = {node.name for node in selected}
    if inventory != names:
        raise RuntimeError(
            f"function inventory drift for {source}: {sorted(inventory)}"
        )
    isolated = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *selected,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(isolated)
    exec(compile(isolated, str(source), "exec"), namespace)
    return {name: namespace[name] for name in names}


def load_oracle(config_source: pathlib.Path, dataset_source: pathlib.Path):
    dataset = source_functions(
        dataset_source,
        DATASET_SOURCE_SHA256,
        {
            "get_distribution_parameters",
            "get_parameters_string",
            "get_p_net_dataset_dir_from_setting",
            "get_v_nets_dataset_dir_from_setting",
        },
        {"__name__": "_virne_dataset_oracle", "os": os},
    )
    config_namespace: dict[str, Any] = {
        "__name__": "_virne_utils_config_oracle",
        "Any": Any,
        "Dict": dict,
        "Union": object,
        "DictConfig": DictConfig,
        "OmegaConf": OmegaConf,
        "open_dict": open_dict,
        "os": os,
        "get_p_net_dataset_dir_from_setting":
            dataset["get_p_net_dataset_dir_from_setting"],
        "get_v_nets_dataset_dir_from_setting":
            dataset["get_v_nets_dataset_dir_from_setting"],
    }
    return source_functions(
        config_source,
        CONFIG_SOURCE_SHA256,
        {
            "generate_run_id",
            "resolve_config_to_dict",
            "add_simulation_into_config",
            "get_run_id_dir",
        },
        config_namespace,
    )


def attribute(name: str, kind: str, **distribution: Any) -> DotDict:
    result = DotDict(name=name, type=kind)
    result.update(distribution)
    return result


def simulation_config(variant: int) -> DictConfig:
    physical_nodes = [
        attribute("cpu", "resource", distribution="uniform", low=1, high=9),
        attribute("max_cpu", "extrema"),
        attribute("tag", "status"),
    ]
    physical_links = [
        attribute("bw", "resource", distribution="uniform", low=2, high=10),
        attribute("max_bw", "extrema"),
    ]
    virtual_nodes = [
        attribute("cpu", "resource", distribution="uniform", low=0, high=4),
        attribute("status", "status"),
        attribute("pos", "position"),
        attribute("gpu", "resource", distribution="uniform", low=1, high=3),
    ]
    virtual_links = [
        attribute("bw", "resource", distribution="uniform", low=1, high=8),
        attribute("status", "status"),
        attribute("ltc", "latency"),
    ]
    return DictConfig(
        p_net_setting=DotDict(
            output=DotDict(save_dir=f"oracle/p{variant}"),
            topology=DotDict(
                file_path="",
                num_nodes=6 + variant,
                type="waxman",
                wm_alpha=0.5,
                wm_beta=0.2,
            ),
            node_attrs_setting=physical_nodes,
            link_attrs_setting=physical_links,
        ),
        v_sim_setting=DotDict(
            output=DotDict(save_dir=f"oracle/v{variant}"),
            num_v_nets=12 + variant,
            v_net_size=DotDict(low=2, high=5),
            topology=DotDict(type="random"),
            lifetime=DotDict(distribution="exponential", scale=50),
            arrival_rate=DotDict(lam=0.25),
            node_attrs_setting=virtual_nodes,
            link_attrs_setting=virtual_links,
        ),
        experiment=DotDict(seed=7 + variant),
        rl=DotDict(
            feature_constructor=DotDict(
                extracted_attr_types=["resource", "extrema", "latency"]
            )
        ),
    )


def hash_byte(checksum: int, value: int) -> int:
    return ((checksum ^ value) * FNV_PRIME) & MASK64


def hash_u64(checksum: int, value: int) -> int:
    value &= MASK64
    for _ in range(8):
        checksum = hash_byte(checksum, value & 0xFF)
        value >>= 8
    return checksum


def hash_text(checksum: int, value: str) -> int:
    raw = value.encode("utf-8")
    checksum = hash_u64(checksum, len(raw))
    for byte in raw:
        checksum = hash_byte(checksum, byte)
    return checksum


def summary_checksum(summary: dict[str, Any]) -> int:
    checksum = FNV_OFFSET
    checksum = hash_text(checksum, summary["p_net_dataset_dir"])
    checksum = hash_text(checksum, summary["v_nets_dataset_dir"])
    fields = [
        "p_net_setting_num_nodes",
        "p_net_setting_num_node_attrs",
        "p_net_setting_num_link_attrs",
        "p_net_setting_num_node_resource_attrs",
        "p_net_setting_num_link_resource_attrs",
        "p_net_setting_num_node_extrema_attrs",
        "p_net_setting_num_link_extrema_attrs",
        "v_sim_setting_num_node_attrs",
        "v_sim_setting_num_link_attrs",
        "v_sim_setting_num_node_resource_attrs",
        "v_sim_setting_num_link_resource_attrs",
        "v_sim_setting_num_node_non_status_attrs",
        "v_sim_setting_num_link_non_status_attrs",
    ]
    for field in fields:
        checksum = hash_u64(checksum, summary[field])
    feature = summary["feature_constructor"]
    for field in [
        "num_extracted_p_node_attrs",
        "num_extracted_p_link_attrs",
        "num_extracted_v_node_attrs",
        "num_extracted_v_link_attrs",
        "p_num_nodes",
    ]:
        checksum = hash_u64(checksum, feature[field])
    return checksum


def serialize_summary(config: DictConfig, variant: int) -> dict[str, Any]:
    simulation = config.simulation
    feature = config.rl.feature_constructor
    result = {
        "variant": variant,
        "p_net_dataset_dir": simulation.p_net_dataset_dir,
        "v_nets_dataset_dir": simulation.v_nets_dataset_dir,
        "p_net_setting_num_nodes": simulation.p_net_setting_num_nodes,
        "p_net_setting_num_node_attrs": simulation.p_net_setting_num_node_attrs,
        "p_net_setting_num_link_attrs": simulation.p_net_setting_num_link_attrs,
        "p_net_setting_num_node_resource_attrs":
            simulation.p_net_setting_num_node_resource_attrs,
        "p_net_setting_num_link_resource_attrs":
            simulation.p_net_setting_num_link_resource_attrs,
        "p_net_setting_num_node_extrema_attrs":
            simulation.p_net_setting_num_node_extrema_attrs,
        "p_net_setting_num_link_extrema_attrs":
            simulation.p_net_setting_num_link_extrema_attrs,
        "v_sim_setting_num_node_attrs": simulation.v_sim_setting_num_node_attrs,
        "v_sim_setting_num_link_attrs": simulation.v_sim_setting_num_link_attrs,
        "v_sim_setting_num_node_resource_attrs":
            simulation.v_sim_setting_num_node_resource_attrs,
        "v_sim_setting_num_link_resource_attrs":
            simulation.v_sim_setting_num_link_resource_attrs,
        "v_sim_setting_num_node_non_status_attrs":
            simulation.v_sim_setting_num_node_non_status_attrs,
        "v_sim_setting_num_link_non_status_attrs":
            simulation.v_sim_setting_num_link_non_status_attrs,
        "feature_constructor": {
            "num_extracted_p_node_attrs": feature.num_extracted_p_node_attrs,
            "num_extracted_p_link_attrs": feature.num_extracted_p_link_attrs,
            "num_extracted_v_node_attrs": feature.num_extracted_v_node_attrs,
            "num_extracted_v_link_attrs": feature.num_extracted_v_link_attrs,
            "p_num_nodes": feature.p_num_nodes,
        },
    }
    result["checksum"] = summary_checksum(result)
    return result


def fixed_run_id(generate_run_id) -> dict[str, Any]:
    stream = random_module.Random(42)
    fake_time = ModuleType("time")
    fake_socket = ModuleType("socket")
    fake_random = ModuleType("random")

    def strftime(format_text: str) -> str:
        if format_text != "%Y%m%dT%H%M%S":
            raise RuntimeError("generate_run_id time format drift")
        return "20240229T030405"

    fake_time.strftime = strftime
    fake_socket.gethostname = lambda: "worker-a"
    fake_random.randint = stream.randint
    replacements = {
        "time": fake_time,
        "socket": fake_socket,
        "random": fake_random,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    try:
        sys.modules.update(replacements)
        value = generate_run_id()
    finally:
        for name, prior in previous.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior
    return {"value": value, "next_randint_0_9999": stream.randint(0, 9999)}


def run_directory_case(get_run_id_dir, root: str, solver: str, run_id: str):
    config = DictConfig(
        experiment=DotDict(save_root_dir=root, run_id=run_id),
        solver=DotDict(solver_name=solver),
    )
    path = get_run_id_dir(config)
    return {
        "save_root_dir": root,
        "solver_name": solver,
        "run_id": run_id,
        "path": path,
    }


def expected_payload(oracle: dict[str, Any]) -> dict[str, Any]:
    summaries = []
    for variant in range(3):
        config = simulation_config(variant)
        result = oracle["add_simulation_into_config"](config)
        if result is not None:
            raise RuntimeError("add_simulation_into_config return drift")
        summaries.append(serialize_summary(config, variant))

    order = [summary["checksum"] for summary in summaries]
    batch_checksum = hash_u64(FNV_OFFSET, len(summaries))
    for checksum in order:
        batch_checksum = hash_u64(batch_checksum, checksum)

    resolved_input = DictConfig(
        resolved_copy=17,
        resolved_text="value-17",
    )
    resolved = oracle["resolve_config_to_dict"](resolved_input)
    raw = {"value": 3}
    raw_result = oracle["resolve_config_to_dict"](raw)
    if raw_result is not raw:
        raise RuntimeError("plain dictionary identity drift")
    raw_result["value"] = 8
    invalid_rejected = False
    try:
        oracle["resolve_config_to_dict"]([1, 2])
    except ValueError:
        invalid_rejected = True

    return {
        "version": 1,
        "mode": "differential",
        "run_id": fixed_run_id(oracle["generate_run_id"]),
        "summaries": summaries,
        "batch": [
            {"workers": workers, "checksum": batch_checksum, "order": order}
            for workers in (0, 1, 2, 8)
        ],
        "run_directory": {
            "direct": run_directory_case(
                oracle["get_run_id_dir"],
                "direct-root",
                "direct-solver",
                "direct-run",
            ),
            "config": run_directory_case(
                oracle["get_run_id_dir"],
                "oracle-runs",
                "typed-solver",
                "config-run-17",
            ),
        },
        "config_resolution": {
            "resolved_copy": resolved["resolved_copy"],
            "resolved_text": resolved["resolved_text"],
            "raw_alias_value": raw["value"],
            "invalid_mapping_rejected": invalid_rejected,
        },
        "status": "PASS",
    }


def run_harness(path: pathlib.Path) -> dict[str, Any]:
    process = subprocess.run(
        [str(path), "differential"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"utils-config harness failed: {process.stderr.strip()}"
        )
    try:
        return json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"malformed utils-config harness JSON: {process.stdout!r}"
        ) from error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-source", type=pathlib.Path, required=True)
    parser.add_argument("--dataset-source", type=pathlib.Path, required=True)
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    oracle = load_oracle(args.config_source, args.dataset_source)
    expected = expected_payload(oracle)
    actual = run_harness(args.harness.resolve())
    if actual != expected:
        raise RuntimeError(
            "utils-config differential mismatch:\n"
            f"C++={json.dumps(actual, indent=2, sort_keys=True)}\n"
            f"Python={json.dumps(expected, indent=2, sort_keys=True)}"
        )

    shared_cases = [
        "generate_run_id",
        "run_id_rng_continuation",
        "simulation_variant_0",
        "simulation_variant_1",
        "simulation_variant_2",
        "run_directory_direct",
        "run_directory_config_adapter",
        "resolve_dict_config",
        "resolve_plain_dict_identity",
        "resolve_invalid_type",
    ]
    native_extensions = [
        "batch_workers_0",
        "batch_workers_1",
        "batch_workers_2",
        "batch_workers_8",
        "fixed_attribute_kind_slots",
        "typed_summary_return",
    ]
    record = {
        "config_source_sha256": CONFIG_SOURCE_SHA256,
        "dataset_source_sha256": DATASET_SOURCE_SHA256,
        "shared_cases": shared_cases,
        "native_extension_cases": native_extensions,
        "python_only_boundaries": PYTHON_ONLY_BOUNDARIES,
        "case_count": (
            len(shared_cases)
            + len(native_extensions)
            + len(PYTHON_ONLY_BOUNDARIES)
        ),
        "summary_checksums": [item["checksum"] for item in expected["summaries"]],
        "batch_checksum": expected["batch"][0]["checksum"],
        "result": "PASS",
    }
    if args.output:
        args.output.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        "utils config differential: PASS "
        f"({len(shared_cases)} shared + {len(native_extensions)} native + "
        f"{len(PYTHON_ONLY_BOUNDARIES)} boundaries)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
