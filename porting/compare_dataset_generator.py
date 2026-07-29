#!/usr/bin/env python3
"""Pinned direct-source differential for the non-ML dataset Generator."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import pathlib
import random
import struct
import subprocess
import sys
import types
from typing import Any

import numpy as np

import compare_virtual_network_request_simulator as simulator_diff


SOURCE_SHA256 = "43d5dbe625fcd15f273067700b3c9d0b69cf931e064f6542c65802b0a4ba4e5c"
BOUNDARIES = {
    "typed_config_boundary":
        "Arbitrary mappings and DictConfig reflection are replaced by the completed Config API and one resolved subtree snapshot.",
    "persistence_filesystem":
        "Save-path composition and filesystem effects are covered by the Generator unit plus frozen dataset/network persistence gates.",
    "torch_seed_boundary":
        "The unused eager torch import and Torch/CUDA global seeding remain deferred with the ML adapter.",
    "native_seed_width":
        "RandomContext deliberately accepts the frozen non-negative uint32 seed domain rather than Python arbitrary integers.",
}


def verify(path: pathlib.Path, expected: str, label: str) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"{label} source hash drift: {actual}")


class GeneratorVirtualNetwork:
    def __init__(
        self,
        graph=None,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if graph is not None:
            raise RuntimeError("Generator oracle expects constructed topology")
        self.config = copy.deepcopy(config or {})
        for name, value in self.config.get("graph_attrs_setting", {}).items():
            setattr(self, name, value)
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.nodes: list[int] = []
        self.edges: list[tuple[int, int]] = []
        self.node_values: dict[str, list[Any]] = {}
        self.link_values: dict[str, list[Any]] = {}

    def set_graph_attribute(self, name: str, value: Any) -> None:
        setattr(self, name, value)

    def generate_topology(self, num_nodes: int, **topology: Any) -> None:
        if topology.get("type") != "path":
            raise RuntimeError(f"unexpected Generator topology: {topology!r}")
        self.nodes = list(range(int(num_nodes)))
        self.edges = [(node, node + 1) for node in range(int(num_nodes) - 1)]

    def generate_attrs_data(self) -> None:
        for spec in self.config.get("node_attrs_setting", []):
            self.node_values[spec["name"]] = (
                simulator_diff.generate_data_with_distribution(
                    size=len(self.nodes), **spec
                )
            )
        for spec in self.config.get("link_attrs_setting", []):
            self.link_values[spec["name"]] = (
                simulator_diff.generate_data_with_distribution(
                    size=len(self.edges), **spec
                )
            )


class StubPhysicalNetwork:
    def __init__(self, setting: dict[str, Any]) -> None:
        topology = setting["topology"]
        if topology.get("type") != "path":
            raise RuntimeError("Generator physical oracle expects path")
        count = int(topology["num_nodes"])
        self.nodes = list(range(count))
        self.edges = [(node, node + 1) for node in range(count - 1)]

    @staticmethod
    def from_setting(setting: dict[str, Any]):
        return StubPhysicalNetwork(copy.deepcopy(setting))

    def save_dataset(self, *_: Any, **__: Any) -> None:
        raise RuntimeError("save is outside the direct Generator differential")


_GENERATOR_RESTORE: dict[str, types.ModuleType | None] | None = None
_ATTRIBUTE_RESTORE: dict[str, tuple[bool, Any]] | None = None
_TORCH_SEED_CALLS = 0


def load_generator_oracle(
    generator_source: pathlib.Path,
    simulator_source: pathlib.Path,
):
    global _GENERATOR_RESTORE, _ATTRIBUTE_RESTORE, _TORCH_SEED_CALLS
    verify(generator_source, SOURCE_SHA256, "dataset Generator")
    simulator_module = simulator_diff.load_oracle(simulator_source)
    simulator_module.VirtualNetwork = GeneratorVirtualNetwork

    names = (
        "torch",
        "virne.network.physical_network",
        "virne.network.dataset_generator",
    )
    _GENERATOR_RESTORE = {name: sys.modules.get(name) for name in names}
    utils = sys.modules["virne.utils"]
    network = sys.modules["virne.network"]
    _ATTRIBUTE_RESTORE = {}
    for owner, name in (
        (utils, "set_seed"),
        (utils, "get_p_net_dataset_dir_from_setting"),
        (utils, "get_v_nets_dataset_dir_from_setting"),
        (network, "physical_network"),
        (network, "dataset_generator"),
    ):
        key = f"{id(owner)}:{name}"
        _ATTRIBUTE_RESTORE[key] = (hasattr(owner, name), getattr(owner, name, None))

    _TORCH_SEED_CALLS = 0

    def unused_torch_seed(*_: Any, **__: Any) -> None:
        global _TORCH_SEED_CALLS
        _TORCH_SEED_CALLS += 1

    torch = types.ModuleType("torch")
    torch.seed = unused_torch_seed
    physical_module = types.ModuleType("virne.network.physical_network")
    physical_module.PhysicalNetwork = StubPhysicalNetwork

    def unexpected_path(*_: Any, **__: Any) -> str:
        raise RuntimeError("save path helper called with save=False")

    utils.set_seed = simulator_diff.set_seed
    utils.get_p_net_dataset_dir_from_setting = unexpected_path
    utils.get_v_nets_dataset_dir_from_setting = unexpected_path
    network.physical_network = physical_module
    sys.modules["torch"] = torch
    sys.modules["virne.network.physical_network"] = physical_module

    try:
        spec = importlib.util.spec_from_file_location(
            "virne.network.dataset_generator", generator_source
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("unable to create Generator oracle spec")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        network.dataset_generator = module
        return module
    except Exception:
        unload_generator_oracle()
        raise


def unload_generator_oracle() -> None:
    global _GENERATOR_RESTORE, _ATTRIBUTE_RESTORE
    if _GENERATOR_RESTORE is None:
        return
    utils = sys.modules.get("virne.utils")
    network = sys.modules.get("virne.network")
    if _ATTRIBUTE_RESTORE is not None:
        for owner, name in (
            (utils, "set_seed"),
            (utils, "get_p_net_dataset_dir_from_setting"),
            (utils, "get_v_nets_dataset_dir_from_setting"),
            (network, "physical_network"),
            (network, "dataset_generator"),
        ):
            if owner is None:
                continue
            existed, previous = _ATTRIBUTE_RESTORE[f"{id(owner)}:{name}"]
            if existed:
                setattr(owner, name, previous)
            elif hasattr(owner, name):
                delattr(owner, name)
    for name, previous in _GENERATOR_RESTORE.items():
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    _GENERATOR_RESTORE = None
    _ATTRIBUTE_RESTORE = None
    simulator_diff.unload_oracle()


def make_config(seed: int | None, request_count: int = 8) -> dict[str, Any]:
    config: dict[str, Any] = {
        "experiment": {"seed": 23},
        "p_net_setting": {
            "topology": {
                "type": "path", "num_nodes": 4,
                "wm_alpha": 0.5, "wm_beta": 0.2,
            },
            "node_attrs_setting": [],
            "link_attrs_setting": [],
            "output": {"save_dir": "unused-physical"},
        },
        "v_sim_setting": {
            "num_v_nets": request_count,
            "topology": {"type": "path"},
            "v_net_size": {
                "distribution": "uniform", "dtype": "int",
                "low": 2, "high": 3,
            },
            "arrival_rate": {
                "distribution": "poisson", "dtype": "float", "lam": 1.0,
            },
            "lifetime": {
                "distribution": "exponential", "dtype": "float", "scale": 2,
            },
            "node_attrs_setting": [{
                "name": "cpu", "type": "resource", "owner": "node",
                "distribution": "uniform", "dtype": "int",
                "generative": True, "low": 0, "high": 3,
            }],
            "link_attrs_setting": [{
                "name": "bandwidth", "type": "resource", "owner": "link",
                "distribution": "uniform", "dtype": "int",
                "generative": True, "low": 0, "high": 5,
            }],
            "output": {
                "save_dir": "unused-virtual",
                "events_file_name": "events.yaml",
                "setting_file_name": "v_sim_setting.yaml",
            },
        },
    }
    if seed is not None:
        config["seed"] = seed
    return config


def double_token(value: float) -> str:
    return "d:" + struct.pack(">d", float(value)).hex()


def attr_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "b1" if value else "b0"
    if isinstance(value, int):
        return "i" + str(value)
    if isinstance(value, float):
        return double_token(value)
    if isinstance(value, str):
        return "s" + value
    raise RuntimeError(f"unexpected Generator attribute value: {value!r}")


def values_payload(values: dict[str, list[Any]], name: str) -> str:
    if name not in values:
        return "none"
    return "[" + ",".join(attr_scalar(value) for value in values[name]) + "]"


def simulator_payload(simulator) -> str:
    networks = []
    for request in simulator.v_nets:
        networks.append(
            f"id={int(request.id)},a={double_token(request.arrival_time)},"
            f"l={double_token(request.lifetime)},n={len(request.nodes)},"
            f"e={len(request.edges)},"
            f"cpu={values_payload(request.node_values, 'cpu')},"
            f"bw={values_payload(request.link_values, 'bandwidth')}"
        )
    events = "/".join(
        f"{int(event.id)}:{int(event.type)}:{int(event.v_net_id)}:"
        f"{double_token(event.time)}"
        for event in simulator.events
    )
    return (
        f"V={len(simulator.v_nets)}:{len(simulator.events)};"
        f"nets=[{'/'.join(networks)}];events=[{events}]"
    )


def generated_payload(generated, next_state: bool = True) -> str:
    physical, simulator = generated
    physical_token = (
        f"{len(physical.nodes)}:{len(physical.edges)}"
        if physical is not None else "none"
    )
    result = f"P={physical_token};" + (
        simulator_payload(simulator) if simulator is not None else "V=none"
    )
    if next_state:
        result += f";next_py={random.getrandbits(32)}"
        result += f";next_np={simulator_diff.next_numpy_word()}"
    return result


def run_generation(module, config, p_net: bool, v_nets: bool, initial: int):
    simulator_diff.set_seed(initial)
    return generated_payload(
        module.Generator.generate_dataset(
            config, p_net=p_net, v_nets=v_nets, save=False
        )
    )


def oracle_cases(module) -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []
    selections = (
        ("selection_none", False, False),
        ("selection_physical", True, False),
        ("selection_virtual", False, True),
        ("selection_both", True, True),
    )
    for name, physical, virtuals in selections:
        cases.append((
            name,
            run_generation(
                module, make_config(41), physical, virtuals, 777
            ),
        ))
    for workers in (0, 2, 8):
        cases.append((
            f"selection_both_w{workers}",
            run_generation(module, make_config(41), True, True, 777),
        ))

    cases.append((
        "absent_seed_both",
        run_generation(module, make_config(None), True, True, 83),
    ))
    # Native composed mode maps experiment.seed explicitly. The standalone
    # Python compatibility input exposes that same value at root seed.
    cases.append((
        "composed_seed_virtual",
        run_generation(module, make_config(23), False, True, 0),
    ))

    for workers in (0, 1, 2, 8):
        simulator_diff.set_seed(0)
        simulator = (
            module.Generator.generate_changeable_v_nets_dataset_from_config(
                make_config(53), save=False
            )
        )
        cases.append((
            f"changeable_w{workers}",
            generated_payload((None, simulator)),
        ))

    simulator_diff.set_seed(72)
    failed = False
    try:
        module.Generator.generate_v_nets_dataset_from_config(
            {"seed": 99}, save=False
        )
    except Exception:
        failed = True
    cases.append((
        "missing_subtree",
        f"error={int(failed)};next_py={random.getrandbits(32)};"
        f"next_np={simulator_diff.next_numpy_word()}",
    ))

    simulator_diff.set_seed(0)
    failed = False
    try:
        module.Generator.generate_changeable_v_nets_dataset_from_config(
            make_config(91, 6), save=False
        )
    except Exception:
        failed = True
    cases.append((
        "invalid_changeable_count",
        f"error={int(failed)};next_py={random.getrandbits(32)};"
        f"next_np={simulator_diff.next_numpy_word()}",
    ))
    return cases


def parse_cpp(path: pathlib.Path) -> list[tuple[str, str]]:
    process = subprocess.run(
        [str(path)], check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"dataset Generator harness failed: {process.stderr.strip()}")
    result: list[tuple[str, str]] = []
    for line in process.stdout.splitlines():
        fields = line.split("|")
        if (
            len(fields) != 3
            or not fields[0].startswith("case=")
            or fields[1] != "ok"
        ):
            raise RuntimeError(f"malformed Generator harness line: {line!r}")
        result.append((fields[0][5:], bytes.fromhex(fields[2]).decode("utf-8")))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--simulator-source", type=pathlib.Path, required=True)
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    module = load_generator_oracle(args.source, args.simulator_source)
    try:
        expected = oracle_cases(module)
        if _TORCH_SEED_CALLS != 0:
            raise RuntimeError("Generator unexpectedly called imported torch.seed")
    finally:
        unload_generator_oracle()
    actual = parse_cpp(args.harness.resolve())
    if actual != expected:
        for got, wanted in zip(actual, expected):
            if got != wanted:
                raise RuntimeError(
                    f"Generator mismatch {wanted[0]}: C++={got!r}, Python={wanted!r}"
                )
        raise RuntimeError(
            f"Generator inventory mismatch: C++={actual!r}, Python={expected!r}"
        )

    payload = {
        "source_sha256": SOURCE_SHA256,
        "simulator_source_sha256": simulator_diff.SOURCE_SHA256,
        "numpy_version": np.__version__,
        "shared_cases": [name for name, _ in expected],
        "native_extension_cases": [],
        "python_only_boundaries": BOUNDARIES,
        "case_count": len(expected) + len(BOUNDARIES),
        "result": "PASS",
        "torch_seed_calls": 0,
    }
    if args.output:
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        "dataset Generator differential: PASS "
        f"({len(expected)} shared + 0 native + {len(BOUNDARIES)} boundaries)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
