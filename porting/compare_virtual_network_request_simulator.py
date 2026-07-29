#!/usr/bin/env python3
"""Pinned direct-source differential for VirtualNetworkRequestSimulator."""

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


SOURCE_SHA256 = "970e63f9dac59f60e2ed1786606dc87d3271af062ba1d8d6c67aef8d3c7478e1"
BOUNDARIES = {
    "arbitrary_setting_protocols":
        "Dynamic mapping, DictConfig, reflection, and mutation protocols stay at the Python boundary.",
    "torch_seed_side_effects":
        "Torch/CUDA seeding is deliberately excluded with the deferred ML boundary.",
    "stochastic_topologies":
        "Random and Waxman topology RNG behavior is frozen by the dedicated topology differential; this leaf uses path.",
    "persistence_io":
        "GML/YAML persistence belongs to the already frozen network, dataset, and setting components.",
}
NATIVE_CASES = [
    ("native_invalid_size_kind",
     "code=invalid_size_distribution;operation=validate_config"),
    ("native_invalid_enum_lookup", "id=none"),
]

_RESTORE_MODULES: dict[str, types.ModuleType | None] | None = None


def verify(path: pathlib.Path) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(
            f"VirtualNetworkRequestSimulator source hash drift: {actual}"
        )


def generate_data_with_distribution(
    size: int, distribution: str, dtype: str, **kwargs: Any
) -> list[Any]:
    """Minimal exact copy of the pinned helper branches used by this matrix."""
    if distribution not in ("uniform", "normal", "exponential", "poisson"):
        raise AssertionError(distribution)
    if dtype not in ("int", "float", "bool"):
        raise AssertionError(dtype)
    if distribution == "normal":
        data = np.random.normal(kwargs.get("loc", 0.0), kwargs.get("scale", 1.0), size)
    elif distribution == "uniform":
        low, high = kwargs.get("low"), kwargs.get("high")
        if dtype == "int":
            data = np.random.randint(low, high + 1, size)
        elif dtype == "float":
            data = np.random.uniform(low, high, size)
        else:
            raise UnboundLocalError("uniform bool is uninitialized in the pinned helper")
    elif distribution == "exponential":
        data = np.random.exponential(kwargs.get("scale"), size)
    else:
        lam = kwargs.get("lam")
        if kwargs.get("reciprocal", False):
            lam = 1 / lam
        data = np.random.poisson(lam, size)
    return data.astype(dtype).tolist()


def set_seed(seed: int | None = None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


class StubVirtualNetwork:
    """Path-only leaf used to observe simulator-owned fields and ordering."""

    def __init__(self, graph=None, config: dict[str, Any] | None = None, **kwargs: Any):
        if graph is not None:
            raise RuntimeError("the simulator oracle must construct topology itself")
        self.config = copy.deepcopy(config or {})
        for name, value in self.config.get("graph_attrs_setting", {}).items():
            setattr(self, name, value)
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.nodes: list[int] = []
        self.edges: list[tuple[int, int]] = []

    def set_graph_attribute(self, name: str, value: Any) -> None:
        setattr(self, name, value)

    def generate_topology(self, num_nodes: int, **topology: Any) -> None:
        if topology.get("type") != "path":
            raise RuntimeError(f"unexpected topology in simulator oracle: {topology!r}")
        if not isinstance(num_nodes, int):
            raise TypeError(f"path size must be int, got {type(num_nodes).__name__}")
        self.nodes = list(range(num_nodes))
        self.edges = [(node, node + 1) for node in range(num_nodes - 1)]

    def generate_attrs_data(self) -> None:
        if self.config.get("node_attrs_setting") or self.config.get("link_attrs_setting"):
            raise RuntimeError("simulator differential expects empty attribute settings")


def load_oracle(source: pathlib.Path):
    global _RESTORE_MODULES
    if _RESTORE_MODULES is not None:
        raise RuntimeError("VirtualNetworkRequestSimulator oracle is already loaded")
    source = source.resolve()
    verify(source)

    names = (
        "omegaconf",
        "virne",
        "virne.utils",
        "virne.utils.dataset",
        "virne.network",
        "virne.network.virtual_network",
        "virne.network.virtual_network_request_simulator",
    )
    _RESTORE_MODULES = {name: sys.modules.get(name) for name in names}

    omegaconf = types.ModuleType("omegaconf")
    omegaconf.DictConfig = type("DictConfig", (dict,), {})

    class OmegaConf:
        @staticmethod
        def to_container(value, resolve: bool = True):
            del resolve
            return copy.deepcopy(dict(value))

    omegaconf.OmegaConf = OmegaConf

    virne = types.ModuleType("virne")
    virne.__path__ = []
    utils = types.ModuleType("virne.utils")
    utils.__path__ = []
    utils.read_setting = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("read_setting is outside the differential")
    )
    utils.write_setting = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("write_setting is outside the differential")
    )
    utils.generate_data_with_distribution = generate_data_with_distribution
    dataset = types.ModuleType("virne.utils.dataset")
    dataset.set_seed = set_seed
    network = types.ModuleType("virne.network")
    network.__path__ = []
    virtual_network = types.ModuleType("virne.network.virtual_network")
    virtual_network.VirtualNetwork = StubVirtualNetwork
    virne.utils = utils
    virne.network = network
    utils.dataset = dataset
    network.virtual_network = virtual_network

    sys.modules.update({
        "omegaconf": omegaconf,
        "virne": virne,
        "virne.utils": utils,
        "virne.utils.dataset": dataset,
        "virne.network": network,
        "virne.network.virtual_network": virtual_network,
    })
    try:
        spec = importlib.util.spec_from_file_location(
            "virne.network.virtual_network_request_simulator", source
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("unable to create simulator oracle spec")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        network.virtual_network_request_simulator = module
        return module
    except Exception:
        unload_oracle()
        raise


def unload_oracle() -> None:
    global _RESTORE_MODULES
    if _RESTORE_MODULES is None:
        return
    for name, previous in _RESTORE_MODULES.items():
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    _RESTORE_MODULES = None


def double_token(value: float) -> str:
    return "d:" + struct.pack(">d", float(value)).hex()


def integer_vector_token(values) -> str:
    return "[" + ",".join(str(int(value)) for value in values) + "]"


def double_vector_token(values) -> str:
    return "[" + ",".join(double_token(value) for value in values) + "]"


def optional_event_token(value: int | None) -> str:
    return "none" if value is None else str(int(value))


def make_config(
    arrival_dtype: str, with_max_latency: bool, request_count: int = 5
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "num_v_nets": request_count,
        "v_net_size": {
            "distribution": "uniform", "dtype": "int", "low": 2, "high": 5,
        },
        "lifetime": {
            "distribution": "exponential", "dtype": "float", "scale": 3.25,
        },
        "arrival_rate": {
            "distribution": "poisson", "dtype": arrival_dtype, "lam": 1.25,
        },
        "topology": {"type": "path"},
        "node_attrs_setting": [],
        "link_attrs_setting": [],
        "output": {
            "events_file_name": "events.yaml",
            "setting_file_name": "v_sim_setting.yaml",
        },
    }
    if with_max_latency:
        config["max_latency"] = {
            "distribution": "uniform", "dtype": "float", "low": 0.25, "high": 2.0,
        }
    return config


def make_tie_config() -> dict[str, Any]:
    config = make_config("bool", False, 3)
    config["v_net_size"] = {
        "distribution": "uniform", "dtype": "int", "low": 2, "high": 2,
    }
    config["lifetime"] = {
        "distribution": "poisson", "dtype": "float", "lam": 0.0,
    }
    config["arrival_rate"] = {
        "distribution": "poisson", "dtype": "bool", "lam": 0.0,
    }
    return config


def arrangement_payload(simulator) -> str:
    maximum = (
        double_vector_token(simulator.v_nets_max_latency)
        if hasattr(simulator, "v_nets_max_latency") else "none"
    )
    return (
        "sizes=" + integer_vector_token(simulator.v_nets_size)
        + ";lifetimes=" + double_vector_token(simulator.v_nets_lifetime)
        + ";arrivals=" + double_vector_token(simulator.v_nets_arrival_time)
        + ";max=" + maximum
    )


def graph_payload(value: StubVirtualNetwork) -> str:
    nodes = ",".join(str(node) for node in value.nodes)
    edges = ",".join(f"{left}-{right}" for left, right in value.edges)
    return f"N=[{nodes}];E=[{edges}]"


def networks_payload(values) -> str:
    encoded = []
    for value in values:
        encoded.append(
            f"id={int(value.id)},arrival={double_token(value.arrival_time)},"
            f"lifetime={double_token(value.lifetime)},max="
            + (double_token(value.max_latency) if hasattr(value, "max_latency") else "none")
            + "," + graph_payload(value)
        )
    return "[" + "/".join(encoded) + "]"


def events_payload(events) -> str:
    return "[" + ",".join(
        f"{int(event.id)}:{int(event.type)}:{int(event.v_net_id)}:{double_token(event.time)}"
        for event in events
    ) + "]"


def lookup_payload(simulator) -> str:
    return "[" + ",".join(
        f"{int(value.id)}:"
        f"{optional_event_token(simulator.v2event_dict.get((value.id, 1)))}:"
        f"{optional_event_token(simulator.v2event_dict.get((value.id, 0)))}"
        for value in simulator.v_nets
    ) + "]"


def full_payload(simulator, next_python: int, next_numpy: int) -> str:
    return (
        arrangement_payload(simulator)
        + f";num_vnets={simulator.num_v_nets};num_events={simulator.num_events}"
        + ";networks=" + networks_payload(simulator.v_nets)
        + ";events=" + events_payload(simulator.events)
        + ";lookup=" + lookup_payload(simulator)
        + f";next_py={next_python};next_np={next_numpy}"
    )


def next_numpy_word() -> int:
    return int(np.random.randint(0, 2**32, dtype=np.uint32))


def oracle_cases(module) -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []
    arrangement_specs = (
        ("integer", "int", False, 701),
        ("floating", "float", True, 702),
        ("boolean", "bool", True, 703),
    )
    for name, dtype, with_max_latency, seed in arrangement_specs:
        for workers in (0, 1, 2, 8):
            np.random.seed(seed)
            simulator = module.VirtualNetworkRequestSimulator.from_setting(
                make_config(dtype, with_max_latency)
            )
            simulator.arrange_v_nets()
            cases.append((
                f"arrange_{name}_w{workers}",
                arrangement_payload(simulator) + f";next_np={next_numpy_word()}",
            ))

    for workers in (0, 1, 2, 8):
        simulator = module.VirtualNetworkRequestSimulator.from_setting(
            make_config("int", True), seed=991
        )
        simulator.renew()
        next_python = random.getrandbits(32)
        next_numpy = next_numpy_word()
        cases.append((
            f"renew_path_w{workers}",
            full_payload(simulator, next_python, next_numpy),
        ))

    tie = module.VirtualNetworkRequestSimulator.from_setting(
        make_tie_config(), seed=313
    )
    tie.renew()
    cases.append((
        "tie_order",
        full_payload(tie, random.getrandbits(32), next_numpy_word()),
    ))

    loaded_events = [
        module.VirtualNetworkEvent(id=41, type=1, v_net_id=1_000_000, time=3.0),
        module.VirtualNetworkEvent(id=42, type=0, v_net_id=1_000_000, time=4.0),
        module.VirtualNetworkEvent(id=43, type=1, v_net_id=1_000_000, time=5.0),
        module.VirtualNetworkEvent(id=44, type=0, v_net_id=7, time=6.0),
    ]
    loaded = module.VirtualNetworkRequestSimulator(
        v_nets=[], events=loaded_events, v_sim_setting={}
    )
    cases.append((
        "loaded_index",
        f"num_vnets={loaded.num_v_nets};num_events={loaded.num_events};"
        f"events={events_payload(loaded.events)};"
        f"sparse_arrival={optional_event_token(loaded.v2event_dict.get((1_000_000, 1)))};"
        f"sparse_leave={optional_event_token(loaded.v2event_dict.get((1_000_000, 0)))};"
        f"id7_arrival={optional_event_token(loaded.v2event_dict.get((7, 1)))};"
        f"id7_leave={optional_event_token(loaded.v2event_dict.get((7, 0)))}",
    ))
    return cases


def parse_cpp(path: pathlib.Path) -> list[tuple[str, str]]:
    process = subprocess.run(
        [str(path)], check=False, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"VirtualNetworkRequestSimulator harness failed: {process.stderr.strip()}"
        )
    result: list[tuple[str, str]] = []
    for line in process.stdout.splitlines():
        fields = line.split("|")
        if len(fields) != 3 or not fields[0].startswith("case=") or fields[1] != "ok":
            raise RuntimeError(f"malformed simulator harness line: {line!r}")
        result.append((fields[0][5:], bytes.fromhex(fields[2]).decode("utf-8")))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    torch_before = "torch" in sys.modules
    module = load_oracle(args.source)
    try:
        expected_shared = oracle_cases(module)
        numpy_version = np.__version__
        if not torch_before and "torch" in sys.modules:
            raise RuntimeError("simulator oracle unexpectedly imported Torch")
    finally:
        unload_oracle()
    expected = expected_shared + NATIVE_CASES
    actual = parse_cpp(args.harness.resolve())
    if actual != expected:
        for got, wanted in zip(actual, expected):
            if got != wanted:
                raise RuntimeError(
                    f"Simulator mismatch {wanted[0]}: C++={got!r}, Python={wanted!r}"
                )
        raise RuntimeError(
            f"Simulator inventory mismatch: C++={actual!r}, Python={expected!r}"
        )

    payload = {
        "source_sha256": SOURCE_SHA256,
        "numpy_version": numpy_version,
        "shared_cases": [name for name, _ in expected_shared],
        "native_extension_cases": [name for name, _ in NATIVE_CASES],
        "python_only_boundaries": BOUNDARIES,
        "case_count": len(expected) + len(BOUNDARIES),
        "result": "PASS",
        "torch_imported": False,
    }
    if args.output:
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        f"VirtualNetworkRequestSimulator differential: PASS "
        f"({len(expected_shared)} shared + {len(NATIVE_CASES)} native + "
        f"{len(BOUNDARIES)} boundaries)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
