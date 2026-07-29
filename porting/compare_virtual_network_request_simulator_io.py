#!/usr/bin/env python3
"""Pinned direct-source I/O differential for VirtualNetworkRequestSimulator."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import pathlib
import random
import struct
import subprocess
import sys
import tempfile
import types
from typing import Any

import numpy as np
import yaml


SOURCE_SHA256 = "970e63f9dac59f60e2ed1786606dc87d3271af062ba1d8d6c67aef8d3c7478e1"
BOUNDARIES = {
    "parallel_io_workers":
        "Python exposes only ordered sequential I/O; native workers retain the same semantic output and lowest sorted failure.",
    "typed_error_taxonomy":
        "Native stable error enums normalize Python AssertionError, ValueError, KeyError, and filesystem exceptions.",
    "gml_byte_encoding":
        "This simulator gate compares loaded graph semantics; exact NetworkX-compatible GML bytes remain frozen by the completed GML/network gates.",
}

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
    if distribution == "normal":
        data = np.random.normal(
            kwargs.get("loc", 0.0), kwargs.get("scale", 1.0), size
        )
    elif distribution == "uniform":
        low, high = kwargs.get("low"), kwargs.get("high")
        if dtype == "int":
            data = np.random.randint(low, high + 1, size)
        elif dtype == "float":
            data = np.random.uniform(low, high, size)
        else:
            raise UnboundLocalError("uniform bool is uninitialized")
    elif distribution == "exponential":
        data = np.random.exponential(kwargs.get("scale"), size)
    elif distribution == "poisson":
        lam = kwargs.get("lam")
        if kwargs.get("reciprocal", False):
            lam = 1 / lam
        data = np.random.poisson(lam, size)
    else:
        raise AssertionError(distribution)
    return data.astype(dtype).tolist()


def set_seed(seed: int | None = None) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


def read_setting(path: str | os.PathLike[str], **_: Any) -> Any:
    return yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))


def write_setting(
    value: Any, path: str | os.PathLike[str], **_: Any
) -> Any:
    pathlib.Path(path).write_text(
        yaml.safe_dump(value, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )
    return value


class StubVirtualNetwork:
    """Path-only persistence leaf; GML bytes are gated by frozen components."""

    def __init__(
        self,
        graph: Any = None,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        if graph is not None:
            raise RuntimeError("unexpected incoming graph")
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
            raise RuntimeError(f"unexpected topology: {topology!r}")
        self.nodes = list(range(num_nodes))
        self.edges = [(node, node + 1) for node in range(num_nodes - 1)]

    def generate_attrs_data(self) -> None:
        if self.config.get("node_attrs_setting") or self.config.get(
            "link_attrs_setting"
        ):
            raise RuntimeError("I/O differential expects empty attributes")

    def to_gml(self, path: str | os.PathLike[str]) -> None:
        graph = {
            name: getattr(self, name)
            for name in ("id", "arrival_time", "lifetime", "max_latency")
            if hasattr(self, name)
        }
        pathlib.Path(path).write_text(
            json.dumps({"graph": graph, "nodes": self.nodes, "edges": self.edges}),
            encoding="utf-8",
        )

    @classmethod
    def from_gml(cls, path: str | os.PathLike[str]):
        payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        result = cls(config={"graph_attrs_setting": payload["graph"]})
        result.nodes = [int(node) for node in payload["nodes"]]
        result.edges = [tuple(map(int, edge)) for edge in payload["edges"]]
        return result


def load_oracle(source: pathlib.Path):
    global _RESTORE_MODULES
    if _RESTORE_MODULES is not None:
        raise RuntimeError("simulator I/O oracle is already loaded")
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
    utils.read_setting = read_setting
    utils.write_setting = write_setting
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
            raise RuntimeError("unable to create simulator I/O oracle spec")
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


def make_setting(
    events_name: str = "events.yaml",
    setting_name: str = "v_sim_setting.yaml",
) -> dict[str, Any]:
    return {
        "num_v_nets": 2,
        "v_net_size": {
            "distribution": "uniform", "dtype": "int", "low": 2, "high": 2,
        },
        "lifetime": {
            "distribution": "uniform", "dtype": "float",
            "low": 3.0, "high": 3.0,
        },
        "arrival_rate": {
            "distribution": "poisson", "dtype": "float", "lam": 0.0,
        },
        "topology": {"type": "path"},
        "node_attrs_setting": [],
        "link_attrs_setting": [],
        "output": {
            "events_file_name": events_name,
            "setting_file_name": setting_name,
        },
    }


def make_generated(module, setting: dict[str, Any]):
    simulator = module.VirtualNetworkRequestSimulator.from_setting(
        copy.deepcopy(setting), seed=19
    )
    simulator.renew()
    return simulator


def graph_payload(value: StubVirtualNetwork) -> str:
    nodes = ",".join(str(node) for node in value.nodes)
    edges = ",".join(f"{left}-{right}" for left, right in value.edges)
    return f"N=[{nodes}];E=[{edges}]"


def networks_payload(values: list[StubVirtualNetwork]) -> str:
    encoded = []
    for value in values:
        encoded.append(
            f"id={int(value.id)},arrival={double_token(value.arrival_time)},"
            f"lifetime={double_token(value.lifetime)},max="
            + (
                double_token(value.max_latency)
                if hasattr(value, "max_latency") else "none"
            )
            + "," + graph_payload(value)
        )
    return "[" + "/".join(encoded) + "]"


def filenames_payload(directory: pathlib.Path) -> str:
    return "[" + ",".join(sorted(path.name for path in directory.iterdir())) + "]"


def json_file(path: pathlib.Path) -> str:
    return json.dumps(read_setting(path))


def dataset_payload(directory: pathlib.Path, loaded: Any) -> str:
    return (
        "setting=" + json_file(directory / "v_sim_setting.yaml")
        + ";events=" + json_file(directory / "events.yaml")
        + ";gml_files=" + filenames_payload(directory / "v_nets")
        + ";networks=" + networks_payload(loaded.v_nets)
    )


def layout_target(module, directory: pathlib.Path) -> str:
    try:
        module.VirtualNetworkRequestSimulator.load_dataset(str(directory))
    except AssertionError as error:
        message = str(error)
        if "Dataset directory" in message:
            return "root"
        if "v_nets directory" in message:
            return "v_nets"
        if "events.yaml" in message:
            return "events.yaml"
        if "setting.yaml" in message:
            return "v_sim_setting.yaml"
        raise RuntimeError(f"unknown Python layout assertion: {message}") from error
    raise RuntimeError("expected Python dataset layout error")


def oracle_cases(module, root: pathlib.Path) -> list[tuple[str, str]]:
    simulator_class = module.VirtualNetworkRequestSimulator
    simulator_class._cached_vnets_loads.clear()
    cases: list[tuple[str, str]] = []

    setting = make_setting()
    snapshot_simulator = simulator_class.from_setting(setting)
    setting["num_v_nets"] = 99
    snapshot_path = root / "snapshot.yaml"
    snapshot_simulator.save_setting(str(snapshot_path))
    source = snapshot_simulator.v_sim_setting
    decoded = (
        source["num_v_nets"] == 2
        and source["topology"]["type"] == "path"
        and source["v_net_size"]["dtype"] == "int"
        and source["v_net_size"]["distribution"] == "uniform"
        and not source["node_attrs_setting"]
        and not source["link_attrs_setting"]
        and source["output"]["events_file_name"] == "events.yaml"
        and source["output"]["setting_file_name"] == "v_sim_setting.yaml"
    )
    cases.append((
        "raw_decode_snapshot",
        f"decoded={int(decoded)};snapshot={json_file(snapshot_path)}",
    ))

    generated = make_generated(module, make_setting())
    for workers in (0, 1, 2, 8):
        directory = root / f"roundtrip_w{workers}"
        generated.save_dataset(str(directory))
        loaded = simulator_class.load_dataset(str(directory))
        cases.append((
            f"save_load_w{workers}", dataset_payload(directory, loaded)
        ))

    custom = make_generated(
        module, make_setting("custom-events.yaml", "custom-setting.yaml")
    )
    custom_directory = root / "custom_names"
    custom.save_dataset(str(custom_directory))
    cases.append((
        "hardcoded_load_names",
        "files=" + filenames_payload(custom_directory)
        + ";load=layout:" + layout_target(module, custom_directory),
    ))

    missing_root = root / "missing_root"
    cases.append((
        "layout_missing_root", "layout:" + layout_target(module, missing_root)
    ))
    missing_vnets = root / "missing_vnets"
    missing_vnets.mkdir()
    cases.append((
        "layout_missing_vnets", "layout:" + layout_target(module, missing_vnets)
    ))
    missing_events = root / "missing_events"
    (missing_events / "v_nets").mkdir(parents=True)
    cases.append((
        "layout_missing_events", "layout:" + layout_target(module, missing_events)
    ))
    missing_setting = root / "missing_setting"
    (missing_setting / "v_nets").mkdir(parents=True)
    write_setting([], missing_setting / "events.yaml")
    cases.append((
        "layout_missing_setting",
        "layout:" + layout_target(module, missing_setting),
    ))

    mismatch = root / "event_cardinality"
    generated.save_dataset(str(mismatch))
    events = read_setting(mismatch / "events.yaml")
    events.pop()
    write_setting(events, mismatch / "events.yaml")
    try:
        simulator_class.load_dataset(str(mismatch))
        raise RuntimeError("expected Python event cardinality error")
    except ValueError:
        cases.append(("event_cardinality", "event_count_mismatch"))

    invalid = root / "invalid_event"
    generated.save_dataset(str(invalid))
    write_setting(
        [{"id": 0, "type": 2, "v_net_id": 0, "time": 0.0}],
        invalid / "events.yaml",
    )
    try:
        simulator_class.load_dataset(str(invalid))
        raise RuntimeError("expected Python invalid event error")
    except ValueError:
        cases.append(("invalid_event_setting", "invalid_event_setting"))

    for workers in (1, 8):
        directory = root / f"sorted_error_w{workers}"
        generated.save_dataset(str(directory))
        (directory / "v_nets" / "000-bad").mkdir()
        (directory / "v_nets" / "zzz-bad").mkdir()
        ordered = sorted(os.listdir(directory / "v_nets"))
        try:
            simulator_class.load_dataset(str(directory))
            raise RuntimeError("expected Python sorted entry load error")
        except (IsADirectoryError, OSError, json.JSONDecodeError):
            cases.append((
                f"sorted_entry_error_w{workers}",
                f"first={ordered[0]};index=0",
            ))

    seeded = root / "seed_cached"
    generated.save_dataset(str(seeded))
    first = simulator_class.load_dataset(str(seeded))
    (seeded / "events.yaml").unlink()
    second = simulator_class.load_dataset(str(seeded))
    same = networks_payload(first.v_nets) == networks_payload(second.v_nets)
    cases.append((
        "cache_seed",
        f"reload=ok;same={int(same)};"
        f"graph_independent={int(first.v_nets[0] is not second.v_nets[0])};"
        f"setting_independent={int(first.v_sim_setting is not second.v_sim_setting)};"
        f"stored={int(str(seeded) in simulator_class._cached_vnets_loads)}",
    ))

    ordinary = root / "ordinary_cache"
    generated.save_dataset(str(ordinary))
    simulator_class.load_dataset(str(ordinary))
    (ordinary / "events.yaml").unlink()
    cases.append((
        "cache_nonseed",
        "reload=layout:" + layout_target(module, ordinary)
        + f";stored={int(str(ordinary) in simulator_class._cached_vnets_loads)}",
    ))
    return cases


def parse_cpp(path: pathlib.Path) -> list[tuple[str, str]]:
    process = subprocess.run(
        [str(path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"simulator I/O harness failed: {process.stderr.strip()}"
        )
    result: list[tuple[str, str]] = []
    for line in process.stdout.splitlines():
        fields = line.split("|")
        if (
            len(fields) != 3
            or not fields[0].startswith("case=")
            or fields[1] != "ok"
        ):
            raise RuntimeError(f"malformed simulator I/O harness line: {line!r}")
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
        with tempfile.TemporaryDirectory(prefix="virne_sim_io_diff_") as raw_root:
            expected = oracle_cases(module, pathlib.Path(raw_root))
        if not torch_before and "torch" in sys.modules:
            raise RuntimeError("simulator I/O oracle unexpectedly imported Torch")
    finally:
        unload_oracle()

    actual = parse_cpp(args.harness.resolve())
    if actual != expected:
        for got, wanted in zip(actual, expected):
            if got != wanted:
                raise RuntimeError(
                    f"Simulator I/O mismatch {wanted[0]}: "
                    f"C++={got!r}, Python={wanted!r}"
                )
        raise RuntimeError(
            f"Simulator I/O inventory mismatch: C++={actual!r}, Python={expected!r}"
        )

    payload = {
        "source_sha256": SOURCE_SHA256,
        "numpy_version": np.__version__,
        "pyyaml_version": yaml.__version__,
        "shared_cases": [name for name, _ in expected],
        "native_extension_cases": [],
        "python_only_boundaries": BOUNDARIES,
        "case_count": len(expected) + len(BOUNDARIES),
        "result": "PASS",
        "torch_imported": False,
    }
    if args.output:
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        "VirtualNetworkRequestSimulator I/O differential: PASS "
        f"({len(expected)} shared + 0 native + {len(BOUNDARIES)} boundaries)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
