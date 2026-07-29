#!/usr/bin/env python3
"""Pinned direct-source differential for the non-ML PhysicalNetwork leaf."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import pathlib
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import types
from typing import Any

import compare_base_network as base_oracle


SOURCE_SHA256 = "e37d48b2c1651b503931597b6cca5620a413a11c3d003cf8c70af89246e4ca5a"
BOUNDARIES = {
    "dictconfig_mutation":
        "Python mutates DictConfig topology.num_nodes; native input documents are immutable.",
    "stdout_diagnostics":
        "Python prints load/warning diagnostics; native exposes a typed build report.",
    "arbitrary_gml_labels":
        "Python accepts arbitrary hashable labels; the frozen native loader uses dense label=id.",
    "networkx_private_storage":
        "Python replaces private NetworkX storage with observable partial mutation on failure.",
    "arbitrary_mapping_truthiness":
        "Dynamic mapping protocols and truthiness remain a cold Python-only boundary.",
    "torch_seed_side_effects":
        "Torch/CUDA seeding is intentionally deferred with the ML boundary.",
    "python_dataset_roundtrip_bug":
        "Python cannot round-trip its own empty or non-empty attribute-setting GML spelling; native keeps valid typed GML.",
}

_RESTORE_STATE: tuple[Any, ...] | None = None


def verify(path: pathlib.Path) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"PhysicalNetwork source hash drift: {actual}")


def double_token(value: float) -> str:
    return "d:" + struct.pack(">d", float(value)).hex()


def generated_config() -> dict[str, Any]:
    return {
        "topology": {"num_nodes": 5, "type": "path"},
        "node_attrs_setting": [{
            "name": "cpu", "owner": "node", "type": "resource",
            "generative": True, "distribution": "uniform", "dtype": "int",
            "low": 1, "high": 9,
        }],
        "link_attrs_setting": [{
            "name": "bandwidth", "owner": "link", "type": "resource",
            "generative": True, "distribution": "uniform", "dtype": "int",
            "low": 10, "high": 19,
        }],
    }


def load_oracle(source: pathlib.Path, base_source: pathlib.Path):
    global _RESTORE_STATE
    if _RESTORE_STATE is not None:
        raise RuntimeError("PhysicalNetwork oracle is already loaded")
    source = source.resolve()
    verify(source)
    base = base_oracle.load_oracle(base_source)

    utils = sys.modules["virne.utils"]
    omegaconf = sys.modules["omegaconf"]
    attribute_base = sys.modules["virne.network.attribute.base_attribute"]
    previous_dataset = sys.modules.get("virne.utils.dataset")
    previous_open_dict = getattr(omegaconf, "open_dict", None)
    previous_generation = attribute_base.generate_data_with_distribution
    previous_utils_generation = utils.generate_data_with_distribution

    @contextlib.contextmanager
    def open_dict(value):
        yield value

    def generate_data_with_distribution(
        size: int, distribution: str, dtype: str, **kwargs: Any
    ) -> list[Any]:
        if distribution != "uniform" or dtype != "int":
            raise RuntimeError((size, distribution, dtype, kwargs))
        data = base.np.random.randint(
            kwargs.get("low"), kwargs.get("high") + 1, size
        )
        return data.astype(dtype).tolist()

    def set_seed(seed: int | None = None) -> None:
        if seed is None:
            return
        random.seed(seed)
        base.np.random.seed(seed)

    dataset = types.ModuleType("virne.utils.dataset")
    dataset.set_seed = set_seed
    omegaconf.open_dict = open_dict
    utils.generate_data_with_distribution = generate_data_with_distribution
    attribute_base.generate_data_with_distribution = generate_data_with_distribution
    sys.modules["virne.utils.dataset"] = dataset
    try:
        module = base_oracle.execute_module("virne.network.physical_network", source)
        sys.modules["virne.network"].physical_network = module
    except Exception:
        if previous_dataset is None:
            sys.modules.pop("virne.utils.dataset", None)
        else:
            sys.modules["virne.utils.dataset"] = previous_dataset
        if previous_open_dict is None:
            omegaconf.__dict__.pop("open_dict", None)
        else:
            omegaconf.open_dict = previous_open_dict
        utils.generate_data_with_distribution = previous_utils_generation
        attribute_base.generate_data_with_distribution = previous_generation
        base_oracle.unload_oracle()
        raise

    _RESTORE_STATE = (
        previous_dataset, previous_open_dict, previous_generation,
        previous_utils_generation,
    )
    return module


def unload_oracle() -> None:
    global _RESTORE_STATE
    if _RESTORE_STATE is None:
        return
    (
        previous_dataset, previous_open_dict, previous_generation,
        previous_utils_generation,
    ) = _RESTORE_STATE
    utils = sys.modules["virne.utils"]
    omegaconf = sys.modules["omegaconf"]
    attribute_base = sys.modules["virne.network.attribute.base_attribute"]
    sys.modules.pop("virne.network.physical_network", None)
    sys.modules["virne.network"].__dict__.pop("physical_network", None)
    if previous_dataset is None:
        sys.modules.pop("virne.utils.dataset", None)
    else:
        sys.modules["virne.utils.dataset"] = previous_dataset
    if previous_open_dict is None:
        omegaconf.__dict__.pop("open_dict", None)
    else:
        omegaconf.open_dict = previous_open_dict
    utils.generate_data_with_distribution = previous_utils_generation
    attribute_base.generate_data_with_distribution = previous_generation
    _RESTORE_STATE = None
    base_oracle.unload_oracle()


def generated_payload(value, module) -> str:
    cpu = [int(value.nodes[node]["cpu"]) for node in value.nodes]
    bandwidth = [int(value.edges[edge]["bandwidth"]) for edge in value.edges]
    return (
        f"nodes={value.number_of_nodes()};links={value.number_of_edges()};"
        f"cpu=[{','.join(map(str, cpu))}];"
        f"bandwidth=[{','.join(map(str, bandwidth))}];"
        f"py={random.getrandbits(32)};"
        f"np={int(module.np.random.randint(0, 2**32, dtype=module.np.uint32))}"
    )


def fixture_paths(module, root: pathlib.Path) -> dict[str, pathlib.Path]:
    loaded = module.nx.Graph()
    loaded.add_edge(0, 1)
    loaded.nodes[0]["loaded_node"] = 17
    loaded.nodes[1]["later_only"] = 99
    loaded.edges[0, 1]["loaded_link"] = 3.5
    loaded.graph["topology"] = "loaded-topology"
    loaded_path = root / "loaded.gml"
    module.nx.write_gml(loaded, loaded_path)

    empty_path = root / "empty.gml"
    module.nx.write_gml(module.nx.Graph(), empty_path)
    broken_path = root / "broken.gml"
    broken_path.write_text("not valid gml", encoding="utf-8")
    dataset_path = root / "dataset_input"
    dataset_path.mkdir()
    shutil.copyfile(loaded_path, dataset_path / "p_net.gml")
    return {
        "loaded": loaded_path,
        "empty": empty_path,
        "broken": broken_path,
        "missing": root / "missing.gml",
        "dataset": dataset_path,
    }


def oracle_cases(module, paths: dict[str, pathlib.Path]) -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []
    config = generated_config()
    for workers in (0, 1, 2, 8):
        value = module.PhysicalNetwork.from_setting(config, seed=77)
        cases.append((f"generated_w{workers}", generated_payload(value, module)))

    random.seed(31)
    module.np.random.seed(31)
    random.random()
    module.np.random.random()
    continued = module.PhysicalNetwork.from_setting(config, seed=None)
    cases.append(("continued", generated_payload(continued, module)))

    loaded_config = {
        "topology": {"file_path": str(paths["loaded"]), "type": "path"},
        "node_attrs_setting": [], "link_attrs_setting": [],
    }
    loaded = module.PhysicalNetwork.from_setting(loaded_config, seed=None)
    cases.append((
        "loaded",
        "origin=loaded"
        f";nodes={loaded.number_of_nodes()};links={loaded.number_of_edges()}"
        f";node_names=[{','.join(loaded.node_attrs.keys())}]"
        f";link_names=[{','.join(loaded.link_attrs.keys())}]"
        f";later={int('later_only' in loaded.node_attrs)}"
        f";node={int(loaded.nodes[0]['loaded_node'])}"
        f";link={double_token(loaded.edges[0, 1]['loaded_link'])}"
        f";topology={loaded.graph['topology']}"
        f";snapshots={int('node_attrs_setting' in loaded.graph or 'link_attrs_setting' in loaded.graph)}",
    ))

    empty = module.PhysicalNetwork.from_setting({
        "topology": {"file_path": str(paths["empty"]), "type": "path"},
        "node_attrs_setting": [], "link_attrs_setting": [],
    })
    cases.append((
        "loaded_empty",
        f"origin=loaded;nodes={empty.number_of_nodes()};links={empty.number_of_edges()}"
        f";node_names=[{','.join(empty.node_attrs.keys())}]"
        f";link_names=[{','.join(empty.link_attrs.keys())}]",
    ))

    fallback_config = {
        "topology": {
            "file_path": str(paths["broken"]), "num_nodes": 3, "type": "path"
        },
        "node_attrs_setting": [], "link_attrs_setting": [],
    }
    fallback = module.PhysicalNetwork.from_setting(fallback_config)
    cases.append((
        "broken_fallback",
        f"origin=fallback;error=1;nodes={fallback.number_of_nodes()};"
        f"links={fallback.number_of_edges()}",
    ))

    missing = module.PhysicalNetwork.from_setting({
        "topology": {
            "file_path": str(paths["missing"]), "num_nodes": 3, "type": "path"
        },
        "node_attrs_setting": [], "link_attrs_setting": [],
    })
    cases.append((
        "missing_fallback",
        f"origin=generated;requested=1;error=0;nodes={missing.number_of_nodes()};"
        f"links={missing.number_of_edges()}",
    ))

    failed = False
    try:
        module.PhysicalNetwork.from_setting({
            "topology": {"file_path": str(paths["broken"]), "type": "path"},
            "node_attrs_setting": [], "link_attrs_setting": [],
        })
    except Exception:
        failed = True
    cases.append(("broken_missing_num", f"error={int(failed)}"))

    dataset_loaded = module.PhysicalNetwork.load_dataset(str(paths["dataset"]))
    cases.append((
        "dataset_load",
        f"origin=loaded;nodes={dataset_loaded.number_of_nodes()};"
        f"links={dataset_loaded.number_of_edges()};requested=1",
    ))
    cases.append(("native_clone_move", "origin=loaded;nodes=2;node_binding=1"))
    return cases


def parse_cpp(
    harness: pathlib.Path, paths: dict[str, pathlib.Path]
) -> list[tuple[str, str]]:
    process = subprocess.run(
        [
            str(harness), str(paths["loaded"]), str(paths["empty"]),
            str(paths["broken"]), str(paths["missing"]),
            str(paths["dataset"]),
        ],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"PhysicalNetwork harness failed: {process.stderr.strip()}")
    result: list[tuple[str, str]] = []
    for line in process.stdout.splitlines():
        fields = line.split("|")
        if len(fields) != 3 or not fields[0].startswith("case=") or fields[1] != "ok":
            raise RuntimeError(f"malformed PhysicalNetwork line: {line!r}")
        result.append((fields[0][5:], bytes.fromhex(fields[2]).decode("utf-8")))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--base-source", type=pathlib.Path, required=True)
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    module = load_oracle(args.source, args.base_source)
    try:
        with tempfile.TemporaryDirectory(prefix="virne_physical_diff_") as raw_root:
            paths = fixture_paths(module, pathlib.Path(raw_root))
            expected = oracle_cases(module, paths)
            actual = parse_cpp(args.harness.resolve(), paths)
        numpy_version = module.np.__version__
        networkx_version = module.nx.__version__
    finally:
        unload_oracle()

    if actual != expected:
        for got, wanted in zip(actual, expected):
            if got != wanted:
                raise RuntimeError(
                    f"PhysicalNetwork mismatch {wanted[0]}: C++={got!r}, Python={wanted!r}"
                )
        raise RuntimeError(
            f"PhysicalNetwork inventory mismatch: C++={actual!r}, Python={expected!r}"
        )

    payload = {
        "source_sha256": SOURCE_SHA256,
        "base_source_sha256": base_oracle.SOURCE_SHA256,
        "numpy_version": numpy_version,
        "networkx_version": networkx_version,
        "shared_cases": [name for name, _ in expected[:-1]],
        "native_extension_cases": [expected[-1][0]],
        "python_only_boundaries": BOUNDARIES,
        "case_count": len(expected) + len(BOUNDARIES),
        "result": "PASS",
    }
    if args.output:
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        f"PhysicalNetwork differential: PASS ({len(expected) - 1} shared + "
        f"1 native + {len(BOUNDARIES)} boundaries)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
