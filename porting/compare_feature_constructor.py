#!/usr/bin/env python3
"""Exact dense ObservationHandler/FeatureConstructor parity gate.

The pinned leaves are AST-loaded without importing Virne, Gym, OmegaConf, or
their package initializers. Timing is optional and only measures repeated
``construct`` calls after one-time preparation on both sides.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional, Type

import networkx as nx
import numpy as np


FEATURE_SOURCE_SIZE = 19445
FEATURE_SOURCE_SHA256 = (
    "0f3c3377545a4b33ba8801415c2826cd7c1409301a0a2856111217d6a9245965"
)
OBS_SOURCE_SIZE = 20291
OBS_SOURCE_SHA256 = (
    "61ce92fdef5544ef3a4384872a13bede8c53521db53f8277439f3092a9e99058"
)
FIXTURE_VERSION = 1
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1
MAX_REPORTED_MISMATCHES = 48


def verify_source(
    source: pathlib.Path,
    expected_size: int,
    expected_sha256: str,
    label: str,
) -> bytes:
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != expected_size or digest != expected_sha256:
        raise RuntimeError(
            f"{label} source drift: size={len(payload)}, sha256={digest}; "
            f"expected size={expected_size}, sha256={expected_sha256}"
        )
    return payload


@dataclass(frozen=True)
class FixtureAttribute:
    name: str
    owner: str
    type: str
    originator: Optional[str] = None

    def get_data(self, network: "FixtureNetwork") -> list[float]:
        if self.owner == "node":
            return list(nx.get_node_attributes(network, self.name).values())
        return list(nx.get_edge_attributes(network, self.name).values())

    def get_aggregation_data(
        self,
        network: "FixtureNetwork",
        aggr: str = "sum",
        normalized: bool = False,
    ) -> np.ndarray:
        if normalized:
            raise RuntimeError("fixture does not use normalized adjacency")
        if aggr not in ("sum", "mean", "max", "min"):
            raise NotImplementedError(f"Aggregation '{aggr}' is not supported.")
        matrix = nx.to_numpy_array(
            network,
            nodelist=list(network.nodes),
            weight=self.name,
            dtype=float,
        )
        if aggr == "sum":
            return np.asarray(matrix.sum(axis=0))
        if aggr == "mean":
            return np.asarray(matrix.mean(axis=0))
        if aggr == "max":
            return matrix.max(axis=0)
        return matrix.min(axis=0)


class FixtureNetwork(nx.Graph):
    """Small BaseNetwork-compatible Graph with deterministic attribute order."""

    def __init__(
        self,
        node_attrs: Iterable[FixtureAttribute],
        link_attrs: Iterable[FixtureAttribute],
    ) -> None:
        super().__init__()
        self.node_attrs = {attribute.name: attribute for attribute in node_attrs}
        self.link_attrs = {attribute.name: attribute for attribute in link_attrs}

    @property
    def num_nodes(self) -> int:
        return self.number_of_nodes()

    @property
    def num_links(self) -> int:
        return self.number_of_edges()

    @property
    def links(self):
        return self.edges

    def get_node_attrs(
        self,
        types: Optional[Iterable[str]] = None,
        names: Optional[Iterable[str]] = None,
    ) -> list[FixtureAttribute]:
        if types is None and names is None:
            return list(self.node_attrs.values())
        if types is not None:
            selected = set(types)
            return [
                value for value in self.node_attrs.values()
                if value.type in selected
            ]
        selected = set(names or ())
        return [
            value for value in self.node_attrs.values()
            if value.name in selected
        ]

    def get_link_attrs(
        self,
        types: Optional[Iterable[str]] = None,
        names: Optional[Iterable[str]] = None,
    ) -> list[FixtureAttribute]:
        if types is None and names is None:
            return list(self.link_attrs.values())
        if types is not None:
            selected = set(types)
            return [
                value for value in self.link_attrs.values()
                if value.type in selected
            ]
        selected = set(names or ())
        return [
            value for value in self.link_attrs.values()
            if value.name in selected
        ]

    def get_node_attrs_data(
        self,
        attributes: list[FixtureAttribute] | list[str],
    ) -> list[list[float]]:
        if not attributes:
            return []
        if isinstance(attributes[0], str):
            return [
                list(nx.get_node_attributes(self, name).values())
                for name in attributes
            ]
        return [attribute.get_data(self) for attribute in attributes]

    def get_link_attrs_data(
        self,
        attributes: list[FixtureAttribute] | list[str],
    ) -> list[list[float]]:
        if not attributes:
            return []
        if isinstance(attributes[0], str):
            return [
                list(nx.get_edge_attributes(self, name).values())
                for name in attributes
            ]
        return [attribute.get_data(self) for attribute in attributes]

    def get_aggregation_attrs_data(
        self,
        attributes: list[FixtureAttribute],
        aggr: str = "sum",
        normalized: bool = False,
    ) -> list[np.ndarray]:
        return [
            attribute.get_aggregation_data(self, aggr, normalized)
            for attribute in attributes
        ]

    def get_subgraph_view(self, filter_node=None, filter_edge=None):
        return nx.subgraph_view(
            self,
            filter_node=filter_node,
            filter_edge=filter_edge,
        )


@dataclass
class FixtureTopologicalMetrics:
    node_degree_centrality: Optional[np.ndarray] = None
    node_closeness_centrality: Optional[np.ndarray] = None
    node_eigenvector_centrality: Optional[np.ndarray] = None
    node_betweenness_centrality: Optional[np.ndarray] = None


def fixed_topological_metrics(
    network: FixtureNetwork,
) -> FixtureTopologicalMetrics:
    node_ids = np.arange(network.num_nodes, dtype=np.float32)
    reverse = np.float32(network.num_nodes - 1) - node_ids
    return FixtureTopologicalMetrics(
        node_degree_centrality=(node_ids / np.float32(8.0)).reshape(-1, 1),
        node_closeness_centrality=(reverse / np.float32(8.0)).reshape(-1, 1),
        node_eigenvector_centrality=(
            np.mod(node_ids * np.float32(3.0), np.float32(8.0)) /
            np.float32(8.0)
        ).reshape(-1, 1),
        node_betweenness_centrality=(
            np.mod(node_ids * node_ids, np.float32(8.0)) /
            np.float32(8.0)
        ).reshape(-1, 1),
    )


@dataclass
class FixtureAttributeBenchmarks:
    node_attr_benchmarks: Dict[str, float]
    link_attr_benchmarks: Dict[str, float]
    link_sum_attr_benchmarks: Dict[str, float]


def attribute_benchmarks(
    attr_types: list[str],
    attrs: list[FixtureAttribute],
    data: np.ndarray,
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if "extrema" in attr_types:
        for attribute, values in zip(attrs, data):
            if attribute.type == "resource":
                continue
            key = attribute.originator or attribute.name
            result[key] = float(np.max(values))
    else:
        for attribute, values in zip(attrs, data):
            result[attribute.name] = float(np.max(values))
    return result


class FixtureAttributeBenchmarkManager:
    @classmethod
    def get_from_cache(cls, cache_key: str):
        del cache_key
        return None

    @staticmethod
    def get_benchmarks(
        network: FixtureNetwork,
        node_attrs: bool = True,
        link_attrs: bool = True,
        link_sum_attrs: bool = True,
    ) -> FixtureAttributeBenchmarks:
        types = ["resource", "extrema"]
        selected_nodes = network.get_node_attrs(types)
        selected_links = network.get_link_attrs(types)
        node_data = np.asarray(
            network.get_node_attrs_data(selected_nodes), dtype=np.float32)
        link_data = np.asarray(
            network.get_link_attrs_data(selected_links), dtype=np.float32)
        link_sum_data = np.asarray(
            network.get_aggregation_attrs_data(selected_links, aggr="sum"),
            dtype=np.float32,
        )
        return FixtureAttributeBenchmarks(
            node_attr_benchmarks=(
                attribute_benchmarks(types, selected_nodes, node_data)
                if node_attrs else {}
            ),
            link_attr_benchmarks=(
                attribute_benchmarks(types, selected_links, link_data)
                if link_attrs else {}
            ),
            link_sum_attr_benchmarks=(
                attribute_benchmarks(types, selected_links, link_sum_data)
                if link_sum_attrs else {}
            ),
        )


class FixtureTopologicalMetricCalculator:
    @classmethod
    def get_from_cache(cls, cache_key: str):
        del cache_key
        return None

    @staticmethod
    def calculate(
        network: FixtureNetwork,
        degree: bool = True,
        closeness: bool = True,
        eigenvector: bool = True,
        betweenness: bool = True,
    ) -> FixtureTopologicalMetrics:
        metrics = fixed_topological_metrics(network)
        if not degree:
            metrics.node_degree_centrality = None
        if not closeness:
            metrics.node_closeness_centrality = None
        if not eigenvector:
            metrics.node_eigenvector_centrality = None
        if not betweenness:
            metrics.node_betweenness_centrality = None
        return metrics


def assignment_names(node: ast.Assign) -> set[str]:
    return {
        target.id for target in node.targets
        if isinstance(target, ast.Name)
    }


def load_observation_handler(obs_source: pathlib.Path) -> type:
    payload = verify_source(
        obs_source, OBS_SOURCE_SIZE, OBS_SOURCE_SHA256, "ObservationHandler")
    tree = ast.parse(payload.decode("utf-8"), filename=str(obs_source))
    constants = {
        "MAX_NUM_V_NODES",
        "POSITIONAL_EMBEDDING_DIM",
        "P_NODE_STATUS_DIM",
        "V_NODE_STATUS_DIM",
        "V_NET_STATUS_DIM",
        "POSITIONAL_EMBEDDINGS",
    }
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "calc_positional_embeddings":
            selected.append(node)
        elif isinstance(node, ast.Assign) and assignment_names(node) & constants:
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "ObservationHandler":
            selected.append(node)
    if not any(
        isinstance(node, ast.ClassDef) and node.name == "ObservationHandler"
        for node in selected
    ):
        raise RuntimeError("pinned observation source has no ObservationHandler")
    namespace: dict[str, Any] = {
        "__name__": "_virne_observation_handler_oracle",
        "np": np,
        "nx": nx,
        "Dict": Dict,
        "Optional": Optional,
        "BaseNetwork": FixtureNetwork,
        "PhysicalNetwork": FixtureNetwork,
        "VirtualNetwork": FixtureNetwork,
        "TopologicalMetrics": FixtureTopologicalMetrics,
    }
    isolated = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(isolated)
    exec(compile(isolated, str(obs_source), "exec"), namespace)
    result = namespace.get("ObservationHandler")
    if not isinstance(result, type):
        raise RuntimeError("isolated ObservationHandler is not a class")
    return result


class UnusedSpaces:
    def __getattr__(self, name: str):
        raise RuntimeError(f"Gym space '{name}' is outside this exact dense gate")


def load_feature_types(
    feature_source: pathlib.Path,
    observation_handler: type,
) -> dict[str, type]:
    payload = verify_source(
        feature_source,
        FEATURE_SOURCE_SIZE,
        FEATURE_SOURCE_SHA256,
        "FeatureConstructor",
    )
    tree = ast.parse(payload.decode("utf-8"), filename=str(feature_source))
    class_names = {
        "BaseFeatureConstructor",
        "FeatureConstructorRegistry",
        "PNetFeatureConstructor",
        "PNetVNodeFeatureConstructor",
        "PNetVNetFeatureConstructor",
    }
    helper_names = {"get_selected_p_net_nodes", "get_placed_v_net_nodes"}
    selected = [
        node for node in tree.body
        if (
            isinstance(node, ast.ClassDef) and node.name in class_names
        ) or (
            isinstance(node, ast.FunctionDef) and node.name in helper_names
        )
    ]
    namespace: dict[str, Any] = {
        "__name__": "_virne_feature_constructor_oracle",
        "np": np,
        "spaces": UnusedSpaces(),
        "Any": Any,
        "Dict": Dict,
        "Optional": Optional,
        "Type": Type,
        "DictConfig": object,
        "VirtualNetwork": FixtureNetwork,
        "PhysicalNetwork": FixtureNetwork,
        "AttributeBenchmarkManager": FixtureAttributeBenchmarkManager,
        "AttributeBenchmarks": FixtureAttributeBenchmarks,
        "TopologicalMetrics": FixtureTopologicalMetrics,
        "TopologicalMetricCalculator": FixtureTopologicalMetricCalculator,
        "ObservationHandler": observation_handler,
        "Solution": dict,
        "Controller": object,
    }
    isolated = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(isolated)
    exec(compile(isolated, str(feature_source), "exec"), namespace)
    result = {name: namespace.get(name) for name in class_names}
    if any(not isinstance(value, type) for value in result.values()):
        raise RuntimeError("isolated feature source is missing an expected class")
    registry = result["FeatureConstructorRegistry"]
    if list(registry.list_registered()) != ["p_net", "p_net_v_node", "p_net_v_net"]:
        raise RuntimeError("isolated feature registry order/name drift")
    return result  # type: ignore[return-value]


def load_oracle(
    feature_source: pathlib.Path,
    obs_source: pathlib.Path,
) -> tuple[type, dict[str, type]]:
    forbidden_prefixes = ("virne", "gym", "omegaconf")
    before = {
        name for name in sys.modules
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in forbidden_prefixes
        )
    }
    observation_handler = load_observation_handler(obs_source)
    feature_types = load_feature_types(feature_source, observation_handler)
    after = {
        name for name in sys.modules
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in forbidden_prefixes
        )
    }
    imported = sorted(after - before)
    if imported:
        raise RuntimeError(
            "isolated feature oracle unexpectedly imported a forbidden "
            f"package: {imported}"
        )
    return observation_handler, feature_types


def physical_network() -> FixtureNetwork:
    node_attrs = [
        FixtureAttribute("cpu", "node", "resource"),
        FixtureAttribute("gpu", "node", "resource"),
        FixtureAttribute("ram", "node", "resource"),
        FixtureAttribute("max_cpu", "node", "extrema", "cpu"),
        FixtureAttribute("max_gpu", "node", "extrema", "gpu"),
        FixtureAttribute("max_ram", "node", "extrema", "ram"),
    ]
    link_attrs = [
        FixtureAttribute("bw", "link", "resource"),
        FixtureAttribute("latency", "link", "resource"),
        FixtureAttribute("max_bw", "link", "extrema", "bw"),
        FixtureAttribute("max_latency", "link", "extrema", "latency"),
    ]
    network = FixtureNetwork(node_attrs, link_attrs)
    cpu = [100, 80, 60, 90, 70, 50]
    gpu = [40, 60, 20, 80, 50, 30]
    ram = [256, 128, 192, 160, 224, 96]
    for node_id in range(6):
        network.add_node(
            node_id,
            cpu=cpu[node_id],
            gpu=gpu[node_id],
            ram=ram[node_id],
            max_cpu=cpu[node_id],
            max_gpu=gpu[node_id],
            max_ram=ram[node_id],
        )
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (4, 5), (5, 0), (1, 4), (2, 5),
    ]
    bw = [100, 80, 60, 90, 70, 50, 75, 65]
    latency = [8, 16, 24, 12, 20, 28, 10, 18]
    for index, edge in enumerate(edges):
        network.add_edge(
            *edge,
            bw=bw[index],
            latency=latency[index],
            max_bw=bw[index],
            max_latency=latency[index],
        )
    return network


def virtual_network() -> FixtureNetwork:
    node_attrs = [
        FixtureAttribute("cpu", "node", "resource"),
        FixtureAttribute("gpu", "node", "resource"),
        FixtureAttribute("ram", "node", "resource"),
    ]
    link_attrs = [
        FixtureAttribute("bw", "link", "resource"),
        FixtureAttribute("latency", "link", "resource"),
    ]
    network = FixtureNetwork(node_attrs, link_attrs)
    cpu = [20, 35, 15, 25]
    gpu = [5, 10, 15, 20]
    ram = [32, 48, 64, 40]
    for node_id in range(4):
        network.add_node(
            node_id,
            cpu=cpu[node_id],
            gpu=gpu[node_id],
            ram=ram[node_id],
        )
    edges = [(0, 1), (0, 2), (1, 2), (2, 3)]
    bw = [12, 20, 16, 24]
    latency = [4, 6, 8, 10]
    for index, edge in enumerate(edges):
        network.add_edge(
            *edge,
            bw=bw[index],
            latency=latency[index],
        )
    return network


def config(
    *,
    status: bool,
    aggregate: bool,
    degree: bool,
    more_topology: bool,
) -> SimpleNamespace:
    feature = SimpleNamespace(
        extracted_attr_types=["resource"],
        if_use_node_status_flags=status,
        if_use_aggregated_link_attrs=aggregate,
        if_use_degree_metric=degree,
        if_use_more_topological_metrics=more_topology,
    )
    return SimpleNamespace(rl=SimpleNamespace(feature_constructor=feature))


def raw_array_summary(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.name not in (
        "float32", "float64", "int32", "int64", "bool"):
        raise RuntimeError(f"unsupported feature tensor dtype: {array.dtype}")
    payload = array.tobytes(order="C")
    checksum = FNV_OFFSET
    for byte in payload:
        checksum ^= byte
        checksum = (checksum * FNV_PRIME) & MASK64
    width = array.dtype.itemsize
    bits = [
        format(
            int.from_bytes(
                payload[offset:offset + width], "little", signed=False),
            f"0{width * 2}x",
        )
        for offset in range(0, len(payload), width)
    ]
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.name,
        "checksum": str(checksum),
        "bytes": len(payload),
        "bits": bits,
    }


def summarize_outputs(outputs: Dict[str, Any]) -> Dict[str, dict[str, Any]]:
    return {name: raw_array_summary(value) for name, value in outputs.items()}


def observation_cases(
    observation_handler_type: type,
    p_net: FixtureNetwork,
    v_net: FixtureNetwork,
) -> list[dict[str, Any]]:
    handler = observation_handler_type()
    benchmarks = FixtureAttributeBenchmarkManager.get_benchmarks(p_net)
    slots = {0: 2, 1: 4}
    resource_types = ["resource"]
    p_metrics = fixed_topological_metrics(p_net)
    physical = {
        "node_order": handler.get_node_order_obs(p_net),
        "node_degree": handler.get_node_degree_obs(p_net),
        "node_attrs_norm": handler.get_node_attrs_obs(
            p_net, resource_types, benchmarks.node_attr_benchmarks),
        "link_pair": handler.get_link_pair_obs(p_net),
        "link_index": handler.get_link_index_obs(p_net),
        "link_attrs_norm": handler.get_link_attrs_obs(
            p_net, resource_types, benchmarks.link_attr_benchmarks),
        "link_aggr_min": handler.get_link_aggr_attrs_obs(
            p_net, resource_types, "min",
            link_attr_benchmarks=benchmarks.link_attr_benchmarks),
        "link_aggr_mean": handler.get_link_aggr_attrs_obs(
            p_net, resource_types, "mean",
            link_attr_benchmarks=benchmarks.link_attr_benchmarks),
        "link_aggr_max": handler.get_link_aggr_attrs_obs(
            p_net, resource_types, "max",
            link_attr_benchmarks=benchmarks.link_attr_benchmarks),
        "link_aggr_sum": handler.get_link_aggr_attrs_obs(
            p_net, resource_types, "sum",
            link_sum_attr_benchmarks=benchmarks.link_sum_attr_benchmarks),
        "link_filters": handler.get_link_filters(
            p_net, {"bw": 70, "latency": 10}),
        "average_distance_slots": handler.get_average_distance(
            p_net, slots, normalization=True),
        "average_distance_empty": handler.get_average_distance(
            p_net, {}, normalization=True),
        "p_status_none": handler.get_p_net_nodes_status(
            p_net, v_net, slots, None),
        "p_status_curr2": handler.get_p_net_nodes_status(
            p_net, v_net, slots, 2),
        "topology_all": handler.get_node_topological_metrics(
            p_net, p_metrics, True, True, True, True),
    }

    v_metrics = fixed_topological_metrics(v_net)
    virtual = {
        "node_attrs_norm": handler.get_node_attrs_obs(
            v_net, resource_types, benchmarks.node_attr_benchmarks),
        "link_pair": handler.get_link_pair_obs(v_net),
        "link_attrs_norm": handler.get_link_attrs_obs(
            v_net, resource_types, benchmarks.link_attr_benchmarks),
        "v_node_status_curr2": handler.get_v_node_status(
            v_net, 2, p_net.num_nodes),
        "v_node_demand_curr2": handler.get_v_node_demand(
            v_net, 2, resource_types, benchmarks.node_attr_benchmarks),
        "v_link_aggr_mean": handler.get_v_node_aggr_link_demands(
            v_net, 2, "mean", resource_types,
            benchmarks.link_attr_benchmarks),
        "v_link_aggr_max": handler.get_v_node_aggr_link_demands(
            v_net, 2, "max", resource_types,
            benchmarks.link_attr_benchmarks),
        "v_link_aggr_min": handler.get_v_node_aggr_link_demands(
            v_net, 2, "min", resource_types,
            benchmarks.link_attr_benchmarks),
        "v_link_aggr_sum": handler.get_v_node_aggr_link_demands(
            v_net, 2, "sum", resource_types,
            benchmarks.link_sum_attr_benchmarks),
        "v_status_none": handler.get_v_net_nodes_status(
            v_net, slots, None),
        "v_status_neighbor": handler.get_v_net_nodes_status(
            v_net, slots, 2, consist_decision=True, neighbor_flags=True),
        "v_status_end": handler.get_v_net_nodes_status(
            v_net, slots, v_net.num_nodes,
            consist_decision=True, neighbor_flags=False),
        "v_neighbor_flags": handler.get_v_node_neighbor_flags(
            v_net, slots, 2),
        "topology_all": handler.get_node_topological_metrics(
            v_net, v_metrics, True, True, True, True),
    }
    return [
        {
            "name": "obs_physical_multi_resource",
            "status": "ok",
            "outputs": summarize_outputs(physical),
        },
        {
            "name": "obs_virtual_multi_resource",
            "status": "ok",
            "outputs": summarize_outputs(virtual),
        },
    ]


def feature_case(
    name: str,
    constructor_type: type,
    p_net: FixtureNetwork,
    v_net: FixtureNetwork,
    settings: SimpleNamespace,
    slots: Dict[int, int],
    curr_v_node_id: int,
) -> dict[str, Any]:
    constructor = constructor_type(p_net, v_net, settings)
    observation = constructor.construct(
        p_net, v_net, {"node_slots": slots}, curr_v_node_id)
    return {
        "name": name,
        "status": "ok",
        "outputs": summarize_outputs(observation),
    }


def feature_cases(
    feature_types: dict[str, type],
    p_net: FixtureNetwork,
    v_net: FixtureNetwork,
) -> list[dict[str, Any]]:
    all_flags = config(
        status=True, aggregate=True, degree=True, more_topology=True)
    no_flags = config(
        status=False, aggregate=False, degree=False, more_topology=False)
    slots = {0: 2, 1: 4}
    result = [
        feature_case(
            "feature_p_net_all_flags",
            feature_types["PNetFeatureConstructor"],
            p_net, v_net, all_flags, slots, 2),
        feature_case(
            "feature_p_net_v_node_all_flags",
            feature_types["PNetVNodeFeatureConstructor"],
            p_net, v_net, all_flags, slots, 2),
        feature_case(
            "feature_p_net_v_net_all_flags",
            feature_types["PNetVNetFeatureConstructor"],
            p_net, v_net, all_flags, slots, 2),
        feature_case(
            "feature_no_optional_empty_slots",
            feature_types["PNetVNodeFeatureConstructor"],
            p_net, v_net, no_flags, {}, 0),
        feature_case(
            "feature_last_valid_empty_slots",
            feature_types["PNetVNodeFeatureConstructor"],
            p_net, v_net, all_flags, {}, v_net.num_nodes - 1),
    ]

    constructor = feature_types["PNetVNodeFeatureConstructor"](
        p_net, v_net, all_flags)
    end = constructor._construct_v_node_features(
        p_net, v_net, {"node_slots": slots}, v_net.num_nodes)
    result.append({
        "name": "feature_v_node_curr_end",
        "status": "ok",
        "outputs": {"v_node_x": raw_array_summary(end["x"])},
    })
    return result


def expected_report(
    observation_handler_type: type,
    feature_types: dict[str, type],
) -> dict[str, Any]:
    p_net = physical_network()
    v_net = virtual_network()
    cases = observation_cases(observation_handler_type, p_net, v_net)
    cases.extend(feature_cases(feature_types, p_net, v_net))
    return {
        "component": "feature_constructor",
        "fixture_version": FIXTURE_VERSION,
        "cases": cases,
    }


def run_native(
    harness: pathlib.Path,
    *,
    benchmark: bool = False,
    threads: int = 1,
    warmups: int = 2,
    repetitions: int = 5,
    iterations: int = 25,
) -> dict[str, Any]:
    command = [str(harness)]
    if benchmark:
        command.extend([
            "--benchmark",
            "--threads", str(threads),
            "--warmups", str(warmups),
            "--repetitions", str(repetitions),
            "--iterations", str(iterations),
        ])
    process = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"feature-constructor harness failed ({process.returncode}): "
            f"{process.stderr.strip()}"
        )
    lines = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"expected one native JSON line, got {lines!r}")
    report = json.loads(lines[0])
    if not isinstance(report, dict):
        raise RuntimeError("native feature report must be a JSON object")
    return report


def run_python_benchmark(
    feature_types: dict[str, type],
    *,
    warmups: int,
    repetitions: int,
    iterations: int,
) -> dict[str, Any]:
    """Measure only prepared Python ``construct`` calls.

    Constructor setup (attribute/topology benchmark discovery) is deliberately
    outside the timed region, matching the native harness.  The Python oracle
    remains single-threaded, as in the original implementation.
    """
    p_net = physical_network()
    v_net = virtual_network()
    settings = config(
        status=True, aggregate=True, degree=True, more_topology=True)
    constructor = feature_types["PNetVNodeFeatureConstructor"](
        p_net, v_net, settings)
    solution = {"node_slots": {0: 2, 1: 4}}

    for _ in range(warmups):
        for _ in range(iterations):
            constructor.construct(p_net, v_net, solution, 2)

    samples_ns: list[int] = []
    last: dict[str, Any] | None = None
    for _ in range(repetitions):
        begin = time.perf_counter_ns()
        for _ in range(iterations):
            last = constructor.construct(p_net, v_net, solution, 2)
        samples_ns.append(time.perf_counter_ns() - begin)
    if last is None:
        raise RuntimeError("Python benchmark produced no observation")
    return {
        "name": "construct_p_net_v_node_all_flags",
        "threads": 1,
        "warmups": warmups,
        "repetitions": repetitions,
        "iterations": iterations,
        "samples_ns": samples_ns,
        "output": raw_array_summary(last["p_net_x"]),
    }


def compare_benchmark(
    native: dict[str, Any],
    python: dict[str, Any],
) -> dict[str, Any]:
    native_benchmark = native.get("benchmark")
    if not isinstance(native_benchmark, dict):
        raise RuntimeError("native benchmark was requested but is missing")
    for field in ("name", "warmups", "repetitions", "iterations"):
        if native_benchmark.get(field) != python.get(field):
            raise RuntimeError(
                f"benchmark parameter mismatch for {field}: "
                f"native={native_benchmark.get(field)!r}, "
                f"python={python.get(field)!r}"
            )
    if native_benchmark.get("output") != python.get("output"):
        raise RuntimeError(
            "benchmark output checksum differs between Python and native"
        )
    iterations = int(python["iterations"])
    native_samples = [
        int(sample) / iterations
        for sample in native_benchmark["samples_ns"]
    ]
    python_samples = [
        int(sample) / iterations
        for sample in python["samples_ns"]
    ]
    native_median = statistics.median(native_samples)
    python_median = statistics.median(python_samples)
    return {
        "name": native_benchmark["name"],
        "threads": native_benchmark["threads"],
        "warmups": native_benchmark["warmups"],
        "repetitions": native_benchmark["repetitions"],
        "iterations": iterations,
        "native_samples_ns_per_call": native_samples,
        "python_samples_ns_per_call": python_samples,
        "native_median_ns_per_call": native_median,
        "python_median_ns_per_call": python_median,
        "python_over_native_speedup": (
            python_median / native_median
            if native_median > 0.0 else None
        ),
        "output_checksum": native_benchmark["output"]["checksum"],
    }


def short_value(value: Any) -> Any:
    if isinstance(value, list) and len(value) > 12:
        return value[:6] + [f"... {len(value) - 12} omitted ..."] + value[-6:]
    return value


def compare_exact(
    expected: Any,
    actual: Any,
    path: str,
    mismatches: list[dict[str, Any]],
) -> None:
    if len(mismatches) >= MAX_REPORTED_MISMATCHES:
        return
    if type(expected) is not type(actual):
        mismatches.append({
            "path": path,
            "python": short_value(expected),
            "cpp": short_value(actual),
            "reason": "type",
        })
        return
    if isinstance(expected, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            mismatches.append({
                "path": path,
                "python_keys": sorted(expected_keys),
                "cpp_keys": sorted(actual_keys),
                "reason": "keys",
            })
            if len(mismatches) >= MAX_REPORTED_MISMATCHES:
                return
        for key in sorted(expected_keys & actual_keys):
            compare_exact(expected[key], actual[key], f"{path}.{key}", mismatches)
        return
    if isinstance(expected, list):
        if len(expected) != len(actual):
            mismatches.append({
                "path": path,
                "python_length": len(expected),
                "cpp_length": len(actual),
                "reason": "length",
            })
            if len(mismatches) >= MAX_REPORTED_MISMATCHES:
                return
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare_exact(left, right, f"{path}[{index}]", mismatches)
        return
    if expected != actual:
        mismatches.append({
            "path": path,
            "python": expected,
            "cpp": actual,
            "reason": "value",
        })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--harness", "--native", dest="harness",
        type=pathlib.Path, required=True)
    parser.add_argument("--python-source", type=pathlib.Path, required=True)
    parser.add_argument("--obs-source", type=pathlib.Path, required=True)
    parser.add_argument(
        "--json-output", "--output", dest="json_output", type=pathlib.Path)
    parser.add_argument(
        "--benchmark", action="store_true",
        help="measure prepared construct calls on both implementations")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=25)
    args = parser.parse_args()

    for name in ("threads", "warmups", "repetitions", "iterations"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} requires a positive integer")

    if sys.byteorder != "little":
        raise RuntimeError("feature parity format requires a little-endian host")
    if not args.harness.is_file():
        raise RuntimeError(f"feature harness does not exist: {args.harness}")

    observation_handler, feature_types = load_oracle(
        args.python_source, args.obs_source)
    expected = expected_report(observation_handler, feature_types)
    actual = run_native(
        args.harness,
        benchmark=args.benchmark,
        threads=args.threads,
        warmups=args.warmups,
        repetitions=args.repetitions,
        iterations=args.iterations,
    )

    global_mismatches: list[dict[str, Any]] = []
    for field in ("component", "fixture_version"):
        compare_exact(
            expected[field], actual.get(field), field, global_mismatches)
    actual_cases = actual.get("cases")
    if not isinstance(actual_cases, list):
        raise RuntimeError("native feature report has no cases array")
    if len(actual_cases) != len(expected["cases"]):
        global_mismatches.append({
            "path": "cases",
            "python_length": len(expected["cases"]),
            "cpp_length": len(actual_cases),
            "reason": "length",
        })

    case_results = []
    for index, expected_case in enumerate(expected["cases"]):
        mismatches: list[dict[str, Any]] = []
        if index >= len(actual_cases):
            mismatches.append({
                "path": f"cases[{index}]",
                "python": expected_case["name"],
                "cpp": None,
                "reason": "missing",
            })
        else:
            compare_exact(
                expected_case,
                actual_cases[index],
                f"cases[{index}]",
                mismatches,
            )
        case_results.append({
            "name": expected_case["name"],
            "status": "PASS" if not mismatches else "FAIL",
            "mismatches": mismatches,
        })

    passed = sum(item["status"] == "PASS" for item in case_results)
    status = (
        "PASS"
        if not global_mismatches and passed == len(expected["cases"])
        else "FAIL"
    )
    benchmark_report: Any = "DEFERRED_UNTIL_EXACT"
    if args.benchmark:
        python_benchmark = run_python_benchmark(
            feature_types,
            warmups=args.warmups,
            repetitions=args.repetitions,
            iterations=args.iterations,
        )
        benchmark_report = compare_benchmark(actual, python_benchmark)
    report = {
        "component": "feature_constructor",
        "status": status,
        "fixture_version": FIXTURE_VERSION,
        "feature_source_size": FEATURE_SOURCE_SIZE,
        "feature_source_sha256": FEATURE_SOURCE_SHA256,
        "obs_source_size": OBS_SOURCE_SIZE,
        "obs_source_sha256": OBS_SOURCE_SHA256,
        "numpy_version": np.__version__,
        "networkx_version": nx.__version__,
        "benchmark": benchmark_report,
        "exact_cases": {
            "passed": passed,
            "total": len(expected["cases"]),
            "results": case_results,
        },
        "global_mismatches": global_mismatches,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output is not None:
        args.json_output.write_text(encoded, encoding="utf-8")

    compact: dict[str, Any] = {
        "component": "feature_constructor",
        "status": status,
        "exact_cases": f"{passed}/{len(expected['cases'])}",
        "fixture_version": FIXTURE_VERSION,
        "benchmark": benchmark_report,
    }
    if status != "PASS":
        compact["global_mismatches"] = global_mismatches
        compact["failed_cases"] = [
            item for item in case_results if item["status"] != "PASS"]
    print(json.dumps(compact, sort_keys=True, separators=(",", ":")))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
