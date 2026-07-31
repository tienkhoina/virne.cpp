#!/usr/bin/env python3
"""AST-isolated differential for every concrete node-rank solver."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import subprocess
from collections import deque
from types import SimpleNamespace
from typing import Any, Dict, Type, Union

import networkx as nx
import numpy as np


SOLVER_SHA256 = (
    "44a6f63f1a1798935453c057c6981a692f5dc4f03bde6a6535607aecbdc61389"
)
RANK_SHA256 = (
    "428f0deb188e685a9f3eb6177a70df533519f62c560284063fc4cf6ec7624fff"
)
WORKERS = (0, 1, 2, 8)
SOLVER_NAMES = (
    "order_rank",
    "random_rank",
    "grc_rank",
    "ffd_rank",
    "nrm_rank",
    "pl_rank",
    "nea_rank",
    "rw_rank",
)


class FakeSolverRegistry:
    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, solver_name, solver_type="unknown", env_cls=None):
        del env_cls

        def decorator(solver_cls):
            solver_cls.solver_name = solver_name
            solver_cls.type = solver_type
            cls._registry[solver_name] = solver_cls
            return solver_cls

        return decorator


class FakeSolver:
    def __init__(
        self, controller, recorder, counter, logger, config, **kwargs
    ):
        self.controller = controller
        self.recorder = recorder
        self.counter = counter
        self.logger = logger
        self.config = config
        self.reusable = config.reusable
        self.node_ranking_method = config.node_ranking_method
        self.link_ranking_method = config.link_ranking_method
        self.matching_mathod = kwargs.get(
            "matching_mathod", config.matching_mathod
        )
        self.shortest_method = kwargs.get(
            "shortest_method", config.shortest_method
        )
        self.k_shortest = kwargs.get("k_shortest", config.k_shortest)


class FakeSolution(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name, value):
        self[name] = value

    @classmethod
    def from_v_net(cls, virtual_network):
        return cls(
            v_net_id=virtual_network.request_id,
            v_net_lifetime=virtual_network.lifetime,
            v_net_arrival_time=virtual_network.arrival_time,
            v_net_num_nodes=virtual_network.num_nodes,
            v_net_num_edges=len(virtual_network.edges),
            result=False,
            place_result=True,
            route_result=True,
            node_slots={},
            link_paths={},
        )


class DenseCooFacade:
    def __init__(self, network):
        dense = np.zeros(
            (network.num_nodes, network.num_nodes), dtype=np.float64
        )
        for source, target in network.edges:
            dense[source, target] = 1.0
            dense[target, source] = 1.0
        self.row, self.col = np.nonzero(dense)
        self.data = dense[self.row, self.col]
        self._shape = dense.shape

    def tocoo(self):
        return self

    def nonzero(self):
        present = self.data != 0.0
        return self.row[present], self.col[present]

    def toarray(self):
        result = np.zeros(self._shape, dtype=np.float64)
        result[self.row, self.col] = self.data
        return result


def adjacency_matrix_without_scipy(network):
    return DenseCooFacade(network)


# The pinned minimal non-ML image intentionally has no SciPy. RW consumes only
# this tiny COO surface; keeping it local avoids importing or reimplementing a
# general sparse package while preserving the source algorithm's operations.
nx.adjacency_matrix = adjacency_matrix_without_scipy


class FakeNetwork(nx.Graph):
    def __init__(
        self,
        node_resources,
        edges,
        link_resources,
        *,
        request=False,
    ):
        super().__init__()
        self.add_nodes_from(range(len(node_resources)))
        self.add_edges_from(edges)
        self.num_nodes = len(node_resources)
        self.node_resources = list(node_resources)
        ordered_edges = list(self.edges)
        if len(ordered_edges) != len(link_resources):
            raise RuntimeError("edge/resource fixture length mismatch")
        self.link_resources = {
            self.edge_key(edge): value
            for edge, value in zip(ordered_edges, link_resources)
        }
        for edge, value in self.link_resources.items():
            self[edge[0]][edge[1]]["bandwidth"] = value
        self._node_token = object()
        self._link_token = object()
        if request:
            self.request_id = 31
            self.arrival_time = 2.0
            self.lifetime = 10.0

    @staticmethod
    def edge_key(edge):
        source, target = edge
        return (source, target) if source <= target else (target, source)

    @property
    def links(self):
        return list(self.edges)

    def get_node_attrs(self, _types):
        return self._node_token

    def get_link_attrs(self, _types):
        return self._link_token

    def get_node_attrs_data(self, token):
        if token is not self._node_token:
            raise RuntimeError("unexpected node attribute token")
        return [list(self.node_resources)]

    def adjacency_resource_matrix(self, normalized=False):
        matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float64)
        for (source, target), value in self.link_resources.items():
            matrix[source, target] = float(value)
            matrix[target, source] = float(value)
        if normalized:
            sums = np.abs(matrix).sum(axis=1)
            for row, total in enumerate(sums):
                if total != 0.0:
                    matrix[row, :] /= total
        return matrix

    def get_adjacency_attrs_data(self, token, normalized=False):
        if token is not self._link_token:
            raise RuntimeError("unexpected link attribute token")
        return [self.adjacency_resource_matrix(normalized)]

    def get_aggregation_attrs_data(
        self, token, aggr="sum", normalized=False
    ):
        if token is not self._link_token:
            raise RuntimeError("unexpected link attribute token")
        matrix = self.adjacency_resource_matrix(normalized)
        if aggr == "sum":
            row = matrix.sum(axis=0)
        elif aggr == "max":
            row = matrix.max(axis=0)
        elif aggr == "mean":
            row = matrix.mean(axis=0)
        elif aggr == "min":
            row = matrix.min(axis=0)
        else:
            raise NotImplementedError(aggr)
        return [np.asarray(row)]


class FakeNodeMapper:
    def node_mapping(
        self,
        v_net,
        p_net,
        *,
        sorted_v_nodes,
        sorted_p_nodes,
        solution,
        reusable,
        inplace,
        matching_mathod,
    ):
        assert not reusable and inplace and matching_mathod == "greedy"
        candidates = list(sorted_p_nodes)
        for virtual_node in sorted_v_nodes:
            selected = next(
                (
                    node
                    for node in candidates
                    if p_net.node_resources[node]
                    >= v_net.node_resources[virtual_node]
                ),
                None,
            )
            if selected is None:
                return False
            placed, _ = self.place(
                v_net, p_net, virtual_node, selected, solution
            )
            if not placed:
                return False
            candidates.remove(selected)
        return True

    def place(self, v_net, p_net, virtual_node, physical_node, solution):
        demand = v_net.node_resources[virtual_node]
        if p_net.node_resources[physical_node] < demand:
            return False, {}
        p_net.node_resources[physical_node] -= demand
        solution["node_slots"][virtual_node] = physical_node
        return True, {"cpu": demand}


class FakeLinkMapper:
    @staticmethod
    def path(p_net, source, target, demand):
        queue = deque([source])
        parent = {source: None}
        while queue:
            node = queue.popleft()
            if node == target:
                break
            for neighbor in p_net.adj[node]:
                edge = p_net.edge_key((node, neighbor))
                if (
                    neighbor not in parent
                    and p_net.link_resources[edge] >= demand
                ):
                    parent[neighbor] = node
                    queue.append(neighbor)
        if target not in parent:
            return None
        nodes = []
        node = target
        while node is not None:
            nodes.append(node)
            node = parent[node]
        nodes.reverse()
        return list(zip(nodes, nodes[1:]))

    def link_mapping(
        self,
        v_net,
        p_net,
        *,
        solution,
        sorted_v_links,
        shortest_method,
        k,
        inplace,
    ):
        del shortest_method, k
        assert inplace
        for virtual_link in sorted_v_links:
            source, target = virtual_link
            physical_source = solution["node_slots"][source]
            physical_target = solution["node_slots"][target]
            demand = v_net.link_resources[v_net.edge_key(virtual_link)]
            path = self.path(
                p_net, physical_source, physical_target, demand
            )
            if path is None:
                solution["link_paths"][tuple(virtual_link)] = []
                return False
            for link in path:
                edge = p_net.edge_key(link)
                p_net.link_resources[edge] -= demand
                p_net[edge[0]][edge[1]]["bandwidth"] = (
                    p_net.link_resources[edge]
                )
            solution["link_paths"][tuple(virtual_link)] = path
        return True


class FakeController:
    def __init__(self):
        self.node_mapper = FakeNodeMapper()
        self.link_mapper = FakeLinkMapper()
        self.shortest_method = "bfs_shortest"

    def find_candidate_nodes(
        self,
        v_net,
        p_net,
        virtual_node,
        filter=None,
        check_node_constraint=True,
        check_link_constraint=True,
    ):
        filtered = [] if filter is None else filter
        if check_node_constraint:
            suitable = [
                node
                for node in p_net.nodes
                if p_net.node_resources[node]
                >= v_net.node_resources[virtual_node]
            ]
            candidates = list(set(suitable).difference(set(filtered)))
        else:
            candidates = []
        if check_link_constraint:
            v_aggregate = v_net.get_aggregation_attrs_data(
                v_net.get_link_attrs(["resource"]), aggr="max"
            )[0]
            p_aggregate = p_net.get_aggregation_attrs_data(
                p_net.get_link_attrs(["resource"]), aggr="max"
            )[0]
            suitable = [
                node
                for node in p_net.nodes
                if p_net.degree[node] >= v_net.degree[virtual_node]
                and p_aggregate[node] >= v_aggregate[virtual_node]
            ]
            new_filter = set(p_net.nodes).difference(set(candidates))
            candidates = list(set(candidates).difference(new_filter))
        return candidates


def path_to_links(path):
    return list(zip(path, path[1:]))


def load_rankers(source: pathlib.Path):
    source = source.resolve()
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != RANK_SHA256:
        raise RuntimeError(f"node-rank source hash drift: {actual}")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    namespace: dict[str, Any] = {
        "__name__": "node_rank_solver_oracle",
        "abc": __import__("abc"),
        "Any": Any,
        "BaseNetwork": Any,
        "Dict": Dict,
        "Type": Type,
        "Union": Union,
        "np": np,
        "nx": nx,
    }
    isolated = ast.fix_missing_locations(
        ast.Module(body=classes, type_ignores=[])
    )
    exec(compile(isolated, str(source), "exec"), namespace)
    registry = namespace["NodeRankRegistry"]._registry
    if list(registry) != [
        "order",
        "random",
        "ffd",
        "nrm",
        "nea",
        "grc",
        "rw",
        "nps",
    ]:
        raise RuntimeError("node-rank registry drift")
    return registry


def load_solvers(source: pathlib.Path, rankers):
    source = source.resolve()
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != SOLVER_SHA256:
        raise RuntimeError(f"heuristic solver source hash drift: {actual}")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    expected = [
        "BaseNodeRankSolver",
        "OrderRankSolver",
        "RandomRankSolver",
        "GRCRankSolver",
        "FFDRankSolver",
        "NRMRankSolver",
        "PLRankSolver",
        "NEARankSolver",
        "RandomWalkRankSolver",
    ]
    if [node.name for node in classes] != expected:
        raise RuntimeError("heuristic solver class inventory drift")
    FakeSolverRegistry._registry = {}
    namespace: dict[str, Any] = {
        "__name__": "heuristic_node_rank_variants_oracle",
        "nx": nx,
        "path_to_links": path_to_links,
        "Controller": object,
        "Recorder": object,
        "Counter": object,
        "Logger": object,
        "PhysicalNetwork": object,
        "VirtualNetwork": object,
        "Solution": FakeSolution,
        "Solver": FakeSolver,
        "SolverRegistry": FakeSolverRegistry,
        "OrderNodeRank": rankers["order"],
        "RandomNodeRank": rankers["random"],
        "GRCNodeRank": rankers["grc"],
        "FFDNodeRank": rankers["ffd"],
        "NRMNodeRank": rankers["nrm"],
        "DegreeWeightedResoureNodeRank": rankers["nea"],
        "RWNodeRank": rankers["rw"],
    }
    isolated = ast.fix_missing_locations(
        ast.Module(body=classes, type_ignores=[])
    )
    exec(compile(isolated, str(source), "exec"), namespace)
    if tuple(FakeSolverRegistry._registry) != SOLVER_NAMES:
        raise RuntimeError(
            f"heuristic registry drift: {tuple(FakeSolverRegistry._registry)}"
        )
    return FakeSolverRegistry._registry


def virtual_network():
    return FakeNetwork([3, 2], [(0, 1)], [2], request=True)


def physical_network():
    return FakeNetwork(
        [10, 12, 9], [(0, 1), (1, 2), (0, 2)], [10, 8, 6]
    )


def sparse_tie_virtual_network():
    network = FakeNetwork([10], [], [], request=True)
    network.request_id = 37
    network.arrival_time = 0.0
    network.lifetime = 1.0
    return network


def sparse_tie_physical_network():
    return FakeNetwork([0, 10, 0, 0, 0, 0, 0, 0, 10], [], [])


def solver_config():
    return SimpleNamespace(
        reusable=False,
        node_ranking_method="random",
        link_ranking_method="ffd",
        matching_mathod="greedy",
        shortest_method="bfs_shortest",
        k_shortest=10,
    )


def canonical_solution(solution):
    return {
        "result": solution["result"],
        "place_result": solution["place_result"],
        "route_result": solution["route_result"],
        "node_slots": [list(item) for item in solution["node_slots"].items()],
        "link_paths": [
            {
                "virtual": list(link),
                "path": [list(path_link) for path_link in path],
            }
            for link, path in solution["link_paths"].items()
        ],
    }


def python_payload(solvers, workers):
    results = []
    for solver_id, name in enumerate(SOLVER_NAMES):
        if name == "random_rank":
            np.random.seed(77)
        instance = solvers[name](
            FakeController(),
            object(),
            object(),
            object(),
            solver_config(),
            matching_mathod="greedy",
            shortest_method="bfs_shortest",
            k_shortest=10,
        )
        solution = instance.solve(
            {"v_net": virtual_network(), "p_net": physical_network()}
        )
        next_value = None
        if name == "random_rank":
            next_value = int(
                np.random.randint(0, 2**32, dtype=np.uint32)
            )
        results.append(
            {
                "solver": name,
                "solver_id": solver_id,
                "category": "node_ranking",
                "solution": canonical_solution(solution),
                "rng_next": next_value,
            }
        )
    sparse_candidate_ties = []
    for name in ("pl_rank", "nea_rank"):
        instance = solvers[name](
            FakeController(),
            object(),
            object(),
            object(),
            solver_config(),
            matching_mathod="greedy",
            shortest_method="bfs_shortest",
            k_shortest=10,
        )
        solution = instance.solve(
            {
                "v_net": sparse_tie_virtual_network(),
                "p_net": sparse_tie_physical_network(),
            }
        )
        sparse_candidate_ties.append(
            {
                "solver": name,
                "solution": canonical_solution(solution),
            }
        )

    return {
        "component": "solver.heuristic.node_rank.variants",
        "workers": workers,
        "solvers": results,
        "sparse_candidate_ties": sparse_candidate_ties,
    }


def native_payload(harness: pathlib.Path, workers: int):
    completed = subprocess.run(
        [str(harness), "--workers", str(workers)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"native workers={workers} failed:\n{completed.stderr}"
        )
    payload = json.loads(completed.stdout)
    if payload.pop("native_input_physical_unchanged", None) is not True:
        raise RuntimeError("native clone-isolation proof failed")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver-source", required=True, type=pathlib.Path)
    parser.add_argument("--rank-source", required=True, type=pathlib.Path)
    parser.add_argument("--harness", required=True, type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    rankers = load_rankers(args.rank_source)
    solvers = load_solvers(args.solver_source, rankers)
    for workers in WORKERS:
        expected = python_payload(solvers, workers)
        actual = native_payload(args.harness.resolve(), workers)
        if actual != expected:
            raise RuntimeError(
                f"workers={workers} mismatch\n"
                f"python={json.dumps(expected, sort_keys=True)}\n"
                f"native={json.dumps(actual, sort_keys=True)}"
            )

    result = {
        "component": "solver.heuristic.node_rank.variants",
        "status": "PASS",
        "solver_source_sha256": SOLVER_SHA256.upper(),
        "rank_source_sha256": RANK_SHA256.upper(),
        "solvers": list(SOLVER_NAMES),
        "native_workers": list(WORKERS),
        "shared_case_count": len(SOLVER_NAMES) + 2,
        "boundaries": {
            "python_isolation": (
                "Only the pinned solver/ranker class AST executes; Virne "
                "package initialization and learning imports are excluded."
            ),
            "dependencies": (
                "Frozen mapper/rank dependency gates own their full numeric "
                "case matrices; this differential locks concrete wiring, "
                "mapping order, custom candidate policy and RNG continuation."
            ),
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
