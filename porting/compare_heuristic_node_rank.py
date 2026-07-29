#!/usr/bin/env python3
"""Compact AST-isolated differential for heuristic OrderRankSolver."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import subprocess
from collections import deque
from types import SimpleNamespace
from typing import Any


SOURCE_SHA256 = (
    "44a6f63f1a1798935453c057c6981a692f5dc4f03bde6a6535607aecbdc61389"
)
WORKERS = (0, 1, 2, 8)


class FakeNodeRankException(RuntimeError):
    pass


class FakeSolver:
    def __init__(
        self,
        controller,
        recorder,
        counter,
        logger,
        config,
        **kwargs,
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
            "matching_mathod", config.matching_mathod)
        self.shortest_method = kwargs.get(
            "shortest_method", config.shortest_method)
        self.k_shortest = kwargs.get("k_shortest", config.k_shortest)


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


class FakeOrderNodeRank:
    def __call__(self, network):
        if network.num_nodes == 0:
            raise FakeNodeRankException("empty order rank")
        score = 1.0 / network.num_nodes
        return {node: score for node in network.nodes}


class FakeSolution(dict):
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


class FakeNetwork:
    def __init__(
        self,
        node_count: int,
        edges,
        node_resources,
        link_resources,
        *,
        request=False,
    ):
        self.num_nodes = node_count
        self.nodes = list(range(node_count))
        self.edges = [tuple(edge) for edge in edges]
        self.node_resources = list(node_resources)
        self.link_resources = {
            self.edge_key(edge): int(value)
            for edge, value in zip(self.edges, link_resources)
        }
        self.adjacency = [[] for _ in range(node_count)]
        for source, target in self.edges:
            self.adjacency[source].append(target)
            self.adjacency[target].append(source)
        if request:
            self.request_id = 17
            self.arrival_time = 2.0
            self.lifetime = 10.0

    @staticmethod
    def edge_key(edge):
        source, target = edge
        return (source, target) if source <= target else (target, source)

    @property
    def links(self):
        return list(self.edges)


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
        assert not reusable and inplace
        solution["node_slots"].clear()
        candidates = list(sorted_p_nodes)
        for virtual_node in sorted_v_nodes:
            selected = None
            if matching_mathod == "l2s2":
                if candidates:
                    candidate = candidates[0]
                    if (p_net.node_resources[candidate] >=
                            v_net.node_resources[virtual_node]):
                        selected = candidate
            else:
                for candidate in candidates:
                    if (p_net.node_resources[candidate] >=
                            v_net.node_resources[virtual_node]):
                        selected = candidate
                        break
            if selected is None:
                solution["place_result"] = False
                solution["result"] = False
                return False
            p_net.node_resources[selected] -= v_net.node_resources[virtual_node]
            solution["node_slots"][virtual_node] = selected
            candidates.remove(selected)
        return True


class FakeLinkMapper:
    @staticmethod
    def shortest_feasible_path(p_net, source, target, demand):
        queue = deque([source])
        parent = {source: None}
        while queue:
            current = queue.popleft()
            if current == target:
                break
            for neighbor in p_net.adjacency[current]:
                edge = p_net.edge_key((current, neighbor))
                if (neighbor not in parent and
                        p_net.link_resources[edge] >= demand):
                    parent[neighbor] = current
                    queue.append(neighbor)
        if target not in parent:
            return None
        nodes = []
        current = target
        while current is not None:
            nodes.append(current)
            current = parent[current]
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
        solution["link_paths"].clear()
        for virtual_link in sorted_v_links:
            source, target = virtual_link
            physical_source = solution["node_slots"][source]
            physical_target = solution["node_slots"][target]
            demand = v_net.link_resources[v_net.edge_key(virtual_link)]
            path = self.shortest_feasible_path(
                p_net, physical_source, physical_target, demand)
            if path is None:
                solution["link_paths"][tuple(virtual_link)] = []
                return False
            for physical_link in path:
                p_net.link_resources[p_net.edge_key(physical_link)] -= demand
            solution["link_paths"][tuple(virtual_link)] = path
        return True


class FakeController:
    def __init__(self):
        self.node_mapper = FakeNodeMapper()
        self.link_mapper = FakeLinkMapper()


def load_target(source: pathlib.Path):
    source = source.resolve()
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"heuristic node-rank source hash drift: {actual}")

    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    inventory = [node.name for node in classes]
    expected_inventory = [
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
    if inventory != expected_inventory:
        raise RuntimeError(
            f"heuristic node-rank class inventory drift: {inventory}")

    selected = [
        node for node in classes
        if node.name in {"BaseNodeRankSolver", "OrderRankSolver"}
    ]
    FakeSolverRegistry._registry = {}
    namespace: dict[str, Any] = {
        "__name__": "heuristic_node_rank_isolated",
        "Controller": object,
        "Recorder": object,
        "Counter": object,
        "Logger": object,
        "VirtualNetwork": object,
        "PhysicalNetwork": object,
        "Solution": FakeSolution,
        "Solver": FakeSolver,
        "SolverRegistry": FakeSolverRegistry,
        "OrderNodeRank": FakeOrderNodeRank,
    }
    isolated = ast.fix_missing_locations(
        ast.Module(body=selected, type_ignores=[]))
    exec(compile(isolated, str(source), "exec"), namespace)
    registered = FakeSolverRegistry._registry
    if list(registered) != ["order_rank"]:
        raise RuntimeError(f"isolated registry drift: {list(registered)}")
    order_solver = registered["order_rank"]
    if order_solver.type != "node_ranking":
        raise RuntimeError(f"isolated category drift: {order_solver.type}")
    return order_solver


def virtual_network(node_count, edges, node_demands, link_demands):
    return FakeNetwork(
        node_count, edges, node_demands, link_demands, request=True)


def physical_network(node_count, edges, node_capacities, link_capacities):
    return FakeNetwork(node_count, edges, node_capacities, link_capacities)


def cases():
    return [
        (
            "no_link_success",
            virtual_network(1, [], [3], []),
            physical_network(1, [], [10], []),
            False,
        ),
        (
            "link_success",
            virtual_network(2, [(0, 1)], [2, 3], [1]),
            physical_network(3, [(0, 1), (1, 2)], [10, 10, 10], [10, 10]),
            False,
        ),
        (
            "node_failure_after_one",
            virtual_network(2, [], [1, 100], []),
            physical_network(2, [], [10, 10], []),
            False,
        ),
        (
            "disconnected_route_failure",
            virtual_network(2, [(0, 1)], [1, 1], [1]),
            physical_network(2, [], [10, 10], []),
            False,
        ),
        (
            "rank_config_ignored",
            virtual_network(2, [(0, 1)], [2, 3], [1]),
            physical_network(3, [(0, 1), (1, 2)], [10, 10, 10], [10, 10]),
            True,
        ),
    ]


def solver_config(ignored_rank_config):
    return SimpleNamespace(
        reusable=True,
        node_ranking_method=("random" if ignored_rank_config else "order"),
        link_ranking_method=("ffd" if ignored_rank_config else "order"),
        matching_mathod="greedy",
        shortest_method="k_shortest",
        k_shortest=10,
    )


def canonical_solution(name, solution):
    return {
        "name": name,
        "kind": "solution",
        "metadata": {
            "v_net_id": solution["v_net_id"],
            "lifetime": solution["v_net_lifetime"],
            "arrival_time": solution["v_net_arrival_time"],
            "num_nodes": solution["v_net_num_nodes"],
            "num_edges": solution["v_net_num_edges"],
        },
        "result": solution["result"],
        "place_result": solution["place_result"],
        "route_result": solution["route_result"],
        "node_slots": [
            [virtual_node, physical_node]
            for virtual_node, physical_node
            in solution["node_slots"].items()
        ],
        "link_paths": [
            {
                "virtual": list(virtual_link),
                "path": [list(link) for link in path],
            }
            for virtual_link, path in solution["link_paths"].items()
        ],
    }


def python_payload(order_solver, workers):
    results = []
    for name, v_net, p_net, ignored_rank_config in cases():
        config = solver_config(ignored_rank_config)
        instance = order_solver(
            FakeController(),
            object(),
            object(),
            object(),
            config,
            matching_mathod=config.matching_mathod,
            shortest_method=config.shortest_method,
            k_shortest=config.k_shortest,
        )
        solution = instance.solve({"v_net": v_net, "p_net": p_net})
        results.append(canonical_solution(name, solution))

    empty_config = solver_config(False)
    empty_solver = order_solver(
        FakeController(), object(), object(), object(), empty_config)
    try:
        empty_solver.solve({
            "v_net": virtual_network(0, [], [], []),
            "p_net": physical_network(0, [], [], []),
        })
    except FakeNodeRankException:
        results.append({
            "name": "empty_virtual",
            "kind": "exception",
            "exception": {
                "type": "NodeRankException",
                "operation": "reduce",
                "stage": "virtual_rank",
            },
        })
    else:
        raise RuntimeError("isolated empty virtual rank did not raise")

    return {
        "component": "solver.heuristic.node_rank",
        "solver": "order_rank",
        "category": "node_ranking",
        "solver_id": 0,
        "workers": workers,
        "cases": results,
    }


def run_native(harness: pathlib.Path, workers: int):
    completed = subprocess.run(
        [str(harness), "--workers", str(workers)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"native workers={workers} failed ({completed.returncode}):\n"
            f"{completed.stderr}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"native workers={workers} emitted invalid JSON:\n"
            f"{completed.stdout}\n{completed.stderr}") from error
    if payload.pop("native_input_physical_unchanged", None) is not True:
        raise RuntimeError(
            f"native workers={workers} did not prove clone isolation")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=pathlib.Path)
    parser.add_argument("--harness", required=True, type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    order_solver = load_target(args.source)
    for workers in WORKERS:
        expected = python_payload(order_solver, workers)
        actual = run_native(args.harness.resolve(), workers)
        if actual != expected:
            raise RuntimeError(
                f"workers={workers} mismatch\n"
                f"python={json.dumps(expected, sort_keys=True)}\n"
                f"native={json.dumps(actual, sort_keys=True)}")

    result = {
        "component": "solver.heuristic.node_rank",
        "solver": "order_rank",
        "status": "PASS",
        "source_sha256": SOURCE_SHA256.upper(),
        "solution_case_count": 5,
        "shared_case_count": 6,
        "native_workers": list(WORKERS),
        "boundaries": {
            "native_clone": (
                "Native solves on one private physical clone and proves the "
                "const input resources unchanged; Python mapper mutation is "
                "not a shared SolverInstance observable."),
            "dependency_scope": (
                "NodeRank, constraint/resource checks, path search and mapper "
                "numeric tables remain covered by their frozen differentials; "
                "this gate isolates solver ordering and returned mappings."),
            "python_isolation": (
                "Only exact BaseNodeRankSolver and OrderRankSolver AST nodes "
                "execute with minimal deterministic dependency fakes; no "
                "learning package is imported."),
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
