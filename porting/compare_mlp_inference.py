#!/usr/bin/env python3
"""Exact Python/LibTorch parity and one end-to-end MLP inference timing."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any

import torch

import compare_mlp_policy as policy_oracle
import compare_policy_action as action_oracle


COUNT_BATCH = 32
COUNT_NODES = 500
COUNT_P_FEATURES = 14
COUNT_V_FEATURES = 11
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


def source_paths() -> tuple[pathlib.Path, pathlib.Path]:
    learning = (
        pathlib.Path(__file__).resolve().parents[2]
        / "virne"
        / "virne"
        / "solver"
        / "learning"
    )
    return learning / "rl_policy" / "mlp_policy.py", learning / "neural_network" / "mlp.py"


def reduced_summary(value: torch.Tensor, *, include_bits: bool = True) -> dict[str, Any]:
    summary = policy_oracle.tensor_summary(value, include_bits=include_bits)
    return {
        "shape": summary["shape"],
        "dtype": summary["dtype"],
        "checksum": summary["checksum"],
        "bits": summary["bits"],
    }


def make_raw_config(
    batch: int,
    nodes: int,
    p_features: int,
    v_features: int,
    with_mask: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    p_count = batch * nodes * p_features
    v_count = batch * v_features
    p_net_x = (
        torch.arange(p_count, dtype=torch.float32)
        .reshape(batch, nodes, p_features)
        .remainder(47.0)
        .sub(23.0)
        .div(16.0)
    )
    v_node_x = (
        torch.arange(v_count, dtype=torch.float32)
        .remainder(47.0)
        .sub(23.0)
        .div(16.0)
        .reshape(batch, v_features)
    )
    mask = None
    if with_mask:
        mask = torch.remainder(
            torch.arange(batch * nodes, dtype=torch.float32), 5.0
        ).ne(0).reshape(batch, nodes)
    return p_net_x, v_node_x, mask


def make_config(
    batch: int,
    nodes: int,
    p_features: int,
    v_features: int,
    with_mask: bool,
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
    p_net_x, v_node_x, mask = make_raw_config(
        batch, nodes, p_features, v_features, with_mask
    )
    combined = torch.cat(
        [p_net_x, v_node_x.unsqueeze(1).expand(-1, nodes, -1)], dim=-1
    )
    observation: dict[str, torch.Tensor] = {"p_net_x": combined}
    if mask is not None:
        observation["action_mask"] = mask
    return observation, mask


def make_solver(model: torch.nn.Module, temperature: float = 0.75) -> Any:
    class PolicyAdapter:
        def __init__(self, value: torch.nn.Module) -> None:
            self.value = value

        def act(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
            return self.value.actor(observation)

        def evaluate(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
            return self.value.critic(observation)

    solver = action_oracle.ORACLE_RL_SOLVER()
    solver.policy = PolicyAdapter(model)
    solver.softmax_temp = temperature
    solver.config = SimpleNamespace(
        rl=SimpleNamespace(mask_actions=True, maskable_policy=True)
    )
    return solver


def run_case(
    name: str,
    batch: int,
    nodes: int,
    p_features: int,
    v_features: int,
    with_mask: bool,
) -> dict[str, Any]:
    torch.manual_seed(7)
    policy_source, mlp_source = source_paths()
    actor_critic, _, _ = policy_oracle.load_policy_types(
        policy_source, mlp_source
    )
    model = actor_critic(
        p_features + v_features,
        nodes,
        num_layers=3,
        embedding_dim=8,
        dropout_prob=0.0,
        batch_norm=False,
    )
    policy_oracle.fill_state(model)
    model.eval()
    observation, _mask = make_config(
        batch, nodes, p_features, v_features, with_mask
    )
    solver = make_solver(model)
    with torch.inference_mode():
        logits = model.actor(observation)
        values = model.critic(observation)
        probabilities, _candidate = solver.get_action_prob_dist(observation)
        numpy_actions, _numpy_log_probabilities = solver.select_action(
            observation, sample=False
        )
        if hasattr(numpy_actions, "shape"):
            actions = torch.from_numpy(numpy_actions).to(torch.int64)
        else:
            actions = torch.tensor([int(numpy_actions)], dtype=torch.int64)
        _eval_values, log_probabilities, entropy, _other = solver.evaluate_actions(
            observation, actions, return_others=True
        )
    return {
        "name": name,
        "logits": reduced_summary(logits),
        "values": reduced_summary(values),
        "probabilities": reduced_summary(probabilities),
        "actions": reduced_summary(actions),
        "log_probabilities": reduced_summary(log_probabilities),
        "entropy": reduced_summary(entropy),
    }


def exact_cases() -> list[dict[str, Any]]:
    return [
        run_case("single_masked_greedy", 1, 7, 4, 3, True),
        run_case("batch_masked_greedy", 4, 17, 4, 3, True),
        run_case("batch_unmasked_greedy", 3, 9, 5, 2, False),
    ]


def fnv_mix(checksum: int, value: torch.Tensor) -> int:
    payload = bytes(
        value.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8).tolist()
    )
    for byte in payload:
        checksum ^= byte
        checksum = (checksum * FNV_PRIME) & MASK64
    return checksum


def benchmark_python(warmups: int, repetitions: int) -> dict[str, Any]:
    torch.manual_seed(19)
    policy_source, mlp_source = source_paths()
    actor_critic, _, _ = policy_oracle.load_policy_types(
        policy_source, mlp_source
    )
    model = actor_critic(
        COUNT_P_FEATURES + COUNT_V_FEATURES,
        COUNT_NODES,
        num_layers=3,
        embedding_dim=128,
        dropout_prob=0.0,
        batch_norm=False,
    )
    policy_oracle.fill_state(model)
    model.eval()
    raw_p_net_x, raw_v_node_x, raw_mask = make_raw_config(
        COUNT_BATCH, COUNT_NODES, COUNT_P_FEATURES, COUNT_V_FEATURES, True
    )
    actions = torch.zeros(COUNT_BATCH, dtype=torch.int64)
    solver = make_solver(model, temperature=1.0)

    def run_once() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        observation = {
            "p_net_x": torch.cat(
                [
                    raw_p_net_x,
                    raw_v_node_x.unsqueeze(1).expand(-1, COUNT_NODES, -1),
                ],
                dim=-1,
            ),
            "action_mask": raw_mask,
        }
        values, log_probabilities, entropy, _other = solver.evaluate_actions(
            observation, actions, return_others=True
        )
        logits = model.actor(observation)
        return logits, values, log_probabilities, entropy

    checksum: int | None = None
    for _ in range(warmups):
        output = run_once()
        current = fnv_mix(
            fnv_mix(
                fnv_mix(
                    fnv_mix(FNV_OFFSET, output[0]), output[1]
                ), output[2]
            ), output[3]
        )
        checksum = current if checksum is None else checksum
        if current != checksum:
            raise RuntimeError("Python MLP inference checksum drift")
    samples: list[int] = []
    with torch.inference_mode():
        for _ in range(repetitions):
            started = time.perf_counter_ns()
            output = run_once()
            elapsed = time.perf_counter_ns() - started
            samples.append(elapsed)
            current = fnv_mix(
                fnv_mix(
                    fnv_mix(
                        fnv_mix(FNV_OFFSET, output[0]), output[1]
                    ), output[2]
                ), output[3]
            )
            if current != checksum:
                raise RuntimeError("Python MLP inference checksum drift")
    return {
        "samples_ns": samples,
        "checksum": str(checksum),
        "component_checksums": {
            "logits": reduced_summary(output[0], include_bits=False)["checksum"],
            "values": reduced_summary(output[1], include_bits=False)["checksum"],
            "log_probabilities": reduced_summary(output[2], include_bits=False)["checksum"],
            "entropy": reduced_summary(output[3], include_bits=False)["checksum"],
        },
    }


def run_native(path: pathlib.Path, benchmark: bool, warmups: int, repetitions: int) -> dict[str, Any]:
    command = [
        str(path.resolve()),
        "--warmups", str(warmups),
        "--repetitions", str(repetitions),
    ]
    if benchmark:
        command.append("--benchmark")
    process = subprocess.run(
        command, check=True, capture_output=True, text=True
    )
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", "--harness", required=True, type=pathlib.Path)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--json-output", "--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.warmups <= 0 or args.repetitions <= 0:
        parser.error("warmups and repetitions must be positive")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    expected = {"component": "mlp_inference", "cases": exact_cases()}
    native = run_native(args.native, False, args.warmups, args.repetitions)
    mismatches = [] if native == expected else [{
        "path": "exact", "python": expected, "cpp": native,
    }]
    report: dict[str, Any] = {
        "component": "mlp_inference",
        "fixture_version": 1,
        "oracle_backend": "isolated_torch26_mlp_policy_and_rl_solver_methods",
        "python_source": {
            "mlp_policy_sha256": policy_oracle.POLICY_SOURCE_SHA256,
            "mlp_net_sha256": policy_oracle.MLP_SOURCE_SHA256,
            "rl_solver": action_oracle.RL_SOURCE_METADATA,
            "mask": action_oracle.MASK_SOURCE_METADATA,
        },
        "exact_cases": "3/3" if not mismatches else "FAIL",
        "mismatches": mismatches,
        "benchmark_mismatches": [],
        "benchmark": "NOT_REQUESTED",
        "status": "PASS" if not mismatches else "FAIL",
    }
    if mismatches:
        output = json.dumps(report, indent=2, sort_keys=True)
        if args.json_output:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(output + "\n", encoding="utf-8")
        print(json.dumps({"component": report["component"], "status": "FAIL"}))
        return 1

    if args.benchmark:
        native_timed = run_native(
            args.native, True, args.warmups, args.repetitions
        )["benchmark"]
        python_timed = benchmark_python(args.warmups, args.repetitions)
        native_checksums = {
            "logits": native_timed["logits_checksum"],
            "values": native_timed["values_checksum"],
            "log_probabilities": native_timed["log_prob_checksum"],
            "entropy": native_timed["entropy_checksum"],
        }
        if any(
            str(native_checksums[key])
            != str(python_timed["component_checksums"][key])
            for key in native_checksums
        ):
            report["benchmark_mismatches"] = [{
                "path": "component_checksums",
                "python": python_timed["component_checksums"],
                "cpp": native_checksums,
            }]
            report["status"] = "FAIL"
        report["benchmark"] = {
            "name": native_timed["name"],
            "batch": COUNT_BATCH,
            "physical_nodes": COUNT_NODES,
            "native_scope": "DenseObservationBatch -> conversion -> MLP actor/critic -> evaluate",
            "python_scope": "raw p_net/v_node -> concat conversion -> MLP actor/critic -> RLSolver.evaluate_actions under inference_mode",
            "native_samples_ns": native_timed["samples_ns"],
            "python_samples_ns": python_timed["samples_ns"],
            "native_median_ns_per_call": statistics.median(native_timed["samples_ns"]),
            "python_median_ns_per_call": statistics.median(python_timed["samples_ns"]),
            "python_over_native_speedup": statistics.median(python_timed["samples_ns"]) / statistics.median(native_timed["samples_ns"]),
            "native_checksums": native_checksums,
            "python_checksums": python_timed["component_checksums"],
            "python_aggregate_checksum": python_timed["checksum"],
        }

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({
        "component": report["component"],
        "exact_cases": report["exact_cases"],
        "benchmark": report["benchmark"],
        "status": report["status"],
    }, sort_keys=True, separators=(",", ":")))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
