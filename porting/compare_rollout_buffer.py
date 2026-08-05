#!/usr/bin/env python3
"""Exact RolloutBuffer lifecycle parity and focused selection benchmark gate.

The local oracle follows ``virne/solver/learning/rl_core/buffer.py`` while
representing observations/tensors as stable integer IDs, as the native API does.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import struct
import subprocess
import time
from typing import Any, Iterable


FIXTURE_VERSION = 1
BENCHMARK_STEPS = 4096
BENCHMARK_SELECTION = 1024
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


class RolloutBuffer:
    """Small dependency-free oracle for the Python buffer.py lifecycle."""

    def __init__(self) -> None:
        self.curr_idx = 0
        self.basic_items = [
            "observations",
            "actions",
            "rewards",
            "dones",
            "next_observations",
            "logprobs",
            "values",
        ]
        self.calc_items = ["advantages", "returns"]
        self.extend_items = [
            "hidden_states",
            "cell_states",
            "action_mask",
            "entropies",
        ]
        self.reset()

    @property
    def all_items(self) -> list[str]:
        return self.basic_items + self.calc_items + self.extend_items

    def reset(self) -> None:
        self.curr_idx = 0
        for item in self.all_items:
            setattr(self, item, [])

    def size(self) -> int:
        return len(self.logprobs)

    def add(
        self,
        observation: int,
        action: int,
        reward: float,
        done: bool,
        logprob: float,
        value: float,
        next_observation: int,
    ) -> None:
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(bool(done))
        self.logprobs.append(logprob)
        self.values.append(value)
        self.next_observations.append(next_observation)
        self.curr_idx += 1

    def extend(self, item: str) -> None:
        if item not in self.extend_items:
            self.extend_items.append(item)
            setattr(self, item, [])

    def _new_subbuffer(self) -> RolloutBuffer:
        result = RolloutBuffer()
        # The native typed schema fixes buffer.py's dynamic-extension copy gap.
        for item in self.extend_items:
            if item not in result.extend_items:
                result.extend(item)
        return result

    def get_subbuffer(self, indices: Iterable[int]) -> RolloutBuffer:
        indices = list(indices)
        result = self._new_subbuffer()
        for item in self.all_items:
            source = getattr(self, item)
            if not source:
                continue
            destination = getattr(result, item)
            for index in indices:
                destination.append(source[index])
        return result

    def split_with_ratios(self, ratios: list[float]) -> list[RolloutBuffer]:
        assert sum(ratios) == 1
        indices = list(range(self.size()))
        sizes = [int(self.size() * ratio) for ratio in ratios]
        sizes[-1] = self.size() - sum(sizes[:-1])
        groups: list[list[int]] = []
        for size in sizes:
            groups.append(indices[:size])
            del indices[:size]
        return [self.get_subbuffer(group) for group in groups]

    def split_with_instance(self) -> list[RolloutBuffer]:
        indices = list(range(self.size()))
        groups: list[list[int]] = []
        for index in range(self.size()):
            if self.dones[index]:
                groups.append(indices[: index + 1])
                del indices[: index + 1]
        return [self.get_subbuffer(group) for group in groups]

    def merge(self, buffer: RolloutBuffer) -> None:
        for item in self.all_items:
            getattr(self, item).extend(getattr(buffer, item))
        # buffer.py intentionally leaves curr_idx unchanged.

    def compute_returns_and_advantages(
        self,
        last_value: float = 0.0,
        gamma: float = 0.99,
        gae_lambda: float = 0.98,
        method: str = "gae",
    ) -> None:
        assert method in ("gae", "mc", "td")
        size = self.size()
        self.returns = [0.0] * size
        self.advantages = [0.0] * size
        if method == "gae":
            last_gae_lam = 0.0
            for step in reversed(range(size)):
                next_value = last_value if step == size - 1 else self.values[step + 1]
                next_non_terminal = 1.0 - self.dones[step]
                delta = (
                    self.rewards[step]
                    + gamma * next_value * next_non_terminal
                    - self.values[step]
                )
                last_gae_lam = (
                    delta
                    + gamma * gae_lambda * next_non_terminal * last_gae_lam
                )
                self.advantages[step] = last_gae_lam
                self.returns[step] = last_gae_lam + self.values[step]
        elif method == "mc":
            self.returns = []
            discounted_reward = 0.0
            for reward, is_terminal in zip(
                reversed(self.rewards), reversed(self.dones)
            ):
                if is_terminal:
                    discounted_reward = 0.0
                discounted_reward = reward + gamma * discounted_reward
                self.returns.insert(0, discounted_reward)
        else:
            self.returns = []
            for step in reversed(range(size)):
                next_value = last_value if step == size - 1 else self.values[step + 1]
                next_non_terminal = 1.0 - self.dones[step]
                discounted_reward = (
                    self.rewards[step]
                    + gamma * next_value * next_non_terminal
                    - self.values[step]
                )
                self.returns.insert(0, discounted_reward)


class RolloutBufferWithCost(RolloutBuffer):
    def __init__(self) -> None:
        self.safe_rl_items = [
            "costs",
            "cost_values",
            "cost_returns",
            "baseline_cost_returns",
        ]
        super().__init__()

    @property
    def all_items(self) -> list[str]:
        return (
            self.basic_items
            + self.calc_items
            + self.extend_items
            + self.safe_rl_items
        )

    def get_cost_subbuffer(self, indices: Iterable[int]) -> RolloutBufferWithCost:
        indices = list(indices)
        result = RolloutBufferWithCost()
        for item in self.extend_items:
            if item not in result.extend_items:
                result.extend(item)
        for item in self.all_items:
            source = getattr(self, item)
            if not source:
                continue
            destination = getattr(result, item)
            for index in indices:
                destination.append(source[index])
        return result

    def compute_cost_returns(self, gamma: float, method: str) -> None:
        assert method in ("reachability", "cumulative")
        assert len(self.costs) == len(self.rewards)
        if method == "reachability":
            state_cost = 0.0
            for cost, is_terminal in zip(
                reversed(self.costs), reversed(self.dones)
            ):
                if is_terminal:
                    state_cost = -float("inf")
                state_cost = max(cost, gamma * state_cost)
                self.cost_returns.insert(0, state_cost)
        else:
            self.costs = [cost if cost >= 0.0 else 0.0 for cost in self.costs]
            state_cost = 0.0
            for cost, is_terminal in zip(
                reversed(self.costs), reversed(self.dones)
            ):
                if is_terminal:
                    state_cost = 0.0
                state_cost = cost + gamma * state_cost
                self.cost_returns.insert(0, state_cost)


class RolloutBufferWithAvgReturn(RolloutBuffer):
    def compute_returns_and_advantages(
        self,
        last_value: float = 0.0,
        gamma: float = 0.99,
        gae_lambda: float = 0.98,
        method: str = "gae",
    ) -> None:
        assert method in ("gae", "td")
        size = self.size()
        self.returns = [0.0] * size
        self.advantages = [0.0] * size
        self.dones[-1] = False
        mean_reward = sum(self.rewards) / len(self.rewards)
        if method == "gae":
            last_gae_lam = 0.0
            for index in range(size):
                self.rewards[index] -= mean_reward
            for step in reversed(range(size)):
                next_value = last_value if step == size - 1 else self.values[step + 1]
                next_non_terminal = 1.0 - self.dones[step]
                delta = (
                    self.rewards[step]
                    + gamma * next_value * next_non_terminal
                    - self.values[step]
                )
                last_gae_lam = (
                    delta
                    + gamma * gae_lambda * next_non_terminal * last_gae_lam
                )
                self.advantages[step] = last_gae_lam
                self.returns[step] = last_gae_lam + self.values[step]
        else:
            del gamma, gae_lambda
            for step in reversed(range(size)):
                next_value = last_value if step == size - 1 else self.values[step + 1]
                next_non_terminal = 1.0 - self.dones[step]
                self.advantages[step] = (
                    self.rewards[step]
                    - mean_reward
                    + next_value * next_non_terminal
                    - self.values[step]
                )
                self.returns[step] = self.advantages[step] + self.values[step]


def make_buffer() -> RolloutBuffer:
    result = RolloutBuffer()
    rewards = [1.2, -0.4, 0.7, 0.1, -0.3]
    values = [0.2, 0.3, -0.1, 0.4, 0.05]
    dones = [False, False, True, False, True]
    for index in range(len(rewards)):
        result.add(
            100 + index,
            10 + index,
            rewards[index],
            dones[index],
            -0.1 * (index + 1),
            values[index],
            200 + index,
        )
    result.extend("feasibility_flags")
    result.extend("feasibility_budgets")
    result.feasibility_flags = [1.0, 1.0, 0.0, 1.0, 0.0]
    result.feasibility_budgets = [0.5, 0.5, 1.0, 1.5, 2.0]
    return result


def make_cost_buffer() -> RolloutBufferWithCost:
    source = make_buffer()
    result = RolloutBufferWithCost()
    for index in range(source.size()):
        result.add(
            source.observations[index],
            source.actions[index],
            source.rewards[index],
            source.dones[index],
            source.logprobs[index],
            source.values[index],
            source.next_observations[index],
        )
    result.costs = [1.0, -2.0, 0.5, 2.0, -1.0]
    return result


def make_avg_buffer() -> RolloutBufferWithAvgReturn:
    result = RolloutBufferWithAvgReturn()
    rewards = [1.0, -1.0, 2.0]
    values = [0.1, 0.2, 0.3]
    for index in range(len(rewards)):
        result.add(
            index,
            index,
            rewards[index],
            index == 2,
            -0.2,
            values[index],
            0,
        )
    return result


def split_summaries(buffers: list[RolloutBuffer]) -> list[dict[str, Any]]:
    return [
        {
            "size": buffer.size(),
            "curr_idx": buffer.curr_idx,
            "observations": buffer.observations,
        }
        for buffer in buffers
    ]


def expected_outputs() -> dict[str, Any]:
    lifecycle = make_buffer()
    selected = lifecycle.get_subbuffer([4, 1])
    ratio_splits = lifecycle.split_with_ratios([0.4, 0.6])
    instance_splits = lifecycle.split_with_instance()
    merged = ratio_splits[0].get_subbuffer(range(ratio_splits[0].size()))
    merge_curr_idx_before = merged.curr_idx
    merged.merge(ratio_splits[1])

    gae = make_buffer()
    gae.compute_returns_and_advantages(0.25, 0.91, 0.83, "gae")
    mc = make_buffer()
    mc.compute_returns_and_advantages(0.25, 0.91, 0.83, "mc")
    td = make_buffer()
    td.compute_returns_and_advantages(0.25, 0.91, 0.83, "td")

    cost = make_cost_buffer()
    cost.compute_cost_returns(0.8, "reachability")
    reachability = list(cost.cost_returns)
    cost.compute_cost_returns(0.8, "cumulative")
    cost_selected = cost.get_cost_subbuffer([3, 0])

    avg_gae = make_avg_buffer()
    avg_gae.compute_returns_and_advantages(0.4, 0.9, 0.8, "gae")
    avg_td = make_avg_buffer()
    avg_td.compute_returns_and_advantages(0.4, 0.9, 0.8, "td")

    return {
        "lifecycle": {
            "size": lifecycle.size(),
            "curr_idx": lifecycle.curr_idx,
            "observations": lifecycle.observations,
            "actions": lifecycle.actions,
            "rewards": lifecycle.rewards,
            "dones": [int(value) for value in lifecycle.dones],
            "next_observations": lifecycle.next_observations,
            "logprobs": lifecycle.logprobs,
            "values": lifecycle.values,
            "feasibility_flags_enabled": hasattr(
                lifecycle, "feasibility_flags"
            ),
            "feasibility_budgets_enabled": hasattr(
                lifecycle, "feasibility_budgets"
            ),
            "feasibility_flags": lifecycle.feasibility_flags,
            "feasibility_budgets": lifecycle.feasibility_budgets,
        },
        "selected": {
            "size": selected.size(),
            "curr_idx": selected.curr_idx,
            "observations": selected.observations,
            "actions": selected.actions,
            "feasibility_flags": selected.feasibility_flags,
            "feasibility_budgets": selected.feasibility_budgets,
        },
        "ratio_splits": split_summaries(ratio_splits),
        "instance_splits": split_summaries(instance_splits),
        "merge": {
            "curr_idx_before": merge_curr_idx_before,
            "curr_idx_after": merged.curr_idx,
            "size": merged.size(),
            "observations": merged.observations,
            "feasibility_flags": merged.feasibility_flags,
            "feasibility_budgets": merged.feasibility_budgets,
        },
        "returns": {
            "gae": {"returns": gae.returns, "advantages": gae.advantages},
            "mc": {"returns": mc.returns, "advantages": mc.advantages},
            "td": {"returns": td.returns, "advantages": td.advantages},
        },
        "cost": {
            "reachability": reachability,
            "after_cumulative": {
                "costs": cost.costs,
                "cost_returns": cost.cost_returns,
                "selected_costs": cost_selected.costs,
                "selected_cost_returns": cost_selected.cost_returns,
            },
        },
        "average_return": {
            "gae": {
                "dones": [int(value) for value in avg_gae.dones],
                "rewards": avg_gae.rewards,
                "returns": avg_gae.returns,
                "advantages": avg_gae.advantages,
            },
            "td": {
                "dones": [int(value) for value in avg_td.dones],
                "rewards": avg_td.rewards,
                "returns": avg_td.returns,
                "advantages": avg_td.advantages,
            },
        },
    }


def make_benchmark_fixture() -> tuple[RolloutBuffer, list[int]]:
    buffer = RolloutBuffer()
    buffer.extend("feasibility_flags")
    buffer.extend("feasibility_budgets")
    for index in range(BENCHMARK_STEPS):
        reward_integer = (index * 37) % 101 - 50
        value_integer = (index * 19) % 79 - 39
        buffer.add(
            100000 + index,
            index % 97,
            reward_integer * 0.03125,
            (index + 1) % 127 == 0 or index + 1 == BENCHMARK_STEPS,
            -((index % 67) + 1) * 0.015625,
            value_integer * 0.0078125,
            200000 + index,
        )
        buffer.feasibility_flags.append(1.0 if index % 3 == 0 else 0.0)
        buffer.feasibility_budgets.append(((index * 13) % 97) * 0.0625)
    indices = [
        (index * 17) % BENCHMARK_STEPS for index in range(BENCHMARK_SELECTION)
    ]
    return buffer, indices


def update_checksum(checksum: int, payload: bytes) -> int:
    for byte in payload:
        checksum ^= byte
        checksum = (checksum * FNV_PRIME) & MASK64
    return checksum


def output_summary(buffer: RolloutBuffer) -> dict[str, Any]:
    checksum = FNV_OFFSET
    for value in buffer.observations:
        checksum = update_checksum(checksum, struct.pack("<Q", value))
    for value in buffer.actions:
        checksum = update_checksum(checksum, struct.pack("<q", value))
    for value in buffer.rewards:
        checksum = update_checksum(checksum, struct.pack("<d", value))
    for value in buffer.dones:
        checksum = update_checksum(checksum, struct.pack("<B", int(value)))
    for value in buffer.next_observations:
        checksum = update_checksum(checksum, struct.pack("<Q", value))
    for value in buffer.logprobs:
        checksum = update_checksum(checksum, struct.pack("<d", value))
    for value in buffer.values:
        checksum = update_checksum(checksum, struct.pack("<d", value))
    for value in buffer.feasibility_flags:
        checksum = update_checksum(checksum, struct.pack("<d", value))
    for value in buffer.feasibility_budgets:
        checksum = update_checksum(checksum, struct.pack("<d", value))
    return {
        "size": buffer.size(),
        "curr_idx": buffer.curr_idx,
        "checksum": str(checksum),
        "first_observation": buffer.observations[0] if buffer.observations else 0,
        "last_observation": buffer.observations[-1] if buffer.observations else 0,
    }


def benchmark_python(
    threads: int,
    warmups: int,
    repetitions: int,
    iterations: int,
) -> dict[str, Any]:
    del threads  # Selection mutates one ordered buffer result; callers parallelize.
    buffer, indices = make_benchmark_fixture()
    output = RolloutBuffer()
    for _ in range(warmups):
        for _ in range(iterations):
            output = buffer.get_subbuffer(indices)
    samples_ns: list[int] = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            output = buffer.get_subbuffer(indices)
        samples_ns.append(time.perf_counter_ns() - start)
    return {
        "name": "subbuffer_select_4096_to_1024",
        "warmups": warmups,
        "repetitions": repetitions,
        "iterations": iterations,
        "samples_ns": samples_ns,
        "output": output_summary(output),
    }


def compare(expected: Any, actual: Any, path: str = "") -> list[dict[str, Any]]:
    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and expected == actual
    ):
        return []
    if type(expected) is not type(actual):
        return [{"path": path, "python": expected, "cpp": actual, "reason": "type"}]
    if isinstance(expected, dict):
        result: list[dict[str, Any]] = []
        for key in sorted(set(expected) | set(actual)):
            if key not in expected or key not in actual:
                result.append({"path": f"{path}.{key}", "reason": "missing"})
            else:
                result.extend(compare(expected[key], actual[key], f"{path}.{key}"))
            if len(result) >= 48:
                return result[:48]
        return result
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [{
                "path": path,
                "python_length": len(expected),
                "cpp_length": len(actual),
                "reason": "length",
            }]
        result = []
        for index, (left, right) in enumerate(zip(expected, actual)):
            result.extend(compare(left, right, f"{path}[{index}]"))
            if len(result) >= 48:
                return result[:48]
        return result
    return [] if expected == actual else [{
        "path": path,
        "python": expected,
        "cpp": actual,
        "reason": "value",
    }]


def compare_benchmark(
    native: dict[str, Any],
    python: dict[str, Any],
    threads: int,
) -> dict[str, Any]:
    native_benchmark = native.get("benchmark")
    if not isinstance(native_benchmark, dict):
        raise RuntimeError("native benchmark report is missing")
    parameters = {
        "name": python["name"],
        "threads": threads,
        "warmups": python["warmups"],
        "repetitions": python["repetitions"],
        "iterations": python["iterations"],
    }
    for key, expected in parameters.items():
        if native_benchmark.get(key) != expected:
            raise RuntimeError(
                f"benchmark parameter mismatch for {key}: "
                f"native={native_benchmark.get(key)!r}, python={expected!r}"
            )
    output_mismatches = compare(
        python["output"], native_benchmark.get("output"), "benchmark.output"
    )
    if output_mismatches:
        raise RuntimeError(f"benchmark output differs: {output_mismatches[:4]}")
    iterations = int(python["iterations"])
    native_samples = [
        int(value) / iterations for value in native_benchmark["samples_ns"]
    ]
    python_samples = [int(value) / iterations for value in python["samples_ns"]]
    native_median = statistics.median(native_samples)
    python_median = statistics.median(python_samples)
    return {
        "name": python["name"],
        "threads": threads,
        "warmups": python["warmups"],
        "repetitions": python["repetitions"],
        "iterations": iterations,
        "native_samples_ns_per_call": native_samples,
        "python_samples_ns_per_call": python_samples,
        "native_median_ns_per_call": native_median,
        "python_median_ns_per_call": python_median,
        "python_over_native_speedup": (
            python_median / native_median if native_median > 0.0 else None
        ),
        "output_checksum": python["output"]["checksum"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", "--harness", required=True, type=pathlib.Path)
    parser.add_argument("--json-output", "--output", type=pathlib.Path)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=25)
    args = parser.parse_args()
    for name in ("threads", "warmups", "repetitions", "iterations"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} requires a positive integer")

    command = [str(args.native)]
    if args.benchmark:
        command.extend([
            "--benchmark",
            "--threads",
            str(args.threads),
            "--warmups",
            str(args.warmups),
            "--repetitions",
            str(args.repetitions),
            "--iterations",
            str(args.iterations),
        ])
    process = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    if process.returncode != 0:
        raise RuntimeError(f"native failed: {process.stderr.strip()}")
    if len(lines) != 1:
        raise RuntimeError(f"native emitted {len(lines)} JSON lines: {lines!r}")
    actual = json.loads(lines[0])
    expected = {
        "component": "rollout_buffer",
        "fixture_version": FIXTURE_VERSION,
        "outputs": expected_outputs(),
    }
    actual_exact = {key: value for key, value in actual.items() if key != "benchmark"}
    mismatches = compare(expected, actual_exact)
    benchmark_report: Any = "DEFERRED_UNTIL_EXACT"
    if args.benchmark:
        benchmark_report = compare_benchmark(
            actual,
            benchmark_python(
                args.threads, args.warmups, args.repetitions, args.iterations
            ),
            args.threads,
        )
    report = {
        "component": "rollout_buffer",
        "fixture_version": FIXTURE_VERSION,
        "status": "PASS" if not mismatches else "FAIL",
        "exact_outputs": "12/12" if not mismatches else "FAIL",
        "mismatches": mismatches,
        "benchmark": benchmark_report,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        args.json_output.write_text(encoded, encoding="utf-8")
    compact = {
        key: report[key]
        for key in ("component", "status", "exact_outputs", "benchmark")
    }
    if mismatches:
        compact["mismatches"] = mismatches[:4]
    print(json.dumps(compact, sort_keys=True, separators=(",", ":")))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
