"""Pinned Python/C++ differential for the online main-config runtime."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import time
from typing import Any

from hydra import compose, initialize_config_dir

from virne.system import BaseSystem
from virne.utils.config import add_simulation_into_config


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
DEFAULT_WORKERS = (1, 2, 8)
SEED = 7
PHYSICAL_NODES = 12
VIRTUAL_REQUESTS = 3


def common_overrides(solver_name: str) -> list[str]:
    return [
        f"solver.solver_name={solver_name}",
        f"experiment.seed={SEED}",
        f"experiment.run_id=system-e2e-{solver_name}",
        "experiment.num_simulations=1",
        "experiment.if_save_config=false",
        "experiment.if_save_p_net=false",
        "experiment.if_save_v_nets=false",
        "experiment.if_load_p_net=false",
        "experiment.if_load_v_nets=false",
        "recorder.if_save_records=false",
        "recorder.if_temp_save_records=false",
        "logger.backends=[]",
        f"p_net_setting.topology.num_nodes={PHYSICAL_NODES}",
        f"v_sim_setting.num_v_nets={VIRTUAL_REQUESTS}",
        "v_sim_setting.v_net_size.low=2",
        "v_sim_setting.v_net_size.high=4",
        "system.if_offline_system=false",
        "system.if_changeable_v_nets=false",
        "system.if_time_window=false",
    ]


def native_overrides(solver_name: str, workers: int) -> list[str]:
    result = common_overrides(solver_name)
    result.extend(
        [
            "++native.capture_solutions=true",
            "++native.output.report=full",
            "++native.progress.enabled=false",
            f"++native.workers.factory={workers}",
            f"++native.workers.attribute={workers}",
            f"++native.workers.arrangement={workers}",
            f"++native.workers.event={workers}",
            f"++native.workers.io={workers}",
            f"++native.workers.counter={workers}",
            f"++native.workers.recorder={workers}",
            f"++native.workers.mutation={workers}",
            f"++native.workers.rank={workers}",
            f"++native.workers.node_candidate={workers}",
            f"++native.workers.link_topology_constraint={workers}",
            f"++native.workers.link_candidate={workers}",
        ]
    )
    return result


def python_config(settings_dir: pathlib.Path, solver_name: str):
    with initialize_config_dir(
        config_dir=str(settings_dir.resolve()),
        job_name="virne-system-e2e",
        version_base=None,
    ):
        config = compose(
            config_name="main",
            overrides=common_overrides(solver_name),
        )
    add_simulation_into_config(config)
    return config


def scalar(value: Any) -> int | float:
    if isinstance(value, bool):
        return int(value)
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, int):
        return int(value)
    return float(value)


def canonical_solution(solution) -> dict[str, Any]:
    node_slots = [
        [int(virtual), int(physical)]
        for virtual, physical in solution["node_slots"].items()
    ]
    link_paths = []
    for virtual_link, physical_path in solution["link_paths"].items():
        link_paths.append(
            [
                [int(virtual_link[0]), int(virtual_link[1])],
                [
                    [int(link[0]), int(link[1])]
                    for link in physical_path
                ],
            ]
        )
    return {
        "result": bool(solution["result"]),
        "place_result": bool(solution["place_result"]),
        "route_result": bool(solution["route_result"]),
        "early_rejection": bool(solution["early_rejection"]),
        "node_slots": node_slots,
        "link_paths": link_paths,
        "v_net_cost": scalar(solution["v_net_cost"]),
        "v_net_revenue": scalar(solution["v_net_revenue"]),
        "v_net_r2c_ratio": scalar(solution["v_net_r2c_ratio"]),
        "hard_constraint_violation": scalar(
            solution["v_net_total_hard_constraint_violation"]
        ),
        "description": str(solution["description"]),
    }


def failure_reason(solution) -> str:
    if bool(solution["result"]):
        return "none"
    if bool(solution["early_rejection"]):
        return "early_rejection"
    if not bool(solution["place_result"]):
        return "placement"
    if not bool(solution["route_result"]):
        return "routing"
    return "unknown"


def canonical_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "acceptance_rate": scalar(summary["acceptance_rate"]),
        "average_r2c_ratio": scalar(summary["avg_r2c_ratio"]),
        "long_term_time_r2c_ratio": scalar(
            summary["long_term_time_r2c_ratio"]
        ),
        "success_count": int(summary["success_count"]),
        "early_rejection_count": int(summary["early_rejection_count"]),
        "place_failure_count": int(summary["place_failure_count"]),
        "route_failure_count": int(summary["route_failure_count"]),
        "total_cost": scalar(summary["total_cost"]),
        "total_revenue": scalar(summary["total_revenue"]),
        "total_time_cost": scalar(summary["total_time_cost"]),
        "total_time_revenue": scalar(summary["total_time_revenue"]),
        "long_term_r2c_ratio": scalar(summary["long_term_r2c_ratio"]),
        "total_simulation_time": scalar(summary["total_simulation_time"]),
    }


def run_python(settings_dir: pathlib.Path, solver_name: str) -> dict[str, Any]:
    config = python_config(settings_dir, solver_name)
    setup_begin = time.perf_counter_ns()
    system = BaseSystem.from_config(config)
    setup_time_ns = time.perf_counter_ns() - setup_begin

    physical_nodes = int(system.env.p_net.num_nodes)
    physical_links = int(system.env.p_net.num_links)
    run_begin = time.perf_counter_ns()
    system.ready()
    system.env.epoch_id = 0
    system.solver.epoch_id = 0
    instance = system.env.reset(config.experiment.seed)
    steps = []
    accepted = 0
    auto_released_total = 0
    while True:
        event_id = int(system.env.curr_event["id"])
        event_time = scalar(system.env.curr_event["time"])
        request_id = int(instance["v_net"].id)
        solution = system.solver.solve(instance)
        next_instance, _, done, _ = system.env.step(solution)
        next_event_id = int(system.env.curr_event["id"])
        auto_released = next_event_id - event_id - (0 if done else 1)
        auto_released_total += auto_released
        accepted_step = bool(solution["result"])
        accepted += int(accepted_step)
        steps.append(
            {
                "epoch": 0,
                "stage": 0,
                "window": 0,
                "request_id": request_id,
                "event_time": event_time,
                "accepted": accepted_step,
                "failure_reason": failure_reason(solution),
                "auto_released_events": auto_released,
                "solution": canonical_solution(solution),
            }
        )
        if done:
            break
        instance = next_instance
    summary = canonical_summary(
        system.env.recorder.summary_records(system.env.recorder.memory)
    )
    run_time_ns = time.perf_counter_ns() - run_begin
    simulator = system.env.v_net_simulator
    return {
        "mode": "online",
        "solver": solver_name,
        "physical_nodes": physical_nodes,
        "physical_links": physical_links,
        "virtual_requests": int(simulator.num_v_nets),
        "scheduled_events": int(simulator.num_events),
        "setup_time_ns": setup_time_ns,
        "run_time_ns": run_time_ns,
        "steps": steps,
        "epochs": [
            {
                "epoch": 0,
                "arrival_steps": len(steps),
                "accepted": accepted,
                "rejected": len(steps) - accepted,
                "auto_released_events": auto_released_total,
                "summary": summary,
            }
        ],
        "stages": [],
        "windows": [],
    }


def run_native(
    executable: pathlib.Path,
    config_path: pathlib.Path,
    solver_name: str,
    workers: int,
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            str(executable),
            "--config",
            str(config_path),
            *native_overrides(solver_name, workers),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"native main failed for {solver_name}, workers={workers}:\n"
            f"{completed.stderr}"
        )
    return json.loads(completed.stdout)


def comparable(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in report.items()
        if key not in {"solver_id", "setup_time_ns", "run_time_ns"}
    }


def parse_csv_names(value: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or any(item not in allowed for item in result):
        raise ValueError(f"invalid selection: {value}")
    return result


def parse_workers(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(","))
    if not result or any(item < 0 for item in result):
        raise ValueError("workers must be non-negative")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-settings", required=True, type=pathlib.Path)
    parser.add_argument("--native", required=True, type=pathlib.Path)
    parser.add_argument("--native-config", required=True, type=pathlib.Path)
    parser.add_argument("--solvers", default=",".join(SOLVER_NAMES))
    parser.add_argument(
        "--workers", default=",".join(str(item) for item in DEFAULT_WORKERS)
    )
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    solvers = parse_csv_names(args.solvers, SOLVER_NAMES)
    workers = parse_workers(args.workers)
    rows = []
    for solver_name in solvers:
        expected = run_python(args.python_settings, solver_name)
        for width in workers:
            actual = run_native(
                args.native, args.native_config, solver_name, width
            )
            if comparable(actual) != comparable(expected):
                raise RuntimeError(
                    f"main E2E mismatch for {solver_name}, workers={width}\n"
                    f"python={json.dumps(comparable(expected), sort_keys=True)}\n"
                    f"native={json.dumps(comparable(actual), sort_keys=True)}"
                )
            rows.append(
                {
                    "solver": solver_name,
                    "workers": width,
                    "python_setup_time_ns": expected["setup_time_ns"],
                    "python_run_time_ns": expected["run_time_ns"],
                    "native_setup_time_ns": actual["setup_time_ns"],
                    "native_run_time_ns": actual["run_time_ns"],
                }
            )

    result = {
        "component": "system.main_config.online.node_rank",
        "status": "PASS",
        "seed": SEED,
        "solvers": list(solvers),
        "workers": list(workers),
        "shared_case_count": len(rows),
        "fixture": {
            "physical_nodes": PHYSICAL_NODES,
            "virtual_requests": VIRTUAL_REQUESTS,
        },
        "rows": rows,
        "boundary": (
            "Python online BaseSystem is the exact oracle; Python TODO modes "
            "are validated by native typed unit/integration contracts."
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
