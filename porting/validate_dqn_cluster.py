#!/usr/bin/env python3
"""One grouped DQN acceptance gate: reuse saved results and test new boundaries."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import MethodType
import numpy as np
import torch
import compare_learning_session as session
from compare_dqn_update import feature_batch_norm

CASES = ["bn_target_resume", "graph_pending_resume", "fresh_seed_resume", "reseed_resume", "incompatible_checkpoint"]
REPORTS = ["dqn_update_differential_2026-09-09.json", "dqn_action_differential_2026-09-09.json",
    "learning_session_dqn_2026-09-09.json", "dqn_update_benchmark_2026-09-09.json", "learning_session_dqn_benchmark_2026-09-09.json"]


def record(name, value, dtype="float64"):
    return session.environment_oracle.record(name, value, dtype)


def snapshot(solver, solution, buffer, last, loss):
    lines = [record("actions",buffer.actions,"int64")]
    for key, values in (("rewards",buffer.rewards),("values",buffer.values),
                        ("logprobs",np.asarray(buffer.logprobs).reshape(-1)),("returns",buffer.returns)):
        lines.append(record(key,values))
    return lines + [record("dones",buffer.dones,"bool"),record("last_value",last),record("success",solution.result,"bool"),
        record("pending",solver.buffer.size(),"int64"),record("updates",solver.update_time,"int64"),
        "loss|undefined" if loss is None else record("loss",float(loss)),record("epsilon",solver.epsilon),
        record("steps",solver.steps_done,"int64"),*session.parameters(solver)]


def run(name):
    valid, failed = session.environment("ppo"), session.environment("failed")
    original = session.build_policy
    if name == "bn_target_resume":
        def with_batch_norm(kind, obs):
            model = session.training.mlp_namespace["MlpActorCritic"](obs["p_net_x"].shape[1]+len(obs["v_node_x"]),6,
                num_layers=2,embedding_dim=8,dropout_prob=0.,batch_norm=True)
            for child in model.modules():
                if isinstance(child,torch.nn.BatchNorm1d): child.forward = MethodType(feature_batch_norm,child)
            return model
        session.build_policy = with_batch_norm
    try:
        solver = session.build_solver("double_dqn_gcn" if name == "graph_pending_resume" else "double_dqn",valid)
    finally:
        session.build_policy = original
    solver.batch_size, solver.target_update_interval = 3,3
    target_steps = 1000 if name == "graph_pending_resume" else (1 if name == "bn_target_resume" else 7)
    if name == "fresh_seed_resume": np.random.seed(947)
    lines = ["CASE "+name]
    for i in range(6):
        if i == 2 and name == "reseed_resume": np.random.seed(947)
        if i == 3 and name == "bn_target_resume":
            solver.InstanceEnv = lambda *a,**k: valid
            solver.policy.eval()
            solver.select_action = lambda obs,**kwargs: session.DQN.select_action(solver,obs,sample=False)
            solution,buffer,last = session.collect(solver,valid)
            lines += ["EVAL",*snapshot(solver,solution,buffer,last,None)]
            solver.select_action = MethodType(session.DQN.select_action,solver)
        if i == 3 and name == "graph_pending_resume":
            if not solver.buffer.size(): raise RuntimeError("empty graph rollout in reference")
            lines += [record("pending_flush",float(solver.update())),*session.parameters(solver)]
        if i == 4 and name == "incompatible_checkpoint": lines.append(record("incompatible_rejected",True,"bool"))
        env = failed if i in (1,4) else valid
        solver.InstanceEnv = lambda *a,**k: env
        solver.policy.train()
        solution,buffer,last = session.collect(solver,env)
        solver.buffer.merge(buffer)
        loss = solver.update() if solver.buffer.size() >= target_steps else None
        lines += ["EPISODE "+str(i),*snapshot(solver,solution,buffer,last,loss)]
    loss = solver.update() if solver.buffer.size() else None
    return lines + ["flush_loss|undefined" if loss is None else record("flush_loss",float(loss)),*session.parameters(solver)]


def signature(lines): return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def reuse_evidence(directory):
    result = {}
    for name in REPORTS:
        raw = (directory/name).read_bytes(); report = json.loads(raw)
        if report.get("status") != "PASS": raise RuntimeError("nonpassing saved gate: "+name)
        for relative, digest in report["source_sha256"].items():
            if hashlib.sha256((session.ROOT/relative).read_bytes()).hexdigest() != digest: raise RuntimeError("Python source drift: "+relative)
        for relative, digest in report.get("environment_source_sha256",{}).items():
            if hashlib.sha256((session.environment_oracle.ROOT/relative).read_bytes()).hexdigest() != digest: raise RuntimeError("environment source drift: "+relative)
        details = report.get("cases")
        if isinstance(details,dict):
            for value in details.values():
                if value["status"] != "PASS" or value["python_sha256"] != value["cpp_sha256"]: raise RuntimeError("inconsistent saved case: "+name)
            count = len(details); records = sum(value["records"] for value in details.values())
        else:
            if report["python_output_sha256"] != report["native_output_sha256"]: raise RuntimeError("saved output mismatch: "+name)
            count = len(details or []); records = report.get("records",0)
        result[name] = {"status":"PASS","report_sha256":hashlib.sha256(raw).hexdigest(),"cases":count,"records":records,
            "rerun":False,"speedup":report.get("python_over_native_speedup")}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--case",choices=CASES)
    parser.add_argument("--reuse-report",type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    directory = Path(__file__).resolve().parent
    reused = reuse_evidence(directory/"results")
    inventory = json.loads((directory/"results/learning_api_inventory_2026-09-08.json").read_text())
    for item in inventory["modules"]:
        if hashlib.sha256((session.ROOT/item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError("original learning source changed: "+item["path"])
    expected, reference_errors = {}, {}
    selected = [args.case] if args.case else CASES
    for name in selected:
        try: expected[name] = run(name)
        except Exception as error:
            expected[name] = ["CASE "+name]
            reference_errors[name] = type(error).__name__+": "+str(error)
    env = dict(os.environ); env["LD_LIBRARY_PATH"] = str(directory.parent/"libs/libtorch/lib")
    process = subprocess.run([str(args.native.resolve())]+(["--case",args.case] if args.case else []),capture_output=True,text=True,env=env)
    actual, current = {}, None
    for line in process.stdout.splitlines():
        if line.startswith("CASE "): current = line[5:]; actual[current] = []
        if current is not None: actual[current].append(line)
    results = {}
    if args.reuse_report:
        saved = json.loads(args.reuse_report.read_bytes())
        if saved["component"] != "dqn_cluster": raise RuntimeError("wrong reused cluster report")
        for name,entry in saved["new_cases"].items():
            if name not in selected and entry["status"] == "PASS":
                results[name] = {**entry,"reused_from":args.reuse_report.name}
    for name, lines in expected.items():
        other = actual.get(name,[])
        mismatch = session.policies.base.first_difference(lines,other)
        entry = {"status":"PASS" if mismatch is None and name not in reference_errors else "FAIL","records":len(lines),"python_sha256":signature(lines),
            "cpp_sha256":signature(other),"mismatch":mismatch,"native_errors":[line for line in other if line.startswith("ERROR ")],
            "reference_error":reference_errors.get(name)}
        results[name] = entry
        print(json.dumps({"case":name,"status":entry["status"],"records":entry["records"],"mismatch":mismatch,"errors":entry["native_errors"]}),flush=True)
    sources = sorted([*directory.parent.joinpath("virne/solver/learning/rl_core").glob("*.cpp"),
        *directory.parent.joinpath("virne/solver/learning/rl_core").glob("*.h"),directory/"dqn_cluster_harness.cpp",Path(__file__)])
    report = {"component":"dqn_cluster","status":"PASS" if process.returncode == 0 and all(x["status"] == "PASS" for x in results.values()) else "FAIL",
        "scope":"DQN/DoubleDQN CPU instance learning, action selection, target synchronization and native checkpoint continuation",
        "reused_evidence":reused,"new_cases":results,"new_records":sum(x["records"] for x in results.values()),
        "source_files_unchanged":len(inventory["modules"]),"native_returncode":process.returncode,"native_stderr":process.stderr,
        "native_sha256":hashlib.sha256(args.native.read_bytes()).hexdigest(),
        "cpp_source_sha256":{path.relative_to(directory.parent).as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "source_sha256":{item["path"]:item["sha256"] for item in inventory["modules"]},
        "reference":"Original agent and DoubleDQN update; compare the restored native trajectory against uninterrupted Python",
        "reference_corrections":["documented environment candidate/revoke and constant-distance corrections", "feature-last BatchNorm in bn_target_resume"],
        "executed_cases":selected,"reused_cluster_report":args.reuse_report.name if args.reuse_report else None,
        "threads":1,"torch":torch.__version__,"cpu_affinity":sorted(os.sched_getaffinity(0)),
        "recorded_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"old_tests_rerun":False,"benchmarks_rerun":False}
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
