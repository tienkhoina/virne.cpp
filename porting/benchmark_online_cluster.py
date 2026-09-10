#!/usr/bin/env python3
"""Finalize an online/registry cluster and benchmark its validated epoch once."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import torch
import compare_learning_registry as registry

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary-dir",type=Path,required=True)
    parser.add_argument("--online-report",type=Path,required=True)
    parser.add_argument("--registry-report",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--reuse-environment-benchmark",type=Path)
    args=parser.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    components={}
    for name,path in (("online_environment",args.online_report),("registry",args.registry_report)):
        raw=path.read_bytes();report=json.loads(raw)
        if report["status"]!="PASS" or any(case["status"]!="PASS" for case in report["cases"].values()):
            raise RuntimeError("cluster component is not passing: "+name)
        components[name]={"status":"PASS","report":path.name,"report_sha256":hashlib.sha256(raw).hexdigest(),
            "cases":len(report["cases"]),"records":sum(case["records"] for case in report["cases"].values())}
        root=registry.BASE.ROOT if name=="online_environment" else registry.session.ROOT
        for relative,digest in report["source_sha256"].items():
            if hashlib.sha256((root/relative).read_bytes()).hexdigest()!=digest:raise RuntimeError("Python source changed: "+relative)
    native=args.binary_dir/"vne_learning_registry_harness"
    output=registry.session.native_command(native,["--benchmark"])
    native_samples=json.loads(output[0])["samples_ns"];actual=output[1:]
    python_samples=[];expected=None
    for repetition in range(-3,5):
        py_output,py_elapsed=registry.online_case(benchmark=True);py_output=py_output[1:]
        difference=registry.session.policies.base.first_difference(py_output,actual)
        if difference:raise RuntimeError("registry benchmark output mismatch: "+json.dumps(difference))
        if expected is None:expected=py_output
        elif expected!=py_output:raise RuntimeError("registry benchmark output changed between repetitions")
        if repetition>=0:python_samples.append(py_elapsed)
    signature=hashlib.sha256("\n".join(expected).encode()).hexdigest()
    benchmark={"component":"online_registered_learning","status":"PASS","workload":"Registry-created PPO solver learns three online requests, handles departures and flushes rollout; initial registry/model/lifecycle construction, process startup and output serialization excluded; per-request environment setup is timed",
        "native_samples_ns":native_samples,"python_samples_ns":python_samples,
        "native_median_ns":statistics.median(native_samples),"python_median_ns":statistics.median(python_samples),
        "native_p95_ns":float(registry.np.percentile(native_samples,95)),"python_p95_ns":float(registry.np.percentile(python_samples,95)),
        "python_over_native_speedup":statistics.median(python_samples)/statistics.median(native_samples),
        "native_output_sha256":signature,"python_output_sha256":signature,"native_sha256":hashlib.sha256(native.read_bytes()).hexdigest(),
        "warmups":3,"repetitions":5,"threads":1,"cpu_affinity":sorted(os.sched_getaffinity(0)),"torch":torch.__version__,"differential_rerun":False,
        "policy_warmup":"one greedy solve before timing on both sides, including native lazy model construction",
        "process_lifetime":"all three warmups and five samples in one process per runtime",
        "source_sha256":{**registry.SOURCE_HASHES,**registry.session.SOURCE_HASHES},"parity_reused_from":args.registry_report.name}
    benchmark_path=args.output.with_name(args.output.stem+"_registry_benchmark.json")
    benchmark_path.write_text(json.dumps(benchmark,indent=2)+"\n",encoding="utf-8")
    directory=Path(__file__).resolve().parent
    env_benchmark=args.reuse_environment_benchmark or args.output.with_name(args.output.stem+"_environment_benchmark.json")
    if not args.reuse_environment_benchmark:
        command=[sys.executable,str(directory/"compare_online_environment.py"),"--native",str(args.binary_dir/"vne_online_environment_harness"),
            "--benchmark-from",str(args.online_report),"--output",str(env_benchmark)]
        process=subprocess.run(command,capture_output=True,text=True)
        if process.returncode:raise RuntimeError(process.stderr or process.stdout)
    env_result=json.loads(env_benchmark.read_bytes())
    if env_result["status"]!="PASS" or env_result["native_sha256"]!=hashlib.sha256((args.binary_dir/"vne_online_environment_harness").read_bytes()).hexdigest():
        raise RuntimeError("invalid reused online environment benchmark")
    artifacts={}
    for path in (benchmark_path,env_benchmark):
        raw=path.read_bytes();data=json.loads(raw)
        artifacts[path.name]={"status":data["status"],"sha256":hashlib.sha256(raw).hexdigest(),"speedup":data["python_over_native_speedup"]}
    inventory=json.loads((directory/"results/learning_api_inventory_2026-09-08.json").read_bytes())
    for entry in inventory["modules"]:
        if hashlib.sha256((registry.session.ROOT/entry["path"]).read_bytes()).hexdigest()!=entry["sha256"]:raise RuntimeError("original source changed")
    report={"component":"online_environment_solver_registry","status":"PASS","components":components,"benchmarks":artifacts,
        "registered_solvers":len(registry.NAMES),"original_learning_sources_unchanged":len(inventory["modules"]),
        "benchmark_scope_correction":"online_registry_benchmark_scope_correction_2026-09-10.md",
        "old_tests_rerun":False,"accepted_benchmarks_rerun":False,"recorded_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope":"Online RL transition modes, sequential PG/A2C/PPO solver registry and native online learning epoch; main CLI, A3C, recurrent/safe solvers remain outside this cluster"}
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report))
if __name__=="__main__":main()
