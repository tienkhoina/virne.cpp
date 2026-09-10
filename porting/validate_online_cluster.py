#!/usr/bin/env python3
"""Run new online environment and solver registry cases together as one gate."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--reuse-registry",type=Path)
    args=parser.parse_args();directory=Path(__file__).resolve().parent
    components={}
    for label,script,binary in (("online_environment","compare_online_environment.py","vne_online_environment_harness"),
        ("solver_registry","compare_learning_registry.py","vne_learning_registry_harness")):
        report_path=args.output.with_name(args.output.stem+"_"+label+".json")
        command=[sys.executable,str(directory/script),"--native",str(args.binary_dir/binary),"--output",str(report_path)]
        if label=="solver_registry" and args.reuse_registry:command += ["--case","registry_online","--reuse-report",str(args.reuse_registry)]
        process=subprocess.run(command,capture_output=True,text=True)
        if report_path.exists():
            raw=report_path.read_bytes();report=json.loads(raw)
            components[label]={"status":report["status"],"report":report_path.name,"report_sha256":hashlib.sha256(raw).hexdigest(),
                "cases":report["cases"],"exit_code":process.returncode}
        else:components[label]={"status":"FAIL","exit_code":process.returncode,"stderr":process.stderr,"stdout":process.stdout}
        print(json.dumps({"component":label,**components[label]}),flush=True)
    report={"component":"online_environment_solver_registry","status":"PASS" if all(v["status"]=="PASS" for v in components.values()) else "FAIL",
        "components":components,"old_tests_rerun":False,"benchmarks_rerun":False}
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return 0 if report["status"]=="PASS" else 1
if __name__=="__main__":raise SystemExit(main())
