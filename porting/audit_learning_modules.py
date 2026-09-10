#!/usr/bin/env python3
"""Inventory source declarations without importing Virne or running any test."""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
from pathlib import Path


def declarations(nodes, prefix=""):
    result = []
    for node in nodes:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            name = prefix + node.name
            item = {"name": name, "line": node.lineno}
            if isinstance(node, ast.ClassDef):
                item.update(kind="class", bases=[ast.unparse(base) for base in node.bases])
            else:
                item.update(kind="function", signature=ast.unparse(node.args))
            result.append(item)
            result.extend(declarations(node.body, name + "."))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    modules = []
    groups = collections.Counter()
    for path in sorted(args.source.rglob("*.py")):
        relative = path.relative_to(args.source).as_posix()
        payload = path.read_bytes()
        tree = ast.parse(payload.decode("utf-8-sig"), filename=relative)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imports.append({"module": "." * node.level + (node.module or ""),
                                "names": [alias.name for alias in node.names]})
            elif isinstance(node, ast.Import):
                imports.extend({"module": alias.name, "names": []} for alias in node.names)
        entries = declarations(tree.body)
        group = relative.split("/")[0] if "/" in relative else "learning_root"
        groups[group] += 1
        modules.append({"path": relative, "sha256": hashlib.sha256(payload).hexdigest(),
                        "lines": len(payload.splitlines()), "declarations": entries,
                        "imports": imports})
    report = {
        "scope": "virne.solver.learning", "method": "AST only; no imports or tests",
        "completion_inferred_from_names": False,
        "module_count": len(modules),
        "class_count": sum(item["kind"] == "class" for m in modules for item in m["declarations"]),
        "function_count": sum(item["kind"] == "function" for m in modules for item in m["declarations"]),
        "groups": dict(sorted(groups.items())), "modules": modules,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "modules"}))


if __name__ == "__main__":
    main()
