#!/usr/bin/env python3
"""SHA-pinned attention, dual and recurrent policy byte differential."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import pathlib
import statistics
import time
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch
import compare_cnn_gnn_mlp_policy as base

EXPECTED = {
    "rl_policy/att_policy.py": "0a1f08712dd0d57fb5b9da0d29edba6679db049f965faaddbce4596edadac08d",
    "rl_policy/dual_gnn_policy.py": "590a320032ed16bce9be844ad34cab0003dc1702161c2cd7e0ac0498a62c2cdf",
    "rl_policy/gcn_seq2seq_policy.py": "2d5e11b6a2984a0706b434d17346a5bb4136fb2e848563a4fe60444c19fb50c0",
    "reinforcement_learning/gnn_seq2seq_solver/policy_with_encoder.py": "11959fc58e0f71cf4da85fd80403e9fc5ef6ebeb9834858dcf48f6ee523934d3",
}

class InitializedDeepGAT(base.neural.GNN["DeepEdgeFeatureGAT"]):
    """Correct only undefined residual initialization; keep the source file intact."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for index in range(self.num_mid_layers):
            torch.nn.init.orthogonal_(getattr(self, f"weight_{index}"))

def initialized_gnn_class(name):
    return InitializedDeepGAT if name == "deep_edge_gat" else base.neural.GNN["get_gnn_class"](name)

def load(relative):
    path = base.SOURCE_ROOT / relative
    if hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED[relative]:
        raise RuntimeError("source drift: " + relative)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    namespace = {**base.neural.GNN, "torch": torch, "nn": torch.nn, "F": torch.nn.functional,
        "BaseActorCritic": base.CNN["BaseActorCritic"], "ActorCriticRegistry": base.CNN["ActorCriticRegistry"],
        "MLPNet": base.GNN["MLPNet"], "ResNetBlock": base.neural.RES_MODULE.ResNetBlock,
        "to_dense_batch": to_dense_batch}
    if "policy_with_encoder" in relative:
        namespace["get_gnn_class"] = initialized_gnn_class
    exec(compile(tree, relative, "exec"), namespace)
    return namespace

ATT = load("rl_policy/att_policy.py")
DUAL = load("rl_policy/dual_gnn_policy.py")
LEGACY = load("rl_policy/gcn_seq2seq_policy.py")
SEQ = load("reinforcement_learning/gnn_seq2seq_solver/policy_with_encoder.py")

def values(shape):
    return torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape).remainder(19).sub(9).div(16)

def graph(features):
    a = Data(x=values((3, features)), edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]), edge_attr=values((4, 1)))
    b = a.clone()
    b.x = values((4, features))
    return Batch.from_data_list([a, b])

def report(model, outputs, inputs, backward):
    if backward:
        loss = outputs[0].square().sum()
        for output in outputs[1:]:
            loss = loss + output.square().sum()
        loss.backward()
    lines = [base.tensor_line(f"output_{i}", output) for i, output in enumerate(outputs)]
    if backward:
        lines.extend(base.tensor_line(f"input_grad_{i}", value.grad) for i, value in enumerate(inputs))
    return lines + base.module_lines(model, backward) + [base.tensor_line("rng_tail", torch.rand(4))]

def attention(backward, use_double=False):
    torch.manual_seed(301)
    model = ATT["AttActorCritic"](4, 3, 2, num_layers=2, embedding_dim=8, dropout_prob=0.2, batch_norm=False)
    obs = {"p_net_x": values((2, 4, 3)), "v_node_x": values((2, 2))}
    if use_double:
        model.double()
        obs = {name: value.double() for name, value in obs.items()}
    model.train(backward)
    for value in obs.values():
        value.requires_grad_(backward)
    outputs = [model.act(obs), model.evaluate(obs)]
    return report(model, outputs, list(obs.values()), backward)

def attention_encoder(backward):
    torch.manual_seed(307)
    model = ATT["AttEncoder"](4, 3, 2, num_layers=2, embedding_dim=8, num_att_heads=2)
    model.train(backward)
    obs = {"p_net_x": values((2, 4, 3)).requires_grad_(backward), "v_node_x": values((2, 2)).requires_grad_(backward)}
    with torch.set_grad_enabled(backward):
        return report(model, [model(obs)], list(obs.values()), backward)

def dual_model(kind, auxiliary, backward=False, mixed=False):
    stem = ["BiGcn", "BiGat", "BiDeepEdgeFeatureGat"][kind]
    suffix = ["", "WithCost", "WithCostAndLambda"][auxiliary]
    kwargs = {"p_gnn_class": base.neural.GNN["GATConvNet"], "v_gnn_class": base.neural.GNN["GCNConvNet"]} if mixed else {}
    return DUAL[stem + suffix + "ActorCritic"](4, 3, 1, 2, 1, embedding_dim=4,
        dropout_prob=0.2, batch_norm=backward, num_layers=3, **kwargs)

def dual(kind, auxiliary, backward, mixed=False):
    torch.manual_seed(311)
    model = dual_model(kind, auxiliary, backward, mixed)
    model.train(backward)
    obs = {"v_net": graph(2), "p_net": graph(3), "curr_v_node_id": torch.tensor([1, 2])}
    inputs = [obs["v_net"].x, obs["p_net"].x, obs["v_net"].edge_attr, obs["p_net"].edge_attr]
    for value in inputs:
        value.requires_grad_(backward)
    outputs = [model.act(obs), model.evaluate(obs)]
    if auxiliary >= 1:
        outputs.append(model.cost_critic(obs))
    if auxiliary >= 2:
        outputs.append(model.lambda_net(obs))
    return report(model, outputs, inputs, backward)

def seq_model(kind):
    if kind == 0:
        return LEGACY["GcnSeq2SeqActorCritic"](4, 3, 2, embedding_dim=8)
    name = ["GcnSeq2SeqActorCritic", "GATSeq2SeqActorCritic", "DeepEdgeGnnSeq2SeqActorCritic"][kind - 1]
    return SEQ[name](4, 3, 1, 2, embedding_dim=8, dropout_prob=0.2, batch_norm=False)

def sequence(kind, backward):
    torch.manual_seed(313)
    model = seq_model(kind)
    if kind == 3:
        base.neural.fill_parameters(model)
    model.train(backward)
    obs = {"p_net": graph(3), "v_net_x": values((2, 3, 2)).requires_grad_(backward)}
    obs["p_net"].x.requires_grad_(backward)
    lines = [base.tensor_line("initial_hidden", model.get_last_rnn_state())]
    encoded = model.encode(obs)
    obs.update(hidden_state=model.get_last_rnn_state().permute(1, 0, 2), encoder_outputs=encoded.transpose(0, 1),
               p_node_id=torch.tensor([0, 4]), mask=torch.tensor([[1, 1, 0], [1, 0, 1]]))
    logits = model.act(obs)
    hidden = model.get_last_rnn_state()
    value = model.evaluate(obs)
    assert model.get_last_rnn_state() is hidden
    model.set_last_rnn_hidden(torch.zeros_like(hidden))
    lines.append(base.tensor_line("replaced_hidden", model.get_last_rnn_state()))
    return lines + report(model, [encoded, logits, hidden, value], [obs["v_net_x"], obs["p_net"].x], backward)

def additive_attention():
    torch.manual_seed(317)
    model = LEGACY["Attention"](8)
    hidden, encoded = values((1, 2, 8)).requires_grad_(), values((2, 3, 8)).requires_grad_()
    context, weights = model(hidden, encoded, torch.zeros((2, 3), dtype=torch.int64))
    return report(model, [context, weights], [hidden, encoded], True)

def exact(selected=None):
    cases = {"att_eval": lambda: attention(False), "att_train": lambda: attention(True),
        "att_double": lambda: attention(True, True), "att_heads2_eval": lambda: attention_encoder(False),
        "att_heads2_train": lambda: attention_encoder(True)}
    for kind in range(3):
        for aux in range(3):
            cases[f"dual_{kind}_{aux}"] = lambda k=kind, a=aux: dual(k, a, False)
        cases[f"dual_train_{kind}"] = lambda k=kind: dual(k, 2, True)
    cases["dual_mixed"] = lambda: dual(2, 2, False, True)
    for kind in range(4):
        cases[f"seq_{kind}"] = lambda k=kind: sequence(k, False)
        cases[f"seq_train_{kind}"] = lambda k=kind: sequence(k, True)
    cases["additive_mask"] = additive_attention
    if selected is not None:
        cases = {selected: cases[selected]}
    lines = []
    for name, run in cases.items():
        lines.extend(["CASE " + name, *run()])
    return lines

def benchmark_python():
    with torch.no_grad():
        torch.manual_seed(331)
        att = ATT["AttActorCritic"](4, 3, 2, num_layers=2, embedding_dim=8, dropout_prob=0.0).eval()
        dual = dual_model(0, 0).eval()
        seq = seq_model(0).eval()
        dense = {"p_net_x": values((2, 4, 3)), "v_node_x": values((2, 2))}
        graphs = {"v_net": graph(2), "p_net": graph(3), "curr_v_node_id": torch.tensor([1, 2])}
        obs = {"p_net": graph(3), "v_net_x": values((2, 3, 2))}
        encoded = seq.encode(obs)
        obs.update(encoder_outputs=encoded.transpose(0, 1), hidden_state=seq.get_last_rnn_state().permute(1, 0, 2),
                   p_node_id=torch.tensor([0, 4]), mask=None)
        def run():
            return [att.act(dense), att.evaluate(dense), dual.act(graphs), dual.evaluate(graphs), seq.act(obs), seq.evaluate(obs)]
        reference = run()
        for _ in range(3): run()
        samples = []
        for _ in range(5):
            start = time.perf_counter_ns()
            for _ in range(5): output = run()
            samples.append((time.perf_counter_ns() - start) // 5)
            assert all(torch.equal(a, b) for a, b in zip(reference, output))
        return samples, [base.tensor_line(f"output_{i}", value) for i, value in enumerate(reference)]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--case")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.case and args.benchmark: parser.error("benchmark requires the full new group")
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    expected = exact(args.case)
    arguments = ["--threads", args.threads] + (["--case", args.case] if args.case else [])
    actual = base.run_native(args.native, arguments)
    mismatch = base.first_difference(expected, actual)
    def blocks(lines):
        result, name = {}, ""
        for line in lines:
            if line.startswith("CASE "):
                name = line[5:]
                result[name] = []
            result[name].append(line)
        return result
    expected_blocks, actual_blocks = blocks(expected), blocks(actual)
    mismatches = [difference for name, lines in expected_blocks.items()
                  if (difference := base.first_difference(lines, actual_blocks.get(name, []))) is not None]
    result = {"component": "rl_policy_remaining", "status": "PASS" if mismatch is None else "FAIL",
        "threads": args.threads, "source_sha256": EXPECTED, "torch": torch.__version__,
        "cases": [line[5:] for line in expected if line.startswith("CASE ")], "records": len(expected),
        "python_sha256": hashlib.sha256("\n".join(expected).encode()).hexdigest(),
        "cpp_sha256": hashlib.sha256("\n".join(actual).encode()).hexdigest(), "mismatch": mismatch,
        "benchmark": "NOT_REQUESTED", "case_mismatches": mismatches}
    result["reference_corrections"] = {
        "seq_3": "orthogonal initialization of otherwise undefined residual weights",
        "seq_train_3": "orthogonal initialization of otherwise undefined residual weights"}
    if mismatch is None and args.benchmark:
        measured = base.run_native(args.native, ["--benchmark", "--threads", args.threads])
        native_samples = json.loads(measured[0])["samples_ns"]
        samples, outputs = benchmark_python()
        mismatch = base.first_difference(outputs, measured[1:])
        result["benchmark"] = {"native_samples_ns": native_samples, "python_samples_ns": samples,
            "native_median_ns": statistics.median(native_samples), "python_median_ns": statistics.median(samples),
            "speedup": statistics.median(samples) / statistics.median(native_samples),
            "output_sha256": hashlib.sha256("\n".join(outputs).encode()).hexdigest(), "mismatch": mismatch}
        if mismatch: result["status"] = "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "records", "case_mismatches", "benchmark")}))
    return 0 if result["status"] == "PASS" else 1

if __name__ == "__main__": raise SystemExit(main())
