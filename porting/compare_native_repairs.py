#!/usr/bin/env python3
"""Focused corrected expectations for confirmed Python defects; files stay intact."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import torch
import compare_rl_policy_remaining as reference
import compare_training_update as pinned
from compare_learning_session import native_command


class SingleLayerCorrection(ast.NodeTransformer):
    def visit_For(self,node):
        if isinstance(node.target,ast.Name) and node.target.id == "layer_id":
            original = node.body[0]
            node.body[0] = ast.If(test=ast.parse("self.num_layers == 1",mode="eval").body,
                body=ast.parse("lin = nn.Linear(self.input_dim, output_dim)\nnorm = nn.Identity()").body,orelse=[original])
        return self.generic_visit(node)


tree = SingleLayerCorrection().visit(pinned.source("neural_network/mlp.py"))
ast.fix_missing_locations(tree)
mlp = {}
exec(compile(tree,"neural_network/mlp.py","exec"),mlp)


def single_layer():
    torch.manual_seed(739)
    model = mlp["MLPNet"](4,3,num_layers=1,batch_norm=True,dropout_prob=.25)
    value = reference.values((2,4)).requires_grad_()
    return reference.report(model,[model(value)],[value],True)


def residual_initialization():
    torch.manual_seed(743)
    model = reference.InitializedDeepGAT(3,2,1,num_layers=3,embedding_dim=4,dropout_prob=0.)
    data = reference.graph(3); data.x.requires_grad_(); data.edge_attr.requires_grad_()
    return reference.report(model,[model(data)],[data.x,data.edge_attr],True)


def recurrent_clamp():
    torch.manual_seed(751)
    model = reference.SEQ["GcnSeq2SeqActorCritic"](4,3,1,2,embedding_dim=8,dropout_prob=0.,batch_norm=False)
    model.actor.decoder.p_net_num_nodes = 4
    model.critic.decoder.p_net_num_nodes = 4
    obs = {"p_net":reference.graph(3),"v_net_x":reference.values((2,3,2)).requires_grad_(),"mask":None}
    obs["p_net"].x.requires_grad_()
    encoded = model.encode(obs)
    obs.update(hidden_state=model.get_last_rnn_state().permute(1,0,2),encoder_outputs=encoded.transpose(0,1),
        p_node_id=torch.tensor([77,101]))
    logits = model.act(obs); hidden = model.get_last_rnn_state(); value = model.evaluate(obs)
    return reference.report(model,[encoded,logits,hidden,value],[obs["v_net_x"],obs["p_net"].x],True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    expected = []
    for name,run in (("single_layer",single_layer),("residual_initialization",residual_initialization),("recurrent_clamp",recurrent_clamp)):
        expected += ["CASE "+name,*run()]
    actual = native_command(args.native,[])
    mismatch = reference.base.first_difference(expected,actual)
    report = {"component":"native_repairs","status":"PASS" if mismatch is None else "FAIL","mismatch":mismatch,
        "cases":[line[5:] for line in expected if line.startswith("CASE ")],"records":len(expected),
        "reference_corrections":{"single_layer":"one output Linear plus Identity, avoiding indexing an empty hidden-dimension list",
            "residual_initialization":"orthogonal residual matrices after original construction",
            "recurrent_clamp":"decoder bound is physical-node count 4, not 100, for an embedding with 5 rows"},
        "source_sha256":{key:pinned.PINNED[key] for key in ("neural_network/mlp.py","neural_network/gnn.py","reinforcement_learning/gnn_seq2seq_solver/policy_with_encoder.py")},
        "python_output_sha256":hashlib.sha256("\n".join(expected).encode()).hexdigest(),
        "native_output_sha256":hashlib.sha256("\n".join(actual).encode()).hexdigest(),
        "native_sha256":hashlib.sha256(args.native.read_bytes()).hexdigest(),"threads":1,"torch":torch.__version__,
        "benchmark":"Regression-only gate for existing modules; accepted aggregate benchmarks were not rerun"}
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({key:report[key] for key in ("status","records","mismatch")}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
