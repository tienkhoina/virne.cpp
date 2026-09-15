#!/usr/bin/env python3
"""Convert a native VIRNE checkpoint to a Python policy state_dict.

The native runtime stores each tensor with torch::pickle_save and keeps the
policy parameter order identical to the Python module's named_parameters()
order.  The output is a policy-only PyTorch checkpoint, so the original
Python RLSolver.load_model path accepts it without changing ../virne.
"""
from __future__ import annotations

import argparse
import io
import struct
from pathlib import Path

import torch


MAGIC = b"VIRNE_NATIVE_CHECKPOINT_2"
FIELD_COUNT = 16


class Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 0

    def u64(self) -> int:
        if self.offset + 8 > len(self.data):
            raise ValueError("truncated native checkpoint")
        value = struct.unpack_from("<Q", self.data, self.offset)[0]
        self.offset += 8
        return value

    def tensor(self) -> torch.Tensor | None:
        size = self.u64()
        if size == (1 << 64) - 1:
            return None
        end = self.offset + size
        if end > len(self.data):
            raise ValueError("invalid native checkpoint tensor size")
        value = torch.load(
            io.BytesIO(self.data[self.offset:end]),
            map_location="cpu",
            weights_only=False,
        )
        self.offset = end
        if not isinstance(value, torch.Tensor):
            raise ValueError("native checkpoint blob is not a tensor")
        return value


def tensor_required(reader: Reader, name: str) -> torch.Tensor:
    value = reader.tensor()
    if value is None:
        raise ValueError(f"native checkpoint field {name} is undefined")
    return value


def read_checkpoint(path: Path) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int]]:
    data = path.read_bytes()
    if not data.startswith(MAGIC):
        raise ValueError("not a VIRNE_NATIVE_CHECKPOINT_2 file")
    reader = Reader(data[len(MAGIC):])
    configuration_i = tensor_required(reader, "configuration_i")
    tensor_required(reader, "configuration_f")
    dimensions = tensor_required(reader, "dimensions").to(torch.int64).tolist()
    if len(dimensions) != 6:
        raise ValueError("native checkpoint dimensions must contain six values")
    parameter_count = reader.u64()
    parameters = [tensor_required(reader, f"parameter_{i}") for i in range(parameter_count)]
    buffer_count = reader.u64()
    buffers = [tensor_required(reader, f"buffer_{i}") for i in range(buffer_count)]
    tensor_required(reader, "training")
    tensor_required(reader, "update_time")
    tensor_required(reader, "rng")
    optimizer_present = reader.u64()
    moment_count = reader.u64()
    for _ in range(moment_count):
        reader.tensor(); reader.tensor(); reader.tensor(); reader.tensor()
    tensor_required(reader, "optimizer_groups")
    tensor_required(reader, "actions")
    tensor_required(reader, "rewards")
    tensor_required(reader, "returns")
    tensor_required(reader, "logprobs")
    observation_count = reader.u64()
    for _ in range(observation_count):
        for _ in range(FIELD_COUNT):
            reader.tensor()
    q_learning = reader.u64()
    if q_learning:
        target_parameter_count = reader.u64()
        for _ in range(target_parameter_count):
            reader.tensor()
        target_buffer_count = reader.u64()
        for _ in range(target_buffer_count):
            reader.tensor()
        tensor_required(reader, "target_training")
        tensor_required(reader, "dones")
        next_observation_count = reader.u64()
        for _ in range(next_observation_count):
            for _ in range(FIELD_COUNT):
                reader.tensor()
        tensor_required(reader, "exploration_key")
        tensor_required(reader, "exploration_i")
        tensor_required(reader, "exploration_f")
    if reader.offset != len(reader.data):
        raise ValueError("native checkpoint has trailing data")
    if optimizer_present not in (0, 1):
        raise ValueError("invalid optimizer marker")
    return parameters, buffers, dimensions


def make_policy(dimensions: list[int]) -> torch.nn.Module:
    # Import through the source package only after dependency validation. This
    # utility is run in the full Python image; ../virne remains read-only.
    from virne.solver.learning.rl_policy.dual_gnn_policy import (
        BiDeepEdgeFeatureGatActorCritic,
    )

    return BiDeepEdgeFeatureGatActorCritic(
        dimensions[0], dimensions[1], dimensions[4], dimensions[3], dimensions[5],
        embedding_dim=128, dropout_prob=0.0, batch_norm=False, num_layers=3,
    )


def convert(source: Path, destination: Path) -> dict[str, object]:
    parameters, buffers, dimensions = read_checkpoint(source)
    policy = make_policy(dimensions)
    named_parameters = list(policy.named_parameters())
    named_buffers = list(policy.named_buffers())
    if len(named_parameters) != len(parameters):
        raise ValueError(f"policy parameter count differs: python={len(named_parameters)} native={len(parameters)}")
    if len(named_buffers) != len(buffers):
        raise ValueError(f"policy buffer count differs: python={len(named_buffers)} native={len(buffers)}")
    state = policy.state_dict()
    for (name, target), source_tensor in zip(named_parameters, parameters):
        if tuple(target.shape) != tuple(source_tensor.shape):
            raise ValueError(f"parameter shape differs for {name}")
        state[name] = source_tensor.to(dtype=target.dtype)
    for (name, target), source_tensor in zip(named_buffers, buffers):
        if tuple(target.shape) != tuple(source_tensor.shape):
            raise ValueError(f"buffer shape differs for {name}")
        state[name] = source_tensor.to(dtype=target.dtype)
    policy.load_state_dict(state, strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, destination)
    return {
        "source": str(source),
        "destination": str(destination),
        "dimensions": dimensions,
        "parameters": len(parameters),
        "buffers": len(buffers),
        "format": "Python state_dict (optimizer intentionally omitted)",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(convert(args.source, args.destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
