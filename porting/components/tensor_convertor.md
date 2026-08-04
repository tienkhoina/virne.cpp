# TensorConvertor dense policy paths

This leaf ports the Torch-only dense paths from
`rl_core/tensor_convertor.py`: physical-node features, current virtual-node
features, fixed general fields, and the MLP/CNN/MLP-FC/attention compositions.
PyG graph batching, heterogeneous graphs and RNN state conversion remain
separate components.

## Typed C++ API

`virne/solver/learning/rl_core/tensor_convertor.h` defines direct fixed fields:

```cpp
struct DenseObservation {
    torch::Tensor p_net_x;                 // [Np,Fp]
    torch::Tensor v_node_x;                // [Fv]
    std::optional<std::int64_t> curr_v_node_id;
    std::optional<torch::Tensor> action_mask;
    std::optional<double> v_net_size;
};

struct DenseObservationBatch {
    torch::Tensor p_net_x;                 // [B,Np,Fp]
    torch::Tensor v_node_x;                // [B,Fv]
    std::optional<torch::Tensor> curr_v_node_id;
    std::optional<torch::Tensor> action_mask;
    std::optional<torch::Tensor> v_net_size;
};
```

`TensorConvertor` overloads each operation for a single observation and an
already stacked batch:

- `p_net_x_obs_as_tensor` and `v_node_obs_as_tensor`;
- `general_obs_as_tensor`;
- `obs_as_tensor_for_mlp`;
- `obs_as_tensor_for_cnn`;
- `obs_as_tensor_for_mlp_fc`;
- `obs_as_tensor_for_att`.

Outputs also use typed structs (`MlpTensorObservation`,
`MlpFcTensorObservation`, `AttentionTensorObservation` and
`GeneralTensorObservation`). A Python string key is resolved once before this
boundary; no fixed field is stored in a string map.

## Layout and dtype contract

All dense feature/action/size tensors become float32 on the requested device;
`curr_v_node_id` becomes int64. A single observation gains a leading batch
dimension. For MLP, `v_node_x[B,Fv]` is expanded across physical nodes and
concatenated with `p_net_x[B,Np,Fp]`, producing `[B,Np,Fp+Fv]`. CNN adds a
channel dimension, MLP-FC flattens physical nodes before concatenation, and
attention preserves the two dense inputs separately.

For the shipped multi-resource feature schema, `Fp=14`, `Fv=11`, so MLP output
is `[B,500,25]` and feeds `MlpActorCritic` directly. Missing general fields
remain disengaged optionals. Invalid ranks or incompatible batch sizes fail
before Tensor concatenation.

## Performance and devices

Input tensors are converted with `copy=false`: matching float32 tensors already
on the target device are reused, while dtype/device changes allocate only when
required. Virtual-node expansion is a stride-zero view; only the final required
concatenation materializes it. Batch/list assembly and dynamic-key traversal are
removed from the C++ hot boundary.

The API accepts a typed `torch::Device` and has no CPU-only operation. Configure
Torch thread pools once at process startup. This converter does not create
outer workers; independent observation batches can be scheduled by the caller
without changing per-batch order.

## Verification

```powershell
cmake --build build-ml-win-msvc --config Release `
  --target vne_tensor_convertor_unit vne_tensor_convertor_harness --parallel 1
ctest --test-dir build-ml-win-msvc -C Release `
  -R '^vne_tensor_convertor_unit$' --output-on-failure

& .\.deps-cache\python-torch26\Scripts\python.exe `
  .\porting\compare_tensor_convertor.py `
  --harness .\build-ml-win-msvc\porting\Release\vne_tensor_convertor_harness.exe `
  --python-source ..\virne\virne\solver\learning\rl_core\tensor_convertor.py `
  --threads 1 --warmups 3 --repetitions 5 --iterations 20
```

The comparator pins the full Python file hash, AST-extracts only
`TensorConvertor`, rejects Virne/PyG imports, pins NumPy 1.26.4, and requires
bit-exact output for every dense path before accepting timing.
