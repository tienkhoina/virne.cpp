# Dense TensorConvertor exact gate and frozen benchmark - 2026-08-04

## Correctness

| Item | Value |
| --- | --- |
| Host/compiler | Windows x64, MSVC 19.41.34123, Release |
| Python | CPython 3.12.3, NumPy 1.26.4, Torch 2.6.0+cpu |
| C++ | LibTorch 2.6.0+cpu |
| Source | 23,806 bytes |
| Source SHA-256 | `75a13356834688d6dff77f7e5fc78ca64cf49171f183aedce92116ad31035f2b` |
| Thread pools | intra-op 1, inter-op 1 |

`vne_tensor_convertor_unit` passed, including direct composition into the
25-feature MLP policy. Differential result: **9/9 bit-exact PASS**:

- single and batch MLP;
- batch MLP with all optional general fields absent;
- single and batch CNN;
- single and batch MLP-FC;
- single and batch attention.

All output shapes, dtypes, bytes, FNV-1a checksums and individual float/int bits
matched the isolated Python class.

## Frozen benchmark

The representative batch uses float64 source features `p_net_x[8,500,14]` and
`v_node_x[8,11]`, plus fixed general fields, and produces float32
`p_net_x[8,500,25]`. Raw fixtures are created outside timing. Python retains
its original list/NumPy-to-Torch conversion; C++ receives the typed stacked
Tensor boundary and performs dtype conversion, expansion and concatenation.

| Runtime | Five samples, ns per 20 conversions | Median per conversion |
| --- | --- | ---: |
| Python | `12517900, 12999300, 12494800, 7838200, 7439800` | 624,740 ns |
| C++ | `4394000, 4020000, 3992400, 4189900, 4087200` | 204,360 ns |

C++ speedup: **3.0570561754x**. The timed 400,000-byte output checksum is
`16030253265787721722` on both sides.

This benchmark is frozen and must not be rerun or retuned unless the dense
converter is explicitly reopened.
