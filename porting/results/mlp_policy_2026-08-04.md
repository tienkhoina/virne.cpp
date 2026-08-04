# MLP policy exact gate and frozen benchmark - 2026-08-04

## Environment and correctness

| Item | Value |
| --- | --- |
| Host/compiler | Windows x64, MSVC 19.41.34123, Release |
| Python / Torch | CPython 3.12.3 / 2.6.0+cpu |
| C++ LibTorch | 2.6.0+cpu (`libs/libtorch-win`) |
| Thread pools | intra-op 1, inter-op 1 |
| Policy source | 1936 bytes, SHA-256 `c8998d8bbfb14477c55fbcf9240bf4e0d82ff1e5c51da7b4aa82387432a2e407` |
| MLP source | 1759 bytes, SHA-256 `4f6d24dde669ec310fd2961bb1a6c0945f5a2d93f2042402025acf094eba89cb` |

`vne_mlp_policy_unit` passed 1/1. The cross-language gate passed all cases
without a shape, dtype, state, RNG, buffer, byte, or checksum mismatch:

| Case | Result |
| --- | --- |
| composite initialization and next RNG | PASS |
| rank-3 Actor/Critic forward | PASS |
| multi-resource `[8,500,25]` policy input | PASS |
| `action_dim=1` | PASS |
| `action_dim=997` | PASS |
| standalone Actor BatchNorm eval | PASS |
| standalone Critic BatchNorm eval | PASS |

Result: **7/7 bit-exact PASS**. The explicit action-dimension invariant also
passed: values 1 and 997 produced identical state, Actor output and Critic
output.

## Frozen benchmark

The representative operation executes Actor then Critic with three MLP layers,
hidden width 128, no BatchNorm and float32 input `[8,500,25]`. Each timing
sample contains 20 complete Actor+Critic pairs after three warm-up samples.

| Runtime | Five samples, ns per 20 pairs | Median per pair |
| --- | --- | ---: |
| Python | `184086600, 200896300, 205721400, 185170800, 181529900` | 9,258,540 ns |
| C++ | `219614600, 159642500, 155909700, 181171800, 224713100` | 9,058,590 ns |

C++ speedup: **1.0220729716x**. Timed output parity passed:

- Actor `[8,500]`, 16,000 bytes, checksum `11584385858004032579`;
- Critic `[8,1]`, 32 bytes, checksum `6721433511023778797`.

This benchmark is frozen. Do not rerun or retune it unless correctness or
performance changes require reopening the policy leaf.
