# Native shared optimizer

Updated 2026-09-09. `rl_core/shared_adam.{h,cpp}` implements one native owner
shared by worker threads. `apply_gradients` validates/copies a complete packet,
clips and updates under one mutex; `snapshot` returns coherent copies of model
parameters, Adam moments and a monotonically increasing packet version.
`restore` restores both model and optimizer state. Optimizer groups and optional
AMSGrad lanes are retained. The default beta2 is 0.99 as in the source class.

The oracle imports the original `rl_core/shared_adam.py`, SHA256
`99b51f365e9853274a9477cef5dade8a0000e28dd13cc9f1dd7e223f9d9d673b`.
Original Python remains unchanged. Its eager integer step counter is invalid
for Torch 2.6, and its AMSGrad groups lack `max_exp_avg_sq`. Corrected expectations
replace the integer by a scalar tensor and initialize the missing maximum lane.
No other source optimizer logic is replaced.

## Exact gate

`results/shared_adam_differential_2026-09-09.json`: **3 cases, 156 records, PASS**.
Cases cover mixed Float/Double groups, skipped gradients, clipping, AMSGrad,
snapshot/restore, rejected malformed packets and four native worker threads.
The worker case applies 64 identical packets, checks that every version 1..64
appears exactly once and checks moment counters against each coherent snapshot.
The final state matches the equivalent sequential Python updates byte-for-byte.
Python IPC/shared-memory compatibility is not claimed by this native thread API.

## Frozen benchmark

`results/shared_adam_benchmark_2026-09-09.json` reuses the passing report without
rerunning its differential. The workload is 32 sequential gradient packets,
including packet cloning, zeroing gradients, clipping and Adam updates. State
construction and output serialization are outside the timer on both sides.
The same container uses CPU affinity 2 and one Torch thread, with three warmups
and five measured samples.

| Runtime | Median |
| --- | ---: |
| C++ | 1.576935 ms |
| Python | 14.236201 ms |

Measured Python/C++ ratio: **9.027767x**. Final parameters, step counters and
moments match, with output SHA256
`4ad1a4d361a2ed9fab3b4906ceb53c22ee0064ebe5bbca3a061841fbf740c928`.
This is a sequential packet benchmark, not evidence of parallel throughput gain.

Build target: `vne_shared_adam_harness`. Oracle:
`porting/compare_shared_adam.py --native <binary> --output <report>`.
The benchmark uses `--benchmark-from <passing-report>`. Container/library
configuration is the same as `learning_session.md`. Accepted artifacts are
frozen; do not rerun them during subsequent RL work.
