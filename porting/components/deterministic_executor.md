# Internal API: persistent deterministic executor

Added 2026-07-30 for native integration hot paths. This is an internal C++
utility; it does not change Python-compatible result types or ordering.

```cpp
template <typename Function>
void virne::utils::deterministic_parallel_blocks(
    std::size_t count,
    std::size_t requested_workers,
    std::size_t minimum_items_per_worker,
    Function&& function); // function(begin, end)
```

Worker `0/1` executes the complete range sequentially. Wider widths are
capped only by item count and the caller-provided deterministic grain; host
CPU count is never consulted. Blocks are contiguous and input ordered.

The process-wide pool grows to the largest requested width and persists until
shutdown. Each submission is serialized at the executor boundary. A typed
`noexcept` function-pointer callback and stack context avoid `std::function`
allocation in hot dispatch. Completion is atomic; a bounded pause window
amortizes adjacent short batches, while workers excluded by a narrower width
park immediately. Nested executor calls run sequentially on the current thread
to prevent recursive-pool deadlock and oversubscription.

Task publication is exception-safe: the pool grows before stack-backed task
state is published. Every worker snapshots generation, callback, context, and
active width under `ready_mutex_`; therefore a worker may skip an intermediate
generation but can never execute a later generation twice or touch a dead
context. Per-index failures are retained and rethrown in input order.

The utility introduces no strings or dynamic field lookup in hot loops. Focused
NodeMapper, LinkMapper, Controller, Counter, Recorder, and Environment units
pass. A ClangCL stress sweep of 20,000 calls varying count, grain, and worker
width (including width changes from 0 through 9) passed exact one-visit range
coverage. The frozen module benchmarks were rerun once after this shared
executor change; their same-compiler A/B results are in
`porting/results/hot_path_old_vs_new_2026-07-31.md`.
