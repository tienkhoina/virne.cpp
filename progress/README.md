# Progress hot-path contract

`Progress` keeps its existing public signatures while making `update()` cheap
enough for tight loops:

- state changes are checked before any formatting;
- rendering is limited to one refresh per 50 ms, except completion;
- the line and postfix buffers are reused and numbers use `to_chars`;
- `set_postfix()` only copies changed data and defers sorting/formatting until
  the next permitted refresh;
- postfix keys are sorted for deterministic output;
- TTY output uses one carriage return plus padding, while redirected output
  contains newline-terminated text and no terminal escape codes;
- progress is clamped to `total`, zero-total is a valid completed bar, and
  `finish()` is idempotent.

`Progress::update(current)` continues to receive the absolute completed count.
`Progress::advance(delta = 1)` provides a saturating incremental operation,
while `update_safe()` additionally rejects a decreasing absolute count and
reports validation errors through `std::clog`.

`TqdmProgress` is the semantic adapter for the original Virne loops:

```cpp
TqdmProgress pbar(total, "Running");
pbar.update();       // identical meaning to tqdm.update(1)
pbar.update(4);      // advance by four
pbar.set_postfix({{"accepted", 3}});
pbar.close();
```

Its `update(n)`, `set_postfix(...)`, and `close()` behavior covers every
non-Torch tqdm call in the original `system` layer. Both classes expose
read-only `current()`, `total()`, and `finished()` state for orchestration and
tests.

Run behavior tests and the opt-in microbenchmark with:

```bash
ctest --test-dir build -R '^progress_behavior$' --output-on-failure
cmake --build build --target progress_benchmark -j2
build/progress/progress_benchmark
```

The benchmark reports absolute `update`, incremental `advance`, tqdm-adapter
`update`, and changed-postfix throughput. Timings vary by host, so it is kept
outside the default build and results should be recorded from the target
machine rather than treated as a fixed promise.

### Reference Release run (2026-07-24 08:56 UTC)

The final execution reports the median of five samples. Host load affects
absolute throughput; the behavior tests, throttle semantics and allocation
profile are the durable contract.

| Hot-path operation | Calls | Throughput |
|---|---:|---:|
| `Progress::update(absolute)` | 2,000,000 | 13.8371 million calls/s |
| `Progress::advance(delta)` | 2,000,000 | 13.7093 million calls/s |
| `TqdmProgress::update(delta)` | 2,000,000 | 18.2101 million calls/s |
| Changed postfix + update | 100,000 | 1.48981 million calls/s |
