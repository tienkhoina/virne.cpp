# Component: `virne.utils.stats`

State: **COMPLETE** on 2026-07-28.

This document is the implementation contract for this leaf.  The Python
source was audited once before implementation; subsequent implementation work
must use this document and must not reread the Python source unless an exact
differential mismatch requires a focused investigation.

Implementation checkpoint:

- `virne/utils/stats.h` contains the typed wrapper and factories;
- `virne/utils/stats.cpp` contains the wall clock, fixed-four formatter, and
  byte-preserving stream emission;
- `vne_utils_stats` is a standard-library-only leaf target, and the aggregate
  `vne_utils` target links it instead of compiling `stats.cpp` directly;
- the leaf passes a host GCC 14.2 C++17 compile with `-Wall -Wextra
  -Wpedantic -Werror`;
- an ephemeral inline smoke executable passed representative formatting,
  move-only/reference/void, dynamic-name, embedded-NUL, nested-call,
  wall-clock, callable/clock, and sink-failure checks.
- the C++ fixed-four formatter matched Python's formatter for 20,010
  deterministic binary64 bit patterns, including signed zero, subnormals,
  maximum finite values, NaNs, and both infinities.

The permanent differential, concurrency, Docker, sanitizer, and canonical
benchmark gates listed below now supersede that early smoke run.

Focused oracle note (2026-07-27): the first permanent differential run could
not inject its scripted clock through a module-level `stats.time` attribute.
The source identity above was rechecked and the exact binding site was opened:
`import time` occurs inside `test_running_time`, so the standard `time` module
object is held in the returned closure and is not exported from `stats.py`.
This confirms the contract below (module object captured at decoration,
`time.time` resolved on each read).  The runner therefore temporarily replaces
the standard module object's `time` attribute and restores it after every
case; no C++ production change was required.

## Source identity

- Original repository commit:
  `d1ec1e4a20461fc9bad50612ad5026fd31e693a8`.
- Checkout bytes use CRLF; SHA-256:
  `68bee247bcaf9cc81009c290c3ebb4013078ab0418231fcdff58a28d1b8480c9`.
- Canonical LF SHA-256:
  `6e5080b32ed0951720f6f922e94e6a686ce06c64acbc1047fe94076e60c31cf6`.
- Git blob:
  `5ef358b09bf2fd41ce41307d553aab1f1fbf4c90`.

## Python-observable contract

The source exposes only `test_running_time(func) -> wrapper`.  Calling the
wrapper forwards every positional and keyword argument to `func` and returns
the exact result object.

On success, observable evaluation order is:

1. read the wall clock (`t1 = time.time()`);
2. invoke `func` with forwarded arguments;
3. read the wall clock again (`t2 = time.time()`);
4. read the callable's current `__name__`;
5. format `t2 - t1` with fixed precision `2.4f`;
6. print exactly `Running time of <name>: <fixed4>s\n` to standard output,
   without requesting a flush;
7. return the callable's exact result.

The source intentionally uses a wall clock, not a monotonic clock.  Negative
deltas, negative zero, NaN, positive infinity, and negative infinity are
valid formatting inputs.

The `time` module object is captured when decoration occurs, while its `time`
attribute is resolved for each clock read.  Python `functools.wraps` metadata
is a Python-only boundary and has no native C++ reflection equivalent.

Generators and coroutines are timed only until their generator/coroutine
object is created; later iteration or awaiting is outside this wrapper call.

### Exception and side-effect order

- If the first clock read throws, the callable is not invoked and nothing is
  written.
- If the callable throws, there is exactly one clock read, no output, and the
  original callable exception propagates unchanged.
- If the second clock read throws, the callable has completed, there is no
  output, and the clock exception wins over the unreturned result.
- If name access/formatting/output fails, both clock reads and the callable
  have completed, but the result is not returned.  A C++ stream failure must
  surface as `std::ios_base::failure`.
- The name is observed after the callable.  Mutating the wrapper's name during
  the call changes the same invocation's output line.
- Nested wrappers print the inner line before the outer line.

## C++ API

Namespace: `virne::utils`.

```cpp
struct SystemWallClock
{
    double operator()() const noexcept;
};

template <typename Callable, typename Clock = SystemWallClock>
class RunningTimeFunction
{
public:
    RunningTimeFunction(
        std::string function_name,
        Callable callable,
        std::ostream& output,
        Clock clock = {});

    template <typename... Args>
    decltype(auto) operator()(Args&&... args);

    const std::string& function_name() const noexcept;
    void set_function_name(std::string function_name);

    Callable& target() noexcept;
    const Callable& target() const noexcept;
};

template <typename Callable>
auto test_running_time(
    std::string function_name,
    Callable&& callable,
    std::ostream& output = std::cout);

template <typename Callable, typename Clock>
auto test_running_time(
    std::string function_name,
    Callable&& callable,
    std::ostream& output,
    Clock&& clock);
```

The factory stores decayed callable and clock types.  Callers that require
identity with an existing lvalue callable use `std::ref`.  The stream is a
non-owning reference and must outlive the wrapper.

Fixed fields are direct typed members:

- `callable_`;
- `clock_`;
- non-owning `std::ostream*`;
- `function_name_`;
- a cached output prefix derived from the dynamic name.

There is no dynamic map, field ID, string dispatch, `std::function`, or worker
API.  The dynamic callable name is copied and resolved once at the boundary;
`set_function_name` rebuilds the cached prefix.

## Return and forwarding rules

Invocation uses `std::invoke` and perfect forwarding.  The implementation has
explicit compile-time paths for:

- `void` results;
- lvalue and rvalue reference results, preserving identity and value category;
- ordinary value results, including move-only and non-default-constructible
  values.

The callable must run exactly once.  No result may be copied merely to support
timing.  The success line is emitted before a stored value is returned.  A
reference return must refer to the same object as the direct callable result.

## Formatting contract

The suffix is always seconds with exactly four digits after the decimal point:

```text
Running time of <name>: <value>s\n
```

Formatting must be locale-independent and match Python's fixed-four output,
including rounding boundaries and these spellings:

- `-0.0000` for negative zero;
- `nan`;
- `inf`;
- `-inf`.

Names are byte-preserving.  ASCII, UTF-8, embedded newline, and embedded NUL
bytes are all retained in the output.  Cached prefix construction must not use
NUL-terminated assumptions.

## Ownership and thread safety

The wrapper owns its decayed callable and clock and follows their copy/move
capabilities.  `target()` returns the owned object; `std::reference_wrapper`
is the explicit opt-in for external lvalue identity.

Concurrent calls through one wrapper are safe only when all of these hold:

- the callable is safe for those calls;
- the clock is safe for those calls;
- no thread calls `set_function_name` concurrently;
- the selected output stream/sink is safe for the required output behavior.

Separate wrapper instances with separate sinks may run concurrently.  The
component adds no global lock.  Parallel wrapper execution is not part of the
API because it would change callable, clock, and output side-effect order.
External throughput tests may use 1, 2, 4, and 8 independent wrappers.

## Complexity and dependencies

- Construction and name replacement: `O(name length)`.
- Accessors: `O(1)`.
- Invocation overhead beyond the callable: two clock calls plus
  `O(output bytes)` formatting/writing.
- Production dependencies: C++17 standard library only, using
  `std::chrono::system_clock`, `std::invoke`, typed storage, and native
  formatting/stream support.
- The isolated target is `vne_utils_stats`.
- It must not link graph, YAML, Boost, Threads, progress, CSV, config, random,
  or other Virne components.
- `stats.cpp` is removed from the monolithic utility target's direct source
  list and the leaf target is linked back into that aggregate target.

## Required verification matrix

Unit and differential work must cover:

- ASCII, UTF-8, newline-containing, and NUL-containing names;
- scalar value, move-only value, non-default-constructible value, `void`,
  lvalue-reference, and rvalue-reference returns;
- exact argument/reference/value-category forwarding and one invocation;
- clock deltas for zero, negative zero, negative values, rounding boundaries,
  large finite values, NaN, and both infinities;
- name mutation during the callable and cached-prefix replacement;
- nested wrappers and exact inner-before-outer output;
- first-clock, callable, second-clock, and sink exceptions with exact side
  effect counts/order;
- generator-like lazy object construction boundary;
- wrapper ownership, `std::ref` identity, `target()` access, and copy/move
  traits inherited from callable/clock;
- wall-clock default smoke test without asserting a non-negative duration;
- external throughput stress using independent wrappers at 1, 2, 4, and 8
  threads, with no production worker API.

The differential oracle must compare output bytes, return identity/category,
call/clock counts, and exception stage.  Timing benchmarks must keep callable
work identical and separate formatting/sink costs from the callable payload.

## Permanent verification artifacts

The leaf now has a standalone verification stack under `porting/`:

- `stats_unit.cpp` locks C++-specific semantics: typed ownership,
  `std::ref`, perfect forwarding, `void`/value/lvalue-reference/
  rvalue-reference returns, move-only/non-default values, fixed-four
  formatting, byte-preserving names, name mutation, nesting, lazy creation,
  exception order, sink failures, and a default wall-clock smoke case;
- `stats_harness.cpp` emits stable tab-separated differential records and has
  a benchmark mode.  Benchmark worker threads own separate wrappers, clocks,
  callable state, and hash sinks; a start gate excludes thread construction
  and a finish gate excludes thread teardown from the measured interval;
- `compare_stats.py` executes the frozen Python file directly without loading
  the wider Virne package and compares exact UTF-8/output bytes, normalized
  results/identity, invocation and clock counts, lazy state, and exception
  stage against the C++ harness;
- `benchmark_stats.py` compares identical 64-bit payload work in direct and
  wrapped modes.  Both languages use deterministic alternating clocks and
  FNV-1a sinks, and timing is accepted only after return/output checksums,
  byte counts, call counts, and clock counts agree exactly.  It discards five
  warm-ups, records 31 paired alternating samples, and reports median, MAD,
  and p95 for both the direct baseline and wrapped path;
- `sweep_stats_workers.py` repeats that gate for independent 1, 2, 4, and 8
  workers, emits a SHA-256 digest over the non-timing sweep fields, and writes
  the complete machine-readable JSON result when `--json-output` is supplied.

The production component still has no worker API and no Threads dependency;
threading exists only in `stats_harness.cpp`.

### Reproduction commands

From the C++ repository root, with `<build-dir>` outside the frozen trees:

```text
g++ -std=c++17 -Wall -Wextra -Wpedantic -Wconversion \
  -Wsign-conversion -Werror \
  -I virne/utils porting/stats_unit.cpp virne/utils/stats.cpp \
  -o <build-dir>/stats_unit

g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Wconversion \
  -Wsign-conversion -Werror -pthread \
  -I virne/utils porting/stats_harness.cpp virne/utils/stats.cpp \
  -o <build-dir>/stats_harness

<build-dir>/stats_unit
python porting/compare_stats.py \
  --harness <build-dir>/stats_harness --python-root ../virne
python porting/benchmark_stats.py \
  --harness <build-dir>/stats_harness --python-root ../virne \
  --workers 1 --iterations 20000 --warmups 5 --repeats 31
python porting/sweep_stats_workers.py \
  --harness <build-dir>/stats_harness --python-root ../virne \
  --iterations 20000 --warmups 5 --repeats 31 --workers 1 2 4 8 \
  --json-output porting/results/stats_sweep_2026-07-28.json
```

On Windows/MinGW the executable names may carry `.exe`; the Python runners
accept either form.

### Completion record (2026-07-28)

- isolated Release Docker build and permanent unit execution: **PASS**;
- strict production, unit, and harness build with conversion/sign-conversion
  warnings as errors: **PASS**;
- Python syntax compilation for all three runners: **PASS**;
- exact Python/C++ differential: **PASS**, 22 cases;
- canonical Docker sweep with five warm-ups, 31 paired samples, and independent
  worker widths 1/2/4/8: **PASS**;
- exact return/output/byte/call/clock invariants in every timed sample:
  **PASS**;
- ASan/UBSan with leak detection and 100 repeated unit runs: **PASS**;
- full repository Release build and all 18 CTests: **PASS**;
- frozen graph/CSV/config/yaml-cpp integrity: **PASS**.

Canonical total-wrapper speedups for workers 1/2/4/8 are respectively
52.028x, 100.870x, 108.311x, and 170.820x.  Pure wrapper-overhead speedups are
45.106x, 87.161x, 89.168x, and 145.665x.  Thread creation and teardown are
outside the measured interval; production deliberately remains worker-free.
The full timing distributions and checksums are recorded in
`porting/results/stats_2026-07-28.md` and
`porting/results/stats_sweep_2026-07-28.json`.
