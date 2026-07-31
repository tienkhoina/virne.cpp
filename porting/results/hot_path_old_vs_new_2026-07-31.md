# Hot-path benchmark: same-compiler old vs final native build

Date: 2026-07-31 (Asia/Saigon)

Status: **PASS**. The accepted Python/Linux records and their benchmark source
remain frozen. The table below is a controlled native A/B: the old accepted
Windows executable and the final executable use the same ClangCL toolchain,
Release flags, dependency objects, fixture, and benchmark source. Only the
ported implementation is different.

## Toolchain and protocol

- Compiler: ClangCL 19.1.7, C++17, `/O2 /Ob2 /DNDEBUG /MD`, `lld-link`.
- Generator: CMake NMake Makefiles; frozen graph/csv/yaml/random libraries
  were reused.
- One process warm-up, then interleaved old/final samples. Counter, Recorder,
  NodeMapper, and LinkMapper use nine outer samples; Controller and Environment
  use seven outer samples over their built-in median. Controller W1 and
  LinkMapper W1 were confirmed with fifteen and twenty-one samples because
  their sub-millisecond/short fixtures are scheduler-noisy.
- The reported value is the outer median of the benchmark's timed boundary;
  process startup and fixture construction are outside that boundary where the
  benchmark defines them as such.

## Controlled native A/B (milliseconds)

Each cell is `old -> final (old/final)`. A ratio above `1` is a final-build
speedup. All rows passed the exact output gate.

| Module | W1 | W2 | W8 | Output gate |
|---|---:|---:|---:|---|
| Counter | 26.8444 -> 26.5589 (1.011x) | 22.1627 -> 22.2032 (0.998x) | 21.1854 -> 20.1579 (1.051x) | `3910809078534895256` |
| Recorder | 5.3091 -> 5.0645 (1.048x) | 7.4428 -> 5.1795 (1.437x) | 8.6792 -> 5.4048 (1.606x) | `8168332940057982619` |
| Controller | 2.4988 -> 2.4872 (1.005x) | 5.1737 -> 3.6586 (1.414x) | 5.1718 -> 3.3370 (1.550x) | restore `8486823302284311477`; deployed `17514356897791579542` |
| NodeMapper | 5.7027 -> 5.3768 (1.061x) | 17.1777 -> 15.0716 (1.140x) | 26.4855 -> 14.2869 (1.854x) | `15604526718891224062` |
| LinkMapper | 1.0068 -> 0.9709 (1.037x) | 2.4686 -> 2.5779 (0.958x) | 2.7254 -> 2.5227 (1.080x) | `14052633754962558449` |
| Environment | 32.6905 -> 25.8403 (1.265x) | 164.2335 -> 33.2063 (4.946x) | 539.8481 -> 42.9124 (12.580x) | `17358322786803582063`; physical `5251282115348753471` |

The small LinkMapper W2 difference is within the short-fixture scheduler
range; W1 and W8 improve, and the checksum is identical. Worker width remains
an explicit caller setting; no hardware-based auto-tuning was added.

## Frozen Python/Linux reference

These accepted records are retained for Python-to-C++ context only. They are
not mixed into the native A/B ratio because OS, compiler, allocator, and CPU
scheduling differ.

| Module | Python ms | C++ W1 | C++ W2 | C++ W8 | Best accepted speedup |
|---|---:|---:|---:|---:|---:|
| Counter | 85.031 | 23.527 | 17.588 | 19.448 | 4.835x |
| Recorder | 87.621 | 3.518 | 5.136 | 11.834 | 24.907x |
| Controller | 33.624 | 3.892 | 6.484 | 11.786 | 8.639x |
| NodeMapper | 61.280 | 3.290 | 16.008 | 19.886 | 18.628x |
| LinkMapper | 8.862 | 0.629 | 0.987 | 1.720 | 14.092x |
| Environment | 9675.094 | 65.883 | 254.199 | 993.396 | 146.854x |

## Final implementation notes

- `virne::utils::deterministic_parallel_blocks` uses a persistent process-wide
  pool, a typed `noexcept` callback/context, atomic completion, and contiguous
  deterministic blocks. The callback/context/generation/active width are
  snapshotted under one mutex, so changing worker widths cannot duplicate a
  range or access a dead stack context.
- The pool grows before publishing a task, making thread-allocation failure
  exception-safe. Active workers get a bounded pause window; inactive workers
  park immediately. Nested executor calls stay sequential to avoid deadlock
  and oversubscription.
- LinkMapper keeps ordered early exit but uses larger deterministic path
  windows (`128 * requested_workers`) to amortize synchronization. Fixed
  attribute fields and numeric IDs remain in all hot loops.

Focused units for Counter, Recorder, Controller, NodeMapper, LinkMapper, and
Environment passed, as did the System, Transactional System, and complete
heuristic-registry units after relinking all executor users with the final
header. An additional ClangCL stress sweep of 20,000 varying
count/worker/grain calls passed exact one-visit coverage. No meta-heuristic or
machine-learning code is included.
