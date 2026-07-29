# Component: `virne.utils.manager`

State: **COMPLETE** on 2026-07-28.

This note is the source-of-truth public API, compatibility, safety, ownership,
and performance contract for the completed C++ implementation. Frozen graph,
CSV, config, and yaml-cpp behavior remains unchanged.

## Documentation-first rule

The original source was read once because this component had no prior API
document. Future work must begin with this note and must not reopen
`manager.py` or `manager.cpp` merely to rediscover behavior. Source may
be opened again only when an exact differential mismatch is otherwise
unexplained, or when a measured low-level performance optimization genuinely
requires implementation-layout knowledge. Any fact learned that way must be
added here in the same change.

## Source identity and inventory

- Original source: sibling `../virne/virne/utils/manager.py`.
- Pinned original commit:
  `d1ec1e4a20461fc9bad50612ad5026fd31e693a8`.
- Exact source SHA-256:
  `77AFBDAB961D3183D877449BFA76A68BA876929D4E822B0D61EEC65F8441FA3B`.
- Source size: 1,426 bytes, 38 physical lines.
- Python dependencies: standard-library `os` and `shutil` only.
- C++ public header and implementation: `virne/utils/manager.h` and `.cpp`.
- Isolated production target: `vne_utils_manager`; it has no dependency on an
  unported stub, Threads, or a third-party library.
- Unit, differential, and timing targets: `vne_manager_unit`,
  `vne_manager_harness`, and `vne_manager_benchmark`.
- No original test and no internal Python callsite uses any of the three
  functions. `virne.utils.__init__` imports all three into its namespace;
  `virne.utils.__all__` lists only `clean_save_dir` and
  `delete_temp_files`, not `delete_empty_dir`.

Final evidence is recorded in `porting/results/manager_2026-07-28.md`. The
24/24 compatibility differential plus one intentional symlink-escape safety
deviation, strict/sanitizer/stress gates, and full 19/19 CTest all passed.

The Python oracle must direct-load this exact leaf. Importing `virne.utils`
would cross the known eager dataset/Torch boundary and is not part of this
component.

## Original public and observable API

```python
def delete_temp_files(file_path): ...
def clean_save_dir(dir): ...
def delete_empty_dir(config): ...
```

There are no type annotations. All three return `None` after successful
completion. Filesystem exceptions propagate immediately; there is no rollback.
`clean_save_dir` additionally writes one stdout line after each successful run
deletion. None of the functions writes stderr.

### `delete_temp_files(file_path)`

The observable algorithm is:

1. call `os.listdir(file_path)` and retain the returned list;
2. for the first entry, call `os.path.join(del_list, entry)` where `del_list`
   is the entire Python list rather than the input path;
3. only after that unreachable join would it test `os.path.isfile()` and the
   substring `temp`, then remove the file.

The mistaken first join operand is a locked compatibility fact:

- a valid empty directory returns `None` and changes nothing;
- any valid non-empty directory raises Python `TypeError` with the semantic
  reason “path operand is a list” before inspecting or deleting its first
  entry;
- therefore ordinary Python execution never deletes a file, regardless of
  whether its name contains `temp`;
- a missing path, regular-file path, inaccessible directory, or invalid path
  operand raises from the initial `os.listdir` before the legacy join bug;
- no stdout is produced.

This bug must not be silently repaired in the same-named compatibility API.
If useful temp-file deletion is later required, design a separately named,
explicitly safe API and document its filename/symlink policy. A literal
one-line “fix” is not acceptable: the original substring test examines the
full joined path, so a parent directory containing `temp` could cause every
direct file to match.

### `clean_save_dir(dir)`

`dir` is the root save directory. The name shadows Python's built-in `dir` but
has no further effect. Observable behavior is:

1. enumerate direct root entries in native `os.listdir` order;
2. snapshot only entries for which `os.path.isdir(root / entry)` is true;
   regular root files are ignored and directory symlinks are followed;
3. for each captured algorithm directory, enumerate every direct run entry in
   native order without first checking that the run entry is a directory;
4. form the fixed child path `run / "records"` once;
5. recursively delete the complete run entry when `records` does not exist or
   when `records` exists and `os.listdir(records)` is empty;
6. after successful recursive deletion, print exactly
   `Delate {run_path}\n` to stdout. The misspelling `Delate` is public output;
7. retain the run when `records` contains at least one entry.

The local Python list `['models', 'records', 'logs']` is never read. Do not port
it as configuration. Only the literal `records` check affects behavior;
`models` and `logs` are removed along with the rest of a deleted run but never
influence the decision.

Important edge behavior:

- a missing/non-directory/inaccessible root fails during root enumeration;
- Python `os.path.isdir` swallows its own `stat` failures and treats that root
  entry as not-an-algorithm directory; the C++ status probe must distinguish
  this from a later `listdir` failure and preserve the skip behavior;
- an empty algorithm directory is retained; algorithm directories themselves
  are never removed;
- a run containing models/logs but no `records` is deleted recursively;
- an empty `records` directory causes deletion; any entry, including a hidden
  file or subdirectory, retains the run;
- a regular file at `run/records` exists, then fails at `os.listdir` with a
  not-directory error; the run remains;
- a regular file directly below an algorithm directory reaches
  `shutil.rmtree(file)` and fails; it is not removed;
- a final run-entry symlink is rejected by `shutil.rmtree`; however an
  algorithm-directory symlink is followed before the run path is formed;
- deletions/prints completed before a later exception remain observable. There
  is no transactional cleanup or catch-and-continue behavior;
- enumeration and stdout order are native filesystem order. Python does not
  promise a sorted order, so the C++ compatibility implementation must not
  introduce sorting;
- relative input produces relative joined paths in the `Delate` line; native
  path separators are observable.

#### Mandatory safe deviation for symlink escape

The original follows an algorithm-directory symlink. Consequently a path such
as `root/algo_link/run`, where `algo_link` targets a directory outside `root`,
can make Python recursively delete the outside `run`. The C++ port must **not**
reproduce this unsafe escape.

Before recursive deletion, resolve and verify the exact candidate against the
resolved caller-supplied root. Reject a candidate whose traversal escapes that
root with a typed `unsafe_path_escape` error and leave both the symlink and
outside target unchanged. The differential corpus must record this as an
intentional safety deviation, not an unexplained mismatch. A C++17
`std::filesystem` check is not a hard security boundary against a malicious
concurrent rename; a future adversarial filesystem API would require a
separate `dirfd`/no-follow design.

### `delete_empty_dir(config)`

Before entering its loop, Python evaluates these attributes left-to-right and
builds a three-element list:

```text
config.record_dir, config.log_dir, config.save_dir
```

This evaluation order matters. A missing/throwing later attribute fails before
an earlier directory can be removed. Once captured, the three paths are
processed in fixed record, log, save order:

- if `os.path.exists(path)` is false, do nothing;
- otherwise call `os.listdir(path)`;
- if the result is empty, remove only that empty directory with `os.rmdir`;
- if non-empty, retain it;
- return `None` with no stdout after all three paths complete.

Edge behavior:

- a regular-file path exists but fails at `os.listdir`; later paths are not
  visited and prior removals remain;
- a dangling symlink has `exists == false` and is skipped;
- a symlink to a directory is followed for `exists/listdir`, but removing the
  symlink as a directory fails rather than deleting its target;
- duplicate paths are allowed: after the first removal, a later identical path
  no longer exists and is skipped;
- nested paths can intentionally cascade: removing empty record/log children
  can make `save_dir` empty before its final check;
- external races between exists/listdir/rmdir propagate their native failure;
  no retry is performed.

`os.path.exists` itself converts its `stat` errors to false. The C++ existence
probe must likewise treat a status error as absent for this compatibility
surface; errors from the subsequent enumerate/remove operations remain fatal.

Python reflection and attribute getter side effects are not a useful C++ data
model. The C++ boundary uses a fixed typed struct, so missing attributes become
a construction/compile-time issue rather than a runtime `AttributeError`. The
Python pre-loop attribute-order case remains in the oracle as a documented
representation boundary.

## Implemented C++ API

All target names are in `namespace virne::utils`. This is the stable production
surface used by the unit, differential harness, and benchmark.

```cpp
enum class ManagerErrorCode : std::uint8_t {
    legacy_temp_join_type_error,
    enumeration_failed,
    not_directory,
    unsafe_path_escape,
    remove_failed,
    output_failed,
};

enum class ManagerOperation : std::uint8_t {
    list_temp_root,
    list_save_root,
    list_algorithm,
    list_records,
    remove_run_tree,
    remove_empty_directory,
    emit_delete_line,
};

class ManagerException : public std::runtime_error {
public:
    ManagerException(
        ManagerErrorCode code,
        ManagerOperation operation,
        std::filesystem::path path,
        std::error_code system_error,
        std::string message);

    ManagerErrorCode code() const noexcept;
    ManagerOperation operation() const noexcept;
    const std::filesystem::path& path() const noexcept;
    const std::error_code& system_error() const noexcept;
};

struct EmptyDirectoryConfig {
    std::filesystem::path record_dir;
    std::filesystem::path log_dir;
    std::filesystem::path save_dir;
};

void delete_temp_files(const std::filesystem::path& directory);

void clean_save_dir(const std::filesystem::path& directory);
void clean_save_dir(
    const std::filesystem::path& directory,
    std::ostream& output);

void delete_empty_dir(const EmptyDirectoryConfig& config);
```

The one-argument `clean_save_dir` writes compatibility output to `std::cout`.
The stream overload has identical incremental deletion/error semantics and
exists so unit/harness code can capture exact output without redirecting global
streams. It is not a buffered “collect then print” API: a line must be emitted
immediately after each successful deletion so output before a later failure is
retained.

The stream overload writes one exact native-path line without an explicit
flush, matching Python `print(..., flush=False)`. It checks stream state after
the write and throws `output_failed/emit_delete_line` if delivery fails; the
already-deleted run remains deleted and processing stops.

The same-named `delete_temp_files` preserves the legacy empty-success/nonempty-
typed-error behavior. `ManagerException` maps stable semantic categories;
platform-specific Python/C++ message text is not the cross-platform contract.
Where available, the native `std::error_code` and failing path remain exposed
for diagnostics. Do not throw raw `std::filesystem_error` across this component
boundary.

### Implementation routes and typed failure mapping

POSIX builds use the libc `opendir`/`readdir`/`stat`/`rmdir` route for the
enumeration, status, and empty-directory hot paths. This avoids redundant
`std::filesystem` abstraction work while retaining native entry order. Other
platforms use the behavior-equivalent `std::filesystem` fallback. The public
API and differential contract do not depend on which route is selected.

Recursive run deletion retains the same typed component boundary. In
particular, Python `shutil.rmtree()` on a regular run entry raises
`NotADirectoryError`; C++ reports
`ManagerErrorCode::remove_failed`/
`ManagerOperation::remove_run_tree`, preserves the regular file, and stops
before later entries. Native message text may differ, but operation, category,
side effects, and order are exact compatibility facts.

If a signature changes, update this document first and rerun the complete
manager differential; do not introduce an undocumented overload.

## Fixed fields, dynamic paths, and hot-loop policy

This component follows `porting/PERFORMANCE_CONTRACT.md`:

- `EmptyDirectoryConfig` has three direct fixed fields. It must not be a
  `SettingObject`, string-keyed map, or reflective property bag.
- Error and operation discriminants are enums, carried through error handling
  without string comparisons.
- The fixed `records` child is a compile-time path component. Construct its
  path once per run; never search a dynamic property map for it.
- Filesystem entry names are genuinely dynamic. Consume each enumerated name
  once to construct an owned/native `std::filesystem::path`, then carry that
  path through checks and deletion. Repeated string concatenation or UTF
  conversion in the run loop is forbidden.
- A compact ID adds no value when an entry is consumed once. If a future scan
  snapshot revisits entries in hot loops, assign a dense `RunId` once and keep
  typed `{run_path, records_path}` records in a contiguous vector.
- The unused Python `sub_dirs` list must not become runtime allocation.
- Do not sort native directory results. Sorting adds work and changes observable
  deletion/exception/stdout order.

The isolated target is `vne_utils_manager` and links only the C++17 standard
library and platform libc. GCC 11 provides native `std::filesystem`; graph,
CSV, config, yaml-cpp, Boost, `virne_common`, and Threads are not linked.
Completed libraries do not need rebuilding for this leaf.

## Ownership and thread safety

- Functions retain no caller path/config reference after return. All enumerated
  dynamic names needed beyond an iterator increment are owned as paths.
- There is no component-global mutable state. `std::cout` is the only global
  side effect of the compatibility overload.
- Calls operating on provably disjoint directory trees are library-thread-safe,
  subject to the filesystem itself. Calls whose roots overlap are unsupported:
  results, exception order, and partial deletion become race-dependent.
- Never invoke two cleanup operations concurrently on the same root merely to
  gain speed. External writers can already create TOCTOU races; internal
  parallel deletion would make them worse.
- Stream-overload callers own the stream and must serialize access to it.
- Recursive deletion is irreversible and has no rollback. Production callers
  must pass the intended narrow save root, never a workspace root, home
  directory, filesystem root, unresolved environment variable, or glob.
- Unit/differential/benchmark runs may delete only explicitly created temporary
  fixture directories. They must verify candidate paths before calling the
  component and report what was removed.

## Parallelism and performance suitability

The compatibility algorithms are intentionally sequential. Python cannot
parallelize them, but C++ parallel deletion is not a valid optimization because
it would change native enumeration order, stdout order, the first propagated
failure, and the set of deletions completed before that failure. Filesystem
metadata/removal latency also dominates these tiny control loops.

No worker-count API and no 1..N worker sweep should be introduced for this
component. Optimize the sequential path instead:

- one enumeration per level;
- one joined `records` path per run;
- native path representation without repeated string conversion;
- contiguous snapshots only where a snapshot is already required by Python
  behavior;
- error-code filesystem operations internally, constructing an exception only
  on failure;
- no allocation for the unused fixed-name list;
- no redundant exists/status calls beyond those required for parity and safety.

A future API that cleans multiple explicitly independent roots may add a
deterministic batch only as a separate extension. It would require prevalidated
non-overlapping canonical roots, pre-sized per-root result/error slots, input
order, lowest-input-index failure, no shared stdout, and a complete worker
sweep before production use.

## Completed unit and differential matrix

The oracle direct-loads the pinned leaf and operates only below a fresh
temporary root. Each case records return/exception facts, exact stdout bytes,
and a before/after tree containing entry type, relative path, file bytes, and
symlink target. Native exception messages are diagnostic; compare stable
exception class/category, failing operation, errno/error condition, and side
effects.

The final gate passed 24/24 compatible differential cases. One additional
algorithm-directory symlink case records the mandatory safety deviation:
Python demonstrates the outside-root deletion, while C++ returns
`unsafe_path_escape` and preserves both link and target. Sequential order,
partial side effects, exact output, and the complete before/after tree matched
for every compatible case.

### `delete_temp_files`

| Case | Required fact |
|---|---|
| missing path | initial enumeration error; no stdout |
| regular-file path | not-directory enumeration error |
| empty directory | returns `None`/void; unchanged |
| one `temp` file | legacy typed error before deletion; file unchanged |
| one non-temp file | same legacy error; file unchanged |
| directory as first entry | same legacy error before type inspection |
| mixed entries | no entry is removed; failure occurs at first loop body |
| relative/path-like input | accepted path boundary and same facts |

### `clean_save_dir`

| Case | Required fact |
|---|---|
| missing root / root is file | root enumeration error, no mutation/output |
| root regular files only | ignored; returns with no output |
| empty algorithm directory | retained |
| run missing `records` | complete run removed; exact `Delate` line |
| run with empty `records` | complete run removed; exact line |
| run with non-empty `records` | retained; no line |
| run with models/logs but no records | removed recursively |
| `records` is regular file | not-directory error; run retained |
| direct run entry is regular file | recursive-remove category error; file retained |
| final run symlink | rejected; target retained |
| algorithm symlink escapes root | Python unsafe delete demonstrated; C++ `unsafe_path_escape`, all targets retained |
| one deletion then later failure | first deletion/line persist; processing stops at failure |
| relative root | exact native relative path in stdout |
| multiple controlled runs | native enumeration order, no sorting |

### `delete_empty_dir`

| Case | Required fact |
|---|---|
| all paths missing | no-op |
| all three empty | removed in record/log/save order; no stdout |
| all three non-empty | retained |
| mixed empty/non-empty/missing | only empty existing directories removed |
| later config attribute missing | Python fails before any deletion; typed C++ boundary documented |
| first/second path is a file | error stops later processing; prior side effects preserved |
| duplicate paths | first removal followed by missing-path skip |
| nested record/log below save | child removals may make and remove empty save parent |
| dangling/final directory symlink | skip or removal failure exactly as profiled; target retained |
| concurrent disappearance fixture | classified native race error, no retry |

Additional C++-only units cover every `ManagerErrorCode` and
`ManagerOperation`, stream failure behavior, path containment checks, Unicode
and spaces, a native path longer than 260 code units, copy/move behavior of the
fixed config, and strict confirmation that no production/frozen directory is
used as a deletion fixture.

## Benchmark matrix

Correctness and safety gates run before any timing. Fixture construction,
recursive copying, tree hashing, stdout capture normalization, process startup,
and cleanup are outside timed intervals. Python and C++ samples alternate on
the same filesystem/container and use fresh equivalent trees for destructive
rows.

| Row | Fixture/work | Measured operation | Gate |
|---|---|---|---|
| temp empty no-op | one empty directory | `delete_temp_files` success path | C++ median must beat Python; exact unchanged tree |
| temp legacy failure | one file | time to compatibility error | report only; no speedup requirement on exception |
| save retained scan | many algorithms/runs, each records non-empty | complete `clean_save_dir` scan | C++ median should beat Python; tree/checksum exact |
| save missing-record deletion | fresh runs without records | sequential recursive delete plus output sink | report median/MAD/p95; exact tree/stdout first |
| save empty-record deletion | fresh empty records directories | sequential recursive delete plus output sink | report median/MAD/p95; exact tree/stdout first |
| empty-dir retained | three non-empty directories | `delete_empty_dir` no-op scan | C++ median must beat Python; exact tree |
| empty-dir removal | three fresh empty directories | fixed-order removals | report median/MAD/p95; exact tree |

Canonical timing uses five warm-ups and 31 alternating measured samples.
Destructive rows receive a new prebuilt fixture per operation and time only the
function call. The gate stores full before/after semantic and output checksums,
excludes fixture construction/verification and process startup, and records
runtime identity and operation counts because metadata timing is
environment-sensitive.

Worker suitability is **sequential only / not applicable**. The performance
record must say this explicitly rather than fabricating worker speedups.

### Recorded canonical result

The 2026-07-28 paired run passed every semantic/output checksum. Values below
are total median milliseconds for each row's recorded operation count.

| Row | Python median | C++ median | Speedup | Decision |
|---|---:|---:|---:|---|
| `temp_empty` | 76.974788 | 58.034485 | 1.326x | PASS |
| `temp_legacy` | 24.112239 | 27.227970 | 0.886x | report only; exception path has no speed gate |
| `clean_retained` | 54.566752 | 36.039141 | 1.514x | PASS |
| `clean_delete_missing` | 5.555053 | 4.515343 | 1.230x | PASS |
| `clean_delete_empty` | 4.869416 | 3.658289 | 1.331x | PASS |
| `empty_retained` | 161.520913 | 115.746512 | 1.395x | PASS |
| `empty_remove` | 13.927685 | 11.738483 | 1.186x | PASS |

There is no production worker count to select: observable sequential order
forbids internal parallel deletion. Full MAD/p95, operation counts, normalized
`ns/op`, checksums, runtime identity, and artifact hashes are in
`porting/results/manager_2026-07-28.md`.

## Completion record

All nine completion gates passed on 2026-07-28:

1. **PASS** — public API review includes the legacy temp bug and mandatory safe
   symlink-escape deviation;
2. **PASS** — isolated `vne_utils_manager` builds with standard-library/platform
   libc linkage only;
3. **PASS** — 24/24 exact compatible differentials plus one recorded safety
   deviation preserve stdout, order, errors, and tree facts;
4. **PASS** — the canonical sequential benchmark passed every semantic/output
   checksum and every success-path performance gate;
5. **PASS** — strict warnings include `-Wconversion` and
   `-Wsign-conversion` as errors;
6. **PASS** — ASan, UBSan, leak detection, and 100-iteration temporary-tree
   stress;
7. **PASS** — full repository CTest 19/19;
8. **PASS** — frozen graph/CSV/config/yaml-cpp integrity; and
9. **PASS** — this API contract, `porting/results/manager_2026-07-28.md`,
   `PORTING_STATUS.md`, and `porting/README.md` contain the final evidence.

The component is **COMPLETE**. Ordinary consumers should use this document and
must not reopen implementation source merely to discover the API.
