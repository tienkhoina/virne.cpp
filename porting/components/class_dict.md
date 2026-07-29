# Component: `virne.utils.class_dict`

State: **COMPLETE** on 2026-07-27.

## Source and target

- Python source: sibling `../virne/virne/utils/class_dict.py`, commit
  `d1ec1e4a20461fc9bad50612ad5026fd31e693a8`.
- Exact source SHA-256:
  `19637BBC0D4EFF9F240C5BE6B799B84AF96C7423E8C32AD97626468E3E3DE8FE`.
- C++ implementation: `virne/utils/class_dict.h` and `.cpp`.
- Isolated target: `vne_utils_class_dict`; it links only `Threads::Threads`.
- Unit/CTest target: `vne_class_dict_unit`.
- Differential/timing driver: `vne_class_dict_harness` plus
  `porting/compare_class_dict.py`.
- Worker sweep: `porting/sweep_class_dict_workers.py`.

The comparator direct-loads the exact leaf file instead of importing the
`virne` package. This keeps ML and unrelated package initialization outside
the component boundary.

## Ported data contract

`ClassDict` retains insertion-ordered string fields. An overwrite keeps its
position. Missing string lookup, explicit Python `None`, and a caller-provided
default remain distinguishable: C++ uses a null pointer for missing, a
non-null empty `std::any` for explicit `None`, and returns an lvalue default by
reference. Integer indexing follows insertion order, including negative
indices and bounds errors.

`update(ClassDictSnapshot)`, `update(ClassDict)`, and `from_dict()` use the
held C++ value's normal copy semantics. The provided `shared_ptr` recursive
forms therefore retain shallow input identity like Python; by-value containers
copy by value, as called out in the compatibility boundary below. Updates
preserve existing positions and append new fields in input order. The Python
method silently ignores unsupported positional update arguments; the typed C++
API has no overload for such arguments, so an unsupported value cannot enter
production code.

`to_dict()` returns an insertion-ordered `ClassDictSnapshot` and mirrors
Python's `copy.deepcopy()` for the supported recursive data surface:

- empty `std::any`, scalar/value payloads, and nested `ClassDict` values;
- `ClassAnyList` and `ClassMapping` values;
- `ClassAnyListPtr` and `ClassMappingPtr`, with one memo spanning the complete
  snapshot so aliases and recursive graphs are retained in the copy;
- conversion of `node_slots`, `link_paths`, `node_slots_info`, and
  `link_paths_info` from ordered mappings to distinct plain mappings.

The four special field names are classified once when a field is first
inserted. `to_dict()` tests the stored boolean; it does not compare those
strings in its field loop.

## Fixed fields, IDs, and hot access

This component follows `porting/PERFORMANCE_CONTRACT.md`. Dynamic field names
are accepted only at its actual dynamic-data boundary. `set()`,
`find_field_id()`, and `resolve_or_create()` perform one boundary hash
operation. The returned object-local `ClassFieldId` is a compact `uint32_t`
index; `at()`, `at_as()`, `set(ClassFieldId, ...)`, and `field_name()` then use
direct indexed access without hashing or string comparison.

IDs remain stable while fields are inserted or overwritten because fields are
never removed. Copy/move assignment and `swap()` replace the object's schema,
so callers must resolve IDs again for that object. The lowest-overhead hot
path resolves once, obtains a typed reference with `at_as<T>()`, and carries
that reference through the loop. The canonical timing gate measures the
string, ID, and resolved-reference paths separately.

This flexibility is not permission to represent known schemas dynamically.
Future fixed models, especially `Solution`, must use direct typed members and
enums. A genuinely dynamic name may cross the boundary once, after which hot
code must carry `ClassFieldId` or a resolved reference.

## Deterministic batch extensions

`from_dict_batch()` and `to_dict_batch()` process independent objects while
retaining input order. A persistent executor reuses workers and divides the
input into deterministic contiguous ranges. Concurrent top-level callers are
serialized at the executor boundary, and failures are rethrown by the lowest
input index regardless of worker completion order.

A custom value copy operation can re-enter a batch API. Such nested work runs
sequentially rather than trying to acquire the same executor, preventing
deadlock while retaining deterministic output and exception behavior. Unit
coverage includes concurrent callers, failures from multiple background
workers, and a reentrant `std::any` copy hook.

Automatic mode uses the measured aggregate top-level field count. It stays
sequential below 8,192 fields and selects up to eight workers at or above that
point, bounded by input item count and Linux CPU affinity (or hardware
concurrency elsewhere). The 64-field threshold sweep confirmed the sequential
side of the policy. Explicit width remains available, particularly for
nested-heavy payloads whose deep-copy cost is not visible in the top-level
field count.

## Differential coverage

The gate passes **16 exact cases**. The corpus covers empty/missing/default
identity, primitives, positive/negative/bool integer indexing, overwrite and
ID stability, resolve-or-create, ordered updates, shallow input identity,
deep-copy aliasing across fields and nested `ClassDict` values, recursive
lists, the four special mapping fields, missing versus explicit `None`,
unsupported Python update arguments, empty/UTF-8/newline/NUL strings, and
worker-independent batch order.

The line protocol uses a tagged canonical traversal that records insertion
order, scalar type/bit payloads, aliases, and cycles. Every payload and fact is
compared exactly. Timed rows additionally require identical 64-bit checksums
before a speedup is reported.

Those timing checksums are deliberately lightweight recipe guards. Complete
field/alias/special-mapping correctness comes from the differential corpus and
the full-field C++ batch unit, not from treating one benchmark scalar as a
second differential oracle.

## Compatibility boundary

The completed parity claim is deliberately data-only. It does not emulate
Python object reflection. In particular, arbitrary interactions with
`__dict__`, `getattr`, method-shadowing names, descriptors, or other dunder
behavior remain Python-language concerns rather than a C++ data contract.

`std::any` preserves the open type boundary, but arbitrary held C++ types use
their own copy constructor in the fallback path; they do not receive Python's
generic `deepcopy` dispatch. Exact recursive copying is guaranteed only for
the types listed above. By-value list/mapping instances have value semantics,
so identity shared between separate by-value objects cannot be reconstructed;
use the provided `shared_ptr` forms when alias identity matters.

Strong `shared_ptr` cycles are copied with the correct graph identity, but C++
does not have Python's cyclic garbage collector. The owner must break such a
cycle when it is no longer needed. Python tuple behavior and the wider
`Solution` object/value surface are deferred until those typed components are
ported; they are not silently approximated here.

## Verification

- Release build and full repository CTest: PASS, 16/16.
- C++ unit, concurrent-caller, deterministic-error, and reentrant-copy tests:
  PASS.
- Differential: PASS, 16/16 exact cases.
- Canonical explicit-eight timing/checksum gate: PASS for all eight rows, five
  warm-ups and 31 samples.
- Automatic-policy timing/checksum validation: PASS.
- Five-round worker sweep: PASS; every explicit/automatic checksum remained
  invariant.
- AddressSanitizer and UndefinedBehaviorSanitizer unit runs: PASS.
- `-Wall -Wextra -Wpedantic -Werror`: PASS for production, unit, and harness.
- Frozen graph/CSV/config/yaml-cpp integrity: PASS; those directories were not
  edited.

See `porting/results/class_dict_2026-07-27.md` for measurements and
`porting/README.md` for commands.
