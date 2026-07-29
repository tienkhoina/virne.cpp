# Port-wide field and hot-loop performance contract

This rule applies to every new production port and to performance-sensitive
test/benchmark support code.

## Fixed schema

- Represent fixed fields as typed struct/class members and access them
  directly.
- Represent fixed discriminants as enums, not strings or string-keyed maps.
- Do not carry a configuration string through an algorithm after its boundary
  meaning is known.

## Dynamic names

- A dynamic string may be accepted at a public/configuration boundary.
- Resolve it exactly once to the native typed representation: for example
  `TopologyType`, `AttrId`, a column index, or another compact ID.
- Pass the resolved value through downstream calls. Do not repeat string
  hashing, comparison, interning, or dictionary lookup.

## Hot loops

- Attribute loops must use `AttrId` with `AttrMap::find/at/set(AttrId, ...)`.
- A call such as `find("name")`, `at("name")`, `operator[]("name")`,
  `attr_id("name")`, or registry lookup is forbidden inside a node, edge,
  neighbor, candidate, source, request, worker, sample, or retry loop.
- Resolve optional dynamic fields once before the loop and branch on the
  optional ID inside it.
- Fixed request/config values use direct fields; no general-purpose dynamic
  dictionary is allowed for a known schema.

## Parallel work

- Parallelism must not change output order, exception selection, RNG
  consumption, or mutation order.
- Shared Python-compatible RNG streams and retry attempts remain sequential.
- Parallel batches require independent per-item state (normally a fixed enum
  plus an explicit seed), pre-sized result/error slots, and input-index error
  selection.
- Worker count is a typed caller/config input. Zero or one selects the exact
  sequential path; wider values are bounded by independent item count and the
  process CPU allowance. New ports do not embed a machine-specific automatic
  worker policy.

## Benchmark freeze

- A new component receives one compact representative benchmark only after its
  exact differential gate passes. A few configured worker widths are enough to
  prove output invariance, useful parallelism, and a C++ advantage.
- Once those rows pass, the benchmark source, command, result JSON, checksums,
  and recorded timings are frozen. Do not rerun, retune, or update a completed
  component's benchmark during later work unless that component is explicitly
  reopened because its output or performance gate failed.
- Unit, differential, sanitizer, integration, and frozen-integrity regressions
  may be rerun without reopening the accepted benchmark.

## Gate for each component

Before marking a component complete:

1. audit fixed data for typed fields/enums;
2. search every hot loop for string-based lookup;
3. compile with `-Wall -Wextra -Wpedantic -Werror`;
4. pass exact Python differential/checksum tests;
5. compare Python, C++ sequential, and a few representative configured worker
   counts once, then freeze the accepted benchmark;
6. record the resolution points, IDs/enums, and worker policy in the component
   note.

The Python oracle necessarily uses the original project's native dictionaries
and NetworkX string attributes. Oracle code performs at most one native lookup
per individual dynamic dictionary and remains outside C++ timed regions; it is
not a model for production C++ storage.
