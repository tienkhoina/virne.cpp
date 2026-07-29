# Component: `virne.utils.setting`

State: **COMPLETE** on 2026-07-28.

## Documentation-first rule for future work

This file is the public API and behavior contract for the component. Future
ports must read this note, `porting/PERFORMANCE_CONTRACT.md`, and
`DEPENDENCIES.md`; they must not reopen `setting.cpp` merely to discover how to
call the component. Open the implementation only when an API point is genuinely
ambiguous, a failing differential requires diagnosis, or a measured low-level
optimization requires implementation-layout knowledge. Any clarification or
new public behavior learned that way must be added here in the same change.
Consult `setting.h` only to resolve an actual compile-time signature drift; if
the header and this note disagree, update this note immediately so the next
consumer returns to documentation-only use.

Known downstream schemas must not remain in the generic setting tree. Decode a
raw document once into a typed model whose fixed fields are direct members and
whose fixed discriminants are enums. The generic `SettingObject`/`SettingKeyId`
surface exists for truly dynamic input and for the one-time decode boundary.

## Source, target, and reused dependencies

- Python source: sibling `../virne/virne/utils/setting.py`, commit
  `d1ec1e4a20461fc9bad50612ad5026fd31e693a8`.
- Exact Python source SHA-256:
  `89FB0E0D6E40FB10703ADB30AD588731DACD0475F34E2E86F1EBFBB4E2031C8E`.
- Oracle: CPython 3.10.20 and PyYAML 6.0.1. The comparator direct-loads only
  the pinned leaf source, so importing Virne, Torch, or any learning package is
  unnecessary.
- C++ public header and implementation: `virne/utils/setting.h` and `.cpp`.
- Isolated library target: `vne_utils_setting`.
- Unit/CTest target: `vne_setting_unit`.
- Differential/timing executable: `vne_setting_harness`.
- Oracle driver: `porting/compare_setting.py`.
- Worker sweep: `porting/sweep_setting_workers.py`.
- Final evidence record: `porting/results/setting_2026-07-28.md`; canonical and
  sweep JSON artifacts use the same date.

The isolated target reuses, rather than rebuilds or replaces, the dependencies
already present in the workspace:

| Existing target | Pinned implementation | Use in this component |
|---|---|---|
| `virne_boost` | Boost 1.85.0 under `libs/boost` | Boost.JSON streaming parser, Boost.Unordered flat dynamic-key index, and arbitrary-precision support |
| `yaml-cpp::yaml-cpp` | frozen yaml-cpp 0.8.0 under `libs/yaml-cpp` | YAML tokenization/tree parsing |
| `Threads::Threads` | accepted toolchain threads | persistent deterministic batch executor |

No OS/Conda package is installed and no new parser library is introduced.
`config_lib` is deliberately not used: it implements the already-frozen Hydra
configuration semantics, while this leaf is the raw `json.load`/`yaml.load`
compatibility boundary. The component adds only the typed AST conversion,
Python-compatible emitters, file wrapper, and batch execution around the
existing parsers.

## Complete public type contract

All names below are in `namespace virne::utils`.

### Fixed discriminants

```cpp
enum class SettingFormat : std::uint8_t {
    json,
    yaml,
};

enum class SettingMode : std::uint8_t {
    read,
    read_update,
    write,
    write_update,
    append,
    append_update,
    exclusive_create,
    exclusive_create_update,
};

enum class SettingValueKind : std::uint8_t {
    null_value,
    boolean,
    integer,
    real,
    string,
    list,
    object,
};

enum class SettingErrorCode : std::uint8_t {
    unsupported_format,
    invalid_mode,
    open_failed,
    read_failed,
    write_failed,
    parse_error,
    unsupported_yaml_feature,
    serialization_error,
};
```

These enums are the only representations carried through parse, dump, file,
and batch loops. A format/mode string may be resolved at the boundary, but must
never be repeatedly compared inside work.

`SettingErrorCode` has the following stable meanings:

| Value | Meaning |
|---|---|
| `unsupported_format` | the path's final four characters are neither `json` nor `yaml` |
| `invalid_mode` | an unrecognized text mode or invalid enum value was supplied |
| `open_failed` | the file could not be opened, or an exclusive-create target already exists |
| `read_failed` | the selected mode is not readable, or seek/read failed |
| `write_failed` | the selected mode is not writable, or seek/write/flush failed |
| `parse_error` | syntactically invalid input, invalid explicit scalar tag, or multiple YAML documents |
| `unsupported_yaml_feature` | valid YAML lies outside the deliberately restricted setting profile |
| `serialization_error` | an invalid UTF-8 string or a JSON cycle cannot be emitted |

```cpp
class SettingException : public std::runtime_error {
public:
    SettingException(SettingErrorCode code, std::string message);
    SettingErrorCode code() const noexcept;
};
```

`what()` carries the diagnostic string and `code()` carries the machine-stable
classification. Exact Python exception classes/messages are not a compatibility
claim. Programmer misuse can instead throw standard exceptions as described
below.

### Dynamic object key ID

```cpp
struct SettingKeyId {
    std::uint32_t value = std::numeric_limits<std::uint32_t>::max();

    friend bool operator==(SettingKeyId, SettingKeyId) noexcept;
    friend bool operator!=(SettingKeyId, SettingKeyId) noexcept;
};
```

The all-ones default value is invalid. An ID is a dense index local to exactly
one `SettingObject`; equal numeric values from different objects are not
interchangeable. Invalid IDs passed to indexed object access are rejected with
`std::out_of_range`. The object rejects exhaustion of the usable `uint32_t` ID
space with `std::overflow_error`.

### Small/big integer and recursive value aliases

```cpp
class SettingInteger {
public:
    using BigInteger = boost::multiprecision::cpp_int;

    SettingInteger() noexcept;

    template <typename Integer, typename = std::enable_if_t<
        std::is_integral_v<std::remove_reference_t<Integer>> &&
        !std::is_same_v<
            std::remove_cv_t<std::remove_reference_t<Integer>>, bool>>>
    SettingInteger(Integer value) noexcept;

    SettingInteger(BigInteger value);
    explicit SettingInteger(std::string_view decimal);
    explicit SettingInteger(const char* decimal);

    bool is_big() const noexcept;

    template <typename T>
    T convert_to() const;

    SettingInteger& operator*=(std::uint32_t multiplier);
    SettingInteger& operator+=(std::uint32_t addend);
    SettingInteger& operator+=(const SettingInteger& addend);

    friend SettingInteger operator-(const SettingInteger& value);
    friend bool operator==(
        const SettingInteger& left,
        const SettingInteger& right);
    friend bool operator!=(
        const SettingInteger& left,
        const SettingInteger& right);
    friend bool operator>=(
        const SettingInteger& left,
        std::uint64_t right);
};

using SettingList = std::vector<SettingValue>;
using SettingListPtr = std::shared_ptr<SettingList>;
using SettingObjectPtr = std::shared_ptr<SettingObject>;
```

`SettingInteger` preserves Python's arbitrary-precision value without paying
`cpp_int` cost for the common case. Its fixed discriminant selects one of
`int64_t`, `uint64_t`, and `BigInteger`; signed/unsigned inputs up to 64 bits
stay compact and heap-free, while values outside those ranges promote to
Boost's arbitrary-precision representation. `is_big()` exposes that storage
choice for diagnostics/benchmarks only; algorithms must depend on the numeric
value, not on the current representation.

Default construction is compact signed zero. The integral template excludes
`bool` and statically rejects integral types wider than 64 bits. The
`BigInteger` constructor normalizes representable values back to a compact
lane. The two explicit string constructors accept a decimal integer boundary;
they are not implicit string-to-number conversions.

`convert_to<std::string>()` always returns the exact base-10 value. Other
`convert_to<T>()` calls follow the selected compact cast or Boost conversion,
so callers must choose a destination that can represent the result. The three
mutating arithmetic operations and unary minus preserve exact value and
promote when needed; equality is representation-independent. No hot path may
convert an integer to a decimal string merely to compare or update it.

Containers are reference-counted so YAML aliases and cycles have an observable
identity. See the ownership section before copying a document.

### `SettingValue`

```cpp
class SettingValue {
public:
    SettingValue() noexcept;
    SettingValue(std::nullptr_t) noexcept;
    SettingValue(bool value) noexcept;
    SettingValue(std::int64_t value);
    SettingValue(std::uint64_t value);
    SettingValue(SettingInteger value);
    SettingValue(double value) noexcept;
    SettingValue(std::string value);
    SettingValue(std::string_view value);
    SettingValue(const char* value);
    SettingValue(SettingListPtr value);
    SettingValue(SettingObjectPtr value);

    static SettingValue make_list();
    static SettingValue make_object();

    SettingValueKind kind() const noexcept;
    bool is_null() const noexcept;
    bool as_bool() const;
    const SettingInteger& as_integer() const;
    double as_real() const;
    const std::string& as_string() const;
    SettingList& as_list();
    const SettingList& as_list() const;
    SettingObject& as_object();
    const SettingObject& as_object() const;
    const SettingListPtr& list_ptr() const;
    const SettingObjectPtr& object_ptr() const;
};
```

Behavior by entry point:

- Default construction and `nullptr` construct `null_value`.
- Signed/unsigned fixed integers are widened into `SettingInteger`; no range is
  lost. `double` retains its IEEE-754 payload, including negative zero, NaN,
  and infinities.
- String and string-view constructors own a copy. A null `const char*` is
  treated as an empty string, not a null setting value.
- Pointer constructors reject a null shared pointer with
  `std::invalid_argument`; use default/`nullptr` for a null setting value.
- `make_list()` and `make_object()` allocate a non-null empty container.
- `kind()` and `is_null()` do not throw. An `as_*` or `*_ptr()` call whose
  accessor does not match `kind()` throws `std::bad_variant_access`.
- `list_ptr()`/`object_ptr()` expose identity. They are not deep-copy helpers.

### `SettingObject`

```cpp
class SettingObject {
public:
    SettingObject();
    SettingObject(const SettingObject& other);
    SettingObject(SettingObject&& other) noexcept;
    SettingObject& operator=(const SettingObject& other);
    SettingObject& operator=(SettingObject&& other) noexcept;

    std::size_t size() const noexcept;
    bool empty() const noexcept;

    void reserve(std::size_t key_count);

    std::optional<SettingKeyId> find_key_id(std::string_view key) const;
    SettingKeyId resolve_or_create(std::string_view key);
    SettingKeyId set(std::string_view key, SettingValue value);
    SettingKeyId set_owned(std::string key, SettingValue value);

    void set(SettingKeyId id, SettingValue value);
    SettingValue& at(SettingKeyId id);
    const SettingValue& at(SettingKeyId id) const;
    std::string_view key_name(SettingKeyId id) const;
    const std::vector<SettingKeyId>& sorted_key_ids() const;

    void swap(SettingObject& other) noexcept;
};

void swap(SettingObject& left, SettingObject& right) noexcept;
```

Objects contain unique string keys in first-insertion order. Setting a key that
already exists overwrites its value without changing its ID or position. The
string-boundary operations have these contracts:

- `find_key_id(key)` performs one hash lookup and returns no value when absent.
- `resolve_or_create(key)` returns the existing ID, or inserts a null value and
  returns its new dense ID.
- `set(key, value)` returns the existing/new ID and stores the value.
- `set_owned(key, value)` has the same ID/order/overwrite behavior as `set`,
  but takes ownership of an already decoded key string. Parsers use it to avoid
  allocating/copying the same dynamic key a second time; it is still a
  one-hash boundary operation, not a hot-loop API.
- `reserve(key_count)` reserves the contiguous key lane, contiguous entry lane,
  and hash index without changing size, keys, values, IDs, or logical order.
  It deliberately does not reserve or rebuild the lazy YAML sorted-ID cache.
  Use it before bulk insertion and before retaining any entry references or key
  views.

The ID operations perform bounds checking and then direct indexed access; they
do no hashing, allocation, or string comparison. `key_name(id)` returns the
owned dynamic key bytes. `sorted_key_ids()` lazily returns IDs in ascending
lexicographic key order for PyYAML's `sort_keys=True`; its cached order is
reused until a new key or schema replacement. An atomic dirty flag keeps the
already-sorted read path lock-free; the rare rebuild is protected by a mutex,
so concurrent dumps of one immutable object do not duplicate or race the sort.

Both owned key strings and `{key_view, value}` entries use contiguous vectors.
After a key-vector relocation, the object rebuilds its internal string views
and hash index before returning to the caller. This removes deque traversal and
improves parser/serializer locality. It does not make an externally retained
`key_name()` view stable across insert/reserve; use the object-local ID as the
long-lived handle and request a view only at the point of use.

Use the dynamic string exactly once:

```cpp
const auto rate_id = object.find_key_id("rate");
if (!rate_id) {
    // required-key handling at the decode boundary
}
const SettingValue& rate = object.at(*rate_id);
for (std::size_t i = 0; i < work_count; ++i) {
    consume(rate);                  // no key string in the hot loop
}
```

For a known downstream schema, decode `rate` into a direct typed field instead
of retaining even this ID beyond the raw-input layer.

### Fixed document/result fields

```cpp
struct SettingDocument {
    SettingValue root;
};

struct ReturnedSettingError {
    SettingErrorCode code = SettingErrorCode::unsupported_format;
    std::string message;
};

using WriteSettingResult = std::variant<
    std::reference_wrapper<const SettingDocument>,
    ReturnedSettingError>;
```

`root`, `code`, and `message` are fixed direct fields, not dynamic property
bags. Any JSON/YAML root type is allowed by `SettingDocument`.

## Complete public function contract

### Boundary resolution

```cpp
std::optional<SettingFormat> setting_format_from_path(
    std::string_view path) noexcept;

SettingMode parse_setting_mode(std::string_view mode);
```

`setting_format_from_path()` reproduces the Python slice `fpath[-4:]`: it
accepts any path of at least four characters ending exactly in lowercase
`json` or `yaml`, with or without a dot. It rejects `.yml`, uppercase `.JSON`,
and shorter names. It never opens a file and never throws.

`parse_setting_mode()` accepts exactly `r`, `r+`, `w`, `w+`, `a`, `a+`, `x`,
and `x+`, mapping them in enum declaration order. A `b`, `t`, or any other
spelling throws `SettingException{invalid_mode}`. Resolve once, then carry
`SettingMode`.

### In-memory parse and dump

```cpp
SettingDocument parse_setting(
    std::string_view bytes,
    SettingFormat format);

std::string dump_setting(
    const SettingDocument& document,
    SettingFormat format);
```

These are the leaf APIs for tests and high-throughput callers. They perform no
file I/O. Parsing owns all strings and containers in the returned document, so
the input view need only live through the call. Dumping returns exact output
bytes for the supported compatibility profile described below.

### File compatibility surface

```cpp
SettingDocument read_setting(
    const std::string& path,
    SettingMode mode = SettingMode::read);

WriteSettingResult write_setting(
    const SettingDocument& document,
    const std::string& path,
    SettingMode mode = SettingMode::write);

const SettingDocument& write_setting_strict(
    const SettingDocument& document,
    const std::string& path,
    SettingMode mode = SettingMode::write);

void conver_format(
    const std::string& source_path,
    const std::string& destination_path);
```

The file is opened before suffix dispatch, exactly matching the original call
order. Consequences are part of the contract:

- Reading a missing wrong-suffix path fails with `open_failed`; reading an
  existing wrong-suffix path opens it first and then throws
  `unsupported_format`.
- Default write mode creates/truncates a wrong-suffix target before
  `write_setting()` returns `ReturnedSettingError{unsupported_format, ...}`.
  It does **not** throw this historical Python bug.
- `write_setting_strict()` turns that returned error into `SettingException`,
  after the same open/create/truncate side effect. On success it returns the
  exact input document by reference.
- Successful `write_setting()` returns a `reference_wrapper` referring to the
  exact input document, mirroring Python object identity.
- The misspelled `conver_format` is intentional public compatibility. It reads
  with the default mode, calls non-strict `write_setting`, ignores a returned
  wrong-suffix error, and returns `void` like Python's `None`. Thus a wrong
  destination suffix can leave an empty created/truncated file without an
  exception.

The compatibility message stored/raised for an unsupported suffix is exactly
`Only supports settings files in yaml and json format!`. A successful result
or `write_setting_strict()` reference is non-owning and remains valid only as
long as the caller's input `SettingDocument` remains alive.

Mode behavior is:

| Enum | Text | Readable | Writable | Initial write position/effect |
|---|---:|---:|---:|---|
| `read` | `r` | yes | no | existing file, beginning |
| `read_update` | `r+` | yes | yes | existing file, beginning, no truncation |
| `write` | `w` | no | yes | create/truncate, beginning |
| `write_update` | `w+` | yes | yes | create/truncate, beginning |
| `append` | `a` | no | yes | create if needed, end |
| `append_update` | `a+` | yes | yes | create if needed, read/write position at end |
| `exclusive_create` | `x` | no | yes | require absent target, create at beginning |
| `exclusive_create_update` | `x+` | yes | yes | require absent target, create at beginning |

Reads normalize CRLF and lone CR to LF, matching Python text-mode universal
newlines. Append writes concatenate complete serialized documents without an
inserted delimiter; for example, writing JSON integer `1` and then appending it
produces `11`.

### Deterministic batch extensions

```cpp
std::vector<SettingDocument> parse_setting_batch(
    const std::vector<std::string>& inputs,
    SettingFormat format,
    std::size_t worker_count = 0);

std::vector<std::string> dump_setting_batch(
    const std::vector<SettingDocument>& documents,
    SettingFormat format,
    std::size_t worker_count = 0);
```

Both functions pre-size output slots, divide indices into deterministic
contiguous ranges, and preserve input order. They wait for all assigned work,
then rethrow the failure at the lowest input index even if a later item failed
first in wall-clock time. Empty input returns empty output.

Explicit widths are bounded by item count and by Linux process CPU affinity
(hardware concurrency elsewhere). An explicit value of one is sequential. A
value of zero selects the automatic policy:

- batch parse counts total input bytes, remains sequential below 256 KiB, and
  otherwise requests up to eight workers;
- batch dump performs a cycle-safe AST node-count prepass, remains sequential
  below 8,192 nodes, and otherwise requests up to eight workers.

The final five-warmup/31-sample/three-round sweep selected explicit worker 8
for the best aggregate result (1.9998x worker 1). Automatic mode achieved
1.9812x and stayed within 1.0640x of the best explicit median on every row;
checksums and output sizes were invariant for workers 1..8 and automatic.

One persistent executor grows its worker pool as needed and reuses it. Separate
top-level batch callers are serialized at the executor boundary. If a batch API
is re-entered from one of its worker tasks, nested work runs sequentially to
avoid acquiring the same executor and deadlocking. The output/error contract is
unchanged in that fallback.

## Ownership, aliases, copies, moves, and invalidation

The following rules are required for safe use without inspecting source:

- Scalar `SettingValue` copies are value copies. List/object copies copy their
  `shared_ptr`; they are shallow and preserve container identity.
- Copying `SettingDocument` therefore shares a container root. Mutating either
  copy's shared tree is visible through the other.
- Copy-constructing a standalone `SettingObject` duplicates its top-level key
  storage and entry lane, but nested list/object values remain shared. There is
  no public deep-copy operation.
- Moves transfer owned strings, entries, and shared pointers. The moved-from
  object remains valid but must be treated as having a replaced schema; resolve
  all keys again before reuse.
- YAML parse preserves list/object aliases and cycles. YAML dump emits repeated
  containers as `&idNNN`/`*idNNN`, including cycles. JSON dump serializes a
  repeated non-cyclic container at each occurrence, like `json.dump`, but
  rejects an active cycle with `serialization_error`.
- Strong `shared_ptr` cycles are not garbage-collected. The owning application
  must break a cycle when it is no longer needed.

ID and reference lifetime rules:

- Insertion and overwrite retain every existing `SettingKeyId`; there is no
  key removal API.
- IDs are object-local. Copy construction creates a different object, even if
  numeric positions match, so resolve IDs for the copy.
- Copy/move assignment and `swap()` replace schemas and invalidate all cached
  IDs, `SettingValue&`, key views, and sorted-ID views associated with the
  affected destination(s).
- Any string-boundary schema mutator (`resolve_or_create`, `set(string, ...)`,
  or `set_owned`) may reallocate either contiguous lane, including an overwrite
  implementation that probes with owned temporary storage. Existing IDs and
  logical order remain valid, but every externally retained `SettingValue&`,
  and `key_name()` view must be reacquired. If the call actually inserts a new
  key, it also invalidates the semantic sorted-ID view; a pure overwrite does
  not change sorted key order.
- `reserve()` can likewise reallocate key, entry, or hash storage. It preserves
  IDs and logical order, but callers must reacquire entry references and key
  views. It does not touch the sorted-ID cache; as a simple rule, still do all
  capacity preparation before retaining any view into the object.
- `set(id, ...)` cannot relocate the key/entry lanes; it preserves every ID and
  key view, but invalidates references or pointers into the replaced payload. A
  reference into a `SettingList` follows normal `std::vector` invalidation
  rules.
- `sorted_key_ids()`'s returned vector reference remains usable across a pure
  capacity `reserve()`, which does not touch that cache. A new key, assignment,
  move, or swap invalidates its semantic view; do not cache it or its elements
  across those mutations.

## JSON compatibility profile

The parser uses the existing Boost.JSON SAX/streaming parser with a maximum
nesting depth of 1,024 and constructs the compact AST directly; it does not
materialize an intermediate `boost::json::value` tree. Parser/handler scratch
is retained per calling thread, so repeated parses reuse capacity without a
global lock and batch workers never share mutable SAX state.

The numeric route is deliberately two-tiered. Ordinary JSON real lexemes take
Boost.JSON's faster `imprecise` conversion path. Tokens classified as requiring
stronger conversion use the `precise` fallback before a value enters the AST.
The externally visible contract is always the exact CPython binary64 result;
“imprecise” names an internal fast path, not relaxed output correctness. The
precision gate covers 17,513 generated decimal spellings, including subnormal,
halfway/rounding-boundary, exponent, extreme finite, negative-zero, and
overflow-adjacent cases, and compares all 64 payload bits.

### JSON parser hot-path layout

The canonical run on 2026-07-27 found one failed performance row despite all
exact gates passing: JSON scalar parse was 0.964692x Python
(`0.011891828125` ms/document C++ versus `0.01147195703125` ms/document
Python). The batch form was already 1.559355x Python, so this was treated as a
small-document fixed-cost problem rather than a reason to change the public
API, parser, or worker policy.

The 2026-07-28 leaf-only tuning therefore keeps the same data model and applies
the following internal rules:

- a complete Boost.JSON number token is classified through its existing
  `string_view`; it is copied into scratch storage only when Boost delivered an
  earlier partial-number chunk;
- integer-lexeme detection, significant-mantissa digit counting, and exponent
  magnitude classification share one pass rather than scanning the token
  independently;
- the exact fallback thresholds are unchanged: tokens longer than 24 bytes,
  more than 10 significant mantissa digits, or exponent magnitude greater than
  22 use the precise path;
- an integer lexeme routed through `on_double` still receives exact arbitrary-
  precision parsing, so range overflow never silently becomes binary64;
- each parsed object initially reserves 16 entries, covering the common Virne
  setting maps without repeated growth;
- the dynamic boundary key index uses Boost's contiguous open-addressed flat
  table. Owned key bytes and values remain in dense ID-addressable lanes, and
  downstream hot loops still resolve a string once and use `SettingKeyId`.

This tuning does not add a process-global cache or mutable shared parser state.
Parser state remains thread-local, batch order/error selection remains
deterministic, duplicate-key first-position semantics remain unchanged, and
reentrant calls keep the documented sequential fallback. Any later change to
these routes must rerun all 17,513 binary64 cases, arbitrary-integer cases,
duplicate/error cases, and the multi-worker exact gate before performance is
interpreted. Performance must always be measured rather than inferred from this
layout.

The post-tuning quick gate on 2026-07-28 did rerun the complete semantic corpus
and passed all 41 fixtures, 82 fixture transforms, 20 compatible cases, 12
required rejections, 17,513 JSON binary64 cases, 10 Python file contracts, and
the worker-1/worker-8/automatic batch comparisons. Its deliberately short
paired timing used one warmup and seven samples: JSON scalar parse measured
`0.00828329296875` ms/document C++ versus `0.0130947578125` ms/document Python,
or 1.580864x. JSON batch parse measured 1.975393x. This is a regression/signal
check only; it was subsequently confirmed by the quiet canonical protocol.

The quiet post-tuning canonical run used five warmups, 31 paired alternating
samples, 256 scalar operations, two 64-item batch rounds, 200,000 key/ID
iterations, and eight workers. It passed the complete differential again and
all ten performance rows. JSON scalar parse was 1.520856x Python
(`0.01089085546875` versus `0.01656341796875` ms/document); JSON batch parse
was 1.575101x. The other speedups were JSON dump 2.324173x, JSON batch dump
3.331601x, JSON ID access 78.705663x, YAML parse 14.074841x, YAML dump
113.281300x, YAML batch parse 36.574895x, YAML batch dump 181.111379x, and
YAML ID access 61.835138x. The machine-readable artifact is
`porting/results/setting_compare_2026-07-28.json`, SHA-256
`A9F8579BF77FC595F9C6E2B9053602C49708DF9D172DA61863AA8CE11926928C`.

The final post-tuning worker sweep also passed on 2026-07-28 with five
warmups, 31 samples, three rotated/reversed rounds, two operations, batch size
256, explicit widths 1..8, and automatic mode. Best per-row explicit widths
were JSON parse 7, JSON dump 8, YAML parse 8, and YAML dump 8; the best single
aggregate width was 8 at 1.999815x worker 1. Automatic mode was 1.981188x
worker 1 and only 1.0094x slower than that best aggregate width. Its per-row
quality versus the best explicit median was 1.0640x JSON parse, faster for JSON
dump, 1.0371x YAML parse, and 1.0570x YAML dump. JSON checksum/output size
remained `713075939059711629`/`1269`; YAML remained
`7347118398436008945`/`1164`. The artifact is
`porting/results/setting_sweep_2026-07-28.json`, SHA-256
`6FCB03D7D747BE6752DA7AD2638253346DAA68C27D09040A0C7A649D8CD245B0`.

Integer tokens first use the compact signed/unsigned lanes and promote to
`SettingInteger::BigInteger` only on range overflow. Thus normal setting files
avoid arbitrary-precision allocation while huge Python integers remain exact.
For the supported surface the parser mirrors Python 3.10
`json.load`/`json.dump`:

- any null/bool/integer/float/string/list/object root;
- insertion-ordered string-key objects;
- duplicate keys overwrite the value while retaining the first key position;
- arbitrary-precision integer lexemes;
- exact binary64 values, negative zero, and Python's non-standard `NaN`,
  `Infinity`, and `-Infinity` spellings;
- UTF-8 input and `ensure_ascii=True` output, including lowercase `\u` escapes
  and surrogate pairs for non-BMP code points;
- Python default separators `, ` and `: `, insertion order, and no final
  newline;
- strict rejection of comments, trailing commas, extra root values, empty
  input, and a UTF-8 BOM;
- JSON cycles rejected during serialization; repeated acyclic aliases are
  expanded at each occurrence.

Invalid UTF-8 stored programmatically in a string raises
`serialization_error`. Exact floating-point spelling is part of the
differential corpus; new toolchains must rerun the full bit-pattern/output gate
before changing the accepted compiler baseline.

## YAML compatibility and security profile

The parser reuses yaml-cpp for YAML structure and applies the PyYAML 6.0.1
`Loader`'s YAML 1.1 scalar resolution only to the supported data surface:

- empty input/empty scalar, `~`, and case-insensitive `null` become null;
- case-insensitive `yes/no`, `true/false`, and `on/off` become booleans;
- decimal, `0b` binary, `0x` hexadecimal, legacy leading-zero octal,
  sexagesimal, signed, and underscore-separated integers become arbitrary
  precision integers; values fitting 64 bits stay in the compact integer lane
  and only larger values promote;
- implicit floats require the PyYAML-compatible decimal/exponent form;
  exponent-only `1e-5` remains a string, while explicit `!!float 1e-5` is a
  real;
- `.nan`, `.inf`, `-.inf`, negative zero, decimal floats, signed exponents, and
  sexagesimal floats are supported as binary64;
- quoted/plain strings, lists, string-key mappings, empty containers, duplicate
  keys, aliases, and cycles are supported;
- explicit core `str`, `null`, `bool`, `int`, `float`, `seq`, and `map` tags are
  accepted when their value is valid;
- exactly one YAML document is allowed. Empty input maps to a null root.

This is intentionally safer and narrower than Python's unsafe
`yaml.load(..., Loader=yaml.Loader)`. The following valid Python inputs are
required C++ rejections with `unsupported_yaml_feature`:

- Python/object or other non-core tags;
- timestamps/dates;
- `!!binary`;
- non-scalar keys and scalar keys that resolve to a non-string type;
- merge key `<<` (quoted or unquoted);
- unsupported scalar/container tags or node kinds.

Multiple documents and malformed YAML use `parse_error`. This security boundary
must not be weakened merely to make unsafe Python tags pass. A future typed
feature needing one rejected YAML construct requires an explicit profile/API
review and new differential/security cases.

For supported AST values, YAML emission targets PyYAML 6.0.1 default
`yaml.dump` bytes:

- every mapping is sorted lexicographically by key through cached compact IDs;
- block mappings use two-space nesting and mapping-value sequences use
  PyYAML's indentless dash layout;
- empty lists/maps use `[]`/`{}`;
- aliases receive deterministic `id001`, `id002`, ... names in emitted
  traversal order;
- strings that would implicitly resolve are quoted; non-ASCII/control values
  use PyYAML-compatible uppercase `\x`, `\u`, or `\U` escapes because default
  `allow_unicode` is false;
- null/bool/int/float spellings follow PyYAML, including `.nan` and `.inf`;
- output ends with LF, and a plain scalar root additionally receives
  `...\n`.

## Deliberate compatibility boundaries

The exact parity claim applies to successful reads/writes in the supported AST
profiles and to the explicitly tested file-order side effects. It does not
claim to emulate all Python object or operating-system behavior:

- Python `pathlib.Path` slicing/type quirks are outside the C++ `std::string`
  path API.
- Binary/text mode spellings outside the eight modes above are rejected at the
  C++ boundary.
- Python's arbitrary object, tuple, date, binary, non-string-key, and unsafe
  YAML representers are outside `SettingValue`.
- Exception class names and messages differ; consumers branch on
  `SettingErrorCode`.
- The current exclusive-create implementation checks existence before opening;
  it matches ordinary single-process behavior but does not promise Python's
  atomic `O_EXCL` race semantics.
- Serialization is completed in memory after opening and before bytes are
  written. If serialization itself fails, a `w` target has already been
  created/truncated but is empty; Python's streaming dumper can instead leave a
  partial prefix. Partial-file bytes on serialization failure are not parity.
- Text newline behavior is normalized on read and emits LF on write; recorded
  byte parity is the Linux container contract.
- JSON nesting deeper than 1,024 is outside the supported profile.

These boundaries must remain visible in downstream documentation; do not call
the component a complete YAML-language implementation.

## Thread-safety and concurrency contract

- Independent documents/objects can be parsed, dumped, or mutated by separate
  threads.
- Concurrent read-only dumping of the same immutable tree is supported. The
  lazy YAML sorted-ID cache synchronizes its construction.
- No mutation may race with lookup, indexed access, sorting, parse/dump, or
  another mutation of the same `SettingObject`/shared container. The AST is not
  a general concurrent map.
- Because copies can share containers, “different documents” are not
  independent if their pointer graph overlaps and either caller mutates it.
- Concurrent top-level batch calls are safe but serialized by the singleton
  executor. Parallelism occurs within one batch call.
- Result order, chosen exception, serialized bytes, and alias traversal never
  depend on worker completion order.

## Complexity and performance properties

Let `k` be an object's key count, `n` the AST node count, and `b` input/output
bytes.

| Operation | Expected cost | Hot-path note |
|---|---:|---|
| `size`, `empty`, `kind`, scalar access | O(1) | direct fixed fields/discriminant |
| compact `SettingInteger` compare/update | O(1) | signed/unsigned 64-bit lane, no `cpp_int` allocation |
| big `SettingInteger` operation | O(limbs) | exact fallback only when value exceeds compact range |
| `reserve(k)` | O(k + key bytes) only when capacity grows | bulk-build hint; no logical change |
| `find_key_id` | expected O(1) | one string hash at boundary |
| `resolve_or_create`, `set(string, ...)` | expected O(1) | boundary hash; a new key owns bytes |
| `set_owned(string, ...)` | expected O(1) | boundary hash plus ownership transfer; avoids a second key copy |
| `at(id)`, `set(id, ...)`, `key_name(id)` | O(1) | bounds check plus direct index; no hash/string compare |
| cached `sorted_key_ids()` | O(1) | returns cached ID vector |
| rebuild `sorted_key_ids()` | O(k log k) | only after new key/schema replacement |
| standalone `SettingObject` copy | O(k + key bytes) | nested containers remain shared |
| move/swap with standard allocators | O(1) | schema ownership changes; IDs must be reacquired |
| JSON parse/dump | O(b + n) | thread-local SAX capacity reuse; object dump walks dense insertion IDs |
| YAML parse | expected O(b + n) | alias memo uses source mark plus identity confirmation |
| YAML dump | O(n + sum(k log k) + b) first time | sorted-ID caches are reused later |
| batch parse | O(total bytes/work) | static contiguous item blocks |
| batch dump | O(n) prepass plus serialization | cycle-safe unique-container count for auto policy |

String lookup inside a node, item, worker, serializer, or downstream algorithm
loop violates the performance contract. The C++ ID benchmark intentionally
measures `at(SettingKeyId)` separately from Python dictionary lookup.

## Harness protocol v1

`vne_setting_harness --help` describes three machine-readable commands. All
normal and error responses are one JSON object on stdout; successful commands
exit zero, setting/standard exceptions exit 2, and CLI misuse exits 64.

```text
transform <input-format> <output-format>
```

- stdin: one raw document;
- success: `{"ok":true,"output_hex":"...","output_size":N}`;
- failure: `{"ok":false,"error_code":N,"message_hex":"..."}`.

```text
batch <input-format> <output-format> <workers>
```

- stdin: NUL-separated raw documents;
- success: exact ordered `outputs_hex`, `count`, and an FNV-1a 64-bit checksum
  that also folds each output length;
- worker zero selects the automatic policy.

```text
benchmark <format> <workers> <rounds> <batch-size> [id-iterations]
```

- stdin: one raw document;
- warms scalar parse/dump and both batch paths before timing;
- returns `total_ns` and operation counts for `parse`, `dump`, `batch_parse`,
  `batch_dump`, and `id_access`, plus checksum and output size;
- `id-iterations` defaults to 1,000,000 and is measured only for a non-empty
  object root;
- process startup, fixture construction, and the warm calls are outside each
  reported timed interval.

Formats are exactly `json` and `yaml`. Payload bytes are hex-encoded so newline,
NUL, Unicode, and serializer formatting are compared without transport loss.

## Differential corpus and benchmark gate

The comparator pins the exact source hash/runtime and direct-loads the original
leaf. The intended final gate includes:

- 41 physical YAML fixtures from the supplied fixture roots, kept
  as distinct rows even when their bytes match;
- exact YAML-to-YAML and YAML-to-JSON bytes for every fixture;
- 20 compatible synthetic cases covering every root/scalar type, duplicate
  order, binary64 edge bits, huge integers, Unicode/control strings, empty
  containers, aliases, and a cycle;
- 17,513 generated JSON float spellings checked against CPython at exact
  binary64-bit and serialized-byte level, specifically gating the SAX
  imprecise/precise routing;
- 12 required rejection cases covering the security profile and malformed
  JSON/YAML;
- exact ordered batches at worker 1, the selected explicit width, and automatic
  mode;
- canonical AST comparison with exact type, insertion order, arbitrary integer
  decimal value, raw float64 bits, alias identity, and cycle references;
- seven original-Python file-contract checks plus the isolated C++ file unit;
- paired alternating Python/C++ samples for JSON and YAML `parse`, `dump`,
  `batch_parse`, `batch_dump`, and Python-key versus C++-ID access;
- invariant C++ timing checksum/output size before a speedup is interpreted.

Timing checksums are recipe/invariance guards, not a substitute for the exact
byte and canonical-AST differential.

## Reusable commands

Build only this leaf's targets in the existing container; an up-to-date build
will reuse Boost/yaml-cpp/Threads rather than rebuilding completed libraries:

```powershell
docker start virne-cpp-dev
docker exec virne-cpp-dev cmake -S /work -B /work/build `
  -DCMAKE_BUILD_TYPE=Release
docker exec virne-cpp-dev cmake --build /work/build -j 4 `
  --target vne_setting_unit vne_setting_harness
docker exec virne-cpp-dev ctest --test-dir /work/build `
  -R '^vne_setting_unit$' --output-on-failure
```

Run the canonical comparator from the directory containing sibling `virne` and
`virne.cpp` checkouts. Replace `<selected-workers>` only after the sweep; use 8
for the initial full-width measurement and repeat with 0 for automatic mode.

```text
python /workspace/cpp/porting/compare_setting.py \
  --cpp /workspace/cpp/build/porting/vne_setting_harness \
  --python-source /src/virne/virne/utils/setting.py \
  --fixture-root /src/virne/settings \
  --fixture-root /src/virne/tests/settings \
  --fixture-root <additional-final-fixture-root(s)> \
  --expected-fixtures 41 \
  --workers <selected-workers> \
  --warmups 5 --repetitions 31 --operations 4 --batch-size 64 \
  --id-iterations 200000
```

Replace `<additional-final-fixture-root(s)>` with the exact extra root list
used to bring the physical corpus to 41; never satisfy the count by duplicating
a root. Record those resolved roots in the final result artifact.

Use the same pinned oracle image, read-only mounts, disabled network, one-thread
Python numeric environment, and `--cpuset-cpus=0-7` policy recorded in
`porting/README.md`.

Sweep all widths plus automatic mode with rotated/reversed order and invariant
checksums:

```text
python /workspace/cpp/porting/sweep_setting_workers.py \
  --harness /workspace/cpp/build/porting/vne_setting_harness \
  --fixture /src/virne/settings/main.yaml --format both \
  --workers 1,2,3,4,5,6,7,8 \
  --warmups 5 --repetitions 31 --rounds 3 \
  --operations 2 --batch-size 256
```

## Completion record

- Release isolated build/unit: **PASS on 2026-07-28**. The existing Release
  tree rebuilt only `vne_utils_setting`, `vne_setting_unit`, and
  `vne_setting_harness`; the frozen Boost/yaml-cpp dependencies were reused.
  `ctest -R '^vne_setting_unit$' --output-on-failure` passed 1/1 in 0.02 s.
- Exact differential: **POST-TUNING CANONICAL PASS on 2026-07-28**. It passed
  41 physical fixtures, 82 fixture transforms, 20 compatible synthetic cases,
  12 required rejections, 17,513 exact JSON binary64 cases, 10 Python file
  contracts, and exact worker-1/worker-8/automatic batches.
- Canonical Python/C++ timing: **POST-TUNING PASS on 2026-07-28**. All ten rows
  passed under the five-warmup/31-sample protocol. JSON scalar parse, the prior
  failing row, improved from the pre-tuning 0.964692x signal to 1.520856x.
  See the immutable machine-readable artifact and hash above.
- Full worker sweep 1..8 plus automatic: **POST-TUNING PASS on 2026-07-28**;
  invariant checksums/output sizes and selected widths are recorded above.
- Automatic-policy threshold/quality check: **PASS**. Automatic aggregate was
  within 1.0094x of the best single explicit width and every row was within
  1.0640x of its own best explicit median.
- Reentrancy, concurrent caller, synchronized first-use sort cache, and
  lowest-index error tests: **PASS**, including 100 repeated unit runs.
- Full repository Release build and CTest: **PASS**, 18/18.
- `-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Werror`:
  **PASS** for production, unit, and harness.
- ASan/UBSan with leak detection and repeated stress: **PASS** on the final
  post-tuning implementation.
- Frozen graph/CSV/config/yaml-cpp hashes: **PASS**, exact manifest hashes.
- Final measurements note under `porting/results/setting_2026-07-28.md`:
  **PASS**.
