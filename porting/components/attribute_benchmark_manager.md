# Component API: `network.attribute.attribute_benchmark_manager`

State: **COMPLETE** on 2026-07-28. The accepted compact benchmark is frozen;
do not rerun or update its driver/result during later network work.

Source: `../virne/virne/network/attribute/attribute_benchmark_manager.py`,
commit `d1ec1e4a20461fc9bad50612ad5026fd31e693a8`, SHA-256
`8AD9BAB52A40342EAF331E3AC71B3D5F6DB0326742D3D3C7C6AAD7D34AD52D9E`,
7,264 bytes and 200 lines. Completed attribute, graph, dataset/RNG, and
performance documents were read first. The leaf imports NumPy but no Torch or
solver/system code; `BaseNetwork` is type-checking-only.

## Python behavior to retain

- `AttributeBenchmarks` contains optional ordered node, link, and link-sum
  dictionaries. `get_benchmarks` computes enabled groups strictly in that
  order and uses `None` for disabled groups.
- Node/link selection defaults to resource and extrema definitions. A `None`
  type filter selects every definition and derives the type list in definition
  order. The unported `BaseNetwork` owns filtering/data gathering; this leaf
  owns the prepared-data reduction and cache contract.
- Node and link-sum data are cast to a rectangular NumPy `float32` array. Direct
  link data is additionally concatenated with itself on axis one before
  reduction. The prepared C++ input represents that second copy as a column
  repetition count, without allocating it.
- `get_attr_benchmarks` tests whether the requested type list contains exact
  `extrema`. In extrema mode, resource descriptors are skipped and every other
  row is keyed by originator with name fallback. Otherwise every row is keyed
  by name. Rows and descriptors are paired with `zip`, so the shorter first
  dimension wins. Duplicate keys overwrite the value but retain their first
  insertion position.
- Each retained row returns `float(np.max(row))`: comparison occurs in
  float32, including NaN payload/quieting and later-operand signed-zero rules,
  then the result is promoted to binary64. A retained empty row fails; zero
  paired rows return an empty map.
- The class cache is process-global, overwrites an existing string key, returns
  the same stored object identity, returns `None` when absent, and can be
  cleared. Native cache operations are thread-safe without changing identity.
- Python reflection (`getattr`/`str` side effects), non-string dictionary keys,
  ragged/object arrays, arbitrary NumPy dtypes, unbounded integers, and the
  empty-list bug inside the future `BaseNetwork.get_*_attrs_data` adapter are
  recorded dynamic/integration boundaries rather than dynamic native storage.

## Stable C++ surface

```cpp
using AttributeBenchmarkId = std::uint32_t;

struct AttributeBenchmarkDescriptor {
    AttributeDefinitionId definition_id = 0;
    AttributeKind kind = AttributeKind::status;
    std::string name;
    std::optional<std::string> originator_name;
};

struct AttributeBenchmarkMatrix {
    std::size_t rows = 0;
    std::size_t columns = 0;
    std::vector<float> values;
};

struct PreparedAttributeBenchmarkData {
    std::vector<AttributeBenchmarkDescriptor> attributes;
    AttributeBenchmarkMatrix matrix;
    bool extrema_requested = false;
    std::size_t column_repetitions = 1;
};

struct AttributeBenchmarkEntry { std::string name; double value = 0.0; };

class AttributeBenchmarkMap {
public:
    std::optional<AttributeBenchmarkId> bind(std::string_view) const;
    const AttributeBenchmarkEntry& at(AttributeBenchmarkId) const;
    const double* find(std::string_view) const;
    const std::vector<AttributeBenchmarkEntry>& entries() const noexcept;
};

struct AttributeBenchmarks {
    std::optional<AttributeBenchmarkMap> node_attr_benchmarks;
    std::optional<AttributeBenchmarkMap> link_attr_benchmarks;
    std::optional<AttributeBenchmarkMap> link_sum_attr_benchmarks;
};

struct AttributeBenchmarkRequest {
    std::optional<PreparedAttributeBenchmarkData> node;
    std::optional<PreparedAttributeBenchmarkData> link;
    std::optional<PreparedAttributeBenchmarkData> link_sum;
    std::size_t workers = 1;
};

AttributeBenchmarkMap get_attr_benchmarks(
    const PreparedAttributeBenchmarkData&, std::size_t workers = 1);

class AttributeBenchmarkManager {
public:
    explicit AttributeBenchmarkManager(const AttributeBenchmarkRequest&);
    const AttributeBenchmarks& benchmarks() const noexcept;
    static AttributeBenchmarks get_benchmarks(
        const AttributeBenchmarkRequest&);
    static void add_to_cache(
        std::string key, std::shared_ptr<AttributeBenchmarks> value);
    static std::shared_ptr<AttributeBenchmarks> get_from_cache(
        std::string_view key);
    static void clear_cache();
    static std::size_t cache_size();
};
```

Exact typed errors/stages and immutable/mutable accessors are finalized in the
header. The later BaseNetwork port will gather rows in definition order using
completed Node/Link bindings and pass this prepared surface; it must not
reimplement reduction or cache behavior.

## ID, threading, and performance rules

- Kind/filter decisions are enums/direct flags. Descriptor strings are owned
  once. Output names resolve once to compact `AttributeBenchmarkId`; consumers
  use `at(id)` inside repeated normalization loops.
- Matrix validation and active-row selection finish before workers. Row workers
  receive only contiguous float pointers, row indices, fixed repetition count,
  and pre-sized result slots. No worker performs string, map, registry, variant,
  or definition lookup.
- Independent row maxima use deterministic contiguous row blocks at configured
  workers `0/1/2/8`; zero/one is sequential and wider values are count/CPU
  capped. Output insertion and duplicate overwrite remain sequential in input
  order. Thread-launch failure completes unlaunched blocks locally and joins
  launched workers.
- Direct-link column repetition is virtual: it preserves exact reduction order
  without allocating/copying the concatenated matrix. Shared cache mutation is
  protected by a mutex; cache-key hashing occurs once per cold cache call.

## Accepted gate

Cover empty/truncated/extra rows, invalid matrix shapes, retained empty rows,
extrema branch selection, resource skip, originator fallback, duplicate keys
and insertion order, binary32 rounding, infinities, subnormals, all signed-zero
orders, qNaN/sNaN payloads and repetition, workers `0/1/2/8`, concurrent
independent reductions, optional group order, and cache overwrite/identity/
clear/concurrency. Record the deferred BaseNetwork empty-filter and Python
reflection/object/ragged/unbounded boundaries explicitly.

The isolated unit covers empty/truncated/extra rows, invalid shapes and
repetitions, retained empty rows, extrema/resource selection, originator
fallback, duplicate overwrite/order, compact-ID copy/move lifetime, binary32
rounding, infinities, subnormals, signed zero, qNaN/sNaN payloads, virtual
repetition, workers `0/1/2/8`, concurrent reductions, group order, and cache
identity/overwrite/clear/concurrency.

The exact direct-source differential passed 19 shared Python cases and seven
native extension/error cases. Six Python-only boundaries are recorded
separately, for 32 total cases. NumPy 2.2.6 outputs are compared as ordered
UTF-8 names plus raw binary64 bits; no tolerance is used.

The one accepted prepared-row benchmark uses 4,096 rows by 128 columns with a
virtual repetition count of two, one warm-up, three samples, and configured
workers `1/2/8`. Fixture creation, process startup, and checksum work are
excluded. All rows retain checksum `16589509004670834835`, 80,810 output bytes,
and 4,096 entries.

| Workers | Python median | C++ median | Speedup |
|---:|---:|---:|---:|
| 1 | 17.032365 ms | 1.857281 ms | 9.171x |
| 2 | 17.032365 ms | 1.837471 ms | 9.269x |
| 8 | 17.032365 ms | 1.484077 ms | 11.477x |

Strict GCC 11 warnings-as-errors, ASan, UBSan, leak checks, and full CTest
28/28 pass. Detailed evidence and immutable artifact hashes are in
`../results/attribute_benchmark_manager_2026-07-28.md`.
