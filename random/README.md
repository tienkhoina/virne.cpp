# Random compatibility contract

This file is the canonical public-API and determinism contract for the Random
component. The declarations in `py_random.h`, `numpy_random_state.h`,
`random_context.h`, and `ndarray.h` must remain synchronized with the inventory
below. The root `README.md` should only summarize this component and link here;
it must not maintain a second, partial API list.

`PyRandom` reproduces CPython 3.10 `random.Random` state consumption for this
repository's fixed C++ subset and non-negative 64-bit integer seeds.
`NumpyRandomState` is an independent MT19937 stream matching NumPy 1.26.4
legacy `numpy.random.RandomState` for the subset used by Virne. Value order and
the exact next state are part of the contract, not only the output
distribution.

Parity covers returned values, ordering, validation/state-consumption order,
and continuation of the stream. C++ exception classes and messages are the C++
surface; they are not required to have Python exception names or text unless a
specific rule below says otherwise.

## Complete public API

### `PyRandom`

The complete non-template and template surface in `py_random.h` is:

```cpp
explicit PyRandom(uint64_t seed);

void seed(uint64_t value);

// One raw tempered MT19937 word. This advances the same stream used by every
// higher-level operation.
uint32_t genrand_uint32();

double random();
double uniform(double a, double b);

uint32_t getrandbits32();
uint64_t getrandbits(int k);              // 0 <= k <= 64

uint64_t randrange(uint64_t stop);
int64_t randrange(int64_t start, int64_t stop);
int64_t randrange(int64_t start, int64_t stop, int64_t step);
int64_t randint(int64_t a, int64_t b);    // inclusive at both ends

template<typename T>
T& choice(std::vector<T>& population);

template<typename T>
const T& choice(const std::vector<T>& population);

template<typename T>
void shuffle(std::vector<T>& values);

template<typename T>
std::vector<T> choices(
    const std::vector<T>& population,
    size_t k);

template<typename T>
std::vector<T> choices(
    const std::vector<T>& population,
    const std::vector<double>& weights,
    size_t k);

template<typename T>
std::vector<T> choices_weights(
    const std::vector<T>& population,
    const std::vector<double>& weights,
    size_t k);

template<typename T>
std::vector<T> choices_cum_weights(
    const std::vector<T>& population,
    const std::vector<double>& cumulative_weights,
    size_t k);
```

The Python mapping is:

| C++ API | CPython 3.10 equivalent |
|---|---|
| constructor / `seed(uint64_t)` | `Random.seed(int)` inside the documented fixed-width seed domain |
| `random()` | `Random.random()` |
| `uniform(a, b)` | `Random.uniform(a, b)`; the exact formula is `a + (b-a) * random()`, so reversed bounds are permitted and endpoint inclusion depends on floating rounding |
| `getrandbits32()` | `Random.getrandbits(32)` |
| `getrandbits(k)` | `Random.getrandbits(k)` for `0 <= k <= 64` |
| `randrange(...)` | `Random.randrange(...)` |
| `randint(a, b)` | `Random.randint(a, b)` |
| `choice(vector)` | `Random.choice(sequence)` |
| `choices(vector, k)` | `Random.choices(population, k=k)` |
| `choices(vector, weights, k)` / `choices_weights(...)` | `Random.choices(population, weights=weights, k=k)` |
| `choices_cum_weights(...)` | `Random.choices(population, cum_weights=..., k=k)` |
| `shuffle(vector)` | `Random.shuffle(list)` |

`genrand_uint32()` is a supported low-level C++ escape hatch rather than a
separate Python algorithm. `getrandbits32()` currently consumes the same one raw
word. Calling either raw API changes every subsequent result, so application
code must not insert raw draws into a compatibility sequence accidentally.

CPython assembles 33--64-bit `getrandbits` values in little-endian MT-word
order and deliberately uses rejection sampling for powers of two. The C++
implementation preserves both details. Arbitrary-size or negative Python seed
integers and the rest of Python's distribution APIs are outside this surface. A
signed interval whose cardinality is the complete `2^64` domain is rejected
because CPython's required `getrandbits(65)` result cannot be represented by the
fixed-width contract.

The empty-population state rules are intentional:

- `choice(empty)` throws without consuming the stream.
- Unweighted `choices(empty, 0)` returns an empty vector without consuming the
  stream.
- Unweighted `choices(empty, k)` for `k > 0` consumes exactly one `random()`
  draw and then throws, matching CPython's evaluation order.
- Weighted and cumulative-weight forms validate the weight input before any
  draw. An empty weight array therefore throws without consuming the stream.

### `NumpyRandomState`

`Shape` is an alias for `NdArray<double>::Shape`, which is
`std::vector<std::size_t>`. Scalar defaults apply only to the scalar overloads;
the vector and shape overloads require their distribution parameters
explicitly.

```cpp
using Shape = NdArray<double>::Shape;

explicit NumpyRandomState(std::uint32_t seed_value = 0);
void seed(std::uint32_t seed_value);

// One raw tempered legacy-MT19937 word.
std::uint32_t next_uint32();

double random();
std::vector<double> random(std::size_t size);
NdArray<double> random(const Shape& shape);

double rand();
NdArray<double> rand(const Shape& dimensions);

template<typename... Dimensions>
NdArray<double> rand(Dimensions... dimensions); // one or more integral dims

double uniform(double low = 0.0, double high = 1.0);
std::vector<double> uniform(
    double low,
    double high,
    std::size_t size);
NdArray<double> uniform(
    double low,
    double high,
    const Shape& shape);

std::int64_t randint(std::int64_t high);
std::int64_t randint(std::int64_t low, std::int64_t high);
std::vector<std::int64_t> randint(
    std::int64_t low,
    std::int64_t high,
    std::size_t size);
NdArray<std::int64_t> randint(
    std::int64_t high,
    const Shape& shape);
NdArray<std::int64_t> randint(
    std::int64_t low,
    std::int64_t high,
    const Shape& shape);

// Named high-only vector convenience overload.
std::vector<std::int64_t> randints(
    std::int64_t high,
    std::size_t size);

double normal(double location = 0.0, double scale = 1.0);
std::vector<double> normal(
    double location,
    double scale,
    std::size_t size);
NdArray<double> normal(
    double location,
    double scale,
    const Shape& shape);

double exponential(double scale = 1.0);
std::vector<double> exponential(double scale, std::size_t size);
NdArray<double> exponential(double scale, const Shape& shape);

std::int64_t poisson(double lambda = 1.0);
std::vector<std::int64_t> poisson(
    double lambda,
    std::size_t size);
NdArray<std::int64_t> poisson(
    double lambda,
    const Shape& shape);
```

The complete `choice`, `shuffle`, and `permutation` overload set is:

```cpp
std::int64_t choice(std::int64_t population_size);
std::int64_t choice(
    std::int64_t population_size,
    const std::vector<double>& probabilities);

std::vector<std::int64_t> choice(
    std::int64_t population_size,
    std::size_t size,
    bool replace = true);
std::vector<std::int64_t> choice(
    std::int64_t population_size,
    std::size_t size,
    const std::vector<double>& probabilities,
    bool replace = true);

// NumPy positional spelling: (a, size, replace, p).
std::vector<std::int64_t> choice(
    std::int64_t population_size,
    std::size_t size,
    bool replace,
    const std::vector<double>& probabilities);

template<typename T>
const T& choice(const std::vector<T>& population);

template<typename T>
const T& choice(
    const std::vector<T>& population,
    const std::vector<double>& probabilities);

template<typename T>
std::vector<T> choice(
    const std::vector<T>& population,
    std::size_t size,
    bool replace = true);

template<typename T>
std::vector<T> choice(
    const std::vector<T>& population,
    std::size_t size,
    const std::vector<double>& probabilities,
    bool replace = true);

template<typename T>
void shuffle(std::vector<T>& values);

template<typename T>
void shuffle(NdArray<T>& values);

template<typename T>
std::vector<T> permutation(const std::vector<T>& values);

template<typename T>
NdArray<T> permutation(const NdArray<T>& values);

std::vector<std::int64_t> permutation(std::size_t size);
```

The mapping is:

| C++ group | NumPy 1.26.4 legacy equivalent |
|---|---|
| constructor / `seed(uint32_t)` | `RandomState(seed)` with a scalar seed |
| `random()` / `random(size)` / `random(shape)` | `random_sample(size)` |
| `rand()` / `rand(dimensions...)` | `RandomState.rand(*dimensions)` |
| `uniform(low, high, ...)` | `uniform(low, high, size)` |
| `randint(...)` / `randints(high, size)` | `randint(low, high, size, dtype=int64)` |
| `choice(vector, ...)` | `choice(array_like, size, replace, p)` |
| `choice(population_size, ...)` | `choice(a_as_int, size, replace, p)` without materializing `arange(a)` |
| `shuffle(vector)` / `permutation(vector)` | one-dimensional `shuffle` / `permutation` |
| `shuffle(NdArray)` / `permutation(NdArray)` | axis-0 `shuffle` / `permutation` |
| `normal(...)` | `normal(loc, scale, size)` |
| `exponential(...)` | `exponential(scale, size)` |
| `poisson(...)` | `poisson(lam, size)` |

`next_uint32()` is the supported raw legacy-MT output hook. NumPy does not
expose it as a distinct `RandomState` distribution method. It advances the same
state as all higher-level calls and is intended for algorithm implementation,
oracle work, and low-level profiling.

The C++ boundary deliberately supports scalar distribution parameters only.
Broadcast parameters, ndarray views/strides/dtypes, array seed objects,
arbitrary-axis permutation, and the newer `numpy.random.Generator` API remain
outside this class. Numeric-to-`bool` or numeric-to-integer conversions done by
Virne after a draw belong to the caller, not to RNG state generation.

### `NdArray<T>`

`NdArray<T>` is a small owning shape-and-storage value, not a general NumPy
implementation. Its complete public surface is:

```cpp
using value_type = T;
using Shape = std::vector<std::size_t>;
using Storage = std::vector<T>;
using iterator = typename Storage::iterator;
using const_iterator = typename Storage::const_iterator;

NdArray();
explicit NdArray(Shape shape);
NdArray(Shape shape, Storage values);

static std::size_t checked_size(const Shape& shape);

const Shape& shape() const noexcept;
std::size_t ndim() const noexcept;
std::size_t size() const noexcept;
bool empty() const noexcept;

Storage& flat() noexcept;
const Storage& flat() const noexcept;
T* data() noexcept;
const T* data() const noexcept;

T& operator[](std::size_t index) noexcept;
const T& operator[](std::size_t index) const noexcept;

T& at(std::size_t index);
const T& at(std::size_t index) const;
T& at(const Shape& indices);
const T& at(const Shape& indices) const;

iterator begin() noexcept;
const_iterator begin() const noexcept;
const_iterator cbegin() const noexcept;
iterator end() noexcept;
const_iterator end() const noexcept;
const_iterator cend() const noexcept;
```

Elements are contiguous in NumPy C order. `operator[]`, `data()`, and iterator
access are unchecked; both `at` forms are checked. Shape products are checked
before allocation or RNG consumption. An explicit empty shape, `Shape{}`,
represents a 0-D array containing one scalar. Any ordinary shape with a zero
dimension has empty storage. The default constructor is deliberately different:
it creates the empty one-dimensional shape `{0}`.

`shuffle(NdArray&)` and `permutation(const NdArray&)` operate along axis 0. A
0-D array is invalid for either operation. There is one important NumPy state
distinction for shapes such as `{3, 0}` whose storage is empty but whose axis 0
has multiple rows:

- `shuffle` returns without generating swaps and does not consume RNG state.
- `permutation` still permutes the axis-0 indices, consumes the corresponding
  bounded-integer draws, and returns an empty array with the same shape.

### `RandomContext` and process-global accessors

The complete context API is:

```cpp
explicit RandomContext(std::uint32_t seed_value = 0);

PyRandom& python() noexcept;
const PyRandom& python() const noexcept;
NumpyRandomState& numpy() noexcept;
const NumpyRandomState& numpy() const noexcept;

void set_seed(
    std::optional<std::uint32_t> seed_value = std::nullopt);

RandomContext& global_random_context();
PyRandom& global_py_random();
NumpyRandomState& global_numpy_random();

void set_seed(
    std::optional<std::uint32_t> seed_value = std::nullopt);
```

A context owns two intentionally independent streams. Supplying a seed resets
both streams with that same scalar, while `set_seed()` and
`set_seed(std::nullopt)` are no-ops. `NumpyRandomState()`, `RandomContext()`, and
the process-global context are deterministically initialized with seed `0`.
They do **not** reproduce the entropy-seeded behavior of a no-argument Python
module RNG or `np.random.RandomState()`.

Prefer an explicit `RandomContext&` in new code. The process-global accessors
exist for ports of Python module-global APIs; they expose mutable state and are
not internally synchronized.

This context covers only the CPython and NumPy-compatible streams. Virne's
Python `utils.dataset.set_seed` also seeds PyTorch/CUDA and configures
`PYTHONHASHSEED`, deterministic cuDNN behavior, and
`CUBLAS_WORKSPACE_CONFIG`. A future LibTorch/PyG adapter must perform those
Torch, CUDA, backend, and environment steps separately. It must not add a
LibTorch dependency to this Random component or silently treat
`RandomContext::set_seed` as the complete Torch seeding policy.

## Hot-loop and bulk-state rule

MT19937 state has one owner and one logical order. Performance work must obey
all of the following rules:

- Resolve graph string keys and attributes once outside a hot loop. Random code
  that feeds graph kernels receives contiguous arrays of pre-resolved numeric
  node/edge/attribute IDs; it must not perform string lookup per draw.
- When the number of draws is known, call a vector or shape overload once, then
  traverse `data()`, `flat()`, or the returned `std::vector` by numeric index.
- When the number of draws is data-dependent, keep scalar calls in the exact
  Python/NumPy logical order. Do not batch across a conditional boundary merely
  to improve throughput.
- Do not look up a global context, reseed, allocate a temporary population, or
  dispatch by a string inside a hot loop. Pass the explicit stream and resolved
  contiguous data into the loop.
- Worker threads may transform disjoint output ranges only after the owning
  thread has advanced MT state in canonical order. They must never draw from or
  mutate one shared RNG concurrently.
- Raw-word calls are not a faster substitute for a distribution unless the
  implementation exactly preserves its rejection algorithm and continuation
  state and the differential oracle covers that path.

These rules preserve the original design: convenient string-facing setup at
the boundary, but IDs, indices, direct addresses, and bulk buffers in every hot
path.

## Accepted libstdc++ output-buffer hack

Large vector-valued draws overwrite every scalar, so zero-initializing the
entire `std::vector` first adds a redundant memory pass. On the frozen Release
baseline—**GCC 11.4.0 + libstdc++ 11** (`_GLIBCXX_RELEASE=11`,
`__GLIBCXX__=20230528`)—`DirectOutputVector<T>` reserves raw storage, begins
the lifetime of each trivial scalar without value-initializing it, and advances
libstdc++'s internal `_M_impl._M_finish` pointer before filling the output. The
optimization is restricted to trivial, trivially destructible scalar types.

For `randint` outputs of at least `2^20` values whose complete result range
fits signed 32-bit, the pinned GNU path also pipelines the independent output
conversion. The owner thread alone advances MT19937 and compacts accepted
masked-rejection offsets, in exact stream order, into the lower half of each
disjoint `vector<int64_t>` chunk. Worker threads only widen those stored
`uint32_t` offsets backwards in place and add `low`; they never read or mutate
RNG state. Backward widening prevents a destination write from overwriting an
unread offset, while disjoint chunks prevent inter-worker overlap. Release is
compiled with `-fno-strict-aliasing` for this deliberate representation reuse.

The unit suite compares every value across `2^20 + 17` draws and the following
RNG value against scalar state consumption. It separately forces the
262,144-value two-pass rejection path with a narrow interval above signed
32-bit and compares every output plus the following state. Empty vector
overloads return without pointer arithmetic or RNG consumption, and extreme
valid Poisson lambdas reject out-of-range floating candidates before conversion
to `int64_t`.

This is an accepted implementation-layout risk, not a standard C++ vector API.
Non-GNU standard libraries take the portable zero-initializing fallback. GNU
libstdc++ also takes that fallback unless the compiler/library tuple is exactly
GCC 11.4/libstdc++ 11 (`20230528`); only that pinned tuple enables the direct
output path. A toolchain upgrade requires deliberately updating that guard
after a layout audit, then passing the complete bit-exact differential suite,
the large-`randint` continuation checks, ASan/UBSan, and the Random benchmark.
The production dependency and toolchain policy is frozen in
`../DEPENDENCIES.md`.

## Oracle, tests, and benchmark scope

The NumPy-compatible algorithms are adapted from NumPy's BSD-licensed legacy
random implementation; see `third_party/NOTICE.md` and
`third_party/NUMPY_LICENSE.txt`.

The unit contract test includes hard-coded CPython 3.10 and NumPy 1.26.4
oracles, mixed continuation-state checks, invalid inputs, 0-D/zero-sized shapes,
empty-population evaluation order, empty ndarray shuffle/permutation behavior,
and both large `randint` paths:

```bash
ctest --test-dir build -R '^random_python_compat$' --output-on-failure
```

The benchmark and differential harness are `EXCLUDE_FROM_ALL` developer
targets, so normal builds do not compile them:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target benchmark_random random_differential_harness
./build/random/benchmark_random
./.venv/bin/python random/benchmark_compare.py \
  --cpp build/random/benchmark_random
./.venv/bin/python random/differential_test.py \
  --cpp build/random/random_differential_harness
# Or build and run the pinned differential oracle in one step:
cmake --build build --target random_differential
```

`differential_test.py` requires the repository-local test-only environment with
CPython 3.10 and NumPy 1.26.4. It currently checks 1,260 CPython values across
nine seeds, 2,368 NumPy values across eight seeds, and 262,145 values in the
large-`randint` output/next-state case. Production C++ targets never import or
link Python or NumPy.

The sanitizer commands used by this repository are an out-of-tree build and do
not install anything into the environment. LeakSanitizer is deliberately
disabled here because this is the ASan/UBSan gate and managed ptrace runners
cannot execute LSan reliably:

```bash
cmake -S . -B /tmp/virne-random-sanitize \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined'
cmake --build /tmp/virne-random-sanitize \
  --target test_random random_differential_harness -j2
ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  /tmp/virne-random-sanitize/random/test_random
ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  ./.venv/bin/python random/differential_test.py \
    --cpp /tmp/virne-random-sanitize/random/random_differential_harness
```

There is currently no repository-owned, reproducible TSan target, so TSan is
not part of the documented release gate and must not be cited as one. Add such a
target and command before promoting it to this contract.

`benchmark_compare.py` uses paired, alternating-order repetitions and enforces
strict `C++ < CPython/NumPy` timing for exactly seven measured rows:

- `PyRandom.random`
- `PyRandom.randint(-1000,1000)`
- `NumpyRandomState.random`
- `NumpyRandomState.randint(-1000,1001)`
- `NumpyRandomState.normal(0,1)`
- `NumpyRandomState.exponential(1)`
- `NumpyRandomState.poisson(20)`

Passing those seven gates proves only that every currently measured row is
faster on the recorded baseline. It is not evidence that every public overload
in this inventory is faster. `choice`, `choices`, `shuffle`, `permutation`,
`randrange`, shape construction, raw-word access, and context access remain
correctness-tested or utility APIs unless a dedicated benchmark row is added.
