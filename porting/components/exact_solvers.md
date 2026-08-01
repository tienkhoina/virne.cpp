# Component API: `solver.exact`

State: **IMPLEMENTED / FOCUSED MULTI-RESOURCE GATE PASS** on 2026-08-01. The
native OR-Tools target, three compatibility registry entries and
multi-resource formulation are present. The separate `exact_with_risk` leaf is
documented in `exact_with_risk.md`. The focused Docker GCC 11 unit passed 1/1
CTest; the reported CTest wall time includes process/test-runner overhead and
need not equal direct executable runtime. This is a focused implementation
gate, not a full or frozen cross-language parity claim.

Python oracles are pinned at commit
`d1ec1e4a20461fc9bad50612ad5026fd31e693a8`:

| Source | SHA-256 |
|---|---|
| `virne/solver/exact/mip.py` | `7E68586641C81AA556110A8CEAF3AC1A96ADDC564890D6B6FBE3F1D984693B2A` |
| `virne/solver/exact/d_rounding.py` | `D71138F3CF50AF5D0F66BF8B99E70A1434B125282DD774EA17AF5F87A9215E3E` |
| `virne/solver/exact/r_rounding.py` | `3E5215AA8E06EB626ED157AC43369E41687B8C596EBACED13A1C6BDD528DF9FD` |

The original Python formulations assert one node resource (`cpu`) and one link
resource (`bw`). The native formulation intentionally removes that limitation:
every resource selected by the prepared Controller is a separate hard capacity
lane, and all lanes share one placement and one route.

## Local OR-Tools dependency

OR-Tools C++ 9.15.6755 is workspace-local and is never installed into the OS
or a global compiler prefix:

| Platform | Required path | CMake target |
|---|---|---|
| Ubuntu 22.04 x86-64 | `libs/ortools` | imported `virne_ortools` |
| Visual Studio 2022 x64 | `libs/ortools-win` | imported `virne_ortools` |

`libs/` is intentionally Git-ignored. The official archive URLs, exact
SHA-256 values and fresh-clone extraction commands are in `DEPENDENCIES.md`
and `DEPENDENCIES.sha256`. CMake does not call `find_package(ortools)` and must
not fall back to an OS/Conda package. Both platform payloads are installed in
the prepared workspace at the paths above. The exact target follows the
official package contract: C++20 for MSVC and C++17 for GCC/Clang; this does
not change the language level of the frozen libraries.

## Public API and registry

All names below are in `virne::solver::exact`.

```cpp
enum class ExactAlgorithm : std::uint8_t {
    mixed_integer,
    deterministic_rounding,
    randomized_rounding,
};

struct ExactSolverParameters {
    std::uint64_t time_limit_ms = 10'000U;
    std::uint64_t search_node_limit = 5'000'000U;
    std::size_t workers = 1U;
};

class ExactSolver final : public Solver {
public:
    ExactSolver(SolverDependencies, SolverConfig, ExactAlgorithm,
                PyRandom&, ExactSolverParameters = {});
    ExactAlgorithm algorithm() const noexcept;
    const ExactSolverParameters& parameters() const noexcept;
    core::Solution solve(const SolverInstance&) override;
    MutableSolverResult solve_mutable(
        const MutableSolverInstance&) override;
};

struct ExactSolverIds {
    SolverId mip;
    SolverId d_round;
    SolverId r_round;
};

ExactSolverIds register_exact_solvers(
    SolverRegistry&, PyRandom&, ExactSolverParameters = {});
```

Registration order and categories are stable:

| Direct ID field | Name | Category | Algorithm tag |
|---|---|---|---|
| `mip` | `mip` | `exact` | `mixed_integer` |
| `d_round` | `d_round` | `rounding` | `deterministic_rounding` |
| `r_round` | `r_round` | `rounding` | `randomized_rounding` |

The numeric `SolverId` depends on what was registered earlier; Main registers
the complete heuristic catalog first. Code therefore retains the returned
direct fields, or resolves a CLI name once before the request loop, and never
hard-codes a global numeric value.

`mip` uses the integral OR-Tools model (SCIP, then CBC/SAT fallback). The two
rounding entries solve the multi-resource LP with GLOP, select a
capacity-fitting non-reused physical node for each virtual node, then hand the
fixed placement to the prepared Controller for normal topology/resource
mapping. Each capacity-feasible candidate is weighted by its LP placement
value times virtual meta-edge flow. `d_round` takes the largest weight with
lowest-ID tie breaking using the same strict `>` comparison as Python (no
epsilon tie window); `r_round` samples those weights through the caller-owned
`PyRandom` stream and uses a uniform draw only when the total weight is exactly
zero.
`time_limit_ms` applies to every backend; `search_node_limit` is wired to
SCIP's solver-specific node bound. `workers` resolves from
`native.workers.exact` and values above one are passed once to the MIP backend;
GLOP remains single-threaded because its MPSolver interface rejects that
control. Zero and values outside the backend `int` domain fail closed. The
central registrar requires a
caller-owned `PyRandom&`, so `r_round` cannot be registered with a null stream.

`exact_with_risk` is registered separately and keeps the same feasibility
formulation while adding a normalized residual-fragmentation tie-breaker; see
`porting/components/exact_with_risk.md` for its parameters and proof that route
length remains lexicographically dominant.

## Multi-resource formulation

Let `x[v,p]` be a placement variable and `q[e,a]` the amount of a virtual
link's base flow on directed physical arc `a`. `x` is binary for `mip` and
continuous `[0,1]` in GLOP. For integer resource lanes, `u[e]` is their
positive-demand GCD and `q` remains integer so every integer journal amount is
exact. When an edge has only floating positive lanes, the first positive
floating demand becomes `u[e]` and `q` is continuous inside the otherwise
mixed-integer model. Every resource lane keeps the fixed ratio
`demand[e,r] / u[e]`.

- Every virtual node is assigned once: `sum_p x[v,p] = 1`.
- With `reusable=false`, each physical node hosts at most one virtual node.
- For every physical node `p` and every selected node resource `r`,
  `sum_v demand[v,r] * x[v,p] <= capacity[p,r]`.
- A candidate variable has upper bound zero if any selected node-resource lane
  is individually infeasible.
- Integral conservation is
  `out(q)-in(q)=u[e]*(x[source,p]-x[target,p])`. It permits the same integer
  split flow as Python MIP when no single path has enough capacity.
- Floating-only links use the same conservation row with continuous `q`, so a
  non-integral base demand such as `1.5` remains feasible and is journaled as a
  floating amount.
- GLOP retains compact source/target meta-edge flow variables. Their aggregate
  `META_BW=9999` bound and the rounding score follow Python without allocating
  its unused all-pairs variables.
- For every physical edge `l` and every selected link resource `r`, both arc
  directions share the same undirected capacity:
  `sum_e ratio[e,r] * (q[e,l+] + q[e,l-]) <= capacity[l,r]`.
- Each path-variable upper bound is tightened once from every resource lane's
  residual capacity; exhausted physical edges do not enter the solve domain.
- Integral `mip` minimizes total physical flow plus the constant placement
  demand term. GLOP weights flow by reciprocal physical capacity and placement
  by demand/capacity before rounding.

The result journal writes every selected node resource to `node_slots_info`
and every selected link amount (`q * ratio`) to `link_paths_info`; a MIP route
may therefore contain several flow branches. Integer-lane journal amounts are
calculated with checked `int64` division/multiplication (not a double
round-trip), while floating lanes retain floating arithmetic. `solve` leaves
the physical network unchanged. `solve_mutable` deploys through the prepared
Controller and reports `committed` only after every resource lane is applied.
It preserves a caller-open transaction on success. If no transaction is open,
it begins and commits its own transaction; deployment failure rolls that owned
transaction back. On exception, only a transaction opened by this call is
rolled back; a caller-owned active transaction remains active for the caller
to inspect or roll back. A failed model remains detached.

This model covers prepared hard node/link resource selections only. It fails
closed when node/link selections contain a non-resource constraint, whenever
any path/graph constraint is selected, when a virtual or physical graph
contains a self-loop, or when a positive link demand is below the supported numeric
threshold (`0 < demand <= 1e-9`). It also fails closed when an integer
coefficient exceeds the exact MPSolver `double` range (`2^53`) or a derived
resource ratio is non-finite. Extracted
integral journals are checked again with typed resource arithmetic before
success. Nothing is silently omitted; each additional exact constraint needs
an explicit model row and focused test.

## Fixed-field and hot-loop contract

Solver controls, algorithm, registry IDs, graph vertices/edges, resource
bindings and Solution slots are typed fields. A dynamic resource name is used
only once while binding the independent virtual and physical registries. Model
preparation then stores graph-local `AttrId`s and contiguous row-major numeric
buffers. Variable, constraint, capacity and extraction loops use numeric IDs
and direct vector offsets only; no string map lookup is permitted there.

OR-Tools placement/path variables are anonymous: the model passes an empty
name and retains each `MPVariable*` directly in dense numeric-indexed tables.
This avoids constructing diagnostic strings such as `x_v_p`/`y_e_a` for every
variable and prevents name lookup from entering model or extraction loops.

## Build and focused test

After reconstructing `libs/ortools` as documented in `DEPENDENCIES.md`, the
canonical Linux Release commands are:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target vne_exact_solver_unit -j
ctest --test-dir build -R '^vne_exact_solver_unit$' --output-on-failure
```

The focused unit fixture shifts physical registry IDs with leading status
attributes, then uses two node resources (`cpu`, `gpu`) and two link resources
(`bw`, `spectrum`). It covers all three registry entries, feasible output,
const-input immutability, rejection caused only by the second node/link lane,
aggregate reusable capacity, forced split flow, exact `int64` split journals,
floating-only link flow, the self-loop/tiny-demand/`2^53`/path-level guards,
complete resource journals, mutable deploy and release restoration. A Windows
clang-cl/NMake ABI smoke against the installed `libs/ortools-win` passed. The
current Docker GCC 11 focused gate is:

```text
1/1 vne_exact_solver_unit ... Passed
```

## Validation and benchmark record

The focused unit timing is not a Python differential benchmark. Do not copy
timings from heuristic, Controller, OR-Tools examples, another compiler or a
different fixture into this note. The accepted cross-language result must
record compiler/build mode, backend, time limit, resource-lane count, output
feasibility/checksum, Python and C++ wall time on the same fixture, and whether
the Python run is the one-resource compatibility case or the documented native
multi-resource extension.

The focused one-resource compatibility measurement is recorded in
`porting/results/exact_solvers_2026-07-31.md`: the structural signature matched
and the native median was **1.955914x** faster (Python 16.8899 ms, C++ 8.6353
ms). It is a small module signal, not canonical/full parity: symmetric concrete
mappings are not compared, only one MIP fixture is timed, and the host Python
and native packages use OR-Tools 9.12 and local 9.15 respectively.
