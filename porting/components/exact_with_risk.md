# Component API: `solver.exact_with_risk`

State: **IMPLEMENTED / FOCUSED OBJECTIVE GATE PASS** on 2026-08-01.

`exact_with_risk` is an intentionally separate MIP leaf. Its implementation is
copied into `virne/solver/exact/exact_with_risk.cpp` so the original `mip`,
`d_round` and `r_round` objective paths remain unchanged. Feasibility rows,
resource binding, split-flow journal extraction and mutable deployment use the
same exact API. The only optimization change is the objective coefficient.
The leaf inherits the exact solver's fail-closed guards (including virtual or
physical self-loops and unsupported tiny positive link demands), checked integer-lane
journal arithmetic, and transaction ownership rules: an exception rolls back
only a transaction opened by the current call, never a caller-owned active
transaction.

## API and configuration

```cpp
struct ExactRiskParameters {
    double scarcity_weight = 1.0;
    double balance_weight = 1.0;
    double criticality_weight = 1.0;
};

class ExactWithRiskSolver final : public Solver {
public:
    ExactWithRiskSolver(
        SolverDependencies,
        SolverConfig,
        PyRandom&,
        ExactSolverParameters = {},
        ExactRiskParameters = {});
};

SolverId register_exact_with_risk_solver(
    SolverRegistry&,
    PyRandom&,
    ExactSolverParameters = {},
    ExactRiskParameters = {});
```

Main resolves these cold configuration fields once:

```text
native.exact_risk.scarcity_weight
native.exact_risk.balance_weight
native.exact_risk.criticality_weight
native.workers.exact
```

The registry name is `exact_with_risk` and its category is `exact`. The solver
still processes one VNR at a time against the current physical residual state.

## Objective

The primary term is the original total base flow:

```text
route_cost = sum(q[virtual_edge, physical_arc])
```

For each placement or flow variable, a fixed-state secondary coefficient is
computed from numeric residual-capacity buffers:

1. `scarcity`: marginal demand divided by the current residual capacity;
2. `balance`: change in squared skew of the normalized residual resource
   vector across CPU/GPU/RAM or BW/spectrum lanes;
3. `criticality`: the scarcity term on a physical bridge, computed once with a
   cold Tarjan traversal.

The construction is a linear marginal surrogate for the Resource Fragmentation
Degree (RFD) idea of Lu and Zhang, *Resource Fragmentation-Aware Embedding in
Dynamic Network Virtualization Environments*, IEEE TNSM 19(2), 936--948
(2022), DOI [10.1109/TNSM.2022.3152309](https://doi.org/10.1109/TNSM.2022.3152309).
The published RFD neighborhood metric is nonlinear; this implementation does
not claim to reproduce its full formula. It freezes the current residual
state, making the surrogate compatible with the existing exact feasibility
rows and multi-resource journal.

For integral path-flow variables, risk coefficients are divided by a bound
strictly larger than both the complete absolute risk range and every raw
coefficient. The single objective is therefore:

```text
min route_cost + normalized_risk
```

with `|normalized_risk| < 1` over the complete model domain. For the integral
MIP, one unit of route flow always dominates every possible risk difference.
Thus a longer path can never be selected for a lower risk score; risk only
breaks equal-flow-length ties.

Floating-only resource lanes use continuous path flow. Because no finite
scalar can dominate every arbitrarily small continuous objective difference,
that case uses two phases: first prove the minimum route flow, lock it with an
equality row, then minimize normalized risk. In both modes the solver accepts
only `OPTIMAL`; a time/node-limited `FEASIBLE` incumbent is rejected rather
than weakening the route-dominance contract. Focused tests cover integral
dominance and floating-only feasibility.

## Performance and field contract

Resource names resolve once when the model is prepared. Risk computation uses
`AttrId`-indexed dense buffers, direct graph IDs and precomputed bridge flags;
no string map lookup occurs in model/objective loops. Node and physical-edge
risk coefficients are frozen once into dense `V_v x V_p` and `E_v x E_p`
buffers, and both directed arcs reuse the same physical-edge coefficient.
Capacity-tight variable bounds exclude unusable edges from normalization;
non-finite derived coefficients fail closed before OR-Tools. The
cold bridge traversal is `O(|V|+|E|)`. Risk is a tie-breaker for one VNR, not a
batch optimizer over future requests. It does not implement nonlinear global
entropy or exact future-demand fragmentation.

## Validation

The focused `vne_exact_solver_unit` fixture uses two equal two-hop routes with
different BW/spectrum residual vectors and confirms that `exact_with_risk`
chooses the lower-risk route. A second fixture offers a spacious three-hop
route and confirms that the two-hop route remains selected. Risk-specific
coverage also exercises split flow, floating-only flow and inherited
self-loop/tiny-demand guards. The same unit retains the original exact
multi-resource, journal and transaction checks.

Run the focused gate with:

```bash
cmake --build build --target vne_exact_solver_unit -j
ctest --test-dir build -R '^vne_exact_solver_unit$' --output-on-failure
```

The Docker GCC 11 gate passed both registered CTests: the focused unit and the
native Main multi-resource smoke. Main accepted 3/3 generated requests through
`exact_with_risk`, including a direct run with `native.workers.exact=8`. This
is an implementation gate, not a replacement for a separately frozen Python
differential benchmark.
