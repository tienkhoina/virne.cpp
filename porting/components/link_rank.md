# Component API: `solver.rank.LinkRank`

State: **IN PROGRESS** on 2026-07-29.

Python oracle: `../virne/virne/solver/rank/link_rank.py`, commit
`d1ec1e4a20461fc9bad50612ad5026fd31e693a8`, SHA-256
`C4D52BD389A004A91D8FCC7E0827BB9620252D9619D4F9121D11BA4E17FF0880`,
2,912 bytes and 80 physical lines. All completed porting component/result
documents were read before this leaf. Frozen Graph, attribute, BaseNetwork,
CSV, YAML/config, and random implementations remain closed and unchanged.

## Scope and compatibility boundary

This independent leaf owns only deterministic ranking of the live undirected
`BaseNetwork` edge order:

- `order`: score edge position `0..E-1`, optionally stable-sort descending;
- `ffd`: sum every selected link-resource row by edge position, then
  optionally stable-sort descending.

The pinned Python `FFDLinkRank` calls
`network.get_attrs('link', 'resource')`, but pinned `BaseNetwork` exposes
`get_link_attrs(...)`; ordinary Python BaseNetwork execution therefore raises
`AttributeError` before gathering data. Node FFD uses the expected owner-specific
API, so native `ffd` implements the clear functional intent through typed
`AttributeKind::resource` selection. The original typo is retained as an
explicit differential boundary, not silently described as exact working Python
behavior.

Node ranking, solver registries/execution, candidate search, heuristics, MCF,
system orchestration, RL, Torch, CUDA, and learning code are outside this leaf.

## Stable typed API

All names are in `virne::solver::rank`.

```cpp
using LinkRankResourceId =
    network::attribute::AttributeRegistryId;
using LinkRankEdgeId = std::uint32_t;

enum class LinkRankMethod : std::uint8_t { order, ffd };

struct LinkRankSelection {
    // nullopt selects every link resource in registry order.
    std::optional<std::vector<LinkRankResourceId>> resources;
};

struct LinkRankOptions {
    bool sort = true;
    std::size_t workers = 1U;
};

struct LinkRankEntry {
    LinkRankEdgeId edge_id = 0U;
    Vertex source = 0U;
    Vertex target = 0U;
    double value = 0.0;
};

using LinkRanking = std::vector<LinkRankEntry>;

enum class LinkRankErrorCode : std::uint8_t;
enum class LinkRankOperation : std::uint8_t;
class LinkRankException : public std::runtime_error;

LinkRankMethod link_rank_method_from_string(std::string_view);
std::string_view link_rank_method_name(LinkRankMethod) noexcept;

class PreparedLinkRanker;

class LinkRanker {
public:
    explicit LinkRanker(LinkRankSelection = {});
    PreparedLinkRanker prepare(const network::BaseNetwork&) const;
};

class PreparedLinkRanker {
public:
    LinkRanking rank(
        LinkRankMethod, LinkRankOptions = {}) const;
    LinkRanking rank_order(LinkRankOptions = {}) const;
    LinkRanking rank_ffd(LinkRankOptions = {}) const;
};
```

Exact error metadata and accessors are finalized in the public header. A rank
entry carries both the stable edge ID and current ordered endpoints. Edge IDs
may contain holes, so no score vector is indexed by `num_edges()` or by raw
edge ID; the ordered result and any private work records are built in the same
`Graph::edges()` traversal.

## Fixed fields, IDs, and numeric behavior

Method, sort flag, workers, errors, operations, endpoints, edge IDs, and result
entries are direct fields/enums. A compatibility string resolves once to
`LinkRankMethod`. Default resource selection uses the typed resource kind;
explicit selections are registry IDs. Preparation validates each ID and binds
its value once to a graph-local `AttrId`. Every scoring loop uses only edge
descriptors/IDs, `AttrId`, direct `AttrMap::find(id)`, numeric variants, and
pre-sized slots. No edge, resource, sort, or worker loop hashes or compares a
string.

FFD models NumPy's resource-major matrix exactly for the supported native
numeric domain:

- bool and int rows use an int64 reduction lane with explicit modulo-`2^64`
  arithmetic, then convert each score to binary64;
- the presence of any double selects binary64, starts each column at positive
  zero, and accumulates resources in registry/selection order;
- edge order is the live frozen `Graph::edges()` order, not edge-ID order;
- `sort=false` retains that order; `sort=true` is stable descending, so equal
  scores and signed-zero ties retain edge order;
- zero edges with at least one resource returns an empty ranking; an empty
  resource selection is a typed error for FFD but remains irrelevant to order;
- missing values reproduce the completed LinkAttribute row-compaction model.
  Unequal compact row lengths are ragged; equal short rows finish numeric
  reduction before the final ranking-length error, matching Python stage
  order;
- strings, recursive values, and unsupported numeric protocols are typed
  nonnumeric boundaries. Arbitrary Python integers remain outside the frozen
  int64 graph domain.

`order` never touches resource values. With sorting enabled its unique
increasing scores permit direct reverse emission without a comparison sort.

## Ownership and parallelism

`PreparedLinkRanker` is a non-owning view of one BaseNetwork and its prepared
graph-local IDs. It may be reused while that network's graph/registry identity
is unchanged. Value and ordinary edge mutation are observed on the next call;
moving/replacing/rebinding the network or racing mutation invalidates the
prepared object. Concurrent read-only calls are supported, as are independent
rankers over independent immutable networks.

Worker zero/one is the canonical sequential route. For FFD, wider caller
widths split independent edge columns into deterministic contiguous blocks
only after sequential selection, shape, numeric-lane, and first-error
validation. Every worker accumulates that edge in the original resource order
and writes one pre-sized score slot. Final ordering is sequential and stable.
There is no host-derived automatic width or compiled benchmark-selected policy.
Order ranking stays sequential because it is already one contiguous emission.

## Acceptance gate

The compact gate covers order sorted/unsorted, empty/one/non-monotonic edges,
edge-ID holes, explicit/default/duplicate resources, bool/int64 modular and
mixed-double lanes, negatives, ties, signed zero, infinities, classified NaN
ordering, missing/ragged/short/nonnumeric rows, workers `0/1/2/8`, prepared
reuse after value/edge changes, invalidation boundaries, and concurrent
read-only callers. The AST oracle locks the source hash, class inventory,
functional FFD intent through a narrow fake, and the real BaseNetwork typo as
a separate recorded boundary.

After correctness passes, run exactly one compact FFD benchmark at configured
workers `1/2/8`, one warm-up and three samples. Preparation, fixture creation,
process startup, serialization, and checksum work stay outside timing. Gate
ordered endpoints, edge IDs, raw score bits, output bytes, and checksum before
accepting runtime. Freeze the benchmark immediately; never rerun completed
dependency benchmarks.
