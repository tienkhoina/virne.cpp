# K-shortest/Yen integration optimization - 2026-07-30

## API

No public Graph, `SearchMask`, `yen_k_shortest_paths`,
`ShortestSimplePathGenerator`, or `nx::shortest_simple_paths` signature changed.
Path cost, insertion tie order, base-mask semantics, and eager/lazy output remain
the frozen contract.

## Performance design

Unweighted Yen searches now keep one private dense workspace per eager call or
lazy generator. Forward/reverse visit state and root-node/edge bans use
generation-stamped arrays indexed by `Vertex` and stable edge ID. BFS fringes
retain capacity between spur searches.

The removed path allocated a complete `SearchMask`, scanned every node and live
edge, and allocated fresh BFS state for every spur. Reset cost is now
proportional to the current root and blocked-edge set; traversal remains
proportional to the actually explored graph. A caller mask is read directly.
Weighted Yen keeps the existing ID-resolved bidirectional-Dijkstra path.

There are no strings or endpoint maps in the traversal loop. The optional
weight name is still resolved once at the public boundary. Workspaces are not
shared, so concurrent callers retain independent deterministic state.

## Verification

`graph_k_shortest_workspace_test` covers directed masked tie order, repeated
spur reset, lazy/eager parity, independent enumerators, removed edge-ID holes,
undirected traversal, and the weighted fallback. The focused Release/Werror
build and test pass. Frozen Graph benchmarks and accepted artifacts were not
rerun or modified.
