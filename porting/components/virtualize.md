# Component boundary: `virne.utils.virtualize`

State: **DEFERRED / VISUALIZATION-ONLY** on 2026-07-29.

Python source: `../virne/virne/utils/virtualize.py`, commit
`d1ec1e4a20461fc9bad50612ad5026fd31e693a8`, SHA-256
`3CF3913FFA590D4F2258E32FD7835CB3A88F79052B599EA839751DAFD9255E31`,
2,731 bytes.

The file contains only Matplotlib/NetworkX interactive visualization:
`draw_graph` computes a spring layout, draws, optionally shows and saves a
figure; `virtualize` constructs a random Waxman demo, repeatedly redraws
statistics, physical nodes and path requests, and pauses the GUI. It ignores
its `p_net` argument and calls `virtualize()` unconditionally at import time.

There is no graph calculation, simulation state, resource mutation, CSV/YAML,
solver, or benchmarkable core API to port. Pulling a GUI/plotting backend into
`vne_utils` would violate the dependency and performance boundary. Native
`virtualize.h/.cpp` therefore remain empty and are not compiled; no placeholder
or no-op behavior is claimed.

If visualization is requested later, implement it as an optional external
adapter consuming only const completed Graph/network views. Layout choice,
render backend, random seed, show/save policy and animation timing must be typed
cold options; they must not enter graph or Environment hot paths.
