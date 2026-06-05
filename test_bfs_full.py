import time
import networkx as nx

G = nx.Graph()

with open(
    "graph_dump_cpp.txt",
    "r",
    encoding="utf8",
) as f:

    for line in f:

        parts = line.split()

        if not parts:
            continue

        if parts[0] != "EDGE":
            continue

        u = int(parts[1])
        v = int(parts[2])

        G.add_edge(
            u,
            v,
            weight=1.0)

print(
    "NODES",
    G.number_of_nodes())

print(
    "EDGES",
    G.number_of_edges())

#
# DEGREE
#

t0 = time.perf_counter()

degree = nx.degree_centrality(
    G)

degree_ms = (
    time.perf_counter()
    - t0
) * 1000.0

#
# EIGEN
#

t0 = time.perf_counter()

eigen = nx.eigenvector_centrality(
    G,
    max_iter=10000,
    tol=1e-6)

eigen_ms = (
    time.perf_counter()
    - t0
) * 1000.0

#
# CLOSE
#

t0 = time.perf_counter()

close = nx.closeness_centrality(
    G)

close_ms = (
    time.perf_counter()
    - t0
) * 1000.0

#
# BETWEEN
#

t0 = time.perf_counter()

between = nx.betweenness_centrality(
    G,
    weight="weight")

between_ms = (
    time.perf_counter()
    - t0
) * 1000.0

checksum = (
    sum(degree.values())
    + sum(eigen.values())
    + sum(close.values())
    + sum(between.values())
)

print(
    "CHECKSUM",
    checksum)

print(
    "\n===== PERFORMANCE =====")

print(
    f"DEGREE(ms)   "
    f"{degree_ms:.6f}")

print(
    f"EIGEN(ms)    "
    f"{eigen_ms:.6f}")

print(
    f"CLOSE(ms)    "
    f"{close_ms:.6f}")

print(
    f"BETWEEN(ms)  "
    f"{between_ms:.6f}")