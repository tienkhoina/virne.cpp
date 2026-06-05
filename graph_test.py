import time
import networkx as nx
import numpy as np

INPUT_FILE = "graph_dump_cpp.txt"
OUTPUT_FILE = "graph_dump_py.txt"

#
# LOAD GRAPH FROM CPP DUMP
#

G = nx.Graph()

current_node = None
current_edge = None

with open(
    INPUT_FILE,
    "r",
    encoding="utf8",
) as f:

    lines = [
        line.strip()
        for line in f
        if line.strip()
    ]

for line in lines:

    parts = line.split()

    tag = parts[0]

    if tag == "NODE":

        current_node = int(
            parts[1])

        G.add_node(
            current_node)

        continue

    if tag == "CPU":

        if current_node is not None:

            G.nodes[
                current_node
            ]["cpu"] = int(
                parts[1])

        continue

    if tag == "GPU":

        if current_node is not None:

            G.nodes[
                current_node
            ]["gpu"] = int(
                parts[1])

        continue

    if tag == "EDGE":

        u = int(parts[1])
        v = int(parts[2])

        G.add_edge(
            u,
            v)

        current_edge = (
            u,
            v)

        continue

    if tag == "EDGE_ID":

        if current_edge is not None:

            G.edges[
                current_edge
            ]["edge_id"] = int(
                parts[1])

        continue

    if tag == "WEIGHT":

        if current_edge is not None:

            G.edges[
                current_edge
            ]["weight"] = float(
                parts[1])

        continue

    if tag == "BW":

        if current_edge is not None:

            G.edges[
                current_edge
            ]["bw"] = float(
                parts[1])

        continue

print(
    "===== GRAPH =====")

print(
    "NODES",
    G.number_of_nodes())

print(
    "EDGES",
    G.number_of_edges())

#
# ATTRIBUTES
#

node_cpu = nx.get_node_attributes(
    G,
    "cpu")

node_gpu = nx.get_node_attributes(
    G,
    "gpu")

edge_weight = nx.get_edge_attributes(
    G,
    "weight")

edge_bw = nx.get_edge_attributes(
    G,
    "bw")

#
# MATRICES
#

import time
import networkx as nx

def benchmark_scan(G):
    s = 0

    t0 = time.perf_counter()

    for _ in range(G.number_of_nodes()):
        for u in G:
            for v in G[u]:
                s += v

    return (
        time.perf_counter()
        - t0
    ) * 1000.0


def benchmark_bfs_nopred(G):

    n = G.number_of_nodes()

    t0 = time.perf_counter()

    for source in G:

        seen = {source}
        queue = [source]

        head = 0

        while head < len(queue):

            u = queue[head]
            head += 1

            for v in G[u]:

                if v in seen:
                    continue

                seen.add(v)
                queue.append(v)

    return (
        time.perf_counter()
        - t0
    ) * 1000.0


print(
    f"SCAN_FULL(ms) "
    f"{benchmark_scan(G):.6f}"
)

print(
    f"BFS_NOPRED(ms) "
    f"{benchmark_bfs_nopred(G):.6f}"
)

import time
import networkx as nx

t0 = time.perf_counter()

for source in G:
    nx.single_source_shortest_path_length(
        G,
        source)



print(
    "SSSP_FULL(ms)",
    (
        time.perf_counter()
        - t0
    ) * 1000.0
)

t0 = time.perf_counter()

nx.closeness_centrality(
    G)

print(
    "CLOSE(ms)",
    (
        time.perf_counter()
        - t0
    ) * 1000.0
)

t0 = time.perf_counter()

total = 0

for source in G:
    d = nx.single_source_shortest_path_length(
        G,
        source)

    total += len(d)
    total += sum(d.values())

print(f"TOTAL: {total}")

print(
    (time.perf_counter()-t0)*1000
)

print(
    "\n===== MATRICES =====")

t0 = time.perf_counter()

A = nx.adjacency_matrix(
    G).tocoo()

adj_ms = (
    time.perf_counter()
    - t0
) * 1000.0

print(
    "ADJ NNZ",
    A.nnz)

t0 = time.perf_counter()

W, _ = nx.attr_sparse_matrix(
    G,
    edge_attr="weight")

W = W.tocoo()

attr_ms = (
    time.perf_counter()
    - t0
) * 1000.0

print(
    "WEIGHT NNZ",
    W.nnz)

#
# CENTRALITY
#

print(
    "\n===== CENTRALITY =====")

t0 = time.perf_counter()

degree = nx.degree_centrality(
    G)

degree_ms = (
    time.perf_counter()
    - t0
) * 1000.0

t0 = time.perf_counter()

eigen = nx.eigenvector_centrality(
    G,
    max_iter=10000,
    tol=1e-6)

eigen_ms = (
    time.perf_counter()
    - t0
) * 1000.0

t0 = time.perf_counter()

for v in G.nodes():
    nx.single_source_shortest_path_length(
        G,
        v)

bfs_full_ms = (
    time.perf_counter()
    - t0
) * 1000.0

print(
    f"BFS_FULL(ms)  "
    f"{bfs_full_ms:.12f}")



close = nx.closeness_centrality(
    G)

close_ms = (
    time.perf_counter()
    - t0
) * 1000.0

t0 = time.perf_counter()

between = nx.betweenness_centrality(
    G,
    weight="weight")

between_ms = (
    time.perf_counter()
    - t0
) * 1000.0

#
# DUMP PY RESULTS
#

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf8",
) as out:

    out.write(
        f"NUM_NODES {G.number_of_nodes()}\n")

    out.write(
        f"NUM_EDGES {G.number_of_edges()}\n")

    #
    # NODES
    #

    for n in sorted(
        G.nodes()):

        out.write(
            f"NODE {n}\n")

        out.write(
            f"CPU "
            f"{node_cpu[n]}\n")

        out.write(
            f"GPU "
            f"{node_gpu[n]}\n")

    #
    # EDGES
    #

    for u, v, d in G.edges(
        data=True):

        out.write(
            f"EDGE {u} {v}\n")

        out.write(
            f"EDGE_ID "
            f"{d['edge_id']}\n")

        out.write(
            f"WEIGHT "
            f"{d['weight']:.12f}\n")

        out.write(
            f"BW "
            f"{d['bw']:.12f}\n")

    #
    # ADJ MATRIX
    #

    out.write(
        f"ADJ_NNZ "
        f"{A.nnz}\n")

    adj_entries = sorted(
        zip(
            A.row,
            A.col,
            A.data))

    for r, c, val in adj_entries:

        out.write(
            f"ADJ "
            f"{r} "
            f"{c} "
            f"{float(val):.12f}\n")

    #
    # WEIGHT MATRIX
    #

    out.write(
        f"WEIGHT_NNZ "
        f"{W.nnz}\n")

    weight_entries = sorted(
        zip(
            W.row,
            W.col,
            W.data))

    for r, c, val in weight_entries:

        out.write(
            f"W "
            f"{r} "
            f"{c} "
            f"{float(val):.12f}\n")

    #
    # CENTRALITY
    #

    for v in sorted(
        G.nodes()):

        out.write(
            f"DEGREE "
            f"{v} "
            f"{degree[v]:.12f}\n")

        out.write(
            f"EIGEN "
            f"{v} "
            f"{eigen[v]:.12f}\n")

        out.write(
            f"CLOSE "
            f"{v} "
            f"{close[v]:.12f}\n")

        out.write(
            f"BETWEEN "
            f"{v} "
            f"{between[v]:.12f}\n")

print(
    "\n===== TOP CENTRALITY =====")

top_degree = sorted(
    degree.items(),
    key=lambda x: x[1],
    reverse=True)

top_eigen = sorted(
    eigen.items(),
    key=lambda x: x[1],
    reverse=True)

top_close = sorted(
    close.items(),
    key=lambda x: x[1],
    reverse=True)

top_between = sorted(
    between.items(),
    key=lambda x: x[1],
    reverse=True)

print(
    "\nTOP DEGREE")

for v, score in top_degree[:5]:

    print(
        v,
        score)

print(
    "\nTOP EIGEN")

for v, score in top_eigen[:5]:

    print(
        v,
        score)

print(
    "\nTOP CLOSE")

for v, score in top_close[:5]:

    print(
        v,
        score)

print(
    "\nTOP BETWEEN")

for v, score in top_between[:5]:

    print(
        v,
        score)

print(
    "\n===== PERFORMANCE =====")

print(
    f"ADJ(ms)      "
    f"{adj_ms:.12f}")

print(
    f"ATTR(ms)     "
    f"{attr_ms:.12f}")

print(
    f"DEGREE(ms)   "
    f"{degree_ms:.12f}")

print(
    f"EIGEN(ms)    "
    f"{eigen_ms:.12f}")

print(
    f"CLOSE(ms)    "
    f"{close_ms:.12f}")

print(
    f"BETWEEN(ms)  "
    f"{between_ms:.12f}")

print(
    "\nOUTPUT:",
    OUTPUT_FILE)

print(
    "\nCOMPARE:")

print(
    "diff graph_dump_cpp.txt graph_dump_py.txt")

