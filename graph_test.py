
import math
import time
import networkx as nx


def fmt(x):
    if math.isinf(x):
        return "inf"
    return f"{x:.12f}"


def bench(fn):
    start = time.perf_counter()

    result = fn()

    end = time.perf_counter()

    return result, (end - start) * 1000.0


def path_cost(G, path):
    cost = 0.0

    for u, v in zip(
        path[:-1],
        path[1:]):
        cost += G[u][v]["weight"]

    return cost


def print_path(path):
    print(
        " ".join(
            str(v)
            for v in path))


def load_graph(path):
    G = nx.Graph()

    source = None
    target = None

    with open(
        path,
        "r",
        encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            tag = parts[0]

            if tag == "NODES":

                n = int(parts[1])

                G.add_nodes_from(
                    range(n))

            elif tag == "SOURCE":

                source = int(
                    parts[1])

            elif tag == "TARGET":

                target = int(
                    parts[1])

            elif tag == "EDGE":

                u = int(parts[1])
                v = int(parts[2])

                w = float(
                    parts[3])

                G.add_edge(
                    u,
                    v,
                    weight=w)

    return (
        G,
        source,
        target)


G, source, target = load_graph(
    "graph_dump.txt")

print("===== GRAPH =====")

print(
    "nodes=",
    G.number_of_nodes())

print(
    "edges=",
    G.number_of_edges())

print()

print("===== EDGE LIST =====")

for u, v, data in G.edges(data=True):

    print(
        u,
        v,
        fmt(
            data["weight"]))

print()

#
# BFS
#

print("===== BFS =====")

bfs_path, bfs_ms = bench(
    lambda:
    nx.shortest_path(
        G,
        source,
        target))

bfs_len = nx.shortest_path_length(
    G,
    source,
    target)

print(
    "shortest_path_length",
    bfs_len)

print(
    "shortest_path",
    end=" ")

print_path(
    bfs_path)

print()

dist = nx.single_source_shortest_path_length(
    G,
    source)

print(
    "single_source_shortest_path_length")

for v in sorted(dist):

    print(
        v,
        dist[v])

print()

#
# Dijkstra
#

print("===== DIJKSTRA =====")

dijkstra_path, dijkstra_ms = bench(
    lambda:
    nx.dijkstra_path(
        G,
        source,
        target,
        weight="weight"))

dijkstra_cost = nx.dijkstra_path_length(
    G,
    source,
    target,
    weight="weight")

print(
    "dijkstra_path_length",
    fmt(
        dijkstra_cost))

print(
    "dijkstra_path",
    end=" ")

print_path(
    dijkstra_path)

print()

dist, sssp_ms = bench(
    lambda:
    nx.single_source_dijkstra_path_length(
        G,
        source,
        weight="weight"))

print(
    "single_source_dijkstra_path_length")

for v in sorted(dist):

    print(
        v,
        fmt(
            dist[v]))

print()

#
# Floyd
#

print("===== FLOYD WARSHALL =====")

fw, fw_ms = bench(
    lambda:
    nx.floyd_warshall(
        G,
        weight="weight"))

for i in range(
    G.number_of_nodes()):

    row = []

    for j in range(
        G.number_of_nodes()):

        row.append(
            fmt(
                fw[i][j]))

    print(
        " ".join(row))

print()

#
# Yen
#

print("===== YEN =====")

def run_yen():

    paths = []

    for path in nx.shortest_simple_paths(
        G,
        source,
        target,
        weight="weight"):

        paths.append(
            path)

        if len(paths) >= 10:
            break

    return paths

paths, yen_ms = bench(
    run_yen)

print(
    "count",
    len(paths))

last_cost = -1.0

for i, path in enumerate(
    paths):

    cost = path_cost(
        G,
        path)

    assert cost >= last_cost

    last_cost = cost

    print(
        "path",
        i,
        end=" ")

    print_path(
        path)

    print(
        "cost",
        fmt(
            cost))

print()

#
# Performance
#

print("===== PERFORMANCE =====")

print(
    "BFS(ms)",
    fmt(
        bfs_ms))

print(
    "DIJKSTRA(ms)",
    fmt(
        dijkstra_ms))

print(
    "SSSP(ms)",
    fmt(
        sssp_ms))

print(
    "FLOYD(ms)",
    fmt(
        fw_ms))

print(
    "YEN(ms)",
    fmt(
        yen_ms))

print()
print("ALL PASS")

