import networkx as nx

N = 100
ALPHA = 0.5
BETA = 0.2
SEED = 42

G = nx.waxman_graph(
    n=N,
    alpha=ALPHA,
    beta=BETA,
    seed=SEED
)

print("nodes =", G.number_of_nodes())
print("edges =", G.number_of_edges())

degrees = [d for _, d in G.degree()]

print("min_degree =", min(degrees))
print("max_degree =", max(degrees))
print("avg_degree =", sum(degrees) / len(degrees))

print("connected =", nx.is_connected(G))