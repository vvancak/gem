import networkx as nx
import math


def global_norm(graph: nx.Graph) -> nx.Graph:
    max_w = max(d.get('weight', 1) for u, v, d in graph.edges.data())

    for u, v, d in graph.edges.data():
        d['weight'] /= max_w

    return graph


def log_norm(graph: nx.Graph) -> nx.Graph:
    for u, v, d in graph.edges.data():
        w = d.get('weight', 1)
        d['weight'] = math.log2(w + 1)

    return graph


def log_global_norm(graph: nx.Graph) -> nx.Graph:
    graph = log_norm(graph)
    return global_norm(graph)
