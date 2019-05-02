import networkx as nx
import numpy as np
import copy


def hide_edges(graph: nx.Graph, percentage: int) -> nx.Graph:
    graph = copy.deepcopy(graph)

    edge_usages = np.zeros(graph.number_of_nodes())
    edges = []
    for u, v, w in graph.edges(data='weight', default=1):
        edge_usages[u] += 1
        edge_usages[v] += 1
        edges.append((u, v, w))
    edges = np.array(edges)

    to_hide = (percentage / 100.0) * graph.number_of_edges()
    while to_hide > 0:
        e = np.random.randint(len(edges))
        u, v, w = edges[e]
        u, v = int(u), int(v)

        if not graph.has_edge(u, v):
            continue

        if edge_usages[u] > 0 and edge_usages[v] > 0:
            edge_usages[u] -= 1
            edge_usages[v] -= 1

            graph.remove_edge(u, v)
            to_hide -= 1

    return graph
