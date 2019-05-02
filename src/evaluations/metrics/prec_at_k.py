import reconstruction.direct_reconstr as dr
import embeddings.embedding_base as eb
import networkx as nx
import numpy as np


def precision_at_k(graph: nx.Graph, learning_graph: nx.Graph, embedding: eb.EmbeddingBase, max_k: int) -> (np.array, np.array):
    precisions = []

    top_k_edges = dr.get_best_new_edges(graph, learning_graph, embedding, max_k)
    correct_edges = 0

    for k, (w, (u, v)) in enumerate(top_k_edges):
        correct_edges += float(graph.has_edge(u, v)) * 100.0
        prak = correct_edges / (k + 1.0)
        precisions.append(round(prak, 5))

    return {
        "PRECISIONS": precisions
    }
