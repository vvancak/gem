import reconstruction.direct_reconstr as dr
import embeddings.embedding_base as eb
import networkx as nx
import numpy as np
import typing as t


def precision_at_k(src_v: int, max_k: int, graph: nx.Graph, embedding: eb.EmbeddingBase) -> (np.array, np.array):
    precisions = []
    recall_deltas = []

    top_k_edges = dr.get_best_edges(graph, embedding, src_v, max_k)

    correct_edges = 0
    for k, (w, (u, v)) in enumerate(top_k_edges):
        edge_observed = float(graph.has_edge(u, v))

        correct_edges += edge_observed
        precisions.append(correct_edges / (k + 1.0))
        recall_deltas.append(edge_observed)

    return np.array(precisions), np.array(recall_deltas)


def average_precision(src_v: int, max_k: int, graph: nx.Graph, embedding: eb.EmbeddingBase) -> float:
    precisions, recall_deltas = precision_at_k(src_v, max_k, graph, embedding)
    up = np.multiply(precisions, recall_deltas)
    up = np.sum(up)
    down = np.sum(recall_deltas)
    return up / down if down > 0 else 0


def mean_avg_prec(graph: nx.Graph, embedding: eb.EmbeddingBase, max_k: int) -> t.Dict:
    avp = []
    for src_v in range(graph.number_of_nodes()):
        ap = average_precision(src_v, max_k, graph, embedding)
        avp.append(ap)

    avp = np.array(avp)
    return {"MAP": round(np.mean(avp) * 100, 3)}
