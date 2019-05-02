import reconstruction.direct_reconstr as dr
import normalization.eval_ns_norm as evn
import embeddings.embedding_base as eb
import networkx as nx
import numpy as np


class EdgeNsReconstruction():
    def __init__(self, graph: nx.Graph, embedding: eb.EmbeddingBase):
        self._embedding = embedding
        self._graph = graph

    def _get_embedding_edges(self, edges: np.array, ns: np.array):
        ns_embed_value = evn.compute_ns_embedding(self._embedding, ns)
        for i in range(len(edges)):
            w = edges[i][0]
            edges[i][0] = np.exp(w) / (np.exp(w) + ns_embed_value)

        return edges

    def _get_graph_edges(self, edges: np.array, ns: np.array):
        ns_graph_value = evn.compute_ns_graph(self._graph, ns)

        weights = []
        for (_, (u, v)) in edges:
            if (self._graph.has_edge(u, v)):
                weight = self._graph.edges[u, v].get('weight', 1)
                weight = np.exp(weight) / (np.exp(weight) + ns_graph_value)
                weights.append((weight, (u, v)))
            else:
                weights.append((0, (u, v)))
        return np.array(weights)

    def get_best_edges_ns(self, src_v: int, max_k: int, ns_size: int, batch_size: int):
        edges = dr.get_best_edges(self._graph, self._embedding, src_v, max_k, batch_size)
        ns = evn.get_negative_sample(self._graph, np.array([src_v]), ns_size)

        # Compute negative_sampled edge weights (for embedding and graph - based evaluations)
        return self._get_embedding_edges(edges, ns), self._get_graph_edges(edges, ns)

    def get_best_new_edges_ns(self, learning_graph: nx.Graph, max_k: int, ns_size: int, batch_size: int):
        edges = dr.get_best_new_edges(self._graph, learning_graph, self._embedding, max_k, batch_size)
        ns = evn.get_negative_sample(self._graph, np.array([]), ns_size)

        return self._get_embedding_edges(edges, ns), self._get_graph_edges(edges, ns)


def get_top_edges_ns(graph: nx.Graph, embedding: eb.EmbeddingBase, src_v: int, max_k: int, ns_size: int, batch_size: int = 32):
    return EdgeNsReconstruction(graph, embedding).get_best_edges_ns(src_v, max_k, ns_size, batch_size)


def get_top_new_edges_ns(graph: nx.Graph, learning_graph: nx.Graph, embedding: eb.EmbeddingBase, max_k: int, ns_size: int, batch_size: int = 32):
    return EdgeNsReconstruction(graph, embedding).get_best_new_edges_ns(learning_graph, max_k, ns_size, batch_size)
