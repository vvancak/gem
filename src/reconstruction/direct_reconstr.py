import reconstruction.helpers.edge_heap as ehpq
import embeddings.embedding_base as eb
import scipy.sparse as sp
import networkx as nx
import typing as t
import numpy as np


class EdgeReconstruction():
    def __init__(self, graph: nx.graph, embedding: eb.EmbeddingBase, max_k: int):
        self._heap = ehpq.Heap(max_k)
        self._embedding = embedding
        self._graph = graph

    def _process_batch(self, batch: np.array, src_v: int):
        if len(batch) == 0: return

        weights = self._embedding.estimate_weights(np.array([src_v]), batch)
        for i, tar_v in enumerate(batch):
            w = weights[0][i]
            self._heap.add((src_v, tar_v), w)

    def get_best_edges(self, src_v: int, batch_size: int) -> np.array:
        batch = []
        for tar_v in self._graph.nodes:
            if tar_v <= src_v: continue

            batch.append(tar_v)
            if len(batch) == batch_size:
                self._process_batch(np.array(batch), src_v)
                batch = []
        self._process_batch(np.array(batch), src_v)

        return self._heap.get_sorted()

    def get_best_new_edges(self, learning_graph: nx.Graph, batch_size: int):
        batch = []
        for src_v in self._graph.nodes:
            for tar_v in self._graph.nodes:
                if tar_v <= src_v: continue
                if learning_graph.has_edge(src_v, tar_v): continue

                batch.append(tar_v)
                if len(batch) == batch_size:
                    self._process_batch(np.array(batch), src_v)
                    batch = []

            self._process_batch(np.array(batch), src_v)
            batch = []

        return self._heap.get_sorted()

    def reconstruct_adj_matrix(self, batch_size: int) -> sp.coo_matrix:
        edges = np.array([])
        for src_v in self._graph.nodes:
            srcve = self.get_best_edges(src_v, batch_size)
            edges = np.insert(edges, srcve)

        rows = []
        columns = []
        weights = []

        for (w, (u, v)) in edges:
            rows.append(u)
            columns.append(v)
            weights.append(w)

        return sp.coo_matrix(
            (np.array(weights), (np.array(rows), np.array(columns))),
            shape=(self._graph.number_of_nodes(), self._graph.number_of_nodes())
        )


def get_best_edges(graph: nx.Graph, embedding: eb.EmbeddingBase, src_v: int, max_k: int, batch_size: int = 32) -> np.array:
    return EdgeReconstruction(graph, embedding, max_k).get_best_edges(src_v, batch_size)


def get_best_new_edges(graph: nx.Graph, learning_graph: nx.Graph, embedding: eb.EmbeddingBase, max_k: int, batch_size: int = 32) -> np.array:
    return EdgeReconstruction(graph, embedding, max_k).get_best_new_edges(learning_graph, batch_size)


def get_adj_matrix(graph: nx.Graph, embedding: eb.EmbeddingBase, max_k: int, batch_size: int = 32) -> sp.coo_matrix:
    return EdgeReconstruction(graph, embedding, max_k).reconstruct_adj_matrix(batch_size)
