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

    def _process_batch(self, src_v: np.array, tar_v: np.array, learning_graph: nx.Graph = None):
        if len(tar_v) == 0 or len(src_v) == 0: return

        weights = self._embedding.estimate_weights(src_v, tar_v)
        for i, u in enumerate(src_v):
            for j, v in enumerate(tar_v):

                # Avoid processing edges twice
                if u <= v: continue

                # Do not predict already observed edges in link prediction
                if learning_graph is not None:
                    if learning_graph.has_edge(u, v): continue
                    if learning_graph.has_edge(v, u): continue

                w = weights[i][j]
                self._heap.add((u, v), w)

    def get_best_edges(self, src_v: int, batch_size: int) -> np.array:
        batch = []
        for tar_v in self._graph.nodes:
            if src_v <= tar_v: continue

            batch.append(tar_v)
            if len(batch) == batch_size:
                self._process_batch(np.array([src_v]), np.array(batch))
                batch = []

        # Last Batch
        self._process_batch(np.array([src_v]), np.array(batch))
        batch = []

        return self._heap.get_sorted()

    def get_best_new_edges(self, learning_graph: nx.Graph, batch_size: int):
        src_v_batch = []
        for src_v in self._graph.nodes:

            # Ensure we have enough src_v nodes
            src_v_batch.append(src_v)
            if len(src_v_batch) < batch_size:
                continue

            # Process tar_v nodes
            tar_v_batch = []
            for tar_v in self._graph.nodes:

                # Process only half of the matrix, other half is symmetrical
                if tar_v < src_v_batch[0]: continue

                # Ensure we have enough tar_v nodes
                tar_v_batch.append(tar_v)
                if len(tar_v_batch) < batch_size:
                    continue

                # Process [src_v X tar_v matrix]
                self._process_batch(np.array(src_v_batch), np.array(tar_v_batch), learning_graph)
                tar_v_batch = []

            # Reset src_v batch for next round
            src_v_batch = []

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
