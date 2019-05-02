import embeddings.embedding_base as eb
import networkx as nx
import numpy as np


class Random(eb.EmbeddingBase):
    def __init__(self, graph: nx.Graph, dimension: int, seed: int) -> None:
        super().__init__(nx.to_undirected(graph), dimension)
        self._dimension = dimension

    # region === OVERRIDES ===
    def learn(self) -> float:
        self._embedding = np.random.rand(self._graph.number_of_nodes(), self._dimension) * 2 - 1.0
        return 0

    def _estimate_weights(self, src_v: np.array, tar_v: np.array) -> np.ndarray:
        src_embeddings = self._embedding[src_v, :]
        src_embeddings = np.expand_dims(src_embeddings, axis=1)
        src_embeddings = np.repeat(src_embeddings, len(tar_v), axis=1)

        tar_embeddings = self._embedding[tar_v, :]
        tar_embeddings = np.expand_dims(tar_embeddings, axis=0)
        tar_embeddings = np.repeat(tar_embeddings, len(src_v), axis=0)

        return np.linalg.norm(src_embeddings - tar_embeddings, axis=2)
    # endregion
