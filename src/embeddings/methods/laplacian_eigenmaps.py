import embeddings.embedding_base as eb
import scipy.sparse.linalg as slg
import networkx as nx
import numpy as np
import time


class LaplacianEigenmaps(eb.EmbeddingBase):
    def __init__(self, graph: nx.Graph, dimension: int) -> None:
        super().__init__(nx.to_undirected(graph), dimension)
        self._dimension = dimension

    # region === OVERRIDES ===
    def learn(self) -> float:
        start = time.time()

        # Eigenvalues and eigenvectors for the generalized eigenvector problem Ly = λDy
        # Remove scaling by normalizing the laplacian matrix - i.e. D becomes I
        L = nx.normalized_laplacian_matrix(self._graph)
        eigenvals, eigenvectors = slg.eigs(L, self._dimension + 1, which="SM")

        # Determine the zero eigenvalue and remove its eigenvector. Resulting rows are embeddings
        zero_eigenval_idx = np.argmin(np.abs(eigenvals))
        self._embedding = np.delete(eigenvectors, zero_eigenval_idx, axis=1)
        self._embedding = np.real(self._embedding)

        return time.time() - start

    def _estimate_weights(self, src_v: np.array, tar_v: np.array) -> np.ndarray:
        src_embeddings = self._embedding[src_v, :]
        src_embeddings = np.expand_dims(src_embeddings, axis=1)
        src_embeddings = np.repeat(src_embeddings, len(tar_v), axis=1)

        tar_embeddings = self._embedding[tar_v, :]
        tar_embeddings = np.expand_dims(tar_embeddings, axis=0)
        tar_embeddings = np.repeat(tar_embeddings, len(src_v), axis=0)

        return np.linalg.norm(src_embeddings - tar_embeddings, axis=2)

    # endregion
