import embeddings.embedding_base as eb
import graphs.node_mapper as mp
import networkx as nx
import typing as t
import numpy as np


class SDNE_Base(eb.EmbeddingBase):
    def __init__(self,
                 graph: nx.Graph,
                 dimension: int,
                 iterations: int,
                 batch_size: int) -> None:
        super().__init__(graph, dimension)
        self._network = None
        self._batch_size = batch_size
        self._dimension = dimension
        self._iterations = iterations
        self._adj_matrix = nx.to_scipy_sparse_matrix(graph, dtype=float)

    # region === OVERRIDES ===
    def _estimate_weights(self, src_v: np.array, tar_v: np.array) -> np.ndarray:
        src_embeddings = self._embedding[src_v, :]
        tar_embeddings = self._embedding[tar_v, :]

        return self._network.get_edge_weight(src_v, src_embeddings, tar_v, tar_embeddings)

    def save_embedding(self, mapper: mp.NodeMapper, output_file: str) -> None:
        super().save_embedding(mapper, output_file)
        self._network.save(output_file)

    def load_embedding(self, mapper: mp.NodeMapper, input_file: str) -> None:
        super().load_embedding(mapper, input_file)
        self._network.load(input_file)

    # endregion

    # region === PRIVATE ===
    def _edge_batches(self) -> t.Generator[(np.array, np.array, np.array)]:
        edges = np.array(self._graph.edges())
        batch = self._empty_batch()

        # Randomly add edges to the batch
        for u, v in np.random.permutation(edges):
            w = self._graph.edges[u, v].get("weight", 1)
            batch = self._add_to_batch(batch, u, v, w)

            # Full Batches
            if len(batch[0]) == self._batch_size:
                yield batch
                batch = self._empty_batch()

        # Last One
        yield batch

    def _print_ep_loss(self, ep: int, batch_results: t.List) -> None:
        br_nparr = np.array(batch_results)

        l1 = np.mean(br_nparr[:, 0]) / self._batch_size
        l2 = np.mean(br_nparr[:, 1]) / self._batch_size
        lr = np.mean(br_nparr[:, 2]) / self._batch_size

        print(f"#{ep}: Loss: 1st {l1:.2f} | 2nd {l2:.2f} | reg {lr:.2f}")

    def _get_batch_rows(self, batch):
        x1_batch, x2_batch, w_batch = batch
        x1_batch, x2_batch, w_batch = np.array(x1_batch), np.array(x2_batch), np.array(w_batch)

        x1_batch = self._adj_matrix[x1_batch].toarray()
        x2_batch = self._adj_matrix[x2_batch].toarray()

        return x1_batch, x2_batch, w_batch

    def _train_batch(self, batch) -> float:
        x1_batch, x2_batch, w_batch = self._get_batch_rows(batch)
        loss = self._network.train(x1_batch, x2_batch, w_batch)
        return loss

    def _empty_batch(self) -> (np.array, np.array, np.array):
        return ([], [], [])

    def _add_to_batch(self, batch, x1, x2, w) -> (np.array, np.array, np.array):
        x1b, x2b, wb = batch
        x1b.append(x1)
        x2b.append(x2)
        wb.append(w)
        return batch

    def _get_embedding(self) -> None:
        embeddings = np.empty(shape=(0, self._dimension))

        for x in range(self._graph.number_of_nodes()):
            x = self._adj_matrix[x].toarray()
            x_emb = self._network.get_embedddings(x)
            embeddings = np.append(embeddings, x_emb, axis=0)

        self._embedding = embeddings

    def _estimate_weight(self, src_v: int, tar_v: int) -> float:
        src_embedding = np.expand_dims(self._embedding[src_v], axis=0)
        tar_embedding = np.expand_dims(self._embedding[tar_v], axis=0)

        return self._network.get_edge_weight(
            np.array([src_v]),
            src_embedding,
            np.array([tar_v]),
            tar_embedding
        )[0]
    # endregion
