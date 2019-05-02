import embeddings.embedding_base as eb
import embeddings.networks.line_network as ln
import embeddings.networks.alias_sampling as als
import networkx as nx
import numpy as np
import time


class LINE(eb.EmbeddingBase):
    def __init__(self, graph: nx.Graph,
                 dimension: int,
                 seed: int,
                 first_learning_rate: float,
                 second_learning_rate: float,
                 epochs: int,
                 batch_size: int, negative_samples: int) -> None:
        super().__init__(graph, dimension)
        self._init_sampling()
        self._epochs = epochs
        self._batch_size = batch_size
        self._negative_samples = negative_samples

        # First-Order proximity
        self._first_order_network = ln.LineNetwork(seed=seed)
        self._first_order_network.construct(
            self._graph.number_of_nodes(),
            int(np.floor(dimension / 2)),
            order=1,
            learn_rate=first_learning_rate,
            neg_sample_size=negative_samples
        )

        # Second-Order proximity
        self._second_order_network = ln.LineNetwork()
        self._second_order_network.construct(
            self._graph.number_of_nodes(),
            int(np.ceil(dimension / 2)),
            order=2,
            learn_rate=second_learning_rate,
            neg_sample_size=negative_samples
        )

    # region === OVERRIDES ===
    def learn(self) -> float:
        start = time.time()
        for i in range(self._epochs):
            self._train_epoch(i)
        end = time.time()

        self._embedding = np.concatenate((self._first_order_network.get_embedddings(), self._second_order_network.get_embedddings()), axis=1)

        self._second_order_network = None
        self._first_order_network = None

        return end - start

    def _estimate_weights(self, src_v: np.array, tar_v: np.array) -> np.ndarray:
        src_embeddings = self._embedding[src_v, :]
        src_embeddings = src_embeddings / np.linalg.norm(src_embeddings, ord=2, axis=1, keepdims=True)

        tar_embeddings = self._embedding[tar_v, :]
        tar_embeddings = tar_embeddings / np.linalg.norm(tar_embeddings, ord=2, axis=1, keepdims=True)

        return np.matmul(src_embeddings, np.transpose(tar_embeddings))

    # endregion

    # region === PRIVATE ===
    def _init_sampling(self) -> None:
        node_degrees = [self._graph.degree(v) for v in self._graph.nodes]
        node_degrees = np.array(node_degrees)
        self._node_sampling = als.AliasSampling(np.array(node_degrees), np.array(self._graph.nodes))

        edge_weights = [[u, v, w] for u, v, w in self._graph.edges.data("weight", default=1)]
        edge_weights = np.array(edge_weights)
        self._edge_sampling = als.AliasSampling(edge_weights[:, 2], edge_weights[:, :2])

    def _train_epoch(self, ep: int) -> None:
        edges = self._edge_sampling.sample(self._batch_size)
        neg_samples = [self._node_sampling.sample(self._negative_samples) for _ in edges]

        loss_1 = self._first_order_network.train(edges[:, 0], edges[:, 1], np.array(neg_samples))
        loss_2 = self._second_order_network.train(edges[:, 0], edges[:, 1], np.array(neg_samples))

        print(f"#{ep} Loss: First-Order {loss_1:2f} Second-Order {loss_2:2f}")
    # endregion
