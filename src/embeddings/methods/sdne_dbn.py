import embeddings.networks.sdne_dbn_network as sdbn
import embeddings.methods.sdne_base as sdb
import tensorflow as tf
import networkx as nx
import typing as t
import time


class SDNE_DBN(sdb.SDNE_Base):
    def __init__(self,
                 graph: nx.Graph,
                 dimension: int,
                 seed: int,
                 rbm_iterations: int,
                 sgd_iterations: int,
                 learning_rate: float,
                 alpha: float,
                 beta: float,
                 gamma: float,
                 batch_size: int,
                 layers: t.List) -> None:
        super().__init__(graph, dimension, sgd_iterations, batch_size)
        self._layers = layers

        self._learning_rate = learning_rate
        self._alpha = alpha
        self._beta = beta

        self._rbm_iterations = rbm_iterations
        self._sgd_iterations = sgd_iterations

        self._network = sdbn.SdneDeepBeliefNetwork(
            seed, dimension, graph.number_of_nodes(), gamma
        )

    # region === OVERRIDES ===
    def learn(self) -> float:
        time = 0.0

        # Pre-Training DBN
        self._network.create_dbn_autoencoder(self._graph.number_of_nodes(), self._dimension, self._layers)
        time += self._pre_train()

        # Training SGD
        self._network.construct(self._dimension,
                                self._graph.number_of_nodes(),
                                self._learning_rate,
                                self._alpha,
                                self._beta,
                                tf.train.GradientDescentOptimizer)
        time += self._sgd_train()

        self._get_embedding()
        return time

    # endregion

    # region === PRIVATE ===
    def _sgd_train(self) -> float:
        start = time.time()

        # Epochs
        for i in range(self._sgd_iterations):
            batch_results = []

            # Batches
            for batch in self._edge_batches():
                br = self._train_batch(batch)
                batch_results.append(br)

            # Loss printing
            self._print_ep_loss(i, batch_results)

        return time.time() - start

    def _pre_train(self):
        start = time.time()
        # [encoder]-embedding-[decoder]-output
        layers = 2 * len(self._layers) + 2

        for i in range(layers):
            for ep in range(self._rbm_iterations):
                ep_loss = 0.0
                for batch in self._edge_batches():
                    batch_transformed = self._network.get_representation(batch[0], i)

                    loss = self._network.pre_train(batch_transformed, i)
                    ep_loss += loss

                print(f"Pre-Train layer #{i} ep loss {ep_loss:.3f}")
        return time.time() - start
