import embeddings.networks.sdne_bd_network as sbd
import embeddings.methods.sdne_base as sdb
import tensorflow as tf
import networkx as nx
import typing as t
import time


class SDNE_BD(sdb.SDNE_Base):
    def __init__(self,
                 graph: nx.Graph,
                 dimension: int,
                 seed: int,
                 iterations: int,
                 learning_rate: float,
                 alpha: float,
                 beta: float,
                 gamma: float,
                 batch_size: int,
                 layers: t.List,
                 dropouts: t.List) -> None:
        super().__init__(graph, dimension, iterations, batch_size)
        self._network = sbd.SdneBatchNormDropoutNetwork(
            seed,
            dimension,
            layers,
            dropouts,
            graph.number_of_nodes(),
            gamma
        )
        self._network.construct(dimension, graph.number_of_nodes(), learning_rate, alpha, beta, tf.train.AdamOptimizer)

    # region === OVERRIDES ===
    def learn(self) -> float:
        start = time.time()

        # Epochs
        for i in range(self._iterations):
            batch_results = []

            # Batches
            for batch in self._edge_batches():
                br = self._train_batch(batch)
                batch_results.append(br)

            # Loss printing
            self._print_ep_loss(i, batch_results)

        end = time.time()

        self._get_embedding()
        return end - start
    # endregion
