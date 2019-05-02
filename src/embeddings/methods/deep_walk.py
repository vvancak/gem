import embeddings.networks.alias_sampling as als
import embeddings.embedding_base as eb
import embeddings.networks.skip_gram as sg
import networkx as nx
import typing as t
import numpy as np
import time


class DeepWalk(eb.EmbeddingBase):
    def __init__(self, graph: nx.Graph,
                 dimension: int,
                 seed: int,
                 weighted_var: bool,
                 walk_length: int,
                 walks_per_node: int,
                 learning_rate: float,
                 negative_samples: int,
                 batch_size: int) -> None:
        super().__init__(graph, dimension)

        self._walk_length = walk_length
        self._walks_per_node = walks_per_node
        self._weighted_var = weighted_var

        self._batch_size = batch_size
        self._negative_samples = negative_samples
        self._learning_rate = learning_rate

        self._precompute_nodes()
        self._skip_gram = sg.SkipGram(seed=seed)
        self._skip_gram.construct(len(self._graph.nodes), dimension, learning_rate, negative_samples)

    # region === OVERRIDES ===
    def learn(self) -> float:
        start = time.time()
        for t in range(self._walks_per_node):
            loss = self._train()
            print(f"Walk #{t + 1}: loss:{loss:2f}")

        t = time.time() - start
        self._embedding = self._skip_gram.get_embedddings()
        self._skip_gram = None

        return t

    def _estimate_weights(self, src_v: np.array, tar_v: np.array) -> np.ndarray:
        src_embeddings = self._embedding[src_v, :]
        src_embeddings = src_embeddings / np.linalg.norm(src_embeddings, ord=2, axis=1, keepdims=True)

        tar_embeddings = self._embedding[tar_v, :]
        tar_embeddings = tar_embeddings / np.linalg.norm(tar_embeddings, ord=2, axis=1, keepdims=True)

        return np.matmul(src_embeddings, np.transpose(tar_embeddings))

    # endregion

    # region === PRIVATE ===
    def _precompute_nodes(self):
        node_probs = {}

        for u in self._graph.nodes:
            nbr_nodes = []
            nbr_weights = []
            for v, d in self._graph[u].items():
                nbr_nodes.append(v)
                nbr_weights.append(d.get('weight', 1))

            nbr_weights = np.array(nbr_weights)
            if len(nbr_weights > 0):
                node_probs[u] = als.AliasSampling(nbr_weights / sum(nbr_weights), np.array(nbr_nodes))

        self._node_probs = node_probs

    def _sample_neighbour(self, node) -> np.array:
        sampling = self._node_probs.get(node, None)

        if sampling is None:
            return None
        else:
            return sampling.sample(1)[0]

    def _sample_walk(self, node) -> np.array:
        walk = [node]
        curr_node = node
        for _ in range(self._walk_length):
            # sample neighbourhood
            successor = self._sample_neighbour(curr_node)

            # No more neighbours
            if successor is None:
                return np.array(walk)

            # append to the current random walk and continue
            walk.append(successor)
            curr_node = successor
        return np.array(walk)

    def _train_skip_gram(self, walks) -> float:
        walks = np.array(walks)
        num_walks = np.shape(walks)[0]

        batch_walks = []
        for i in range(num_walks):
            src_node = walks[i][0]
            for j in walks[i][1:]:
                batch_walks.append([src_node, j])

        batch_walks = np.array(batch_walks)
        return self._skip_gram.train(batch_walks[:, 0], batch_walks[:, 1])

    def _train(self) -> t.Union[float, np.array]:
        walks = []
        losses = []
        random_node_seq = np.random.permutation(self._graph.nodes)
        for node in random_node_seq:
            walk = self._sample_walk(node)
            walks.append(walk)

            if len(walks) == self._batch_size:
                loss = self._train_skip_gram(np.array(walks))
                losses.append(loss)
                walks = []

        # Final batch - fill with more random walks
        if len(walks) > 0:
            while len(walks) < self._batch_size:
                random_node = np.random.choice(self._graph.nodes)
                walk = self._sample_walk(random_node)
                walks.append(walk)

            loss = self._train_skip_gram(np.array(walks))
            losses.append(loss)

        return np.mean(losses)
    # endregion
