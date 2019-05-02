import embeddings.networks.alias_sampling as als
import embeddings.networks.skip_gram as sg
import embeddings.embedding_base as eb
import networkx as nx
import typing as t
import numpy as np
import time


class Node2Vec(eb.EmbeddingBase):
    def __init__(self, graph: nx.Graph,
                 dimension: int,
                 seed: int,
                 q: float,
                 p: float,
                 walk_length: int,
                 walks_per_node: int,
                 learning_rate: float,
                 negative_samples: int,
                 batch_size: int) -> None:
        super().__init__(graph, dimension)

        self._p = p
        self._q = q
        self._walk_length = walk_length
        self._walks_per_node = walks_per_node

        self._batch_size = batch_size
        self._negative_samples = negative_samples
        self._learning_rate = learning_rate
        self._skip_gram = sg.SkipGram(seed=seed)
        self._skip_gram.construct(len(self._graph.nodes), dimension, learning_rate, negative_samples)

        self._precompute_edges()
        self._precompute_nodes()

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
    def _precompute_edges(self):
        edge_probs = {}

        for u, v, w in self._graph.edges.data("weight", default=1):
            # === DIRECTION U -> V ===
            next_nodes = []
            next_weights = []
            for t, d in self._graph[v].items():
                w = d.get('weight', 1)
                next_nodes.append(t)

                # BFS - DFS node2vec sampling strategy
                if t == u:
                    prob = 1.0 / self._p
                elif self._graph.has_edge(t, u):
                    prob = 1.0
                else:
                    prob = 1.0 / self._q

                # Append to unnormalized probs
                next_weights.append(prob * w)

            # Generate alias sampling for edge
            next_weights = np.array(next_weights)
            if len(next_weights > 0):
                edge_probs[(u, v)] = als.AliasSampling(next_weights / sum(next_weights), np.array(next_nodes))

            # === DIRECTION V -> U ===
            next_nodes = []
            next_weights = []
            for t, d in self._graph[u].items():
                w = d.get('weight', 1)
                next_nodes.append(t)

                # BFS - DFS node2vec sampling strategy
                if t == v:
                    prob = 1.0 / self._p
                elif self._graph.has_edge(t, v):
                    prob = 1.0
                else:
                    prob = 1.0 / self._q

                # Append to unnormalized probs
                next_weights.append(prob * w)

            # Generate alias sampling for edge
            next_weights = np.array(next_weights)
            if len(next_weights > 0):
                edge_probs[(u, v)] = als.AliasSampling(next_weights / sum(next_weights), np.array(next_nodes))

        # Store the sampled results
        self._edge_probs = edge_probs

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

    def _sample_neighbour(self, current_node, prev_node) -> np.array:
        if prev_node is None:
            sampling = self._node_probs.get(current_node, None)
        else:
            sampling = self._edge_probs.get((prev_node, current_node), None)

        if sampling is None:
            return None
        else:
            return sampling.sample(1)[0]

    def _sample_walk(self, node) -> np.array:
        walk = [None, node]
        curr_node = node
        for _ in range(self._walk_length):
            # sample neighbourhood
            successor = self._sample_neighbour(curr_node, walk[-2])

            # No more neighbours
            if successor is None:
                return np.array(walk[1:])

            # append to the current random walk and continue
            walk.append(successor)
            curr_node = successor

        return np.array(walk[1:])

    def _train_skip_gram(self, walks):
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
