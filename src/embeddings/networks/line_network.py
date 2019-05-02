import tensorflow as tf
import numpy as np


class LineNetwork:
    def __init__(self, threads=4, seed=42) -> None:
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self._session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                      intra_op_parallelism_threads=threads))

    # region === PUBLIC ===
    def construct(self, num_nodes: int, embedding_size: int, order: int, learn_rate: float, neg_sample_size: int = None) -> None:
        with self._session.graph.as_default():
            self._source_embeddings = tf.Variable(tf.random_uniform([num_nodes, embedding_size], -1.0, 1.0))

            # Placeholders for inputs
            self._src_edges = tf.placeholder(tf.int32, shape=[None])
            self._tar_edges = tf.placeholder(tf.int32, shape=[None])
            self._nes_edges = tf.placeholder(tf.int32, shape=[None, neg_sample_size])

            # Source embeddings
            src_embed = tf.nn.embedding_lookup(self._source_embeddings, self._src_edges)

            # Target embeddings
            if order == 1:
                # Same embedding as source
                tar_embed = tf.nn.embedding_lookup(self._source_embeddings, self._tar_edges)
                nes_embed = tf.nn.embedding_lookup(self._source_embeddings, self._nes_edges)
            else:
                # Separate target embeddings for second-order
                self._target_embeddings = tf.Variable(tf.random_uniform([num_nodes, embedding_size], -1.0, 1.0))
                tar_embed = tf.nn.embedding_lookup(self._target_embeddings, self._tar_edges)
                nes_embed = tf.nn.embedding_lookup(self._target_embeddings, self._nes_edges)

            # Loss
            negative = []
            for i in range(neg_sample_size):
                ns_dot = tf.multiply(nes_embed[:, i], src_embed)
                ns_dot = tf.reduce_sum(ns_dot, axis=1)
                ns_sigma = tf.log_sigmoid(-ns_dot)
                negative.append(ns_sigma)
            negative = tf.reduce_sum(negative, axis=0)

            positive = tf.multiply(tar_embed, src_embed)
            positive = tf.reduce_sum(positive, axis=1)
            self._loss = -tf.reduce_mean(tf.log_sigmoid(positive) + negative)

            # We use the SGD optimizer.
            global_step = tf.train.create_global_step()
            self._training = tf.train.AdamOptimizer(learn_rate).minimize(self._loss, global_step=global_step)

            # Initialize variables
            self._session.run(tf.global_variables_initializer())

    def train(self, source_edges, target_edges, negative_samples) -> float:
        _, loss = self._session.run([self._training, self._loss],
                                    feed_dict={self._src_edges: source_edges, self._tar_edges: target_edges, self._nes_edges: negative_samples})
        return loss

    def get_embedddings(self) -> np.array:
        return self._source_embeddings.eval(self._session)

    # endregion
