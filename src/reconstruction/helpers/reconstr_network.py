import tensorflow as tf
import numpy as np
import math


class ReconstrNetwork:
    def __init__(self, threads: int = 4, seed: int = 42) -> None:
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self._session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                      intra_op_parallelism_threads=threads))

    # region === PUBLIC ===
    def construct(self, embed_dim: int, learning_rate: float) -> None:
        with self._session.graph.as_default():
            self._x1 = tf.placeholder(tf.float32, shape=[None, embed_dim])
            self._x2 = tf.placeholder(tf.float32, shape=[None, embed_dim])
            self._weights = tf.placeholder(tf.float32, shape=[None])

            net = tf.concat((self._x1, self._x2), axis=1)

            # TODO: Parametrize this, suitable for our networks only !!
            net = tf.keras.layers.Dense(embed_dim, activation=tf.nn.softplus)(net)

            self._predictions = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(net)
            self._predictions = tf.reduce_sum(self._predictions, axis=1)

            self._loss = tf.losses.mean_squared_error(self._weights, self._predictions)

            global_step = tf.train.create_global_step()
            self._training = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self._loss, global_step=global_step)

            # Initialize variables
            self._session.run(tf.global_variables_initializer())

    def train(self, x1_embed: np.array, x2_embed: np.array, weights: np.array) -> float:
        _, loss = self._session.run([self._training, self._loss], feed_dict={self._x1: x1_embed, self._x2: x2_embed, self._weights: weights})
        return loss

    def predict(self, x1_embed: np.array, x2_embed: np.array):
        return self._session.run([self._predictions], feed_dict={self._x1: x1_embed, self._x2: x2_embed})[0]

# endregion
