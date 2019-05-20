# Skip-Gram architecture
# Modified version of the https://www.tensorflow.org/tutorials/representation/word2vec tutorial

import tensorflow as tf
import numpy as np
import math


class SkipGram:
    def __init__(self, threads: int = 4, seed: int = 42) -> None:
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        config = tf.ConfigProto(inter_op_parallelism_threads=threads,
                                intra_op_parallelism_threads=threads)
        config.gpu_options.allow_growth = True
        self._session = tf.Session(graph=graph, config=config)

    # region === PUBLIC ===
    def construct(self, num_nodes: int, embedding_size: int, learning_rate: float, sample_size: int) -> None:
        with self._session.graph.as_default():
            self._embeddings = tf.Variable(tf.random_uniform([num_nodes, embedding_size], -1.0, 1.0))

            # NCE weights and biases
            nce_weights = tf.Variable(tf.truncated_normal([num_nodes, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([num_nodes]))

            # Placeholders for inputs
            self._train_inputs = tf.placeholder(tf.int32, shape=[None])
            self._train_labels = tf.placeholder(tf.int32, shape=[None])
            self._train_labels_exp = tf.expand_dims(self._train_labels, 1)

            # Get embedded inputs
            embed = tf.nn.embedding_lookup(self._embeddings, self._train_inputs)

            # Compute the NCE loss, using a sample of the negative labels each time.
            self._loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                               biases=nce_biases,
                               labels=self._train_labels_exp,
                               inputs=embed,
                               num_sampled=sample_size,
                               num_classes=num_nodes))

            # We use the SGD optimizer, as Adam / RMS Prop do not achieve good results on skip-gram models
            global_step = tf.train.create_global_step()
            self._training = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self._loss, global_step=global_step)

            # Initialize variables
            self._session.run(tf.global_variables_initializer())

    def train(self, inputs: np.array, labels: np.array) -> float:
        _, loss = self._session.run([self._training, self._loss], feed_dict={self._train_inputs: inputs, self._train_labels: labels})
        return loss

    def get_embedddings(self) -> np.array:
        return np.array(self._embeddings.eval(self._session))
    # endregion
