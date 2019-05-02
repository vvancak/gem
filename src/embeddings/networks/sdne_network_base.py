from abc import abstractmethod
import tensorflow as tf
import numpy as np
import typing as t


class SdneNetwork:
    def __init__(self, seed: int = 42, threads: int = 4) -> None:
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self._session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                      intra_op_parallelism_threads=threads))

    # region === ABSTRACT ===
    @abstractmethod
    def _get_encoder(self) -> t.Callable[[tf.Tensor, bool], tf.Tensor]:
        pass

    @abstractmethod
    def _get_decoder(self) -> t.Callable[[tf.Tensor, bool], tf.Tensor]:
        pass

    @abstractmethod
    def _get_reg_loss(self) -> t.Union[tf.Tensor, np.array]:
        pass

    # endregion

    # region === PUBLIC ===
    @property
    def session(self):
        return self._session

    def construct(self,
                  embed_dim: int,
                  num_nodes: int,
                  learning_rate: float,
                  alpha: float,
                  beta: float,
                  optimizer: t.Type[tf.train.Optimizer]):
        with self._session.graph.as_default():
            # === TRAINING ===
            encoder = self._get_encoder()
            decoder = self._get_decoder()

            # Placeholders
            self._x1 = tf.placeholder(tf.float32, shape=[None, num_nodes])
            self._x2 = tf.placeholder(tf.float32, shape=[None, num_nodes])
            self._w = tf.placeholder(tf.float32, shape=[None, 1])

            # Embeddings
            x1_embed = encoder(self._x1, True)
            x2_embed = encoder(self._x2, True)

            # Outputs
            self._x1_hat = decoder(x1_embed, True)
            self._x2_hat = decoder(x2_embed, True)

            # l1 loss - laplacian eigenmaps loss between embeddings
            self._l1_loss = self._loss_1st(x1_embed, x2_embed, self._w, alpha)

            # l2 loss - direct autoencoder graph_reconstruction loss
            l2_loss1 = self._loss_2nd(self._x1, self._x1_hat, beta)
            l2_loss2 = self._loss_2nd(self._x2, self._x2_hat, beta)
            self._l2_loss = l2_loss1 + l2_loss2

            # lr loss - regularization loss
            self._lr_loss = self._get_reg_loss()
            self._lr_loss = tf.reduce_mean(self._lr_loss)

            # Training
            self._loss = self._l1_loss + self._l2_loss + self._lr_loss
            global_step = tf.train.create_global_step()
            self._training = optimizer(learning_rate).minimize(self._loss, global_step=global_step)

            # === EVALUATION ===
            # Embeddings
            self._x = tf.placeholder(tf.float32, shape=[None, num_nodes])
            self._x_emb = encoder(self._x, False)

            # Weight prediction
            self._embedding = tf.placeholder(tf.float32, shape=[None, embed_dim])
            self._x_pred = decoder(self._embedding, False)

            self._saver = tf.train.Saver()

            # Initialize variables
            self._session.run(tf.global_variables_initializer())

    def get_embedddings(self, x: np.array) -> np.array:
        return self._session.run([self._x_emb], feed_dict={self._x: x})[0]

    def get_edge_weight(self, x1: np.array, x1_emb: np.array, x2: np.ndarray, x2_emb: np.ndarray) -> np.array:
        pred_adj_x1 = self._session.run([self._x_pred], feed_dict={self._embedding: x1_emb})[0]
        pred_adj_x2 = self._session.run([self._x_pred], feed_dict={self._embedding: x2_emb})[0]

        x1_weights = pred_adj_x1[:, x2]
        x2_weights = pred_adj_x2[:, x1]

        return (x1_weights + np.transpose(x2_weights)) / 2

    def save(self, path) -> None:
        return self._saver.save(self._session, f"{path}.w")

    def load(self, path) -> None:
        return self._saver.restore(self._session, f"{path}.w")

    def train(self, x1: np.array, x2: np.array, w: np.array) -> (float, float, float):
        w = np.expand_dims(w, axis=-1)
        _, l1, l2, lr, x1h = self._session.run([self._training, self._l1_loss, self._l2_loss, self._lr_loss, self._x1_hat],
                                               feed_dict={self._x1: x1, self._x2: x2, self._w: w})
        return l1, l2 / 2.0, lr

    # endregion

    # reigon === PRIVATE ===
    def _loss_1st(self, x1_embed: tf.Tensor, x2_embed: tf.Tensor, w: tf.Tensor, alpha: float) -> float:
        loss_1st = tf.norm(x2_embed - x1_embed, axis=1) ** 2
        loss_1st = tf.multiply(loss_1st, w)
        return alpha * tf.reduce_sum(loss_1st)

    def _loss_2nd(self, x: tf.Tensor, x_hat: tf.Tensor, beta: float) -> float:
        b = tf.where(x == 0, tf.zeros_like(x, dtype=tf.float32), tf.ones_like(x, dtype=tf.float32) * beta)
        l = tf.norm(tf.multiply(x_hat - x, b), axis=1)
        return tf.reduce_sum(l)
    # endregion
