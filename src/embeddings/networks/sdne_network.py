from abc import abstractmethod
import tensorflow as tf
import numpy as np
import typing as t


class SdneNetwork:
    def __init__(self,
                 seed: int,
                 embed_dim: int,
                 layers: t.List,
                 dropouts: t.List,
                 num_nodes: int,
                 gamma: float,
                 threads: int = 4):
        # Create an empty graph and a session
        self._seed = seed
        self._embed_dim = embed_dim
        self._layers = np.array(layers)
        self._dropouts = np.array(dropouts)
        self._num_nodes = num_nodes
        self._gamma = gamma

        graph = tf.Graph()
        graph.seed = seed
        config = tf.ConfigProto(inter_op_parallelism_threads=threads,
                                intra_op_parallelism_threads=threads)
        config.gpu_options.allow_growth = True
        self._session = tf.Session(graph=graph, config=config)

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
    # Losses
    def _loss_1st(self, x1_embed: tf.Tensor, x2_embed: tf.Tensor, w: tf.Tensor, alpha: float) -> float:
        loss_1st = tf.norm(x2_embed - x1_embed, axis=1) ** 2
        loss_1st = tf.multiply(loss_1st, w)
        return alpha * tf.reduce_sum(loss_1st)

    def _loss_2nd(self, x: tf.Tensor, x_hat: tf.Tensor, beta: float) -> float:
        b = tf.where(x == 0, tf.zeros_like(x, dtype=tf.float32), tf.ones_like(x, dtype=tf.float32) * beta)
        l = tf.norm(tf.multiply(x_hat - x, b), axis=1)
        return tf.reduce_sum(l)

    def _get_reg_loss(self) -> t.Union[tf.Tensor, np.array]:
        return tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # Network structure
    def _get_encoder(self) -> t.Callable[[tf.Tensor, bool], tf.Tensor]:
        return self._get_layers(self._embed_dim, None, self._dropouts, self._layers, self._gamma)

    def _get_decoder(self) -> t.Callable[[tf.Tensor, bool], tf.Tensor]:
        return self._get_layers(self._num_nodes, None, np.zeros_like(self._layers), reversed(self._layers), self._gamma)

    def _get_layers(self,
                    final_dim: int,
                    final_activation: t.Union[str, None],
                    dropouts: t.Iterable,
                    layers: t.Iterable,
                    gamma: float) -> t.Callable[[tf.Tensor, bool], tf.Tensor]:
        net_layers = []

        # Initializers and Regularizers
        kinit = tf.contrib.layers.xavier_initializer(seed=self._seed)
        kreg = tf.contrib.layers.l2_regularizer(scale=gamma)

        # Construct network layer-wise
        for dim, drp in zip(layers, dropouts):
            dense = tf.layers.Dense(dim, activation=tf.nn.relu, kernel_initializer=kinit, kernel_regularizer=kreg)
            batchnorm = tf.layers.BatchNormalization()
            activation = tf.keras.layers.ReLU()
            if drp > 0:
                dropout = tf.keras.layers.Dropout(rate=drp)
            else:
                dropout = lambda x, training: x

            # Append to model
            net_layers.append((dense, batchnorm, activation, dropout))

        final_layer = tf.layers.Dense(final_dim,
                                      activation=final_activation,
                                      kernel_initializer=kinit,
                                      kernel_regularizer=kreg)

        # Final model function
        def model(x: tf.Tensor, training: bool) -> tf.Tensor:
            self._is_training = training
            for dense, batchnorm, activation, dropout in net_layers:
                x = dense(x)
                x = batchnorm(x)
                x = activation(x)
                x = dropout(x, training=training)

            x = final_layer(x)
            return x

        return model

    # endregion
