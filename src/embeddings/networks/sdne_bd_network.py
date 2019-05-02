import embeddings.networks.sdne_network_base as sdnb
import tensorflow as tf
import typing as t
import numpy as np


class SdneBatchNormDropoutNetwork(sdnb.SdneNetwork):
    def __init__(self,
                 seed: int,
                 embed_dim: int,
                 layers: t.List,
                 dropouts: t.List,
                 num_nodes: int,
                 gamma: float):
        super().__init__(seed)
        self._seed = seed
        self._embed_dim = embed_dim
        self._layers = layers
        self._dropouts = dropouts
        self._num_nodes = num_nodes
        self._gamma = gamma

    # region === OVERRIDES ===
    def _get_encoder(self) -> t.Callable[[tf.Tensor, bool], tf.Tensor]:
        return self._get_layers(self._embed_dim, tf.nn.tanh, self._dropouts, self._layers, self._gamma)

    def _get_decoder(self) -> t.Callable[[tf.Tensor, bool], tf.Tensor]:
        return self._get_layers(self._num_nodes, tf.nn.sigmoid, reversed(self._dropouts), reversed(self._layers), self._gamma)

    def _get_reg_loss(self) -> t.Union[tf.Tensor, np.array]:
        return tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # endregion

    # region === PRIVATE ===
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
            dropout = tf.keras.layers.Dropout(rate=drp)

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
