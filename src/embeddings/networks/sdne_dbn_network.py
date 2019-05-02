import embeddings.networks.sdne_network_base as sdnb
import tfrbm.tfrbm.gbrbm as gbrbm
import tensorflow as tf
import typing as t
import numpy as np


class SdneDeepBeliefNetwork(sdnb.SdneNetwork):
    def __init__(self,
                 seed: int,
                 embed_dim: int,
                 num_nodes: int,
                 gamma: float):
        super().__init__(seed)

        self._embed_dim = embed_dim
        self._num_nodes = num_nodes
        self._gamma = gamma

    # region === OVERRIDES ===
    def _get_encoder(self) -> t.Callable[[tf.Tensor, bool], tf.Tensor]:
        return self._dbn2nn(self._rbms_encoder)

    def _get_decoder(self) -> t.Callable[[tf.Tensor, bool], tf.Tensor]:
        return self._dbn2nn(self._rbms_decoder)

    def _get_reg_loss(self) -> t.Union[tf.Tensor, np.array]:
        return tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # endregion

    # region === PUBLIC ===
    def create_dbn_autoencoder(self, num_nodes: int, embed_dim: int, layers: t.List) -> None:
        encoder_rbms = self._create_belief_network(num_nodes, embed_dim, layers)
        decoder_rbms = self._create_belief_network(embed_dim, num_nodes, reversed(layers))

        self._rbms_encoder = encoder_rbms
        self._rbms_decoder = decoder_rbms

        self._rbms = encoder_rbms + decoder_rbms

    def pre_train(self, batch: np.array, layer_idx: int):
        layer = self._rbms[layer_idx]
        layer.partial_fit(batch)
        return layer.get_err(batch)

    def get_representation(self, batch: np.array, layer_idx: int):
        for i in range(layer_idx):
            batch = self._rbms[i].transform(batch)
        return batch

    # endregion

    # region === PRIVATE ===
    def _create_belief_network(self, init_dim: int, last_dim: int, layers: t.Iterable) -> t.List:
        rbms = []
        for dim in layers:
            r = gbrbm.GBRBM(init_dim, dim)
            init_dim = dim
            rbms.append(r)

        r = gbrbm.GBRBM(init_dim, last_dim)
        rbms.append(r)

        return rbms

    def _dbn2nn(self, dbn: t.Iterable) -> t.Callable[[tf.Tensor, bool], tf.Tensor]:
        layers = []
        for rbm in dbn:
            # Layer weights
            weights, visible_bias, hidden_bias = rbm.get_weights()
            kinit = tf.constant_initializer(weights)
            binit = tf.constant_initializer(hidden_bias)
            kreg = tf.contrib.layers.l2_regularizer(scale=self._gamma)

            # Create nn layer
            layer = tf.layers.Dense(rbm.n_hidden, activation=tf.nn.sigmoid,
                                    kernel_initializer=kinit,
                                    bias_initializer=binit,
                                    kernel_regularizer=kreg
                                    )
            layers.append(layer)

        def model(x: tf.Tensor, training: bool):
            for layer in layers:
                x = layer(x)
            return x

        return model

    # endregion
