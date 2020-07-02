import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Softmax, ReLU, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import binary_crossentropy

import modules.util as util
import modules.backbones as backbones
from modules.hyperparameters import *


class Prototypical(tf.keras.Model):
    """
    Reference: https://github.com/schatty/prototypical-networks-tf/blob/master/prototf/models/prototypical.py
    Implemenation of Prototypical Network.
    """

    def __init__(self,
                 # few shot parameters
                 # n_support, n_query,

                 # inputs parameters
                 w, h, c,

                 # Model parameters
                 backbone: tf.keras.Model,

                 # SL bosster
                 sl_classifier: tf.keras.Model = None,

                 # others
                 **kwargs):
        """
        Args:
            n_support (int): number of support examples.
            n_query (int): number of query examples.
            w (int): image width .
            h (int): image height.
            c (int): number of channels.
            backbone (tf.keras.Model): the encoder model as backbone.
        """
        super(Prototypical, self).__init__()
        self.w, self.h, self.c = w, h, c
        self.backbone = backbone

        self.sl_classifier = sl_classifier

    def call(self, support, query, sl_args: list = None):
        if self.sl_classifier is None:
            return self.call_proto(support, query)
        else:
            assert sl_args is not None

            return self.call_proto_sl(support, query, sl_args)

    def call_proto(self, support, query):
        n_class = support.shape[0]
        n_support = support.shape[1]
        n_query = query.shape[1]
        y = np.tile(np.arange(n_class)[:, np.newaxis], (1, n_query))
        y_onehot = tf.cast(tf.one_hot(y, n_class), tf.float32)

        # correct indices of support samples (just natural order)
        target_inds = tf.reshape(tf.range(n_class), [n_class, 1])
        target_inds = tf.tile(target_inds, [1, n_query])

        # merge support and query to forward through encoder
        cat = tf.concat([
            tf.reshape(support, [n_class * n_support,
                                 self.w, self.h, self.c]),
            tf.reshape(query, [n_class * n_query,
                               self.w, self.h, self.c])], axis=0)
        z = self.backbone(cat)

        # Divide embedding into support and query
        z_prototypes = tf.reshape(z[:n_class * n_support],
                                  [n_class, n_support, z.shape[-1]])
        # Prototypes are means of n_support examples
        z_prototypes = tf.math.reduce_mean(z_prototypes, axis=1)
        z_query = z[n_class * n_support:]

        # Calculate distances between query and prototypes
        dists = util.calc_euclidian_dists(z_query, z_prototypes)

        # log softmax of calculated distances
        log_p_y = tf.nn.log_softmax(-dists, axis=-1)
        log_p_y = tf.reshape(log_p_y, [n_class, n_query, -1])

        loss_few = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_onehot, log_p_y), axis=-1), [-1]))
        eq = tf.cast(
            tf.equal(
                tf.cast(tf.argmax(log_p_y, axis=-1), tf.int32),
                tf.cast(y, tf.int32)
            ), tf.float32
        )
        acc_few = tf.reduce_mean(eq)

        return loss_few, acc_few

    def call_proto_sl(self, support, query, sl_args: list):
        [sl_x, sl_y, sl_test_x, sl_test_y] = sl_args
        sl_embed_x, sl_embed_test_x = self.backbone(sl_x), self.backbone(sl_test_x)
        sl_y_pred, sl_test_y_pred = self.sl_classifier(sl_embed_x), self.sl_classifier(sl_embed_test_x)

        loss_sl = binary_crossentropy(sl_y, sl_y_pred) + binary_crossentropy(sl_test_y, sl_test_y_pred)

        sl_eq = tf.cast(
            tf.equal(
                tf.cast(tf.argmax(sl_y_pred, axis=-1), tf.int32),
                tf.cast(tf.argmax(sl_y, axis=-1), tf.int32)
            ), tf.float32
        )

        sl_eq_test = sl_eq = tf.cast(
            tf.equal(
                tf.cast(tf.argmax(sl_test_y_pred, axis=-1), tf.int32),
                tf.cast(tf.argmax(sl_test_y, axis=-1), tf.int32)
            ), tf.float32
        )

        sl_eq_t = tf.concat([sl_eq, sl_eq_test], axis=0)

        acc_sl = tf.reduce_mean(sl_eq_t)

        loss_few, acc_few = self.call_proto(support, query)
        return loss_few, acc_few, loss_sl, acc_sl