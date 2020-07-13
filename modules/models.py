import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Softmax, ReLU, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy
from copy import deepcopy

import modules.util as util
import modules.backbones as backbones
from modules.hyperparameters import *


class FewShot(tf.keras.Model):
    def __init__(self,
                 # few shot parameters
                 # n_support, n_query,

                 # inputs parameters
                 w, h, c,

                 # Model parameters
                 backbone: tf.keras.Model,

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
        super(FewShot, self).__init__()
        self.w, self.h, self.c = w, h, c
        self.backbone = backbone

    def set_support(self, support):
        raise NotImplementedError

    def apply_query(self, query):
        raise NotImplementedError


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
        self.alpha = kwargs.get("alpha", 1.0)

        self.possible_rotations = [0, 90, 180, 270]
        self.possible_rotations_to_one_hot = {
            rot: util.c_idx2one_hot(idx, np.zeros(len(self.possible_rotations), dtype=int))
            for idx, rot in enumerate(self.possible_rotations)
        }
        self.possible_k = range(4)

        self.z_prototypes = None
        self.n_class = None
        self.n_support = None

        self.sl_x = None
        self.sl_y = None

    def call(self,  _query, training=None, mask=None):
        loss_few, acc_few = self.apply_query(_query)

        if self.sl_classifier is None:
            return loss_few, acc_few
        else:
            loss_sl = self.call_proto_sl(self.sl_x, self.sl_y, *self.get_sl_set_args(tf.identity(_query)))
            return loss_few + self.alpha*loss_sl, acc_few

    def set_support(self, support):
        self.n_class = support.shape[0]
        self.n_support = support.shape[1]
        support_reshape = tf.reshape(support, [self.n_class * self.n_support,
                                               self.w, self.h, self.c])
        z = self.backbone(support_reshape)

        self.z_prototypes = tf.reduce_mean(
            tf.reshape(z, [self.n_class, self.n_support, z.shape[-1]])
            , axis=1
        )

        if self.sl_classifier is not None:
            self.sl_x, self.sl_y = self.get_sl_set_args(tf.identity(support))

    def apply_query(self, query):
        n_query = query.shape[1]
        y = np.tile(np.arange(self.n_class)[:, np.newaxis], (1, n_query))
        y_onehot = tf.cast(tf.one_hot(y, self.n_class), tf.float32)

        z_query = self.backbone(tf.reshape(query, [self.n_class * n_query,
                                                   self.w, self.h, self.c]))

        # Calculate distances between query and prototypes
        dists = util.calc_euclidian_dists(z_query, self.z_prototypes)

        # log softmax of calculated distances
        log_p_y = tf.nn.log_softmax(-dists, axis=-1)
        log_p_y = tf.reshape(log_p_y, [self.n_class, n_query, -1])

        loss_few = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_onehot, tf.cast(log_p_y, tf.float32)), axis=-1), [-1]))
        eq = tf.cast(
            tf.equal(
                tf.cast(tf.argmax(log_p_y, axis=-1), tf.int32),
                tf.cast(y, tf.int32)
            ), tf.float32
        )
        acc_few = tf.reduce_mean(eq)

        return loss_few, acc_few

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

    def call_proto_sl(self, *sl_args):
        [sl_x, sl_y, sl_test_x, sl_test_y] = sl_args

        sl_y_pred = self.sl_classifier(self.backbone(sl_x))
        sl_test_y_pred = self.sl_classifier(self.backbone(sl_test_x))

        lb0 = categorical_crossentropy(sl_y, sl_y_pred)
        # del sl_y
        # del sl_y_pred

        lb1 = categorical_crossentropy(sl_test_y, sl_test_y_pred)
        # del sl_test_y
        # del sl_test_y_pred

        loss_sl = (tf.reduce_mean(lb0) + tf.reduce_mean(lb1)) / 2

        # sl_eq = tf.cast(
        #     tf.equal(
        #         tf.cast(tf.argmax(sl_y_pred, axis=-1), tf.int32),
        #         tf.cast(tf.argmax(sl_y, axis=-1), tf.int32)
        #     ), tf.float32
        # )
        #
        # sl_eq_test = sl_eq = tf.cast(
        #     tf.equal(
        #         tf.cast(tf.argmax(sl_test_y_pred, axis=-1), tf.int32),
        #         tf.cast(tf.argmax(sl_test_y, axis=-1), tf.int32)
        #     ), tf.float32
        # )
        #
        # sl_eq_t = tf.concat([sl_eq, sl_eq_test], axis=0)
        #
        # acc_sl = tf.reduce_mean(sl_eq_t)

        return loss_sl

    def get_sl_args(self, support, query):
        _n_way, _n_shot, _w, _h, _c = support.shape
        _n_way, _n_query, _w, _h, _c = query.shape

        support_reshape = tf.reshape(support, shape=[_n_way * _n_shot, _w, _h, _c])
        query_reshape = tf.reshape(query, shape=[_n_way * _n_query, _w, _h, _c])

        sl_y_r = np.random.choice(self.possible_k, support_reshape.shape[0])
        sl_x = tf.map_fn(lambda _i: tf.image.rot90(support_reshape[_i], sl_y_r[_i]),
                         tf.range(support_reshape.shape[0]), dtype=tf.float32)

        sl_y = tf.cast(tf.one_hot(sl_y_r, len(self.possible_k)), tf.int16)

        sl_test_y_r = np.random.choice(self.possible_k, query_reshape.shape[0])
        sl_test_x = tf.map_fn(lambda _i: tf.image.rot90(query_reshape[_i], sl_test_y_r[_i]),
                              tf.range(query_reshape.shape[0]), dtype=tf.float32)

        sl_test_y = tf.cast(tf.one_hot(sl_test_y_r, len(self.possible_k)), tf.int16)

        return [sl_x, sl_y, sl_test_x, sl_test_y]

    def get_sl_set_args(self, _set):
        _n_way, _n_shot, _w, _h, _c = _set.shape

        _set_reshape = tf.reshape(_set, shape=[_n_way * _n_shot, _w, _h, _c])

        sl_y_r = np.random.choice(self.possible_k, _set_reshape.shape[0])
        sl_x = tf.map_fn(lambda _i: tf.image.rot90(_set_reshape[_i], sl_y_r[_i]),
                         tf.range(_set_reshape.shape[0]), dtype=tf.float32)

        sl_y = tf.cast(tf.one_hot(sl_y_r, len(self.possible_k)), tf.int16)

        return sl_x, sl_y


class CosineClassifier(tf.keras.Model):
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
        super(CosineClassifier, self).__init__()
        self.w, self.h, self.c = w, h, c
        self.backbone = backbone

        self.sl_classifier = sl_classifier
        self.alpha = kwargs.get("alpha", 1.0)

        self.possible_rotations = [0, 90, 180, 270]
        self.possible_rotations_to_one_hot = {
            rot: util.c_idx2one_hot(idx, np.zeros(len(self.possible_rotations), dtype=int))
            for idx, rot in enumerate(self.possible_rotations)
        }
        self.possible_k = range(4)

        # self.weight_base = tf.Tensor()
        # self.weight_novel = tf.Tensor()
        self.n_class = None
        self.n_support = None

        self.sl_x = None
        self.sl_y = None

    def call(self,  _query, training=None, mask=None):
        # loss_few, acc_few = self.call_proto(support, query)
        loss_few, acc_few = self.apply_query(_query)

        if self.sl_classifier is None:
            return loss_few, acc_few
        else:
            loss_sl = self.call_proto_sl(self.sl_x, self.sl_y, *self.get_sl_set_args(_query))
            return loss_few + self.alpha*loss_sl, acc_few

    def set_support(self, support):
        self.n_class = support.shape[0]
        self.n_support = support.shape[1]
        support_reshape = tf.reshape(support, [self.n_class * self.n_support,
                                               self.w, self.h, self.c])
        z = self.backbone(support_reshape)

        self.z_prototypes = tf.reduce_mean(
            tf.reshape(z, [self.n_class, self.n_support, z.shape[-1]])
            , axis=1
        )

        if self.sl_classifier is not None:
            self.sl_x, self.sl_y = self.get_sl_set_args(support)

    def apply_query(self, query):
        n_query = query.shape[1]
        y = np.tile(np.arange(self.n_class)[:, np.newaxis], (1, n_query))
        y_onehot = tf.cast(tf.one_hot(y, self.n_class), tf.float32)

        z_query = self.backbone(tf.reshape(query, [self.n_class * n_query,
                                                   self.w, self.h, self.c]))

        # Calculate distances between query and prototypes
        dists = util.calc_euclidian_dists(z_query, self.z_prototypes)

        # log softmax of calculated distances
        log_p_y = tf.nn.log_softmax(-dists, axis=-1)
        log_p_y = tf.reshape(log_p_y, [self.n_class, n_query, -1])

        loss_few = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_onehot, log_p_y), axis=-1), [-1]))
        eq = tf.cast(
            tf.equal(
                tf.cast(tf.argmax(log_p_y, axis=-1), tf.int32),
                tf.cast(y, tf.int32)
            ), tf.float32
        )
        acc_few = tf.reduce_mean(eq)

        return loss_few, acc_few

    def get_sl_set_args(self, _set):
        _n_way, _n_shot, _w, _h, _c = _set.shape

        _set_reshape = tf.reshape(_set, shape=[_n_way * _n_shot, _w, _h, _c])

        sl_y_r = np.random.choice(self.possible_k, _set_reshape.shape[0])
        sl_x = tf.map_fn(lambda _i: tf.image.rot90(_set_reshape[_i], sl_y_r[_i]),
                         tf.range(_set_reshape.shape[0]), dtype=tf.float32)

        sl_y = tf.cast(tf.one_hot(sl_y_r, len(self.possible_k)), tf.int16)

        return sl_x, sl_y

