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


class FewShot(tf.keras.Model):
    def __init__(self,
                 # inputs parameters
                 w, h, c,

                 # Model parameters
                 backbone: tf.keras.Model,

                 # SL bosster
                 sl_model: tf.keras.Model = None,

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
        self.sl_model = sl_model
        self.alpha = kwargs.get("alpha", 1.0)

        self.sl_support_loss = None
        self.sl_query_loss = None

    def set_support(self, support):
        raise NotImplementedError

    def apply_query(self, query):
        raise NotImplementedError


class Prototypical(FewShot):
    """
    Reference: https://github.com/schatty/prototypical-networks-tf/blob/master/prototf/models/prototypical.py
    Implemenation of Prototypical Network.
    """

    def __init__(self,
                 # inputs parameters
                 w, h, c,

                 # Model parameters
                 backbone: tf.keras.Model,

                 # SL bosster
                 sl_model: tf.keras.Model = None,

                 # others
                 **kwargs):
        """
        Args:
            w (int): image width .
            h (int): image height.
            c (int): number of channels.
            backbone (tf.keras.Model): the encoder model as backbone.

        """
        super(Prototypical, self).__init__(
            w, h, c,
            backbone,
            sl_model,
            **kwargs
        )

        self.z_prototypes = None
        self.n_class = None
        self.n_support = None

    def call(self,  _inputs, training=None, mask=None):
        return self.call_proto(*_inputs)

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

        if self.sl_model is not None:
            self.sl_support_loss, _ = self.sl_model.call(support)

    def apply_query(self, _query):
        loss_few, acc_few = self.apply_query_proto(_query)

        if self.sl_model is None:
            return loss_few, acc_few
        else:
            self.sl_query_loss, _ = self.sl_model.call(_query)
            sl_loss = tf.reduce_mean(tf.concat([self.sl_support_loss, self.sl_query_loss], axis=0))
            return loss_few + self.alpha * sl_loss, acc_few

    def apply_query_proto(self, query):
        n_query = query.shape[1]
        y = np.tile(np.arange(self.n_class)[:, np.newaxis], (1, n_query))
        y_onehot = tf.cast(tf.one_hot(y, self.n_class), tf.float32)

        # correct indices of support samples (just natural order)
        target_inds = tf.reshape(tf.range(self.n_class), [self.n_class, 1])
        target_inds = tf.tile(target_inds, [1, n_query])

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
        log_p_y = tf.nn.log_softmax(-tf.cast(dists, tf.float32), axis=-1)
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


class CosineClassifier(FewShot):
    """
    Reference:
    Implemenation of Cosine Classifier.
    """

    def __init__(self,
                 # inputs parameters
                 w, h, c,

                 # Model parameters
                 backbone: tf.keras.Model,

                 # SL booster
                 sl_model: tf.keras.Model = None,

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
        super(CosineClassifier, self).__init__(
            w, h, c,
            backbone,
            sl_model,
            **kwargs
        )

        self.n_cls_base = kwargs.get("n_cls_base", 1)
        self.nFeat = self.backbone.output_shape[-1]

        self.n_class = None
        self.n_support = None

        self.weight_base = tf.Variable(
            np.random.normal(0.0, np.sqrt(2.0/self.nFeat), size=(self.n_cls_base, self.nFeat)),
            dtype=tf.float32,
            trainable=True
        )
        # self.bias = tf.Variable(0.0, trainable=True)
        self.scale_cls = tf.Variable(10.0, trainable=True)

        self.z_prototypes = None

    def call(self, inputs, training=None, mask=None):
        x_batch, ids_batch, y_batch = inputs
        x_feats = self.backbone(x_batch)  # shape: [n_cls, n_feats]

        # normalization
        x_feats_normalized = tf.math.l2_normalize(x_feats, axis=-1)
        self.weight_base = tf.math.l2_normalize(self.weight_base, axis=-1)

        # similarity [n_cls, n_cls] = x_feats_norm [n_cls, n_feats] \dot w_norm.T [n_feats, n_cls]
        cls_similarity = tf.keras.backend.dot(x_feats_normalized, tf.transpose(self.weight_base))

        log_p_y = tf.nn.log_softmax(self.scale_cls * cls_similarity, axis=-1)

        loss_few = -tf.reduce_mean(
            tf.reshape(
                tf.reduce_sum(
                    tf.multiply(
                        tf.cast(y_batch, tf.float32), tf.cast(log_p_y, tf.float32)
                    ), axis=-1
                ), [-1]
            )
        )
        eq = tf.cast(
            tf.equal(
                tf.cast(tf.argmax(log_p_y, axis=-1), tf.int32),
                tf.cast(ids_batch, tf.int32)
            ), tf.float32
        )
        acc_few = tf.reduce_mean(eq)

        if self.sl_model is None:
            return loss_few, acc_few
        else:
            sl_loss, sl_acc = self.sl_model.call(x_batch)
            return loss_few + self.alpha * sl_loss, acc_few

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

        if self.sl_model is not None:
            self.sl_support_loss, _ = self.sl_model.call(support)

    def apply_query(self, _query):
        loss_few, acc_few = self._apply_query_cosine(_query)

        if self.sl_model is None:
            return loss_few, acc_few
        else:
            self.sl_query_loss, _ = self.sl_model.call(_query)
            sl_loss = tf.reduce_mean(tf.concat([self.sl_support_loss, self.sl_query_loss], axis=0))
            return loss_few + self.alpha * sl_loss, acc_few

    def _apply_query_cosine(self, query):
        n_query = query.shape[1]
        y = np.tile(np.arange(self.n_class)[:, np.newaxis], (1, n_query))
        y_onehot = tf.cast(tf.one_hot(y, self.n_class), tf.float32)

        # correct indices of support samples (just natural order)
        target_inds = tf.reshape(tf.range(self.n_class), [self.n_class, 1])
        target_inds = tf.tile(target_inds, [1, n_query])

        z_query = self.backbone(tf.reshape(query, [self.n_class * n_query,
                                                   self.w, self.h, self.c]))

        # Calculate distances between query and prototypes
        dists = util.calc_cosine_dists(z_query, self.z_prototypes)

        # log softmax of calculated distances
        log_p_y = tf.nn.log_softmax(self.scale_cls * dists, axis=-1)
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


#####################################################################################################
#  Self-learning models
#####################################################################################################


def get_sl_model(backbone, sl_boosted_type: util.SLBoostedType, **kwargs):
    sl_type_to_cls = {
        util.SLBoostedType.ROT: SLRotationModel
    }

    return sl_type_to_cls.get(sl_boosted_type)(backbone, **kwargs) if sl_boosted_type is not None else None


class SLRotationModel(tf.keras.Model):
    def __init__(self, backbone, **kwargs):
        super(SLRotationModel, self).__init__()

        self.possible_k = range(4)
        self.possible_rotations = [90 * _k for _k in self.possible_k]
        self.possible_rotations_to_one_hot = {
            rot: util.c_idx2one_hot(idx, np.zeros(len(self.possible_rotations), dtype=int))
            for idx, rot in enumerate(self.possible_rotations)
        }

        self.backbone = backbone
        self._sl_input_shape = backbone.output_shape[1:]
        self._sl_output_size = len(self.possible_k)
        self._nb_hidden_layer: int = kwargs.get("nb_hidden_layers", 1)
        self._hidden_neurons: list = kwargs.get("hidden_neurons", [4096 for _ in range(self._nb_hidden_layer)])
        self._nb_hidden_layer = len(self._hidden_neurons)

        self.sl_classifier_layers = [
            Flatten(),
            ReLU(),
            *[
                Sequential([
                    Dense(h),
                    BatchNormalization(),
                    ReLU(),
                ], name=f"Dense_Block_{i}")
                for i, h in enumerate(self._hidden_neurons)
            ],
            Dense(self._sl_output_size, name="sl_rot_output_layer"),
            Softmax(dtype=tf.float32),
        ]

        self._cls_input = Input(shape=self._sl_input_shape)
        self._seq = Sequential(self.sl_classifier_layers)
        self._cls = tf.keras.Model(inputs=self._cls_input, outputs=self._seq(self._cls_input))

    def call(self, inputs, training=None, mask=None):
        _in, *_ = inputs
        sl_x, sl_y = self._get_sl_set_args(_in)

        sl_y_pred = self._cls(self.backbone(sl_x))

        loss = categorical_crossentropy(sl_y, sl_y_pred)

        sl_eq = tf.cast(
            tf.equal(
                tf.cast(tf.argmax(sl_y_pred, axis=-1), tf.int32),
                tf.cast(tf.argmax(sl_y, axis=-1), tf.int32)
            ), tf.float32
        )

        acc = tf.reduce_mean(sl_eq)

        return loss, acc

    def _get_sl_set_args(self, _set):
        *_batch_dim, _w, _h, _c = _set.shape

        _set_reshape = tf.reshape(_set, shape=[np.prod(_batch_dim), _w, _h, _c])

        sl_y_r = np.random.choice(self.possible_k, _set_reshape.shape[0])
        sl_x = tf.map_fn(lambda _i: tf.image.rot90(_set_reshape[_i], sl_y_r[_i]),
                         tf.range(_set_reshape.shape[0]), dtype=tf.float32)

        sl_y = tf.cast(tf.one_hot(sl_y_r, len(self.possible_k)), tf.int16)

        return sl_x, sl_y

