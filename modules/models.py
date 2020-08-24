import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Softmax, ReLU, Input
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Sequential

import modules.util as util


#####################################################################################################
#  Base models
#####################################################################################################

class BaseModel(tf.keras.Model):
    def __init__(self, backbone: tf.keras.Model):
        super(BaseModel, self).__init__()
        self.backbone = backbone

    def call(self, inputs, training=None, mask=None):
        """
        Call for batch training
        :param inputs: The inputs of the model (tf.Tensor)
        :param training: True if is in training phase else False (bool)
        :param mask: The mask of the data
        :return: y, logits (tuple)
        """
        raise NotImplementedError

    def compute_batch_logs(self, y, y_pred) -> dict:
        """
        Compute the logs of the batch.
        :param y: The truth values of the classification.
        :param y_pred: The predicted values of the classification or the logits.
        :return: The logs of the classification batch. ex: {"loss": 4.523, "accuracy": 0.726}
        """
        raise NotImplementedError


class SelfLearningModel(BaseModel):
    def __init__(self, backbone):
        super(SelfLearningModel, self).__init__(backbone)
        self._sl_input_shape = backbone.output_shape[1:]

    def call(self, inputs, training=None, mask=None):
        """
        Call for batch training
        :param inputs: The inputs of the model (tf.Tensor)
        :param training: True if is in training phase else False (bool)
        :param mask: The mask of the data
        :return: y, logits (tuple)
        """
        raise NotImplementedError

    def compute_batch_loss_acc(self, y, y_pred):
        """
        Compute the loss and the accuracy of the current batch. (Depreciated -> use compute_batch_logs instead)
        :param y: The truth values of the classification.
        :param y_pred: The predicted values of the classification or the logits.
        :return: (loss, accuracy) (tuple(float or tf.Tensor, float))
        """
        raise NotImplementedError

    def compute_loss_by_inputs(self, inputs, training=None, mask=None):
        """
        Compute directly the loss for batch training
        :param inputs: The inputs of the model (tf.Tensor)
        :param training: True if is in training phase else False (bool)
        :param mask: The mask of the data
        :return: loss (float or tf.Tensor)
        """
        return self.compute_batch_loss_acc(*self.call(inputs, training, mask))

    def compute_batch_logs(self, y, y_pred):
        """
        Compute the logs of the batch.
        :param y: The truth values of the classification.
        :param y_pred: The predicted values of the classification or the logits.
        :return: The logs of the classification batch. ex: {"loss": 4.523, "accuracy": 0.726}
        """
        raise NotImplementedError


class FewShot(BaseModel):
    def __init__(self,
                 # inputs parameters
                 w, h, c,

                 # Model parameters
                 backbone_net: tf.keras.Model,

                 # SL for auxiliary loss
                 sl_model: SelfLearningModel = None,

                 # others
                 **kwargs):
        """
        Args:
            :param w (int): image width .
            :param h (int): image height.
            :param c (int): number of channels.
            :param backbone_net (tf.keras.Model): the encoder model as backbone.
            :param sl_model (SelfLearningModel): The self-learning model
            :param kwargs: {
                :param alpha (float): float value between 0 and 1 used to scale the importance of the auxiliary loss.
                                      -> loss = main_loss + alpha * aux_loss.
            }

        """
        super(FewShot, self).__init__(backbone_net)
        self.w, self.h, self.c = w, h, c
        self.sl_model = sl_model
        self.alpha = kwargs.get("alpha", 1.0)

        self.sl_support_loss = None
        self.sl_query_loss = None

    def call(self, inputs, training=None, mask=None):
        """
        Call for batch training
        :param inputs: The inputs of the model (tf.Tensor)
        :param training: True if is in training phase else False (bool)
        :param mask: The mask of the data
        :return: y, logits (tuple)
        """
        raise NotImplementedError

    def set_support(self, support):
        """
        Set the support tensor data for the few-shot classification.
        :param support: The support tensor in shape: [nb_cls, nb_img, img_size, img_size, nb_channels] or
                                                    [n-way, n-shot, img_size, img_size, nb_channels]
        :return: None
        """
        raise NotImplementedError

    def apply_query(self, query) -> tuple:
        """
        Apply the query for the classification Task. The support tensor must be set before this call -> see set_support
        :param query: The query Tensor in shape: [n-way, n-query, img_size, img_size, nb_channels]
        :return: (y: The truth values of the classification,
                  y_pred: The predicted values of the classification or the logits). (tuple)
        """
        raise NotImplementedError

    def compute_episodic_loss_acc(self, y, y_pred):
        """
        Compute the loss and the accuracy of the current episode. (Depreciated -> use compute_episodic_logs instead)
        :param y: The truth values of the classification.
        :param y_pred: The predicted values of the classification or the logits.
        :return: (loss, accuracy) (tuple(float or tf.Tensor, float))
        """
        raise NotImplementedError

    def compute_batch_logs(self, y, y_pred) -> dict:
        """
        Compute the logs of the current batch.
        :param y: The truth values of the classification.
        :param y_pred: The predicted values of the classification or the logits.
        :return: The logs of the classification batch. ex: {"loss": 4.523, "accuracy": 0.726}
        """
        raise NotImplementedError

    def compute_episodic_logs(self, y, y_pred) -> dict:
        """
        Compute the logs of the current episode.
        :param y: The truth values of the classification.
        :param y_pred: The predicted values of the classification or the logits.
        :return: The logs of the classification batch. ex: {"loss": 4.523, "accuracy": 0.726}
        """
        raise NotImplementedError

    def compute_batch_loss_acc(self, y, y_pred):
        """
        Compute the loss and the accuracy of the current batch. (Depreciated -> use compute_batch_logs instead)
        :param y: The truth values of the classification.
        :param y_pred: The predicted values of the classification or the logits.
        :return: (loss, accuracy) (tuple(float or tf.Tensor, float))
        """
        raise NotImplementedError

    def compute_sl_loss(self):
        """
        Compute the loss of the self-learning as auxiliary loss.
        :return: The loss. (float or tf.Tensor)
        """
        raise NotImplementedError


#####################################################################################################
#  Few-Shot learning models
#####################################################################################################

class Prototypical(FewShot):
    """
    Reference: https://github.com/schatty/prototypical-networks-tf/blob/master/prototf/models/prototypical.py
    Implemenation of Prototypical Network.
    """

    def __init__(self,
                 # inputs parameters
                 w: int, h: int, c: int,

                 # Model parameters
                 backbone_net: tf.keras.Model,

                 # SL bosster
                 sl_model: tf.keras.Model = None,

                 # others
                 **kwargs):
        """
        :param w (int): image width .
        :param h (int): image height.
        :param c (int): number of channels.
        :param backbone_net (tf.keras.Model): the encoder model as backbone.
        :param sl_model (SelfLearningModel): The self-learning model
        :param kwargs: {
            :param alpha (float): float value between 0 and 1 used to scale the importance of the auxiliary loss.
                                  -> loss = main_loss + alpha * aux_loss.
        }
        """
        super(Prototypical, self).__init__(
            w, h, c,
            backbone_net,
            sl_model,
            **kwargs
        )

        self.z_prototypes = None
        self.query = None
        self.n_class = None
        self.n_support = None
        self.n_query = None

    def call(self, _inputs, training=None, mask=None):
        y, y_pred = self.call_proto(*_inputs)

        if self.sl_model is not None:
            self._sl_y, self._sl_y_pred = self.sl_model.call(_inputs)

        return y, y_pred

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
            self.sl_support_loss, _ = self.sl_model.compute_loss_by_inputs(support)

    def apply_query(self, _query):
        self.query = _query
        self.n_query = _query.shape[1]
        y, y_pred = self.apply_query_proto(self.query)
        return y, y_pred

    def apply_query_proto(self, query):
        n_query = query.shape[1]
        y = np.tile(np.arange(self.n_class)[:, np.newaxis], (1, n_query))
        # y_onehot = tf.cast(tf.one_hot(y, self.n_class), tf.float32)

        # correct indices of support samples (just natural order)
        target_inds = tf.reshape(tf.range(self.n_class), [self.n_class, 1])
        target_inds = tf.tile(target_inds, [1, n_query])

        z_query = self.backbone(tf.reshape(query, [self.n_class * n_query,
                                                   self.w, self.h, self.c]))

        # Calculate distances between query and prototypes
        dists = util.calc_euclidian_dists(z_query, self.z_prototypes)

        p_y = -tf.cast(dists, tf.float32)
        return y, p_y

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
        p_y = -tf.cast(dists, tf.float32)
        return y, p_y

    def compute_batch_loss_acc(self, y, y_pred):
        raise NotImplementedError

    def compute_episodic_loss_acc(self, y, y_pred):
        y_onehot = tf.cast(tf.one_hot(y, self.n_class), tf.float32)

        # log softmax of calculated distances
        log_p_y = tf.nn.log_softmax(y_pred, axis=-1)
        log_p_y = tf.reshape(log_p_y, [self.n_class, self.n_query, -1])

        loss_few = -tf.reduce_mean(
            tf.reshape(tf.reduce_sum(tf.multiply(y_onehot, tf.cast(log_p_y, tf.float32)), axis=-1), [-1])
        )
        eq = tf.cast(
            tf.equal(
                tf.cast(tf.argmax(log_p_y, axis=-1), tf.int32),
                tf.cast(y, tf.int32)
            ), tf.float32
        )
        acc_few = tf.reduce_mean(eq)

        sl_loss = self.compute_sl_loss()
        return loss_few + self.alpha * sl_loss, acc_few

    def compute_batch_logs(self, y, y_pred) -> dict:
        raise NotImplementedError

    def compute_episodic_logs(self, y, y_pred) -> dict:
        y_onehot = tf.cast(tf.one_hot(y, self.n_class), tf.float32)

        # log softmax of calculated distances
        log_p_y = tf.nn.log_softmax(y_pred, axis=-1)
        log_p_y = tf.reshape(log_p_y, [self.n_class, self.n_query, -1])

        loss_few = -tf.reduce_mean(
            tf.reshape(tf.reduce_sum(tf.multiply(y_onehot, tf.cast(log_p_y, tf.float32)), axis=-1), [-1])
        )
        eq = tf.cast(
            tf.equal(
                tf.cast(tf.argmax(log_p_y, axis=-1), tf.int32),
                tf.cast(y, tf.int32)
            ), tf.float32
        )
        acc_few = tf.reduce_mean(eq)

        sl_loss = self.compute_sl_loss()
        loss = loss_few + self.alpha * sl_loss
        logs = {"loss": loss, "accuracy": acc_few, "sl_loss": sl_loss}
        return logs

    def compute_sl_loss(self):
        if self.sl_model is None:
            sl_loss = 0.0
        else:
            self.sl_query_loss, _ = self.sl_model.compute_loss_by_inputs(self.query)
            if len(self.sl_query_loss.shape) == 0:
                sl_loss = (self.sl_support_loss * self.n_support + self.sl_query_loss * self.n_query) / (
                        self.n_support + self.n_query)
            else:
                sl_loss = tf.reduce_mean(tf.concat(
                    [tf.squeeze(self.sl_support_loss), tf.squeeze(self.sl_query_loss)],
                    axis=0))
        return sl_loss


class CosineClassifier(FewShot):
    """
    Reference:
    Implemenation of Cosine Classifier.
    """

    def __init__(self,
                 # inputs parameters
                 w, h, c,

                 # Model parameters
                 backbone_net: tf.keras.Model,

                 # SL booster
                 sl_model: tf.keras.Model = None,

                 # others
                 **kwargs):
        """
        :param w (int): image width .
        :param h (int): image height.
        :param c (int): number of channels.
        :param backbone_net (tf.keras.Model): the encoder model as backbone.
        :param sl_model (SelfLearningModel): The self-learning model
        :param kwargs: {
            :param alpha (float): float value between 0 and 1 used to scale the importance of the auxiliary loss.
                                  -> loss = main_loss + alpha * aux_loss.
            :param n_cls_base (int): number of base class.
            :param n_cls_val (int): number of validation class.
        }
        """
        super(CosineClassifier, self).__init__(
            w, h, c,
            backbone_net,
            sl_model,
            **kwargs
        )

        self.n_cls_base = kwargs.get("n_cls_base", 64)
        self.nFeat = self.backbone.output_shape[-1]

        self.n_class = None
        self.n_support = None
        self.n_query = None

        self.weight_base = tf.Variable(
            np.random.normal(0.0, np.sqrt(2.0 / self.nFeat), size=(self.n_cls_base, self.nFeat)),
            dtype=tf.float32,
            trainable=True
        )

        self.n_cls_val = kwargs.get("n_cls_val", 16)
        self.weight_val = lambda: tf.ones((self.n_cls_val, self.nFeat)) * tf.reduce_mean(self.weight_base)
        # self.bias = tf.Variable(0.0, trainable=True)
        self.scale_cls = tf.Variable(10.0, trainable=True)

        self.z_prototypes = None
        self.query = None
        self._sl_y, self._sl_y_pred = None, None

    def call(self, inputs, training=None, mask=None):
        x_batch, ids_batch, y_batch = inputs
        x_feats = self.backbone(x_batch)  # shape: [n_exp, n_feats]

        weights = self.weight_base if training else self.weight_val()

        # similarity [n_exp, n_cls] = x_feats_norm [n_exp, n_feats] \dot w_norm.T [n_feats, n_cls]
        cls_similarity = util.calc_cosine_similarity(x_feats, weights)
        p_y = self.scale_cls * cls_similarity

        if self.sl_model is not None:
            self._sl_y, self._sl_y_pred = self.sl_model.call(x_batch)

        return y_batch, p_y

    def compute_batch_loss_acc(self, y, y_pred):
        log_p_y = tf.nn.log_softmax(y_pred, axis=-1)

        loss_few = -tf.reduce_mean(
            tf.reduce_sum(
                tf.multiply(
                    tf.cast(y, tf.float32), tf.cast(log_p_y, tf.float32)
                ), axis=-1
            )
        )

        eq = tf.cast(
            tf.equal(
                tf.cast(tf.argmax(log_p_y, axis=-1), tf.int32),
                tf.cast(tf.argmax(y, axis=-1), tf.int32)
            ), tf.float32
        )
        acc_few = tf.reduce_mean(eq)

        if self.sl_model is None:
            return loss_few, acc_few
        else:
            sl_loss, sl_acc = self.sl_model.compute_batch_loss_acc(self._sl_y, self._sl_y_pred)
            return loss_few + self.alpha * sl_loss, acc_few

    def compute_batch_logs(self, y, y_pred) -> dict:
        log_p_y = tf.nn.log_softmax(y_pred, axis=-1)

        loss_few = -tf.reduce_mean(
            tf.reduce_sum(
                tf.multiply(
                    tf.cast(y, tf.float32), tf.cast(log_p_y, tf.float32)
                ), axis=-1
            )
        )

        eq = tf.cast(
            tf.equal(
                tf.cast(tf.argmax(log_p_y, axis=-1), tf.int32),
                tf.cast(tf.argmax(y, axis=-1), tf.int32)
            ), tf.float32
        )
        acc_few = tf.reduce_mean(eq)

        if self.sl_model is None:
            sl_loss = 0.0
        else:
            sl_loss, sl_acc = self.sl_model.compute_batch_loss_acc(self._sl_y, self._sl_y_pred)

        loss = loss_few + self.alpha * sl_loss
        logs = {"loss": loss, "accuracy": acc_few, "sl_loss": sl_loss}
        return logs

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
            self.sl_support_loss, _ = self.sl_model.compute_loss_by_inputs(support)

    def apply_query(self, _query):
        self.query = _query
        self.n_query = _query.shape[1]
        y, y_pred = self._apply_query_cosine(self.query)
        return y, y_pred

    def _apply_query_cosine(self, query):
        n_query = query.shape[1]
        y = np.tile(np.arange(self.n_class)[:, np.newaxis], (1, n_query))

        # correct indices of support samples (just natural order)
        target_inds = tf.reshape(tf.range(self.n_class), [self.n_class, 1])
        target_inds = tf.tile(target_inds, [1, n_query])

        z_query = self.backbone(tf.reshape(query, [self.n_class * n_query,
                                                   self.w, self.h, self.c]))

        # Calculate distances between query and prototypes
        similarity = util.calc_cosine_similarity(z_query, self.z_prototypes)

        # log softmax of calculated distances
        p_y = self.scale_cls * similarity
        return y, p_y

    def compute_episodic_loss_acc(self, y, y_pred):
        y_onehot = tf.cast(tf.one_hot(y, self.n_class), tf.float32)

        log_p_y = tf.nn.log_softmax(y_pred, axis=-1)
        log_p_y = tf.reshape(log_p_y, [self.n_class, self.n_query, -1])

        loss_few = -tf.reduce_mean(
            tf.reshape(
                tf.reduce_sum(
                    tf.multiply(y_onehot, tf.cast(log_p_y, tf.float32)), axis=-1),
                [-1])
        )
        eq = tf.cast(
            tf.equal(
                tf.cast(tf.argmax(log_p_y, axis=-1), tf.int32),
                tf.cast(y, tf.int32)
            ), tf.float32
        )
        acc_few = tf.reduce_mean(eq)

        sl_loss = self.compute_sl_loss()
        return loss_few + self.alpha * sl_loss, acc_few

    def compute_episodic_logs(self, y, y_pred) -> dict:
        y_onehot = tf.cast(tf.one_hot(y, self.n_class), tf.float32)

        log_p_y = tf.nn.log_softmax(y_pred, axis=-1)
        log_p_y = tf.reshape(log_p_y, [self.n_class, self.n_query, -1])

        loss_few = -tf.reduce_mean(
            tf.reshape(
                tf.reduce_sum(
                    tf.multiply(y_onehot, tf.cast(log_p_y, tf.float32)), axis=-1),
                [-1])
        )
        eq = tf.cast(
            tf.equal(
                tf.cast(tf.argmax(log_p_y, axis=-1), tf.int32),
                tf.cast(y, tf.int32)
            ), tf.float32
        )
        acc_few = tf.reduce_mean(eq)

        sl_loss = self.compute_sl_loss()
        loss = loss_few + self.alpha * sl_loss
        logs = {"loss": loss, "accuracy": acc_few, "sl_loss": sl_loss}
        return logs

    def compute_sl_loss(self):
        if self.sl_model is None:
            sl_loss = 0.0
        else:
            self.sl_query_loss, _ = self.sl_model.compute_loss_by_inputs(self.query)
            if len(self.sl_query_loss.shape) == 0:
                sl_loss = (self.sl_support_loss * self.n_support + self.sl_query_loss * self.n_query) / (
                        self.n_support + self.n_query)
            else:
                sl_loss = tf.reduce_mean(tf.concat([self.sl_support_loss, self.sl_query_loss], axis=0))
        return sl_loss


class Gen0(FewShot):
    """
    Reference: https://arxiv.org/pdf/2006.09785.pdf
    """

    def __init__(self,
                 # inputs parameters
                 w, h, c,

                 # Model parameters
                 backbone_net: tf.keras.Model,

                 # others
                 **kwargs):
        """
        :param w (int): image width .
        :param h (int): image height.
        :param c (int): number of channels.
        :param backbone_net (tf.keras.Model): the encoder model as backbone.
        :param kwargs: {
            :param alpha (float): float value between 0 and 1 used to scale the importance of the auxiliary loss.
                                  -> loss = main_loss + alpha * aux_loss.
            :param n_cls_base (int): number of base class.
        }
        """
        super(Gen0, self).__init__(
            w, h, c,
            backbone_net,
            None,
            **kwargs
        )

        self.z_prototypes = None
        self.query = None
        self.n_class = None
        self.n_support = None
        self.n_query = None

        self.n_cls_base = kwargs.get("n_cls_base", 64)
        self.n_cls_val = kwargs.get("n_cls_val", 16)
        self.nFeat = self.backbone.output_shape[-1]

        self.cls_classifier_base = Dense(self.n_cls_base, input_shape=(self.nFeat, ), name="Dense-cls_classifier_base")
        self.cls_classifier_val = Dense(self.n_cls_base, input_shape=(self.nFeat,), name="Dense-cls_classifier_val")
        self.rot_classifier = Dense(4, input_shape=(self.n_cls_base, ), name="Dense-rot_classifier")

        self.sl_p_rot = None
        self.sl_y_rot = None

    def call(self, _inputs, training=None, mask=None):
        x_batch, ids_batch, y_batch = _inputs
        x_rot, self.sl_y_rot = self.rotate_x_batch(x_batch, [0, 1, 2, 3])

        x_feats = self.backbone(x_rot)
        p_cls = self.cls_classifier_base(x_feats)
        self.sl_p_rot = self.rot_classifier(p_cls)

        return y_batch, p_cls

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
            self.sl_support_loss, _ = self.sl_model.compute_loss_by_inputs(support)

    def apply_query(self, _query):
        self.query = _query
        self.n_query = _query.shape[1]
        y, y_pred = self.apply_query_proto(self.query)
        return y, y_pred

    def apply_query_proto(self, query):
        n_query = query.shape[1]
        y = np.tile(np.arange(self.n_class)[:, np.newaxis], (1, n_query))
        # y_onehot = tf.cast(tf.one_hot(y, self.n_class), tf.float32)

        # correct indices of support samples (just natural order)
        target_inds = tf.reshape(tf.range(self.n_class), [self.n_class, 1])
        target_inds = tf.tile(target_inds, [1, n_query])

        z_query = self.backbone(tf.reshape(query, [self.n_class * n_query,
                                                   self.w, self.h, self.c]))

        # Calculate distances between query and prototypes
        dists = util.calc_euclidian_dists(z_query, self.z_prototypes)

        p_y = -tf.cast(dists, tf.float32)
        return y, p_y

    def compute_batch_loss_acc(self, y, y_pred):
        raise NotImplementedError

    def compute_episodic_loss_acc(self, y, y_pred):
        raise NotImplementedError

    def compute_batch_logs(self, y, y_pred) -> dict:
        sl_y_pred = tf.nn.softmax(self.sl_p_rot)
        sl_loss = tf.losses.categorical_crossentropy(self.sl_y_rot, sl_y_pred)

        soft_y_pred = tf.nn.softmax(y_pred)
        loss_few = tf.losses.categorical_crossentropy(y, soft_y_pred)
        eq = tf.cast(
            tf.equal(
                tf.cast(tf.argmax(soft_y_pred, axis=-1), tf.int32),
                tf.cast(tf.argmax(y, axis=-1), tf.int32)
            ), tf.float32
        )
        acc_few = tf.reduce_mean(eq)

        loss = loss_few + self.alpha * sl_loss
        logs = {"loss": loss, "accuracy": acc_few, "sl_loss": sl_loss}
        return logs

    def compute_episodic_logs(self, y, y_pred) -> dict:
        y_onehot = tf.cast(tf.one_hot(y, self.n_class), tf.float32)

        # log softmax of calculated distances
        log_p_y = tf.nn.log_softmax(y_pred, axis=-1)
        log_p_y = tf.reshape(log_p_y, [self.n_class, self.n_query, -1])

        loss = -tf.reduce_mean(
            tf.reshape(tf.reduce_sum(tf.multiply(y_onehot, tf.cast(log_p_y, tf.float32)), axis=-1), [-1])
        )
        eq = tf.cast(
            tf.equal(
                tf.cast(tf.argmax(log_p_y, axis=-1), tf.int32),
                tf.cast(y, tf.int32)
            ), tf.float32
        )
        acc_few = tf.reduce_mean(eq)

        logs = {"loss": loss, "accuracy": acc_few}
        return logs

    def compute_sl_loss(self):
        raise NotImplementedError

    def rotate_x_batch(self, x, rotations_k: list = None):
        """

        :param x:
        :param rotations_k:
        :return:
        """
        if rotations_k is None:
            rotations_k = [0, 1, 2, 3]

        *_batch_dim, _w, _h, _c = x.shape

        _x_reshape = tf.reshape(x, shape=[np.prod(_batch_dim), _w, _h, _c])
        # print(_set.shape, _batch_dim, _set_reshape.shape)
        # assert 1 == 0
        x_r = tf.concat(
            [
                tf.image.rot90(_x_reshape, k)
                for k in rotations_k
            ],
            axis=0
        )
        y_r = tf.concat(
            [
                tf.one_hot(
                    tf.ones((_x_reshape.shape[0],), dtype=tf.int32)*k,
                    len(rotations_k)
                )
                for k in rotations_k
            ],
            axis=0
        )

        return x_r, y_r


#####################################################################################################
#  Self-learning models
#####################################################################################################

def get_sl_model(backbone, sl_boosted_type: util.SLBoostedType, **kwargs):
    sl_type_to_cls = {
        util.SLBoostedType.ROT: SLRotationModel,
        util.SLBoostedType.ROT_FEAT: SLDistFeatModel,
    }

    return sl_type_to_cls.get(sl_boosted_type)(backbone, **kwargs) if sl_boosted_type is not None else None


class SLRotationModel(SelfLearningModel):
    def __init__(self, backbone, **kwargs):
        super(SLRotationModel, self).__init__(backbone)

        self.possible_k = range(4)
        self.possible_rotations = [90 * _k for _k in self.possible_k]
        self.possible_rotations_to_one_hot = {
            rot: util.c_idx2one_hot(idx, np.zeros(len(self.possible_rotations), dtype=int))
            for idx, rot in enumerate(self.possible_rotations)
        }

        self._sl_output_size = len(self.possible_k)
        self._classifier_type = kwargs.get("classifier_type", "cosine")

        if self._classifier_type.lower() == "dense":
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

            self.loss_fn = lambda y, y_pred: categorical_crossentropy(y, y_pred)

        elif self._classifier_type.lower() == "cosine":
            self.nFeat = self.backbone.output_shape[-1]

            self._weights = tf.Variable(
                np.random.normal(0.0, np.sqrt(2.0 / self.nFeat), size=(self._sl_output_size, self.nFeat)),
                dtype=tf.float32,
                trainable=True,
                name="sl_rot_weights"
            )
            self.scale_cls = tf.Variable(10.0, trainable=True, name="sl_rot_scale_cls")

            self._cls = lambda feats: tf.nn.log_softmax(
                self.scale_cls * util.calc_cosine_similarity(feats, self._weights),
                axis=-1)

            self.loss_fn = lambda y, y_pred: -tf.reduce_mean(
                tf.reduce_sum(
                    tf.multiply(
                        tf.cast(y, tf.float32), tf.cast(y_pred, tf.float32)
                    ), axis=-1
                )
            )
        else:
            raise ValueError(f"{self._classifier_type} is not a recognizable classifier type")

    def call(self, inputs, training=None, mask=None):
        sl_x, sl_y = self._get_sl_set_args(inputs)
        sl_y_pred = self._cls(self.backbone(sl_x))
        return sl_y, sl_y_pred

    def compute_batch_loss_acc(self, y, y_pred):
        loss = self.loss_fn(y, y_pred)

        sl_eq = tf.cast(
            tf.equal(
                tf.cast(tf.argmax(y_pred, axis=-1), tf.int32),
                tf.cast(tf.argmax(y, axis=-1), tf.int32)
            ), tf.float32
        )

        acc = tf.reduce_mean(sl_eq)
        return loss, acc

    def compute_batch_logs(self, y, y_pred):
        loss, acc = self.compute_batch_loss_acc(y, y_pred)
        return {"loss": loss, "accuracy": acc}

    def _get_sl_set_args(self, _set):
        *_batch_dim, _w, _h, _c = _set.shape

        _set_reshape = tf.reshape(_set, shape=[np.prod(_batch_dim), _w, _h, _c])
        # print(_set.shape, _batch_dim, _set_reshape.shape)
        # assert 1 == 0
        sl_y_r = np.random.choice(self.possible_k, _set_reshape.shape[0])
        sl_x = tf.map_fn(lambda _i: tf.image.rot90(_set_reshape[_i], sl_y_r[_i]),
                         tf.range(_set_reshape.shape[0]), dtype=tf.float32)

        sl_y = tf.cast(tf.one_hot(sl_y_r, len(self.possible_k)), tf.int16)

        return sl_x, sl_y


class SLDistFeatModel(SelfLearningModel):
    def __init__(self, backbone, **kwargs):
        super(SLDistFeatModel, self).__init__(backbone)

        self.nb_k = min([kwargs.get("nb_k", 2), 2])
        self.possible_k = range(self.nb_k)

        self._sl_output_size = len(self.possible_k)
        self._feat_dist = kwargs.get("feat_dist_mth", "l2")

        if self._feat_dist == "cosine":

            self.loss_fn = lambda x, y: util.calc_cosine_similarity(x, y)
        elif self._feat_dist == "l2":
            self.loss_fn = lambda x, y: tf.losses.mse(x, y)
        else:
            raise ValueError(f"feat_dist_mth ({self._feat_dist}) is not a recognizable dist")

    def call(self, inputs, training=None, mask=None):
        x = self._get_sl_set_args(inputs)
        x_feats = self.backbone(x)
        # print(f"_in.shape: {inputs.shape}")
        # print(f"x.shape: {x.shape}")
        # print(f"x_feats.shape: {x_feats.shape}")
        x_feats_k = tf.split(x_feats, self.nb_k, axis=0)
        # print(f"x_feats_k.shape: {[t.shape for t in x_feats_k]}")
        psi, psi_rot = x_feats_k
        return psi, psi_rot

    def _get_sl_set_args(self, _set):
        *_batch_dim, _w, _h, _c = _set.shape

        _set_reshape = tf.reshape(_set, shape=[np.prod(_batch_dim), _w, _h, _c])
        self._in_shape_0 = _set_reshape.shape[0]
        for k in self.possible_k[1:]:
            _set_reshape = tf.concat([_set_reshape, tf.image.rot90(_set_reshape, k)], axis=0)

        return _set_reshape

    def compute_batch_loss_acc(self, y, y_pred):
        loss = self.loss_fn(y, y_pred)
        # print(f"loss.shape: {loss.shape}, \n {loss}")
        # assert 1 == 0
        return loss, -1.0

    def compute_batch_logs(self, y, y_pred):
        loss, acc = self.compute_batch_loss_acc(y, y_pred)
        return {f"{self._feat_dist}_loss": loss, "accuracy": acc}


