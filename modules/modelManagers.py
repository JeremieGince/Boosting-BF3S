import os
import enum
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Softmax, ReLU, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import binary_crossentropy

import modules.util as util
from modules.models import Prototypical, SLRotationModel, CosineClassifier, get_sl_model, BaseModel, Gen0, Gen1
import modules.backbones as backbones

os.environ["PATH"] += os.pathsep + r'C:\Program Files (x86)\Graphviz2.38\bin/'


class NetworkModelManager:
    """
    This class is used to manage a keras model.
    """
    available_backbones = {
        "InceptionResNetV2": tf.keras.applications.InceptionResNetV2,
        "InceptionV3": tf.keras.applications.InceptionV3,
        "MobileNet": tf.keras.applications.MobileNet,
        "MobileNetV2": tf.keras.applications.MobileNetV2,
        "ResNet101": tf.keras.applications.ResNet101,
        "ResNet101V2": tf.keras.applications.ResNet101V2,
        "ResNet152": tf.keras.applications.ResNet152,
        "ResNet152V2": tf.keras.applications.ResNet152V2,
        "ResNet50": tf.keras.applications.ResNet50,
        "ResNet50V2": tf.keras.applications.ResNet50V2,
        "VGG16": tf.keras.applications.VGG16,
        "VGG19": tf.keras.applications.VGG19,
        "conv-4-64": backbones.conv_4_64,
        "conv-4-64_avg_pool": backbones.conv_4_64_avg_pool,
        "conv-4-64_glob_avg_pool": backbones.conv_4_64_glob_avg_pool,
    }

    WEIGHTS_PATH_EXT = "/cp-weights.h5"

    available_metrics = {"loss", "accuracy"}

    def __init__(self, **kwargs):
        """
        Constructor of NetworkModelManager.

        :param kwargs: a dict of parameters (dict){

            :param name: a string of the name of the network manage (str). This name is also used for the name of
            the directory that will contained all the data about this network.

            Optimizer parameters
            :param learning_rate: The learning of the optimizer. (float)
            :param momentum: The momentum of the optimizer. (float)
            :param use_nesterov: For SGD optimizer, use nesterov if True. (bool)
            :param optimizer_args: arguments to pass to the optimizer. (dict)
            :param optimizer: The type of the optimizer. (tensorflow.keras.optimizers)

            Others
            :param output_form: The ouput form of the network (util.OutputForm)
            :param teacher: The teacher of the current model.
            :param weights_path: The path of the initial weights for the model. (str)
        }
        """
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(policy)

        self.name = kwargs.get("name", "network_model")
        os.makedirs("training_data/" + self.name, exist_ok=True)
        self.checkpoint_path = "training_data/" + self.name + self.WEIGHTS_PATH_EXT
        self.history_path = f"training_data/{self.name}/cp-history.json"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.history = dict()
        self.model: BaseModel = None
        self.current_epoch = 0

        self.load_history()
        self.update_curr_epoch()

        self.output_form: util.OutputForm = kwargs.get("output_form", util.OutputForm.LABEL)

        # setting the optimizer
        self.learning_rate = kwargs.get("learning_rate", 1e-3)
        self.momentum = kwargs.get("momentum", 0.9)
        self.use_nesterov = kwargs.get("use_nesterov", True)
        self.optimizer_args = kwargs.get("optimizer_args", {
            "momentum": self.momentum,
            "nesterov": self.use_nesterov,
        })
        self.optimizer = kwargs.get("optimizer", SGD)(self.learning_rate, **self.optimizer_args)

        # metrics
        self.metrics: list = kwargs.get("metrics", ["loss", "accuracy"])
        assert all([m in NetworkModelManager.available_metrics for m in self.metrics]), "Unavailable metrics"

        # others
        self.init_weights_path: str = kwargs.get("weights_path", None)

        # Teaching
        self.is_teacher: bool = kwargs.get("is_teacher", False)

        self.teacher_net_manager: NetworkModelManager = kwargs.get("teacher", None)
        self.teacher_loss: str = kwargs.get("teacher_loss", "klb")
        self.teacher_t: float = kwargs.get("teacher_T", 4.0)
        self.teacher_loss_fn = lambda p, p_t: tf.reduce_mean(
            tf.losses.kullback_leibler_divergence(
                tf.nn.softmax(p / self.teacher_t), tf.nn.softmax(p_t / self.teacher_t)
            )
        )
        self.teacher_gamma: float = kwargs.get("teacher_gamma", 1.0)

        if self.teacher_net_manager is not None:
            self.metrics.append("T_"+self.teacher_loss)
            if "sl_loss" not in self.metrics:
                self.metrics.append("sl_loss")

    def init_model(self):
        """
        Used to initiate the model and its weights.
        :return: None
        """
        self.build_and_compile()
        if self.init_weights_path is not None and self.current_epoch <= 0:
            assert os.path.exists(self.init_weights_path), f"The path: {self.init_weights_path} doesn't exists"
            print(f"{self} is loading the weights from {self.init_weights_path}")
            _ = self.model.load_weights(self.init_weights_path, by_name=True),  # , skip_mismatch=True
            self.save_weights()

    def summary(self) -> str:
        """
        :return: The summary of the model. (str)
        """
        return self.model.summary()

    def load_weights(self):
        """
        Used to load the weights of the model in the current checkpoint_path.
        :return: None
        """
        assert self.model is not None
        self.model.load_weights(self.checkpoint_path)

    def save_weights(self):
        """
        Used the save the weights of the current model at the checkpoint_path.
        :return: None
        """
        self.model.save_weights(self.checkpoint_path)

    def load_history(self):
        """
        Used to load the current history in the attribute self.history.
        :return: None
        """
        import json
        if os.path.exists(self.history_path):
            self.history = json.load(open(self.history_path, 'r'))
        self.update_curr_epoch()

    def save_history(self):
        """
        Used to save the current history at the history_path.
        :return: None
        """
        import json
        json.dump(self.history, open(self.history_path, 'w'), indent=3)

    def update_curr_epoch(self):
        """
        Used to load the current epoch of the model with the current history in the attribute self.current_epoch.
        :return: None
        """
        self.current_epoch = len(self.history.get("train", {}).get("loss", []))

    @staticmethod
    def concat_phase_logs(logs_0: dict, logs_1: dict):
        """
        Used to concatenate two logs dictonary.
        :param logs_0: The first log (dict)
        :param logs_1: The second log (dict).
        :return: The concanate version of logs_0 and logs_1. (dict)
        """
        re_logs = {**logs_0, **logs_1}

        for key, value in re_logs.items():
            if key in logs_0 and key in logs_1:
                if hasattr(logs_0[key], '__iter__') and hasattr(value, '__iter__'):
                    re_logs[key] = list(np.array(list(logs_0[key]) + list(value), dtype=float))
                elif not hasattr(logs_0[key], '__iter__') and hasattr(value, '__iter__'):
                    re_logs[key] = list(np.array([logs_0[key]] + list(value), dtype=float))
                elif hasattr(logs_0[key], '__iter__') and not hasattr(value, '__iter__'):
                    re_logs[key] = list(np.array(list(logs_0[key]) + [value], dtype=float))
                else:
                    re_logs[key] = list(np.array([logs_0[key], value], dtype=float))

            else:
                if hasattr(value, '__iter__'):
                    re_logs[key] = list(np.array(list(value), dtype=float))
                else:
                    re_logs[key] = list(np.array([value], dtype=float))

        return re_logs

    def update_history(self, other: dict):
        """
        Update the history with another history dict.
        :param other: history to concatenate after the current one. (dict)
        :return: None
        """
        for _phase, p_logs in other.items():
            self.history[_phase] = self.concat_phase_logs(self.history.get(_phase, {}), p_logs)
        self.save_history()

    def load(self):
        """
        Load the weights and the history.
        :return: None
        """
        self.load_weights()
        self.load_history()

    def save(self):
        """
        Save the weights and the history.
        :return: None
        """
        self.save_weights()
        self.save_history()

    def build(self):
        """
        Build the current model and stock it in self.model.
        :return: None
        """
        raise NotImplementedError

    def compile(self):
        """
        Used to compile the current model.
        :return: None
        """
        assert self.model is not None
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_function,
            metrics=[
                tf.keras.metrics.Accuracy(),
            ]
        )
        return self.model

    def build_and_compile(self):
        """
        Build and compile the current model.
        :return: the current model. (tf.keras.Model)
        """
        self.model = self.build()
        self.model = self.compile()

        if not os.path.exists(self.checkpoint_path):
            if self.is_teacher:
                warnings.warn(f"This teacher model has no initialized weights!")
            self.save_weights()
        return self.model

    @staticmethod
    def loss_function(y_true, y_pred, **kwargs):
        """
        Used to stock the current loss_function.
        :param y_true: The truth labels.
        :param y_pred: The predicted labels.
        :param kwargs: the args for the loss function.
        :return: the loss.
        """
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred, **kwargs)

    def call(self, *args, **kwargs):
        return self.model.call(*args, **kwargs)

    def call_as_teacher(self, *args, **kwargs):
        return self.model.call_as_teacher(*args, **kwargs)

    def set_support(self, support):
        return self.model.set_support(support)

    def apply_query(self, query):
        return self.model.apply_query(query)

    def compute_batch_metrics(self, *args, **kwargs) -> dict:
        """
        Used to compute the metrics of the current call
        :param args:
        :param kwargs:
        :return:
        """
        y, y_pred = self.model.call(*args, **kwargs)

        if self.teacher_net_manager is None:
            logs = self.model.compute_batch_logs(y, y_pred)
        else:
            itr_logs = self.model.compute_batch_logs(y, y_pred)
            sl_loss = itr_logs.get("sl_loss", 0.0)
            acc = itr_logs.get("accuracy", 0.0)

            teacher_y, teacher_y_pred = self.teacher_net_manager.call_as_teacher(*args, **kwargs)
            teaching_loss = self.teacher_loss_fn(teacher_y_pred, y_pred)
            # print(y_pred, teacher_y_pred)
            # print(teaching_loss, sl_loss)
            loss = teaching_loss + self.teacher_gamma * sl_loss
            logs = {"loss": loss, "accuracy": acc, "T_"+self.teacher_loss: teaching_loss, "sl_loss": sl_loss}
        # print(logs)
        return logs

    def compute_episodic_metrics(self, data_itr, *args, **kwargs):
        _support = next(data_itr)
        self.model.set_support(_support)
        if self.teacher_net_manager is not None:
            self.teacher_net_manager.set_support(_support)

        _query = next(data_itr)
        y, y_pred = self.model.apply_query(_query)

        if self.teacher_net_manager is None:
            logs = self.model.compute_episodic_logs(y, y_pred)
        else:
            epi_logs = self.model.compute_episodic_logs(y, y_pred)
            sl_loss = epi_logs.get("sl_loss", 0.0)
            acc = epi_logs.get("accuracy", 0.0)

            teacher_y, teacher_y_pred = self.teacher_net_manager.apply_query(_query)
            teaching_loss = self.teacher_loss_fn(teacher_y_pred, y_pred)
            # print(y_pred, teacher_y_pred)
            # print(teaching_loss, sl_loss)
            loss = teaching_loss + self.teacher_gamma * sl_loss
            logs = {"loss": loss, "accuracy": acc, "T_"+self.teacher_loss: teaching_loss, "sl_loss": sl_loss}

        # print(y.shape, y_pred.shape)
        # print(logs)
        return logs


class NetworkManagerCallback(tf.keras.callbacks.Callback):
    """
    tf.keras.callbacks.Callback child class for the NetworkManager object.
    """
    def __init__(self, network_manager: NetworkModelManager, **kwargs):
        """
        Contructor of NetworkManagerCallback.
        :param network_manager: The NetworkModelManager associated with the current callback instance.
        :param kwargs: a dict of parameters (dict){

            base parameters
            :param verbose: print stats if True. (bool)
            :param save_freq: The saving frequency (in epoch) of the model. (int)

            EarlyStopping parameters
            :param early_stopping: Enable early stopping if True. (bool)
            :param patience: Patience in epoch of the early stopping. (int)

            Learning rate decay
            :param learning_rate_decay_enabled: Enable learning rate decay if True. (bool)
            :param learning_rate_decay_factor: The learning rate decay applied every step. (float)
            :param learning_rate_decay_freq: learning rate decay frequency in epoch. (int)

        }
        """
        super().__init__()

        # base parameters
        self.network_manager = network_manager
        self.verbose = kwargs.get("verbose", True)
        self.save_freq = kwargs.get("save_freq", 1)

        # EarlyStopping parameters
        self.early_stopping_enabled = kwargs.get("early_stopping", True)
        self.early_stopping_patience = kwargs.get("patience", 100)
        self.early_stopping_triggered = False
        self.val_losses = self.network_manager.history.get("val", {}).get("loss", [])

        # Learning rate decay
        self.learning_rate_decay_enabled = kwargs.get("learning_rate_decay_enabled", True)
        self.learning_rate_decay_factor = kwargs.get("learning_rate_decay_factor", 0.5)
        self.learning_rate_decay_freq = kwargs.get("learning_rate_decay_freq", 10)
        self.init_lr = self.network_manager.model.optimizer.learning_rate

    def on_epoch_end(self, epoch, logs=None):
        self.network_manager.current_epoch = epoch

        self._saving_weights(epoch, logs)
        self._early_stopping_func(epoch, logs)
        self._learning_rate_decay_func(epoch, logs)

        self.network_manager.update_history(logs)

    def _saving_weights(self, epoch, logs=None):
        if epoch % self.save_freq == 0:
            if self.verbose:
                print(f"\n Epoch {epoch}: saving model to {self.network_manager.checkpoint_path} \n")
            self.network_manager.save_weights()

    def _early_stopping_func(self, epoch, logs=None):
        if self.early_stopping_enabled:
            self.val_losses.append(logs.get("val", {}).get("loss"))
            if len(self.val_losses) > self.early_stopping_patience \
                    and max(self.val_losses[-self.early_stopping_patience:]) == self.val_losses[-1]:
                self.early_stopping_triggered = True

    def _learning_rate_decay_func(self, epoch, logs=None):
        if self.learning_rate_decay_enabled:
            if epoch % self.learning_rate_decay_freq == 0:
                running_lr = self.init_lr * (self.learning_rate_decay_factor**int(epoch/self.learning_rate_decay_freq))
                self.network_manager.model.optimizer.learning_rate = running_lr


class SelfLearnerWithImgRotation(NetworkModelManager):
    default_backbone = "conv-4-64"

    def __init__(self, **kwargs):
        """
        Constructor of SelfLearnerWithImgRotation
        :param kwargs: a dict of parameters (dict){

            :param name: a string of the name of the network manage (str). This name is also used for the name of
            the directory that will contained all the data about this network.

            Optimizer parameters
            :param learning_rate: The learning of the optimizer. (float)
            :param momentum: The momentum of the optimizer. (float)
            :param use_nesterov: For SGD optimizer, use nesterov if True. (bool)
            :param optimizer_args: arguments to pass to the optimizer. (dict)
            :param optimizer: The type of the optimizer. (tensorflow.keras.optimizers)

            Others
            :param output_form: The ouput form of the network (util.OutputForm)
            :param teacher: The teacher of the current model.
            :param weights_path: The path of the initial weights for the model. (str)

            :param image_size: size of the input images. (int)
            :param backbone: The name of the backbone. Must be in NetworkModelManager.available_backbones. (str)
            :param backbones_args: Arguments of the given backbone fir its initialization. (dict)
            :param backbones_kwargs: Arguments of the given backbone fir its initialization. (dict)

            Self learning params
            :param sl_type: Type of the self-learning model. (util.SLBoostedType)
        }
        """
        super().__init__(**kwargs)
        self.img_size = kwargs.get("image_size", 84)
        self.output_form = util.OutputForm.ROT

        self._backbone = kwargs.get("backbone", SelfLearnerWithImgRotation.default_backbone)
        assert self._backbone in NetworkModelManager.available_backbones
        self._backbone_args = kwargs.get(
            "backbone_args",
            {
                "include_top": False,
                "weights": None,
                "input_shape": (self.img_size, self.img_size, 3)
            }
        )
        self._backbone_kwargs = kwargs.get("backbone_kwargs", {})
        self.sl_type = kwargs.get("sl_type", util.SLBoostedType.ROT)
        self._kwargs = kwargs

        self.init_model()

    def build(self):
        self.model = get_sl_model(
            backbone=self.available_backbones.get(self._backbone)(
                **self._backbone_args, **self._backbone_kwargs
            ),
            sl_boosted_type=self.sl_type,
            **self._kwargs
        )
        return self.model


class FewShotImgLearner(NetworkModelManager):
    default_backbone = "conv-4-64"

    class Method(enum.Enum):
        PrototypicalNet = 0,
        CosineNet = 1
        Gen0 = 2
        Gen1 = 3

    def __init__(self, **kwargs):
        """
        Constructor of FewShotImgLearner
        :param kwargs: a dict of parameters (dict){

            :param name: a string of the name of the network manage (str). This name is also used for the name of
            the directory that will contained all the data about this network.

            Optimizer parameters
            :param learning_rate: The learning of the optimizer. (float)
            :param momentum: The momentum of the optimizer. (float)
            :param use_nesterov: For SGD optimizer, use nesterov if True. (bool)
            :param optimizer_args: arguments to pass to the optimizer. (dict)
            :param optimizer: The type of the optimizer. (tensorflow.keras.optimizers)

            Others
            :param output_form: The ouput form of the network (util.OutputForm)
            :param teacher: The teacher of the current model.
            :param weights_path: The path of the initial weights for the model. (str)

            :param image_size: size of the input images. (int)
            :param channels: number of channels for the input images. (int)
            :param backbone: The name of the backbone. Must be in NetworkModelManager.available_backbones. (str)
            :param backbones_args: Arguments of the given backbone fir its initialization. (dict)
            :param backbones_kwargs: Arguments of the given backbone fir its initialization. (dict)

            Few shot learning params
            :param method: Type of the few-shot learning model. (FewShotImgLearner.Method)
            :param sl_boosted_type: Type of the self-learning auxiliary task. (util.SLBoostedType)
            :param sl_kwargs: Arguments of the self-learning add on. (dict)
        }
        """
        super().__init__(**kwargs)

        self.img_size = kwargs.get("image_size", 84)
        self.channels = kwargs.get("channels", 3)
        self.output_form = util.OutputForm.FS

        self._backbone = kwargs.get("backbone", FewShotImgLearner.default_backbone)
        assert self._backbone in NetworkModelManager.available_backbones
        self._backbone_args = kwargs.get(
            "backbone_args",
            {
                "include_top": False,
                "weights": None,
                "input_shape": (self.img_size, self.img_size, self.channels)
            }
        )
        self._backbone_kwargs = kwargs.get("backbone_kwargs", {})

        self.method = kwargs.get("method", FewShotImgLearner.Method.PrototypicalNet)

        self._methods_to_build = {
            FewShotImgLearner.Method.PrototypicalNet: self._build_proto_net,
            FewShotImgLearner.Method.CosineNet: self._build_cosine_net,
            FewShotImgLearner.Method.Gen0: self._build_gen0,
            FewShotImgLearner.Method.Gen1: self._build_gen1,
        }

        self.sl_add_on: util.SLBoostedType = kwargs.get("sl_boosted_type", None)
        self.sl_kwargs = kwargs.get("sl_kwargs", {})

        self.kwargs = kwargs

        self.init_model()

    def build(self):
        return self._methods_to_build.get(self.method)()

    def _build_proto_net(self):
        backbone_net = self.available_backbones.get(self._backbone)(**self._backbone_args, **self._backbone_kwargs)
        sl_model = get_sl_model(backbone_net, self.sl_add_on, **self.sl_kwargs) if self.sl_add_on is not None else None

        self.model = Prototypical(
            w=self.img_size,
            h=self.img_size,
            c=self.channels,
            backbone_net=backbone_net,
            sl_model=sl_model,
        )

        return self.model

    def _build_cosine_net(self):
        import logging
        tf.get_logger().setLevel(logging.ERROR)

        backbone_net = self.available_backbones.get(self._backbone)(**self._backbone_args, **self._backbone_kwargs)
        sl_model = get_sl_model(backbone_net, self.sl_add_on, **self.sl_kwargs) if self.sl_add_on is not None else None

        self.model = CosineClassifier(
            w=self.img_size,
            h=self.img_size,
            c=self.channels,
            backbone_net=backbone_net,
            sl_model=sl_model,
            **self.kwargs
        )

        return self.model

    def _build_gen0(self):
        import logging
        tf.get_logger().setLevel(logging.ERROR)

        backbone_net = self.available_backbones.get(self._backbone)(**self._backbone_args, **self._backbone_kwargs)

        self.model = Gen0(
            w=self.img_size,
            h=self.img_size,
            c=self.channels,
            backbone_net=backbone_net,
            **self.kwargs
        )

        return self.model

    def _build_gen1(self):
        import logging
        tf.get_logger().setLevel(logging.ERROR)

        backbone_net = self.available_backbones.get(self._backbone)(**self._backbone_args, **self._backbone_kwargs)

        self.model = Gen1(
            w=self.img_size,
            h=self.img_size,
            c=self.channels,
            backbone_net=backbone_net,
            **self.kwargs
        )

        return self.model


if __name__ == '__main__':
    self_learner = SelfLearnerWithImgRotation()
    self_learner.build_and_compile()
    self_learner.summary()
