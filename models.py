import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Softmax, ReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

import util
from hyperparameters import *

os.environ["PATH"] += os.pathsep + r'C:\Program Files (x86)\Graphviz2.38\bin/'


class NetworkModelManager:
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

    }

    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "network_model")
        os.makedirs("training_data/" + self.name, exist_ok=True)
        self.checkpoint_path = "training_data/" + self.name + "/cp-weights.h5"
        self.history_path = f"training_data/{self.name}/cp-history.json"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.history = dict()
        self.model = None
        self.current_epoch = 0

        self.output_form: util.OutputForm = kwargs.get("output_form", util.OutputForm.LABEL)

    def summary(self):
        return self.model.summary()

    def load_weights(self):
        assert self.model is not None
        self.model.load_weights(self.checkpoint_path)

    def save_weights(self):
        self.model.save(self.checkpoint_path)

    def load_history(self):
        import json
        if os.path.exists(self.history_path):
            self.history = json.load(open(self.history_path, 'r'))
        self.update_curr_epoch()

    def save_history(self):
        import json

        json.dump(self.history, open(self.history_path, 'w'))

    def update_curr_epoch(self):
        self.current_epoch = len(self.history.get("loss", []))

    def update_history(self, other: dict):
        temp = {**self.history, **other}
        for key, value in temp.items():
            if key in self.history and key in other:
                if isinstance(value, list):
                    temp[key] = list(np.array(self.history[key] + value, dtype=float))
                elif isinstance(value, np.ndarray):
                    temp[key] = list(np.array(list(self.history[key]) + list(value), dtype=float))
                else:
                    temp[key] = list(np.array([self.history[key], value], dtype=float))
            else:
                temp[key] = list(value)

        self.history = temp
        self.save_history()

    def load(self):
        self.load_weights()
        self.load_history()

    def save(self):
        self.save_weights()
        self.save_history()

    def build(self):
        raise NotImplementedError

    def compile(self):
        raise NotImplementedError

    def build_and_compile(self):
        self.model = self.build()
        self.model = self.compile()

        if len(os.listdir(self.checkpoint_dir)) == 0:
            self.save_weights()
        return self.model


class NetworkManagerCallback(tf.keras.callbacks.Callback):
    def __init__(self, network_manager: NetworkModelManager, **kwargs):
        super().__init__()
        self.network_manager = network_manager
        self.verbose = kwargs.get("verbose", True)
        self.save_freq = kwargs.get("save_freq", 1)

    def on_epoch_end(self, epoch, logs=None):
        self.network_manager.current_epoch = epoch
        if epoch % self.save_freq == 0:
            if self.verbose:
                print(f"\n Epoch {epoch}: saving model to {self.network_manager.checkpoint_path} \n")
            self.network_manager.save_weights()

        self.network_manager.update_history({k: [v] for k, v in logs.items()})


class SelfLearnerWithImgRotation(NetworkModelManager):
    default_backbone = "InceptionResNetV2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.img_size = kwargs.get("image_size", 80)
        self.learning_rate = kwargs.get("learning_rate", 1e-3)
        self.momentum = kwargs.get("momentum", CLS_MOMENTUM)
        self.output_size = kwargs.get("output_size", 1)
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

        self._nb_hidden_layer: int = kwargs.get("nb_hidden_layers", 1)
        self._hidden_neurons: list = kwargs.get("hidden_neurons", [4096 for _ in range(self._nb_hidden_layer)])

        assert self._nb_hidden_layer == len(self._hidden_neurons)

        self.loss_function = SGD(self.learning_rate, momentum=self.momentum, nesterov=CLS_USE_NESTEROV)

    def build(self):
        self.model = Sequential(
            [
                self.available_backbones.get(self._backbone)(
                    **self._backbone_args, **self._backbone_kwargs
                ),
                Flatten(),
                ReLU(),
                *[
                    Sequential([
                        Dense(h),
                        BatchNormalization(),
                        ReLU(),
                    ], name="Dense_Block")
                    for h in self._hidden_neurons
                ],
                Dense(self.output_size, name="output_layer"),
                Softmax(),
            ]
        )

        return self.model

    def compile(self):
        assert self.model is not None
        self.model.compile(
            optimizer=self.loss_function,
            loss=tf.keras.losses.binary_crossentropy,
            metrics=[
                'accuracy',
            ]
        )
        return self.model


class FewShotImgLearner(NetworkModelManager):
    default_backbone = "InceptionResNetV2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.img_size = kwargs.get("image_size", 80)
        self.learning_rate = kwargs.get("learning_rate", 1e-3)
        self.momentum = kwargs.get("momentum", CLS_MOMENTUM)
        self.output_size = kwargs.get("output_size", 1)

        self._backbone = kwargs.get("backbone", FewShotImgLearner.default_backbone)
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

        self.loss_function = SGD(self.learning_rate, momentum=self.momentum, nesterov=CLS_USE_NESTEROV)

    def build(self):
        self.model = Sequential(
            [
                self.available_backbones.get(self._backbone)(
                    **self._backbone_args, **self._backbone_kwargs
                ),
                Flatten(),
                BatchNormalization(),
                Dense(1024),
                BatchNormalization(),
                Dense(self.output_size),
                Softmax(),
            ]
        )

        return self.model

    def compile(self):
        assert self.model is not None
        self.model.compile(
            optimizer=self.loss_function,
            loss=tf.keras.losses.binary_crossentropy,
            metrics=[
                'accuracy',
            ]
        )
        return self.model


if __name__ == '__main__':
    self_learner = SelfLearnerWithImgRotation()
    self_learner.build_and_compile()
    self_learner.summary()

    self_learner.update_history({"loss": [0.16546546874, ], "accuracy": np.array([0.156496848, ])})
    self_learner.save_history()
    self_learner.load_history()
    print(self_learner.history)

