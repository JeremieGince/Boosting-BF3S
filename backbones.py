import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Softmax, ReLU
from tensorflow.keras.models import Sequential


class Backbone(tf.keras.Model):
    def __init__(self, input_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs, training=None, mask=None):
        raise NotImplementedError()


class Conv_4_64_test(Backbone):
    # Encoder as ResNet like CNN with 4 blocks
    def __init__(self, input_shape, *args, **kwargs):
        super().__init__(input_shape, *args, **kwargs)
        self._core = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2, 2)),

            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2, 2)),

            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2, 2)),

            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2, 2)),
            Flatten()]
        )

    def call(self, inputs, training=None, mask=None):
        return self._core.call(inputs, training, mask)


def Conv_4_64(input_shape, *args, **kwargs):
    return tf.keras.Sequential([

            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2, 2)),

            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2, 2)),

            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2, 2)),

            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2, 2)),

            Flatten()],
            name="Conv-4-64"
        )

