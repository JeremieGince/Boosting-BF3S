import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Softmax, ReLU
from tensorflow.keras.models import Sequential


def conv_4_64(input_shape, *args, **kwargs):
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

            Flatten(dtype=tf.float32)
        ], name="conv-4-64"
        )


def conv_4_64_avg_pool(input_shape, *args, **kwargs):
    seq_0 = tf.keras.Sequential([

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
        ], name="conv-4-64"
        )
    # print("seq_0.(in, out)_shape", seq_0.input_shape, seq_0.output_shape)
    seq_1 = tf.keras.Sequential([
            tf.keras.layers.AveragePooling2D((2, 2), input_shape=seq_0.output_shape[1:]),
    ], name="conv-4-64-avg_pool"
    )
    # print("seq_1.(in, out)_shape", seq_1.input_shape, seq_1.output_shape)
    # assert 1 == 0
    return tf.keras.Sequential([seq_0, seq_1, Flatten(dtype=tf.float32)])


def conv_4_64_glob_avg_pool(input_shape, *args, **kwargs):
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
            tf.keras.layers.GlobalAveragePooling2D(),

            Flatten(dtype=tf.float32)
        ], name="Conv-4-64"
    )
