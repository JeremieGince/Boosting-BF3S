import enum
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import ImageFile
from scipy import ndimage
import warnings

import modules.util as util
from config.user_constants import MINIIMAGETNET_DIR
from modules.util import OutputForm, TrainingPhase

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DatasetBase:
    """
    Base Dataset, used to manage data
    """
    def __init__(
            self,
            data_dir: str,
            **kwargs
    ):
        """
        Constructor of BaseDataset
        :param data_dir: The path to the root of the dataset. (str)
        :param kwargs: {
            :param name: name of the current dataset (str)
            :param labels: labels of the current dataset.
        }
        """
        self.name = kwargs.get("name", "Dataset")
        self.data_dir = data_dir

        self._batch_size = kwargs.get("batch_size", 1)

        self.labels: set = kwargs.get("labels", set([]))
        self._labels_to_one_hot: dict = None if self.labels is None else {
            lbl: util.c_idx2one_hot(idx, np.zeros(len(self.labels), dtype=int))
            for idx, lbl in enumerate(self.labels)
        }
        self._one_hot_to_labels = {
            str(_oh): lbl
            for lbl, _oh in self._labels_to_one_hot.items()
        }

        self.possible_rotations: set = kwargs.get("possible_rotations", {0, 90, 180, 270})
        self.possible_rotations_to_one_hot = {
            rot: util.c_idx2one_hot(idx, np.zeros(len(self.possible_rotations), dtype=int))
            for idx, rot in enumerate(self.possible_rotations)
        }

        self._train_length: int = 0
        self._val_length: int = 0
        self._test_length: int = 0

    @property
    def train_length(self) -> int:
        return self._train_length

    @property
    def val_length(self) -> int:
        return self._val_length

    @property
    def test_length(self) -> int:
        return self._test_length

    @property
    def batch_size(self):
        return self._batch_size

    def get_output_size(self, output_form: OutputForm = OutputForm.LABEL):
        warnings.warn(DeprecationWarning)
        if output_form == OutputForm.LABEL:
            return len(self.labels)

        elif output_form == OutputForm.ROT:
            return len(self.possible_rotations)

    def get_batch_generator(self, phase: TrainingPhase, output_form: OutputForm = OutputForm.LABEL, **kwargs):
        """
        Get the batch generator of the current dataset.
        :param phase: The training phase of the generator. (TrainingPhase)
        :param output_form: The output_form og the generator. (OutputForm)
        :param kwargs: other parameters
        :return: a data generator
        """
        raise NotImplementedError()

    def get_iterator(self, phase: TrainingPhase, output_form: OutputForm = OutputForm.LABEL, **kwargs):
        warnings.warn(DeprecationWarning)
        return iter(self.get_batch_generator(phase, output_form, **kwargs))

    def get_few_shot_generator(self, _n_way, _n_shot, _n_query,
                               phase: TrainingPhase,
                               output_form: OutputForm = OutputForm.FS,
                               **kwargs):
        """
        Get the episodic generator of the current dataset.
        :param _n_way: number of base class. (int)
        :param _n_shot: number of images per class. (int)
        :param _n_query: number of query per class. (int)
        :param phase: The training phase of the generator. (TrainingPhase)
        :param output_form: The output_form og the generator. (OutputForm)
        :param kwargs: other parameters
        :return: a data generator
        """
        warnings.warn("get_few_shot_generator not Implemented yet")

    def preprocess_input(self, _input) -> np.ndarray:
        """
        Root method to preprocess the current set of data
        :param _input: The current data (np.ndarray)
        :return: The preprocess data (np.ndarray)
        """
        return _input

    def augment_data(self, _data) -> np.ndarray:
        """
        Root method to augment the current set of data
        :param _data: The current data (np.ndarray)
        :return: The augmented data (np.ndarray)
        """
        return _data

    def _add_labels(self, _labels):
        warnings.warn(DeprecationWarning)
        self.labels = set.union(self.labels, set(_labels))
        self._labels_to_one_hot = {
            lbl: util.c_idx2one_hot(idx, np.zeros(len(self.labels), dtype=int))
            for idx, lbl in enumerate(self.labels)
        }
        self._one_hot_to_labels = {
            str(_oh): lbl
            for lbl, _oh in self._labels_to_one_hot.items()
        }

    def get_label_from_one_hot(self, one_hot):
        warnings.warn(DeprecationWarning)
        return self._one_hot_to_labels.get(str(np.array(one_hot)))

    def plot_samples(self, nb_samples=5):
        """
        Method use to show samples of the current dataset
        :param nb_samples: number of sample to show (int)
        :return: None
        """
        nb_samples = min(nb_samples, self._batch_size)

        _itr = self.get_iterator(TrainingPhase.TRAIN, OutputForm.LABEL, shuffle=True)

        (_images, _labels) = next(_itr)

        _labels = list(map(self.get_label_from_one_hot, _labels))

        fig, axes = plt.subplots(1, nb_samples, figsize=(10, 10))
        for i in range(nb_samples):
            axes[i].imshow(np.array(_images[i]))
            axes[i].axis('off')
            axes[i].set_title(np.array(_labels[i]))

        plt.tight_layout(pad=0.0)
        os.makedirs("Figures/", exist_ok=True)
        plt.savefig(f"Figures/samples_of_{self.name.replace(' ', '_')}_train.png", dpi=500)
        plt.show()


class MiniImageNetDataset(DatasetBase):
    """
    The dataset class to manage Mini-ImageNet
    """
    def __init__(
            self,
            data_dir=MINIIMAGETNET_DIR,
            **kwargs
    ):
        """
        The constructor of MiniImageNetDataset
        :param data_dir: The current path to the root of the dataset
        :param kwargs: {
            :param name: name of the dataset. (str)
        }
        """
        super().__init__(data_dir, **kwargs)

        self.name = kwargs.get("name", "MiniImageNet Dataset")

        self.train_file = os.path.join(data_dir, "mini-imagenet-cache-train.pkl")
        self.val_file = os.path.join(data_dir, "mini-imagenet-cache-val.pkl")
        self.test_file = os.path.join(data_dir, "mini-imagenet-cache-test.pkl")

        self.phase_to_file: dict = {
            TrainingPhase.TRAIN: self.train_file,
            TrainingPhase.VAL: self.val_file,
            TrainingPhase.TEST: self.test_file,
        }

        self.image_size = 84

    def preprocess_input(self, _input):
        nb_dims = len(_input.shape)
        assert nb_dims >= 3

        _input = _input/255.0

        x_channels_ids = [i for i in range(nb_dims-3, nb_dims)]
        x_means = [np.mean(_input[i]) for i in x_channels_ids]
        x_stds = [max(np.std(_input[i]), 1.0/np.sqrt(_input.shape[i])) for i in x_channels_ids]

        for i, c_id in enumerate(x_channels_ids):
            _input[c_id] = (_input[c_id] - x_means[i]) / x_stds[i]
        return _input

    def augment_data(self, _data):
        assert len(_data.shape) == 4 or len(_data.shape) == 3
        with tf.device("CPU:0"):
            _data = tf.image.random_flip_left_right(_data)
        return _data

    def get_batch_generator(self, phase: TrainingPhase, output_form: OutputForm = OutputForm.LABEL, **kwargs):
        _raw_data = util.load_pickle_data(self.phase_to_file.get(phase))

        # Convert original data to format [n_classes, n_img, w, h, c]
        first_key = list(_raw_data['class_dict'])[0]
        _data = np.zeros((len(_raw_data['class_dict']), len(_raw_data['class_dict'][first_key]), 84, 84, 3))
        for i, (k, v) in enumerate(_raw_data['class_dict'].items()):
            _data[i, :, :, :, :] = _raw_data['image_data'][v, :]
        # print(_data.shape)
        # if phase == TrainingPhase.TRAIN:
        #     _data = np.array(list(map(lambda b: self.augment_data(b), _data)))
        # print(_data.shape)
        _data = self.preprocess_input(_data)
        n_classes, n_img, _w, _h, _c = _data.shape

        if phase == TrainingPhase.TRAIN:
            self._train_length = _data.shape[0] * _data.shape[1]
        elif phase == TrainingPhase.VAL:
            self._val_length = _data.shape[0] * _data.shape[1]
        elif phase == TrainingPhase.TEST:
            self._test_length = _data.shape[0] * _data.shape[1]

        # self._add_labels(_data[0])

        def _gen():
            while True:
                x_batch = np.zeros([n_classes, _w, _h, _c], dtype=np.float32)
                ids_batch = np.zeros([n_classes, 1], dtype=np.int32)
                y_batch = np.zeros([n_classes, n_classes], dtype=np.int32)

                for _i, i_class in enumerate(range(n_classes)):
                    selected = np.random.permutation(n_img)[0]
                    x_batch[_i] = _data[i_class, selected]
                    ids_batch[_i] = i_class

                y_batch = tf.cast(
                    tf.one_hot(tf.convert_to_tensor(ids_batch.squeeze(), tf.int32), n_classes),
                    tf.int32
                ).numpy()

                yield x_batch, ids_batch, y_batch

        output_shapes = (tf.TensorShape([None, self.image_size, self.image_size, 3]),
                         tf.TensorShape([n_classes, 1]),
                         tf.TensorShape([n_classes, n_classes]))

        _ds = tf.data.Dataset.from_generator(
            _gen,
            output_types=(tf.float32, tf.int32, tf.int32),
            output_shapes=output_shapes,
        )

        return _ds.prefetch(tf.data.experimental.AUTOTUNE)

    def get_few_shot_generator(self, _n_way, _n_shot, _n_query,
                               phase: TrainingPhase,
                               output_form: OutputForm = OutputForm.FS,
                               **kwargs):
        _raw_data = util.load_pickle_data(self.phase_to_file.get(phase))

        # Convert original data to format [n_classes, n_img, w, h, c]
        first_key = list(_raw_data['class_dict'])[0]
        _data = np.zeros((len(_raw_data['class_dict']), len(_raw_data['class_dict'][first_key]), 84, 84, 3))
        for i, (k, v) in enumerate(_raw_data['class_dict'].items()):
            _data[i, :, :, :, :] = _raw_data['image_data'][v, :]

        # if phase == TrainingPhase.TRAIN:
        #     _data = np.array(map(lambda b: self.augment_data(b), _data))

        _data = self.preprocess_input(_data)
        # print(f"data.shape: {_data.shape}")
        n_classes, n_img, _w, _h, _c = _data.shape

        def _gen():
            while True:
                support = np.zeros([_n_way, _n_shot, _w, _h, _c], dtype=np.float32)
                query = np.zeros([_n_way, _n_query, _w, _h, _c], dtype=np.float32)
                classes_ep = np.random.permutation(n_classes)[:_n_way]

                for _i, i_class in enumerate(classes_ep):
                    selected = np.random.permutation(n_img)[:_n_shot + _n_query]
                    support[_i] = _data[i_class, selected[:_n_shot]]
                    query[_i] = _data[i_class, selected[_n_shot:]]

                yield support
                yield query

        output_types = tf.float32
        output_shapes = tf.TensorShape([_n_way, None, _w, _h, _c])

        _ds = tf.data.Dataset.from_generator(
            _gen,
            output_types=output_types,
            output_shapes=output_shapes,
        )

        return _ds.prefetch(tf.data.experimental.AUTOTUNE)

    def __iter__(self):
        # TODO: change ça pour pas que ce soit juste train
        warnings.warn(DeprecationWarning)
        return self.get_batch_generator(TrainingPhase.TRAIN)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys
    from tensorflow.keras import layers

    mini_imagenet_dataset = MiniImageNetDataset(data_dir=r"D:\Datasets\mini-imagenet")
    # mini_imagenet_dataset.plot_samples()
    #
    mini_gen = mini_imagenet_dataset.get_few_shot_generator(
        _n_way=6, _n_shot=5, _n_query=4,
        phase=TrainingPhase.TRAIN)

    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    ])

    mini_augmented_gen = mini_gen.ds.map(lambda x: data_augmentation(x, training=True),
                                         num_parallel_calls=tf.data.experimental.AUTOTUNE)

    mini_gen_itr = iter(mini_gen)
    mini_augmented_gen_itr = iter(mini_augmented_gen)
    # print(mini_gen, mini_gen_itr, next(mini_gen_itr), sep='\n')

    support = next(mini_gen_itr)
    print(f"support.shape: {support.shape}")
    query = next(mini_gen_itr)
    print(f"query.shape: {query.shape}")

    support_aug = next(mini_augmented_gen_itr)
    print(f"support_aug.shape: {support_aug.shape}")
    query_aug = next(mini_augmented_gen_itr)
    print(f"query_aug.shape: {query_aug.shape}")

    _l = [support[0][0], support_aug[0][0]]
    _lbl = ["support", "support_aug", ]
    fig = plt.figure()
    axes = fig.subplots(1, len(_l))

    for i, img in enumerate(_l):
        axes[i].imshow(np.squeeze(img))
        axes[i].set_title(_lbl[i])

    plt.show()
