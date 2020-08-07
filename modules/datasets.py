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
    def __init__(
            self,
            data_dir,
            **kwargs
    ):
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
        if output_form == OutputForm.LABEL:
            return len(self.labels)

        elif output_form == OutputForm.ROT:
            return len(self.possible_rotations)

    def get_generator(self, phase: TrainingPhase, output_form: OutputForm = OutputForm.LABEL, **kwargs):
        raise NotImplementedError()

    def get_iterator(self, phase: TrainingPhase, output_form: OutputForm = OutputForm.LABEL, **kwargs):
        return iter(self.get_generator(phase, output_form, **kwargs))

    def get_few_shot_generator(self, _n_way, _n_shot, _n_query,
                               phase: TrainingPhase,
                               output_form: OutputForm = OutputForm.FS,
                               **kwargs):
        warnings.warn("get_few_shot_generator not Implemented yet")

    def preprocess_input(self, _input):
        return _input

    def _add_labels(self, _labels):
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
        return self._one_hot_to_labels.get(str(np.array(one_hot)))

    def plot_samples(self, nb_samples=5):
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
    def __init__(
            self,
            data_dir=MINIIMAGETNET_DIR,
            **kwargs
    ):
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
        # return tf.image.resize(_input/255, (self.image_size, self.image_size))
        return tf.image.per_image_standardization(_input/255.0)  # computes (x - mean) / adjusted_stddev

    def get_generator(self, phase: TrainingPhase, output_form: OutputForm = OutputForm.LABEL, **kwargs):
        _raw_data = util.load_pickle_data(self.phase_to_file.get(phase))

        # Convert original data to format [n_classes, n_img, w, h, c]
        first_key = list(_raw_data['class_dict'])[0]
        _data = np.zeros((len(_raw_data['class_dict']), len(_raw_data['class_dict'][first_key]), 84, 84, 3))
        for i, (k, v) in enumerate(_raw_data['class_dict'].items()):
            _data[i, :, :, :, :] = self.preprocess_input(_raw_data['image_data'][v, :])

        # _data = self.preprocess_input(_data)
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
            _data[i, :, :, :, :] = self.preprocess_input(_raw_data['image_data'][v, :])

        # _data = self.preprocess_input(_data)
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
        return self.get_generator(TrainingPhase.TRAIN)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys
    mini_imagenet_dataset = MiniImageNetDataset(data_dir=r"D:\Datasets\mini-imagenet")
    # mini_imagenet_dataset.plot_samples()
    #
    mini_gen = mini_imagenet_dataset.get_few_shot_generator(
        _n_way=30, _n_shot=5, _n_query=5,
        phase=TrainingPhase.TRAIN, output_form=OutputForm.FS_SL)

    print(mini_gen, iter(mini_gen), next(iter(mini_gen)), sep='\n')

    support, query, [sl_x, sl_y, sl_test_x, sl_test_y] = next(iter(mini_gen))

    b = [e.nbytes for e in [support, query, sl_x, sl_y, sl_test_x, sl_test_y]]
    print(b, sum(b))

    # _l = [support, query, sl_x, sl_test_x]
    # _lbl = ["support", "query", str(sl_y), str(sl_test_y)]
    # fig = plt.figure()
    # axes = fig.subplots(1, len(_l))
    #
    # for i, img in enumerate(_l):
    #     axes[i].imshow(np.squeeze(img))
    #     axes[i].set_title(_lbl[i])
    #
    # plt.show()
