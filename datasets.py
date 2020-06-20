import enum
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import ImageFile
from scipy import ndimage

import util
from hyperparameters import SEED
from user_constants import MINIIMAGETNET_DIR
from util import OutputForm

ImageFile.LOAD_TRUNCATED_IMAGES = True

tf.random.set_seed(SEED)


class DatasetPhase(enum.Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


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

    def get_generator(self, phase: DatasetPhase, output_form: OutputForm = OutputForm.LABEL, **kwargs):
        raise NotImplementedError()

    def get_iterator(self, phase: DatasetPhase, output_form: OutputForm = OutputForm.LABEL, **kwargs):
        return iter(self.get_generator(phase, output_form, **kwargs))

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

        _itr = self.get_iterator(DatasetPhase.TRAIN, OutputForm.LABEL, shuffle=True)

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

        self.file_train_categories_train_phase = os.path.join(
            data_dir, "miniImageNet_category_split_train_phase_train.pickle"
        )
        self.file_train_categories_val_phase = os.path.join(
            data_dir, "miniImageNet_category_split_train_phase_val.pickle"
        )
        self.file_train_categories_test_phase = os.path.join(
            data_dir, "miniImageNet_category_split_train_phase_test.pickle"
        )
        self.file_val_categories_val_phase = os.path.join(
            data_dir, "miniImageNet_category_split_val.pickle"
        )
        self.file_test_categories_test_phase = os.path.join(
            data_dir, "miniImageNet_category_split_test.pickle"
        )

        # TODO: comprendre les phases bizzare de mini-imagenet, genre train-train, val-val, train-val,
        #  train-test, test-test...

        self.phase_to_file: dict = {
            DatasetPhase.TRAIN: self.file_train_categories_train_phase,
            DatasetPhase.VAL: self.file_val_categories_val_phase,
            DatasetPhase.TEST: self.file_test_categories_test_phase,
        }

        self.image_size = kwargs.get("image_size", 80)

    def preprocess_input(self, _input):
        return tf.image.resize(_input, (self.image_size, self.image_size))

    def get_generator(self, phase: DatasetPhase, output_form: OutputForm = OutputForm.LABEL, **kwargs):
        _data = util.load_pickle_data(self.phase_to_file.get(phase))

        if phase == DatasetPhase.TRAIN:
            self._train_length = len(_data["data"])
        elif phase == DatasetPhase.VAL:
            self._val_length = len(_data["data"])
        elif phase == DatasetPhase.TEST:
            self._test_length = len(_data["data"])

        self._add_labels(_data["labels"])

        def _gen():
            for _idx, (_image, _label) in enumerate(zip(_data["data"], _data["labels"])):
                _image = self.preprocess_input(_image)

                if output_form == OutputForm.LABEL:
                    yield _image, self._labels_to_one_hot[_label]

                elif output_form == OutputForm.ROT:
                    _rot = np.random.choice(list(self.possible_rotations))
                    yield ndimage.rotate(_image, _rot, reshape=False), self.possible_rotations_to_one_hot[_rot]

        _ds = tf.data.Dataset.from_generator(
            _gen,
            output_types=(tf.int32, tf.int32),
            output_shapes=(tf.TensorShape([self.image_size, self.image_size, 3]),
                           tf.TensorShape([self.get_output_size(output_form)])),
        )

        if kwargs.get("shuffle", True):
            _ds = _ds.shuffle(buffer_size=len(_data["labels"]), seed=SEED)

        _ds = _ds.batch(self._batch_size)

        return _ds

    def __iter__(self):
        # TODO: change ça pour pas que ce soit juste train
        return self.get_generator(DatasetPhase.TRAIN)


if __name__ == '__main__':
    mini_imagenet_dataset = MiniImageNetDataset(batch_size=64)
    mini_imagenet_dataset.plot_samples()

    mini_gen = mini_imagenet_dataset.get_generator(phase=DatasetPhase.TRAIN, output_form=OutputForm.ROT, shuffle=True)

    print(mini_gen)
    print(iter(mini_gen))
