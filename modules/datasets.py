import enum
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import ImageFile
from scipy import ndimage
import warnings

import modules.util as util
from modules.hyperparameters import SEED
from modules.user_constants import MINIIMAGETNET_DIR
from modules.util import OutputForm, TrainingPhase

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DatasetBase:
    def __init__(
            self,
            data_dir,
            **kwargs
    ):
        tf.random.set_seed(SEED)
        np.random.seed(SEED)

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

        # self.file_train_categories_train_phase = os.path.join(
        #     data_dir, "miniImageNet_category_split_train_phase_train.pickle"
        # )
        # self.file_train_categories_val_phase = os.path.join(
        #     data_dir, "miniImageNet_category_split_train_phase_val.pickle"
        # )
        # self.file_train_categories_test_phase = os.path.join(
        #     data_dir, "miniImageNet_category_split_train_phase_test.pickle"
        # )
        # self.file_val_categories_val_phase = os.path.join(
        #     data_dir, "miniImageNet_category_split_val.pickle"
        # )
        # self.file_test_categories_test_phase = os.path.join(
        #     data_dir, "miniImageNet_category_split_test.pickle"
        # )

        self.train_file = os.path.join(data_dir, "mini-imagenet-cache-train.pkl")
        self.val_file = os.path.join(data_dir, "mini-imagenet-cache-val.pkl")
        self.test_file = os.path.join(data_dir, "mini-imagenet-cache-test.pkl")

        # TODO: comprendre les phases bizzare de mini-imagenet, genre train-train, val-val, train-val,
        #  train-test, test-test...

        # self.phase_to_file: dict = {
        #     TrainingPhase.TRAIN: self.file_train_categories_train_phase,
        #     TrainingPhase.VAL: self.file_val_categories_val_phase,
        #     TrainingPhase.TEST: self.file_test_categories_test_phase,
        # }

        self.phase_to_file: dict = {
            TrainingPhase.TRAIN: self.train_file,
            TrainingPhase.VAL: self.val_file,
            TrainingPhase.TEST: self.test_file,
        }

        self.image_size = 84

    def preprocess_input(self, _input):
        # return tf.image.resize(_input/255, (self.image_size, self.image_size))
        return _input/255.0

    def get_generator(self, phase: TrainingPhase, output_form: OutputForm = OutputForm.LABEL, **kwargs):
        _data = util.load_pickle_data(self.phase_to_file.get(phase))

        if phase == TrainingPhase.TRAIN:
            self._train_length = len(_data["data"])
        elif phase == TrainingPhase.VAL:
            self._val_length = len(_data["data"])
        elif phase == TrainingPhase.TEST:
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
            output_types=(tf.float32, tf.int32),
            output_shapes=(tf.TensorShape([self.image_size, self.image_size, 3]),
                           tf.TensorShape([self.get_output_size(output_form)])),
        )

        if kwargs.get("shuffle", True):
            _ds = _ds.shuffle(buffer_size=len(_data["labels"]), seed=SEED)

        _ds = _ds.batch(self._batch_size)

        return _ds

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

        # print(_data.shape, _data[..., 0])
        _data = self.preprocess_input(_data)
        # print(_data.shape, _data[..., 0])
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

                if output_form == OutputForm.FS:
                    yield support, query
                elif output_form == OutputForm.FS_SL:
                    support_reshape = np.reshape(support, newshape=[_n_way*_n_shot, _w, _h, _c])
                    query_reshape = np.reshape(query, newshape=[_n_way * _n_query, _w, _h, _c])

                    sl_y_r = np.random.choice(list(self.possible_rotations), support_reshape.shape[0])
                    sl_x = np.array(list(map(
                        lambda _t: ndimage.rotate(_t[0], _t[1], reshape=False),
                        zip(support_reshape, sl_y_r)
                    )))
                    sl_y = np.array(list(map(
                        lambda _r: self.possible_rotations_to_one_hot[_r],
                        sl_y_r
                    )))

                    sl_test_y_r = np.random.choice(list(self.possible_rotations), query_reshape.shape[0])
                    sl_test_x = np.array(list(map(
                        lambda _t: ndimage.rotate(_t[0], _t[1], reshape=False),
                        zip(query_reshape, sl_test_y_r)
                    )))
                    sl_test_y = np.array(list(map(
                        lambda _r: self.possible_rotations_to_one_hot[_r],
                        sl_test_y_r
                    )))

                    yield support, query, [sl_x, sl_y, sl_test_x, sl_test_y]
                else:
                    raise NotImplementedError()

        return _gen()

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
