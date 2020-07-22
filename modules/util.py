import enum


def plotHistory_old(history: dict, **kwargs):
    import os
    import matplotlib.pyplot as plt
    # print(history.keys())
    acc = history['accuracy']
    val_acc = history.get('val_accuracy')

    loss = history['loss']
    val_loss = history.get('val_loss')

    epochs_range = range(1, len(acc)+1)

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    if val_acc is not None:
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'Training {"and Validation" if val_acc is not None else ""} Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    if val_loss is not None:
        plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'Training {"and Validation" if val_loss is not None else ""} Loss')

    if kwargs.get("savefig", True):
        os.makedirs("Figures/", exist_ok=True)
        plt.savefig(f"Figures/{kwargs.get('savename', 'training_curve')}.png", dpi=500)
    plt.show()


def plotHistory(history: dict, **kwargs):
    import os
    import matplotlib.pyplot as plt

    assert len(history) > 0

    metrics_to_phases = {_metric_name: {} for _metric_name in history[list(history.keys())[0]]}

    for _p in history:
        for _metric_name, hist_vec in history[_p].items():
            metrics_to_phases[_metric_name][_p] = hist_vec

    plt.figure(figsize=(12, 12))

    for i, metric_name in enumerate(metrics_to_phases):
        plt.subplot(1, len(metrics_to_phases.keys()), i+1)

        for _phase, values in metrics_to_phases[metric_name].items():
            plt.plot(values, label=_phase)

        plt.legend(loc='lower right')
        plt.title(f'{metric_name}')

    if kwargs.get("savefig", True):
        os.makedirs("Figures/", exist_ok=True)
        plt.savefig(f"Figures/{kwargs.get('savename', 'training_curve')}.png", dpi=500)
    plt.show()


def load_pickle_data(file):
    import pickle

    with open(file, "rb") as fo:
        data = pickle.load(fo, encoding="bytes")
        try:
            if isinstance(data, dict):
                data = {k.decode("ascii"): v for k, v in data.items()}
        except:
            pass

    return data


def data_generator_from_pickle(filename):
    meta = load_pickle_data(filename)
    print(meta.keys())
    print(len(meta["data"]))

    for idx, (data, label) in enumerate(zip(meta["data"], meta["labels"])):
        yield idx, (data, label)


def build_label_index(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


def iterate_dataset(dataset, nb_batch: int = None):
    import tensorflow as tf

    if nb_batch is None:
        nb_batch = 1

    if not tf.executing_eagerly():
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        with tf.Session() as sess:
            for idx in range(nb_batch):
                yield idx, sess.run(next_element)
    else:
        for idx, episode in enumerate(dataset):
            if idx == nb_batch:
                break
            yield idx, episode


def plot_batch(images, labels, size_multiplier=1):
    import numpy as np
    import matplotlib.pyplot as plt

    num_examples = len(labels)
    figwidth = np.ceil(np.sqrt(num_examples)).astype('int32')
    figheight = num_examples // figwidth
    figsize = (figwidth * size_multiplier, (figheight + 1.5) * size_multiplier)
    _, axarr = plt.subplots(figwidth, figheight, dpi=300, figsize=figsize)

    for i, ax in enumerate(axarr.transpose().ravel()):
        # Images are between -1 and 1.
        ax.imshow(images[i] / 2 + 0.5)
        ax.set(xlabel=labels[i], xticks=[], yticks=[])
    plt.show()


def convert_label_batch_to_rot_batch(images_batch, possible_rotations_to_one_hot: dict):
    import numpy as np
    from scipy import ndimage

    rn_rotations = np.random.choice(np.array(list(possible_rotations_to_one_hot.keys())), size=images_batch.shape[0])

    def _rotate(_p):
        idx, img = _p
        return ndimage.rotate(img, rn_rotations[idx], reshape=False)

    rot_images = map(_rotate, list(enumerate(images_batch)))

    return np.array(list(rot_images)), np.array(list(map(lambda e: possible_rotations_to_one_hot[e], rn_rotations)))


def c_idx2one_hot(idx, arr):
    arr[idx] = 1
    return arr


class TrainingPhase(enum.Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class OutputForm(enum.Enum):
    LABEL = 0
    ROT = 1
    FS = 2  # Few Shot
    FS_SL = 3  # Few Shot and Self-Learning


def calc_euclidian_dists(x, y):
    import tensorflow as tf
    """
    Calculate euclidian distance between two 3D tensors.
    Args:
        x (tf.Tensor):
        y (tf.Tensor):
    Returns (tf.Tensor): 2-dim tensor with distances.
    """
    n = x.shape[0]
    m = y.shape[0]
    x = tf.tile(tf.expand_dims(x, 1), [1, m, 1])
    y = tf.tile(tf.expand_dims(y, 0), [n, 1, 1])
    return tf.reduce_mean(tf.math.pow(x - y, 2), 2)


def calc_cosine_dists(x, y):
    import tensorflow as tf
    """
    Calculate cosine distance between two 3D tensors.
    Args:
        x (tf.Tensor):
        y (tf.Tensor):
    Returns (tf.Tensor): 2-dim tensor with distances.
    """
    n = x.shape[0]
    m = y.shape[0]
    x = tf.tile(tf.expand_dims(x, 1), [1, m, 1])
    y = tf.tile(tf.expand_dims(y, 0), [n, 1, 1])

    return tf.reduce_mean(tf.multiply(tf.math.l2_normalize(x), tf.math.l2_normalize(y)), 2)


class SLBoostedType(enum.Enum):
    ROT = 0


def get_str_repr_for_config(config: dict):
    _str = ""
    for sec, sec_dict in config.items():
        _str += get_str_repr_for_sec_config(config, sec)
    return _str


def get_str_repr_for_secs_config(config: dict, secs: list):
    _str = ""
    for sec in secs:
        _str += get_str_repr_for_sec_config(config, sec)
    return _str


def get_str_repr_for_sec_config(config: dict, sec: str):
    assert sec in config.keys(), f"The param: 'sec' must be in {config.keys()}"
    _str = ""

    sec_dict = config[sec]
    _str += '-' * 25 + '\n'
    _str += str(sec) + '\n'
    _str += '-' * 25 + '\n'

    if isinstance(sec_dict, dict):
        for param, value in sec_dict.items():
            _str += f"{param}: {value} \n"
    else:
        _str += f"{sec_dict} \n"

    _str += '\n'
    return _str

