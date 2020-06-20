import enum


def load_pickle_data(file):
    import pickle

    with open(file, "rb") as fo:
        data = pickle.load(fo, encoding="bytes")
        if isinstance(data, dict):
            data = {k.decode("ascii"): v for k, v in data.items()}

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


class OutputForm(enum.Enum):
    LABEL = 0
    ROT = 1
