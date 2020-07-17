from modules.datasets import MiniImageNetDataset
from modules.modelManagers import FewShotImgLearner
from modules.trainers import FewShotTrainer
from modules.modelManagers import NetworkManagerCallback
import modules.util as util
from config.prototypical_config import config as proto_config
from config.prototypical_rotation_config import config as proto_rot_config
from config.cosine_config import config as cosine_config
from config.cosine_rotation_config import config as cosine_rot_config

import tensorflow as tf
import numpy as np
import sys


if __name__ == '__main__':
    _mth_to_config = {
        "proto": proto_config,
        "proto_rot": proto_rot_config,
        "cosine": cosine_config,
        "cosine_rot": cosine_rot_config,
    }

    cerebus = False
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        _mth = sys.argv[2]
        cerebus = True
    else:
        data_dir = r"D:\Datasets\mini-imagenet"
        _mth = "proto_rot"
    assert _mth in _mth_to_config, f"Method {_mth} is not recognized"

    opt = _mth_to_config[_mth]
    opt["Dataset_parameters"]["data_dir"] = data_dir
    opt["Model_parameters"]["name"] = opt["Model_parameters"]["name"]+f"{'_c' if cerebus else ''}"

    tf.random.set_seed(opt["Tensorflow_constants"]["seed"])
    np.random.seed(opt["Tensorflow_constants"]["seed"])

    mini_image_net = MiniImageNetDataset(**opt["Dataset_parameters"])

    few_shot_learner = FewShotImgLearner(**opt["Model_parameters"])
    few_shot_learner.build_and_compile()

    network_callback = NetworkManagerCallback(
        network_manager=few_shot_learner,
        **opt["Network_callback_parameters"]
    )

    few_shot_trainer = FewShotTrainer(
        model_manager=few_shot_learner,
        dataset=mini_image_net,
        network_callback=network_callback,
        **opt["Trainer_parameters"],
    )

    print(util.get_str_repr_for_config(opt))
    few_shot_trainer.train(epochs=300, final_testing=False)
    few_shot_trainer.test(n=10)

    util.plotHistory(few_shot_learner.history, savename="training_curve_"+few_shot_learner.name, savefig=not cerebus)
