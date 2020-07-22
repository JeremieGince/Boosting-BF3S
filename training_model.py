from modules.datasets import MiniImageNetDataset
from modules.trainers import FewShotTrainer, Trainer
from modules.modelManagers import NetworkManagerCallback
import modules.util as util
from config.prototypical_config import config as proto_config
from config.prototypical_rotation_config import config as proto_rot_config
from config.cosine_config import config as cosine_config
from config.cosine_rotation_config import config as cosine_rot_config
from config.selflearning_rot_config import config as sl_rot_config

import tensorflow as tf
import numpy as np
import sys


if __name__ == '__main__':
    _mth_to_config = {
        "sl_rot": sl_rot_config,
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
        _mth = "sl_rot"
    assert _mth in _mth_to_config, f"Method {_mth} is not recognized"

    opt = _mth_to_config[_mth]
    opt["Dataset_parameters"]["data_dir"] = data_dir
    opt["Model_parameters"]["name"] = opt["Model_parameters"]["name"]+f"{'_c' if cerebus else ''}"

    tf.random.set_seed(opt["Tensorflow_constants"]["seed"])
    np.random.seed(opt["Tensorflow_constants"]["seed"])

    mini_image_net = MiniImageNetDataset(**opt["Dataset_parameters"])

    network_manager = opt["model_type"](**opt["Model_parameters"])
    network_manager.build_and_compile()

    network_callback = NetworkManagerCallback(
        network_manager=network_manager,
        **opt["Network_callback_parameters"]
    )

    print(util.get_str_repr_for_config(opt))

    if opt["Batch_Trainer_parameters"] is not None:
        batch_trainer = Trainer(
            model_manager=network_manager,
            dataset=mini_image_net,
            network_callback=network_callback,
            **opt["Batch_Trainer_parameters"]
        )
        batch_trainer.train(epochs=300, final_testing=False)
        batch_trainer.test(n=10)

    if opt["FewShot_Trainer_parameters"] is not None:
        few_shot_trainer = FewShotTrainer(
            model_manager=network_manager,
            dataset=mini_image_net,
            network_callback=network_callback,
            **opt["FewShot_Trainer_parameters"],
        )
        few_shot_trainer.train(epochs=300, final_testing=False)
        few_shot_trainer.test(n=10)

    util.plotHistory(network_manager.history, savename="training_curve_" + network_manager.name, savefig=not cerebus)
