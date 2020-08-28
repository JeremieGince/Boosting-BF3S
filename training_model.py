from modules.datasets import MiniImageNetDataset
from modules.trainers import FewShotTrainer, Trainer, get_trainer
from modules.modelManagers import NetworkManagerCallback
import modules.util as util
import config

import tensorflow as tf
import numpy as np
import sys


if __name__ == '__main__':
    cerebus = False
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        _mth = sys.argv[2]
        cerebus = True
    else:
        data_dir = r"D:\Datasets\mini-imagenet"
        _mth = "Gen0"

    opt = config.get_opt(_mth)
    opt["Dataset_parameters"]["data_dir"] = data_dir
    # opt["Model_parameters"]["name"] = opt["Model_parameters"]["name"]+f"{'_c' if cerebus else ''}"

    util.save_opt(opt)

    tf.random.set_seed(opt["Tensorflow_constants"]["seed"])
    np.random.seed(opt["Tensorflow_constants"]["seed"])

    print(f"Physical devices: {tf.config.experimental.list_physical_devices()}")

    mini_image_net = MiniImageNetDataset(**opt["Dataset_parameters"])

    network_manager = opt["model_type"](**opt["Model_parameters"])

    network_callback = NetworkManagerCallback(
        network_manager=network_manager,
        **opt["Network_callback_parameters"]
    )

    print(util.get_str_repr_for_config(opt))

    if opt.get("Batch_Trainer_parameters", None) is not None:
        print("Batch training")
        batch_trainer = Trainer(
            model_manager=network_manager,
            dataset=mini_image_net,
            network_callback=network_callback,
            **opt["Batch_Trainer_parameters"]
        )
        batch_trainer.train(epochs=opt["Batch_Trainer_parameters"]["n_epochs"], final_testing=False)
        if opt["Batch_Trainer_parameters"]["n_test"]:
            results = batch_trainer.test(n=opt["Batch_Trainer_parameters"]["n_test"])
            util.save_test_results(opt, results)

        del batch_trainer

    if opt.get("FewShot_Trainer_parameters", None) is not None:
        print("Episodic training")
        few_shot_trainer = FewShotTrainer(
            model_manager=network_manager,
            dataset=mini_image_net,
            network_callback=network_callback,
            **opt["FewShot_Trainer_parameters"],
        )
        few_shot_trainer.train(epochs=opt["FewShot_Trainer_parameters"]["n_epochs"], final_testing=False)
        if opt["FewShot_Trainer_parameters"]["n_test"]:
            results = few_shot_trainer.test(n=opt["FewShot_Trainer_parameters"]["n_test"])
            util.save_test_results(opt, results)

        del few_shot_trainer

    if opt.get("Trainers_parameters", None) is not None:
        assert isinstance(opt["Trainers_parameters"], list)
        for i, params in enumerate(opt["Trainers_parameters"]):
            print(f"{i}, {params['trainer_type'].name}")
            trainer = get_trainer(
                params["trainer_type"],
                model_manager=network_manager,
                dataset=mini_image_net,
                network_callback=network_callback,
                **params,
            )
            trainer.train(epochs=params["n_epochs"], final_testing=False)
            if params["n_test"]:
                results = trainer.test(n=params["n_test"])
                util.save_test_results(opt, results)

            del trainer

    util.plotHistory(
        network_manager.history,
        savename="training_curve_" + network_manager.name,
        savefig=not cerebus,
        block=not cerebus,
    )
