from modules.datasets import MiniImageNetDataset
from modules.models import FewShotImgLearner
from modules.trainers import FewShotTrainer
from modules.models import NetworkManagerCallback
import modules.util as util

import tensorflow as tf
import sys


if __name__ == '__main__':

    way = 5
    shot = 5
    cerebus = False

    data_dir = r"D:\Datasets\mini-imagenet"

    print(sys.argv)

    if len(sys.argv) > 2:
        data_dir = sys.argv[1]
        cerebus = True

    mini_image_net = MiniImageNetDataset(
        data_dir=data_dir
    )

    few_shot_learner = FewShotImgLearner(
        name=f"prototypical_few_shot_learner-conv-4-64_backbone_{way}way{shot}shot{'_c' if cerebus else ''}",
        image_size=mini_image_net.image_size,
        backbone="conv-4-64",
        optimizer_args={},
        learning_rate=1e-3,
        optimizer=tf.keras.optimizers.Adam,
    )
    few_shot_learner.build_and_compile()

    network_callback = NetworkManagerCallback(
        network_manager=few_shot_learner,
        verbose=False,
        save_freq=1,
        early_stopping=True,
        patience=50,
        learning_rate_decay_enabled=True,
        learning_rate_decay_factor=0.5,
        learning_rate_decay_freq=2,
    )

    few_shot_trainer = FewShotTrainer(
        model_manager=few_shot_learner,
        dataset=mini_image_net,

        # few shot params
        n_way=way,
        n_shot=shot,
        n_query=15,
        n_train_episodes=1_000,
        n_val_episodes=600,
        n_test_episodes=600,

        # callback params
        network_callback=network_callback,
    )

    few_shot_trainer.train(epochs=1)

    util.plotHistory(few_shot_learner.history, savename="training_curve_"+few_shot_learner.name)
