from modules.datasets import MiniImageNetDataset
from modules.modelManagers import FewShotImgLearner
from modules.trainers import FewShotTrainer
from modules.modelManagers import NetworkManagerCallback
import modules.util as util

import tensorflow as tf
import sys


if __name__ == '__main__':

    way = 30
    t_way = 5
    shot = 5
    t_shot = 5
    backbone = "conv-4-64"

    cerebus = not False

    data_dir = r"D:\Datasets\mini-imagenet"

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        cerebus = True

    mini_image_net = MiniImageNetDataset(
        data_dir=data_dir
    )

    few_shot_learner = FewShotImgLearner(
        name=f"prototypical_few_shot_learner-{backbone}_backbone_"
             f"{way}way{shot}shot_{t_way}tway{t_shot}tshot{'_c' if cerebus else ''}",
        image_size=mini_image_net.image_size,
        backbone=backbone,
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
        learning_rate_decay_factor=0.75,
        learning_rate_decay_freq=20,
    )

    few_shot_trainer = FewShotTrainer(
        model_manager=few_shot_learner,
        dataset=mini_image_net,

        # few shot params
        n_way=way,
        n_test_way=t_way,
        n_shot=shot,
        n_test_shot=t_shot,
        n_query=15,
        n_test_query=5,
        n_train_episodes=100,
        n_val_episodes=100,
        n_test_episodes=600,

        # callback params
        network_callback=network_callback,
    )

    print(few_shot_trainer.config)
    # few_shot_trainer.train(epochs=300, final_testing=False)
    few_shot_trainer.test()

    util.plotHistory(few_shot_learner.history, savename="training_curve_"+few_shot_learner.name, savefig=not cerebus)
