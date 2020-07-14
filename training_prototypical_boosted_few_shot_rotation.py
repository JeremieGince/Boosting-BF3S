from modules.datasets import MiniImageNetDataset
from modules.modelManagers import BoostedFewShotLearner
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

    cerebus = False

    data_dir = r"D:\Datasets\mini-imagenet"

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        cerebus = True

    mini_image_net = MiniImageNetDataset(
        data_dir=data_dir
    )

    few_shot_learner = BoostedFewShotLearner(
        name=f"prototypical_boosted_few_shot_rot_learner-{backbone}_"
             f"{way}way{shot}shot_{t_way}tway{t_shot}tshot_Adam_a010_float16"
             f"{'_c' if cerebus else ''}",
        # name="proto_test",
        image_size=mini_image_net.image_size,
        backbone=backbone,
        sl_output_size=mini_image_net.get_output_size(util.OutputForm.ROT),
        alpha=0.1,
        hidden_neurons=[640 for _ in range(4)],
        learning_rate=1e-3,
        # optimizer_args={
        #     "momentum": 0.9,
        #     "decay": 5e-4,
        #     "nesterov": True,
        # },
        # optimizer=tf.keras.optimizers.SGD,
        optimizer_args={},
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
        learning_rate_decay_factor=0.85,
        learning_rate_decay_freq=20,
    )

    few_shot_trainer = FewShotTrainer(
        model_manager=few_shot_learner,
        dataset=mini_image_net,
        train_mini_batch=1 if cerebus else 1,

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
    few_shot_trainer.train(epochs=40, final_testing=False)
    few_shot_trainer.test(n=10)

    util.plotHistory(few_shot_learner.history, savename="training_curve_"+few_shot_learner.name, savefig=not cerebus)
