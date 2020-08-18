import tensorflow as tf
from modules.modelManagers import FewShotImgLearner
from modules import util


way = 30
t_way = 5
shot = 5
t_shot = 5
backbone = "conv-4-64"
sl_classifier_type = "cosine"

batch_epochs = 300

config = {
    "Tensorflow_constants": {
        "seed": 7,
    },

    "Dataset_parameters": {
        "data_dir": r"D:\Datasets\mini-imagenet"
    },

    "model_type": FewShotImgLearner,
    "Model_parameters": {
        "name": f"cosine_boosted_few_shot_rot-{backbone}_"
                f"{way}way{shot}shot_{t_way}tway{t_shot}tshot_Adam",
        "method": FewShotImgLearner.Method.CosineNet,
        "alpha": 1.0,
        "sl_boosted_type": util.SLBoostedType.ROT,
        "sl_kwargs": {
            "hidden_neurons": [640 for _ in range(1)] if sl_classifier_type == "dense" else None,
            "classifier_type": sl_classifier_type,
        },
        "n_cls_base": 64,
        "n_cls_val": 16,
    },

    "Network_callback_parameters": {
        "verbose": False,
        "save_freq": 1,
        "early_stopping": True,
        "patience": batch_epochs + 50,
        "learning_rate_decay_enabled": False,
        "learning_rate_decay_factor": 0.85,
        "learning_rate_decay_freq": 20,
    },

    "Batch_Trainer_parameters": {
        "n_train_batch": 100,
        "n_val_batch": 0,
        "n_epochs": batch_epochs,
        "n_test": 0,

        # optimizer
        "learning_rate": 1e-3,
        "optimizer_args": {
            "momentum": 0.9,
            "decay": 5e-4,
            "nesterov": True,
        },
        "optimizer": tf.keras.optimizers.SGD,
        # "optimizer_args": {},
        # "optimizer": tf.keras.optimizers.Adam,
    },

    "FewShot_Trainer_parameters": {
        "n_way": way,
        "n_test_way": t_way,
        "n_shot": shot,
        "n_test_shot": t_shot,
        "n_query": 15,
        "n_test_query": 5,
        "n_train_episodes": 100,
        "n_val_episodes": 100,
        "n_test_episodes": 1,
        "n_epochs": batch_epochs + 300,
        "n_test": 2_000,

        # optimizer
        "learning_rate": 1e-3,
        "optimizer_args": {
            "momentum": 0.9,
            "decay": 5e-4,
            "nesterov": True,
        },
        "optimizer": tf.keras.optimizers.SGD,
        # "optimizer_args": {},
        # "optimizer": tf.keras.optimizers.Adam,
    }
}
