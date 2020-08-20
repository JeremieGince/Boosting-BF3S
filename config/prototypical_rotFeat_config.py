import tensorflow as tf
from modules.modelManagers import FewShotImgLearner, SelfLearnerWithImgRotation
from modules import util
from modules.util import SLBoostedType
from config.prototypical_rotation_config import config as teacher_config


way = 6
t_way = 5
shot = 5
t_shot = 5

backbone = "conv-4-64"
feat_dist_mth = "cosine"


config = {
    "Tensorflow_constants": {
        "seed": 7,
    },

    "Dataset_parameters": {
        "data_dir": r"D:\Datasets\mini-imagenet"
    },

    "model_type": FewShotImgLearner,
    "Model_parameters": {
        "name": f"prototypical_rotFeat-{backbone}-{feat_dist_mth}_"
                f"{way}way{shot}shot_{t_way}tway{t_shot}tshot"
                f"_1t",
        "method": FewShotImgLearner.Method.PrototypicalNet,
        "alpha": 1.00,

        "sl_boosted_type": SLBoostedType.ROT_FEAT,
        "sl_kwargs": {
            "feat_dist_mth": feat_dist_mth,
            "nb_k": 2,
        },

        # Teaching parameters
        "teacher": teacher_config["model_type"](**teacher_config["Model_parameters"]),
        "weights_path": "training_data/"+teacher_config["Model_parameters"]["name"]
                        + SelfLearnerWithImgRotation.WEIGHTS_PATH_EXT,
        "teacher_loss": "klb",
        "teacher_T": 4.0,
        "teacher_gamma": 1.0,
    },

    "Network_callback_parameters": {
        "verbose": False,
        "save_freq": 1,
        "early_stopping": True,
        "patience": 50,
        "learning_rate_decay_enabled": True,
        "learning_rate_decay_factor": 0.85,
        "learning_rate_decay_freq": 20,
    },

    "Batch_Trainer_parameters": None,

    "FewShot_Trainer_parameters": {
        "n_way": way,
        "n_test_way": t_way,
        "n_shot": shot,
        "n_test_shot": t_shot,
        "n_query": 5,
        "n_test_query": 5,
        "n_train_episodes": 1,
        "n_val_episodes": 1,
        "n_test_episodes": 1,
        "n_epochs": 300,
        "n_test": 2_000,

        # optimizer
        "learning_rate": 1e-3,
        "optimizer_args": {},
        "optimizer": tf.keras.optimizers.Adam,
    }
}
