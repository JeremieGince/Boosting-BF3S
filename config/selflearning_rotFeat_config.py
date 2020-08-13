import tensorflow as tf
from modules.modelManagers import SelfLearnerWithImgRotation
from modules.util import SLBoostedType
from config.prototypical_rotation_config import config as teacher_config

backbone = "conv-4-64"
feat_dist_mth = "l2"


config = {
    "Tensorflow_constants": {
        "seed": 7,
    },

    "Dataset_parameters": {
        "data_dir": r"D:\Datasets\mini-imagenet"
    },

    "model_type": SelfLearnerWithImgRotation,
    "Model_parameters": {
        "name": f"self-learning_rotFeat-{backbone}-{feat_dist_mth}",
        "sl_type": SLBoostedType.ROT_FEAT,
        "feat_dist_mth": feat_dist_mth,
        "nb_k": 2,
        "learning_rate": 1e-3,
        "optimizer_args": {},
        "optimizer": tf.keras.optimizers.Adam,

        "teacher": teacher_config["model_type"](**teacher_config["Model_parameters"]),
        "weights_path": teacher_config["model_type"]["name"] + SelfLearnerWithImgRotation.WEIGHTS_PATH_EXT
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

    "Batch_Trainer_parameters": {
        "n_train_batch": 100,
        "n_val_batch": 100,
        "n_epochs": 300,
        "n_test": 100,
    },

    "FewShot_Trainer_parameters": None
}
