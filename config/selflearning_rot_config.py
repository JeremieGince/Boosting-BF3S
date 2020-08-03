import tensorflow as tf
from modules.modelManagers import SelfLearnerWithImgRotation

backbone = "conv-4-64"
classifier_type = "cosine"


config = {
    "Tensorflow_constants": {
        "seed": 7,
    },

    "Dataset_parameters": {
        "data_dir": r"D:\Datasets\mini-imagenet"
    },

    "model_type": SelfLearnerWithImgRotation,
    "Model_parameters": {
        "name": f"self-learning_rot-{backbone}-{classifier_type}",
        "classifier_type": classifier_type,
        "hidden_neurons": [640 for _ in range(1)] if classifier_type == "dense" else None,
        "learning_rate": 1e-3,
        "optimizer_args": {},
        "optimizer": tf.keras.optimizers.Adam,
    },

    "Network_callback_parameters": {
        "verbose": False,
        "save_freq": 1,
        "early_stopping": True,
        "patience": 75,
        "learning_rate_decay_enabled": True,
        "learning_rate_decay_factor": 0.90,
        "learning_rate_decay_freq": 20,
    },

    "Batch_Trainer_parameters": {
        "n_train_batch": 100,
        "n_val_batch": 100,
        "n_epochs": 300,
        "n_test": 10,
    },

    "FewShot_Trainer_parameters": None
}
