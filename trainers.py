from models import NetworkModelManager, NetworkManagerCallback
from datasets import DatasetPhase
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import os


class Trainer:
    def __init__(self,
                 model_manager: NetworkModelManager,
                 dataset: DatasetPhase,
                 network_callback_args=None,
                 **kwargs):
        if network_callback_args is None:
            network_callback_args = {
                "verbose": True,
                "save_freq": 1
            }
        self.BATCH_SIZE = kwargs.get("batch_size", 256)
        self.IMG_SIZE = kwargs.get("img_size", 80)
        self.modelManager = model_manager
        self.model = model_manager.model
        self.dataset: DatasetPhase = dataset
        self.use_saving_callback = kwargs.get("use_saving_callback", True)
        self.load_on_start = kwargs.get("load_on_start", True)
        self.verbose = kwargs.get("verbose", 1)

        self.network_callback = NetworkManagerCallback(self.modelManager, **network_callback_args)

    def train(self, epochs=1):
        if self.load_on_start:
            self.modelManager.load()
        history = self.model.fit(
            self.dataset.get_generator(DatasetPhase.TRAIN, self.modelManager.output_form),
            steps_per_epoch=self.dataset.train_length//self.dataset.batch_size,
            epochs=epochs,
            validation_data=self.dataset.get_generator(DatasetPhase.VAL, self.modelManager.output_form),
            validation_steps=self.dataset.val_length//self.dataset.batch_size,
            callbacks=[self.network_callback] if self.use_saving_callback else [],
            verbose=self.verbose,
            initial_epoch=self.modelManager.current_epoch,
        )
        self.modelManager.save_weights()
        return history


if __name__ == '__main__':
    from datasets import MiniImageNetDataset, OutputForm
    from models import SelfLearnerWithImgRotation
    from hyperparameters import *
    import util
    import time

    # -----------------------------------------------------------------------------------------------------------------
    # hyper-parameters
    # -----------------------------------------------------------------------------------------------------------------
    tf.random.set_seed(SEED)
    print(get_str_repr_for_hyper_params())

    mini_imagenet_dataset = MiniImageNetDataset(
        image_size=IMG_SIZE,
        batch_size=64
    )

    self_learner = SelfLearnerWithImgRotation(
        name="SelfLearnerWithImgRotation",
        image_size=mini_imagenet_dataset.image_size,
        output_size=mini_imagenet_dataset.get_output_size(OutputForm.ROT),
    )
    self_learner.build_and_compile()
    self_learner.summary()

    # -----------------------------------------------------------------------------------------------------------------
    # Training the self learner with rotation
    # -----------------------------------------------------------------------------------------------------------------
    self_trainer = Trainer(
        model_manager=self_learner,
        dataset=mini_imagenet_dataset
    )

    start_time = time.time()
    self_trainer.train(epochs=5)
    end_feature_training_time = time.time() - start_time
    print(f"--- Elapse feature training time: {end_feature_training_time} [s] ---")

