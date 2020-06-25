import tensorflow as tf
from tqdm import tqdm
import numpy as np

from datasets import DatasetBase
from models import NetworkModelManager, NetworkManagerCallback
import util


class Trainer:
    def __init__(self,
                 model_manager: NetworkModelManager,
                 dataset: DatasetBase,
                 network_callback_args=None,
                 **kwargs):
        if network_callback_args is None:
            network_callback_args = {
                "verbose": True,
                "save_freq": 1
            }

        # setting the model manager
        self.modelManager = model_manager
        self.model = model_manager.model

        # setting the dataset
        self.dataset: DatasetBase = dataset
        self.data_generators = self.set_data_generators()

        # setting training parameters
        self.use_saving_callback = kwargs.get("use_saving_callback", True)
        self.load_on_start = kwargs.get("load_on_start", True)
        self.verbose = kwargs.get("verbose", 1)

        self.network_callback = NetworkManagerCallback(self.modelManager, **network_callback_args)
        self.current_phase = None

        # Setting metrics
        # TODO: get the metrics from ModelManager
        self.running_metrics = {_metric: tf.keras.metrics.Mean() for _metric in ["loss", "accuracy"]}

        # progress bar
        self.progress = None

    def set_data_generators(self):
        self.data_generators = {
            _p: self.dataset.get_generator(_p, self.modelManager.output_form)
            for _p in util.TrainingPhase
        }
        return self.data_generators

    def train(self, epochs=1):
        self.current_phase = util.TrainingPhase.TRAIN

        if self.load_on_start:
            self.modelManager.load()

        # history = self.model.fit(
        #     self.dataset.get_generator(TrainingPhase.TRAIN, self.modelManager.output_form),
        #     steps_per_epoch=self.dataset.train_length // self.dataset.batch_size,
        #     epochs=epochs,
        #     validation_data=self.dataset.get_generator(TrainingPhase.VAL, self.modelManager.output_form),
        #     validation_steps=self.dataset.val_length // self.dataset.batch_size,
        #     callbacks=[self.network_callback] if self.use_saving_callback else [],
        #     verbose=self.verbose,
        #     initial_epoch=self.modelManager.current_epoch,
        # )

        history = {
            _metric: []
            for _metric in ["loss", "accuracy"]  # TODO: get metrics from modelManager
        }

        self.progress = tqdm(
            iterable=range(self.modelManager.current_epoch, epochs),
            unit="epoch",
        )
        self.progress.set_description_str(str(self.current_phase.value))
        for _ in range(len(self.progress)):
            epoch = self.progress.n

            logs = self.do_epoch(epoch, self.current_phase)

            # update history
            for _metric in history:
                history[_metric].append(logs.get(_metric))

            # update progress
            self.progress.set_postfix_str(f"train_loss: {logs.get('loss')} - train_acc: {logs.get('accuracy')}")
            # self.progress.update()

        self.progress.close()
        self.modelManager.save_weights()
        return history

    def loss(self, x, y, training):
        # training=training is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        y_ = self.model(x, training=training)
        return self.model.loss(y_true=y, y_pred=y_)

    def grad(self, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss(inputs, targets, training=True)
        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

    def do_epoch(self, epoch: int, phase: util.TrainingPhase):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.Accuracy()
        metrics = dict(zip(self.model.metrics_names, self.model.metrics))
        print(metrics)

        nb_batch = self.dataset.train_length // self.dataset.batch_size

        # calling the callback
        self.network_callback.on_epoch_begin(epoch)

        # Training loop
        for batch_idx, (x, y) in enumerate(iter(self.data_generators[phase])):

            # Optimize the model
            loss_value, grads = self.grad(x, y)
            self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            epoch_accuracy.update_state(y, self.model(x, training=True))

            # update progress
            self.progress.update(1/nb_batch)
            self.progress.set_postfix_str(f"batches: {batch_idx}/{nb_batch} "
                                          f"- train_loss: {epoch_loss_avg.result():.3f} "
                                          f"- train_acc: {epoch_accuracy.result():.3f}")
            if batch_idx + 1 == nb_batch:
                break

        epoch_logs = {"loss": epoch_loss_avg.result(), "accuracy": epoch_accuracy.result()}

        # calling the callback
        self.network_callback.on_epoch_end(epoch, logs=epoch_logs)

        return epoch_logs


class FewShotTrainer(Trainer):
    def __init__(self, model_manager: NetworkModelManager, dataset: DatasetBase, **kwargs):
        self.n_way = kwargs.get("n_way", 20)
        self.n_shot = kwargs.get("n_shot", 5)
        self.n_query = kwargs.get("n_query", 1)
        self.n_episodes = kwargs.get("n_episodes", 10)

        super().__init__(model_manager, dataset, **kwargs)

    def set_data_generators(self):
        self.data_generators = {
            _p: self.dataset.get_few_shot_generator(
                _n_way=self.n_way,
                _n_shot=self.n_shot,
                _n_query=self.n_query,
                phase=_p,
            )
            for _p in util.TrainingPhase
        }
        return self.data_generators

    def do_epoch(self, epoch: int, phase: util.TrainingPhase):
        # reset the metrics
        for _metric in self.running_metrics:
            self.running_metrics[_metric].reset_states()

        # calling the callback
        self.network_callback.on_epoch_begin(epoch)

        _data_itr = iter(self.data_generators[phase])
        for episode_idx in range(self.n_episodes):
            support, query = next(_data_itr)

            # Optimize the model
            # Forward & update gradients
            with tf.GradientTape() as tape:
                loss, acc = self.model.call(support, query)  # TODO: ask ModelManeger to get metrics dict as logs
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            # Track progress
            # TODO: get metrics automatically
            self.running_metrics["loss"].update_state(loss)
            self.running_metrics["accuracy"].update_state(acc)

            # update progress
            self.progress.update(1 / self.n_episodes)
            self.progress.set_postfix_str(f"episode: {episode_idx}/{self.n_episodes} -> "
                                          + ' - '.join([f"{k}: {v.result():.3f}"
                                                       for k, v in self.running_metrics.items()]))

        epoch_logs = {k: v.result() for k, v in self.running_metrics.items()}

        # calling the callback
        self.network_callback.on_epoch_end(epoch, logs=epoch_logs)

        return epoch_logs


if __name__ == '__main__':
    from datasets import MiniImageNetDataset, OutputForm
    from models import SelfLearnerWithImgRotation
    from hyperparameters import *
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
    self_trainer.train(epochs=2)
    end_feature_training_time = time.time() - start_time
    print(f"--- Elapse feature training time: {end_feature_training_time} [s] ---")
