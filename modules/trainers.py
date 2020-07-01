import tensorflow as tf
from tqdm import tqdm
import numpy as np

from modules.datasets import DatasetBase
from modules.models import NetworkModelManager, NetworkManagerCallback
import modules.util as util


class Trainer:
    def __init__(self,
                 model_manager: NetworkModelManager,
                 dataset: DatasetBase,
                 network_callback: NetworkManagerCallback = None,
                 network_callback_args=None,
                 **kwargs):

        # setting phases
        self.TRAINING_PHASES = [util.TrainingPhase.TRAIN, util.TrainingPhase.VAL]

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

        # network callback
        if network_callback_args is None:
            network_callback_args = {
                "verbose": False,
                "save_freq": 1,
                "early_stopping": True,
                "patience": 50,

            }

        if network_callback is None:
            self.network_callback = NetworkManagerCallback(self.modelManager, **network_callback_args)
        else:
            self.network_callback = network_callback

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

    def train(self, epochs=1, final_testing=True):
        if self.load_on_start:
            self.modelManager.load()

        history = {
            _p.value: {
                _metric: []
                for _metric in self.running_metrics.keys()  # TODO: get metrics from modelManager
            }
            for _p in self.TRAINING_PHASES
        }

        epochs_range = range(self.modelManager.current_epoch, epochs)

        self.progress = tqdm(
            ascii=True,
            iterable=epochs_range,
            unit="epoch",
        )
        for epoch in epochs_range:
            logs = self.do_epoch(epoch)

            # update history
            for _p in self.TRAINING_PHASES:
                for _metric in history[_p.value]:
                    history[_p.value][_metric].append(logs[_p.value].get(_metric))

            # update progress
            self.progress.set_postfix_str(
                ' - '.join([
                    f"{_p.value}_{_metric_name}: {_metric}"
                    for _p in self.TRAINING_PHASES for _metric_name, _metric in logs[_p.value].items()
                    ])
            )
            # self.progress.update()

            # EarlyStopping
            if self.network_callback.early_stopping_triggered:
                break

        self.progress.close()
        self.modelManager.save_weights()

        if final_testing:
            self.test()

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

    def do_epoch(self, epoch: int):
        # calling the callback
        self.network_callback.on_epoch_begin(epoch)

        logs = {
            _p.value: self.do_phase(epoch, _p)
            for _p in self.TRAINING_PHASES
        }

        # calling the callback
        self.network_callback.on_epoch_end(epoch, logs=logs)

        return logs

    def do_phase(self, epoch: int, phase: util.TrainingPhase):
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

    def test(self):
        pass


class FewShotTrainer(Trainer):
    def __init__(self, model_manager: NetworkModelManager, dataset: DatasetBase, **kwargs):
        self.n_way = kwargs.get("n_way", 20)
        self.n_test_way = kwargs.get("n_test_way", self.n_way)
        self.n_shot = kwargs.get("n_shot", 5)
        self.n_test_shot = kwargs.get("n_test_shot", self.n_shot)
        self.n_query = kwargs.get("n_query", 1)
        self.n_test_query = kwargs.get("n_test_query", self.n_query)
        self.n_train_episodes = kwargs.get("n_train_episodes", 10)
        self.n_val_episodes = kwargs.get("n_val_episodes", 10)
        self.n_test_episodes = kwargs.get("n_test_episodes", 600)

        self.n_training_episodes = {
            util.TrainingPhase.TRAIN: self.n_train_episodes,
            util.TrainingPhase.VAL: self.n_val_episodes,
        }

        self.phase_to_few_shot_params = {
            util.TrainingPhase.TRAIN: {
                "way": self.n_way,
                "shot": self.n_shot,
                "query": self.n_query
            },
            util.TrainingPhase.VAL: {
                "way": self.n_test_way,
                "shot": self.n_test_shot,
                "query": self.n_test_query
            },
            util.TrainingPhase.TEST: {
                "way": self.n_test_way,
                "shot": self.n_test_shot,
                "query": self.n_test_query
            },
        }

        super().__init__(model_manager, dataset, **kwargs)

    def set_data_generators(self):
        self.data_generators = {
            _p: self.dataset.get_few_shot_generator(
                _n_way=self.phase_to_few_shot_params[_p]["way"],
                _n_shot=self.phase_to_few_shot_params[_p]["shot"],
                _n_query=self.phase_to_few_shot_params[_p]["query"],
                phase=_p,
            )
            for _p in util.TrainingPhase
        }
        return self.data_generators

    def do_phase(self, epoch: int, phase: util.TrainingPhase):
        self.progress.set_description_str(str(phase.value))

        # reset the metrics
        for _metric in self.running_metrics:
            self.running_metrics[_metric].reset_states()

        total_episodes = sum(list(self.n_training_episodes.values()))

        _data_itr = iter(self.data_generators[phase])
        for episode_idx in range(self.n_training_episodes[phase]):
            support, query = next(_data_itr)

            # Optimize the model
            # Forward & update gradients
            if phase == util.TrainingPhase.TRAIN:
                with tf.GradientTape() as tape:
                    loss, acc = self.model.call(support, query)  # TODO: ask ModelManager to get metrics dict as logs
                grads = tape.gradient(loss, self.model.trainable_variables)
                self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            elif phase == util.TrainingPhase.VAL:
                loss, acc = self.model.call(support, query)
            else:
                raise NotImplementedError(f"Training phase: {phase} not implemented")

            # Track progress
            # TODO: get metrics automatically
            self.running_metrics["loss"].update_state(loss)
            self.running_metrics["accuracy"].update_state(acc)

            # update progress
            self.progress.update(1 / total_episodes)
            self.progress.set_postfix_str(f"episode: {episode_idx}/{self.n_training_episodes[phase]} -> "
                                          + ' - '.join([f"{phase.value}_{k}: {v.result():.3f}"
                                                       for k, v in self.running_metrics.items()]))

        phase_logs = {k: v.result().numpy() for k, v in self.running_metrics.items()}

        return phase_logs

    def test(self):
        phase = util.TrainingPhase.TEST

        self.progress = tqdm(
            ascii=True,
            iterable=range(self.n_test_episodes),
            unit="episode",
        )
        self.progress.set_description_str("Test")

        # reset the metrics
        for _metric in self.running_metrics:
            self.running_metrics[_metric].reset_states()

        _data_itr = iter(self.data_generators[phase])
        for episode_idx in range(self.n_test_episodes):
            support, query = next(_data_itr)

            # Optimize the model
            # Forward & update gradients
            loss, acc = self.model.call(support, query)

            # Track progress
            # TODO: get metrics automatically
            self.running_metrics["loss"].update_state(loss)
            self.running_metrics["accuracy"].update_state(acc)

            # update progress
            self.progress.update(1)
            self.progress.set_postfix_str(' - '.join([f"{phase.value}_{k}: {v.result():.3f}"
                                                     for k, v in self.running_metrics.items()]))

        self.progress.close()
        phase_logs = {k: v.result().numpy() for k, v in self.running_metrics.items()}

        if self.verbose:
            print("\n--- Test results --- \n"
                  f"model: {self.modelManager.name} \n"
                  f"Few Shot params: \n "
                  f"\t Train: {self.n_way}-way {self.n_shot}-shot \n"
                  f"\t Val: {self.n_test_way}-way {self.n_test_shot}-shot \n"
                  f"\t Test: {self.n_test_way}-way {self.n_test_shot}-shot \n \n"
                  f"Train episodes: {self.n_train_episodes * (self.modelManager.current_epoch + 1)} \n"
                  f"Test episodes: {self.n_test_episodes} \n"
                  f"Mean accuracy: {phase_logs.get('accuracy')*100:.2f}% \n"
                  f"{'-'*35}")

        return phase_logs


if __name__ == '__main__':
    from modules.datasets import MiniImageNetDataset, OutputForm
    from modules.models import SelfLearnerWithImgRotation
    from modules.hyperparameters import *
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
