import tensorflow as tf
from tqdm import tqdm
import numpy as np

from modules.datasets import DatasetBase
from modules.modelManagers import NetworkModelManager, NetworkManagerCallback
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

        self.n_train_batch = kwargs.get("n_train_batch", 100)
        self.n_val_batch = kwargs.get("n_val_batch", 100)
        self.n_training_batches = {
            util.TrainingPhase.TRAIN: self.n_train_batch,
            util.TrainingPhase.VAL: self.n_val_batch,
        }

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
        self.progress.set_description_str(str(phase.value))

        # reset the metrics
        for _metric in self.running_metrics:
            self.running_metrics[_metric].reset_states()

        total_episodes = sum(list(self.n_training_batches.values()))

        _data_itr = iter(self.data_generators[phase])
        for batch_idx in range(self.n_training_batches[phase]):
            # Optimize the model
            # Forward & update gradients
            if phase == util.TrainingPhase.TRAIN:
                with tf.GradientTape() as tape:
                    _inputs = next(_data_itr)
                    batch_logs = self.modelManager.compute_metrics(_inputs)  # TODO: ask ModelManager to get metrics dict as logs

                grads = tape.gradient(batch_logs["loss"], self.model.trainable_variables)
                self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            elif phase == util.TrainingPhase.VAL:
                _inputs = next(_data_itr)
                batch_logs = self.modelManager.compute_metrics(_inputs)
            else:
                raise NotImplementedError(f"Training phase: {phase} not implemented")

            # Track progress
            # TODO: get metrics automatically
            self.running_metrics["loss"].update_state(batch_logs["loss"])
            self.running_metrics["accuracy"].update_state(batch_logs["accuracy"])

            # update progress
            self.progress.update(1 / total_episodes)
            self.progress.set_postfix_str(f"batch: {batch_idx}/{self.n_training_batches[phase]} -> "
                                          + ' - '.join([f"{phase.value}_{k}: {v.result():.3f}"
                                                       for k, v in self.running_metrics.items()]))

        phase_logs = {k: v.result().numpy() for k, v in self.running_metrics.items()}

        return phase_logs

    def test(self, n=1):
        if self.load_on_start:
            self.modelManager.load()

        phase_logs = {k: [] for k, v in self.running_metrics.items()}

        phase = util.TrainingPhase.TEST

        self.progress = tqdm(
            ascii=True,
            iterable=range(n),
            unit="episode",
        )
        self.progress.set_description_str("Test")

        _data_itr = iter(self.data_generators[phase])
        # reset the metrics
        for _metric in self.running_metrics:
            self.running_metrics[_metric].reset_states()

        for i in range(n):
            _inputs = next(_data_itr)
            loss, acc = self.model.call(_inputs)

            # Track progress
            # TODO: get metrics automatically
            self.running_metrics["loss"].update_state(loss)
            self.running_metrics["accuracy"].update_state(acc)

            # update progress
            self.progress.update(1)
            self.progress.set_postfix_str(' - '.join([f"{phase.value}_{k}: {v.result():.3f}"
                                                      for k, v in self.running_metrics.items()]))
            for k in phase_logs:
                phase_logs[k].append(self.running_metrics[k].result().numpy())

        self.progress.close()
        phase_logs = {k: np.array(v) for k, v in phase_logs.items()}

        if self.verbose:
            print("\n--- Test results --- \n"
                  f"{self.config}"
                  f"Train epochs: {self.modelManager.current_epoch} \n"
                  f"Test epochs: {n} \n"
                  f"Mean accuracy: {np.mean(phase_logs.get('accuracy')) * 100:.2f}% "
                  f"± {np.std(phase_logs.get('accuracy')) * 100:.2f} \n"
                  f"{'-' * 35}")

        return phase_logs

    @property
    def config(self) -> str:
        return ""


class FewShotTrainer(Trainer):
    def __init__(self, model_manager: NetworkModelManager, dataset: DatasetBase, **kwargs):
        self.n_way = kwargs.get("n_way", 30)
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
                output_form=self.modelManager.output_form,
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
            # Optimize the model
            # Forward & update gradients
            if phase == util.TrainingPhase.TRAIN:
                with tf.GradientTape() as tape:
                    _support = next(_data_itr)
                    self.model.set_support(_support)

                    _query = next(_data_itr)
                    loss, acc = self.model.apply_query(_query)  # TODO: ask ModelManager to get metrics dict as logs

                grads = tape.gradient(loss, self.model.trainable_variables)
                self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            elif phase == util.TrainingPhase.VAL:
                _support = next(_data_itr)
                self.model.set_support(_support)
                del _support

                _query = next(_data_itr)
                loss, acc = self.model.apply_query(_query)
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

    def test(self, n=1):
        if self.load_on_start:
            self.modelManager.load()

        phase_logs = {k: [] for k, v in self.running_metrics.items()}

        phase = util.TrainingPhase.TEST

        self.progress = tqdm(
            ascii=True,
            iterable=range(self.n_test_episodes*n),
            unit="episode",
        )
        self.progress.set_description_str("Test")

        for i in range(n):
            # reset the metrics
            for _metric in self.running_metrics:
                self.running_metrics[_metric].reset_states()

            _data_itr = iter(self.data_generators[phase])
            for episode_idx in range(self.n_test_episodes):
                _support = next(_data_itr)
                self.model.set_support(_support)

                _query = next(_data_itr)
                loss, acc = self.model.apply_query(_query)

                # Track progress
                # TODO: get metrics automatically
                self.running_metrics["loss"].update_state(loss)
                self.running_metrics["accuracy"].update_state(acc)

                # update progress
                self.progress.update(1)
                self.progress.set_postfix_str(' - '.join([f"{phase.value}_{k}: {v.result():.3f}"
                                                         for k, v in self.running_metrics.items()]))
                for k in phase_logs:
                    phase_logs[k].append(self.running_metrics[k].result().numpy())

        self.progress.close()
        phase_logs = {k: np.array(v) for k, v in phase_logs.items()}

        if self.verbose:
            print("\n--- Test results --- \n"
                  f"{self.config}"
                  f"Train episodes: {self.n_train_episodes * self.modelManager.current_epoch} \n"
                  f"Test episodes: {n*self.n_test_episodes} \n"
                  f"Mean accuracy: {np.mean(phase_logs.get('accuracy'))*100:.2f}% "
                  f"± {np.std(phase_logs.get('accuracy'))*100:.2f} \n"
                  f"{'-'*35}")

        return phase_logs

    @property
    def config(self) -> str:
        _config = f"\n Model: {self.modelManager.name} \n" \
                  f"Few Shot params: \n " \
                  f"\t Train: {self.n_way}-way {self.n_shot}-shot \n" \
                  f"\t Val: {self.n_test_way}-way {self.n_test_shot}-shot \n" \
                  f"\t Test: {self.n_test_way}-way {self.n_test_shot}-shot \n \n"
        return _config


if __name__ == '__main__':
    from modules.datasets import MiniImageNetDataset, OutputForm
    from modules.modelManagers import SelfLearnerWithImgRotation
    import time

    # -----------------------------------------------------------------------------------------------------------------
    # hyper-parameters
    # -----------------------------------------------------------------------------------------------------------------

    mini_imagenet_dataset = MiniImageNetDataset(
        image_size=84,
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
