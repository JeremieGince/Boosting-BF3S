import tensorflow as tf
from tqdm import tqdm
import numpy as np
import enum
from modules.datasets import DatasetBase
from modules.modelManagers import NetworkModelManager, NetworkManagerCallback
import modules.util as util


class TrainerType(enum.Enum):
    BatchTrainer = 0
    EpisodicTrainer = 1
    MixedTrainer = 2


class Trainer:
    """
    The base trainer class. Used to train a model of a NetworkModelManager.
    """
    def __init__(self,
                 model_manager: NetworkModelManager,
                 dataset: DatasetBase,
                 network_callback: NetworkManagerCallback = None,
                 network_callback_args=None,
                 **kwargs):
        """

        :param model_manager:
        :param dataset:
        :param network_callback:
        :param network_callback_args:
        :param kwargs:
        """

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

        # setting optimizer if needed
        self.learning_rate = kwargs.get("learning_rate", 1e-3)
        self.optimizer_args = kwargs.get("optimizer_args", {})
        _optim_type = kwargs.get("optimizer", None)
        if _optim_type is not None:
            self.model.optimizer = _optim_type(self.learning_rate, **self.optimizer_args)
        # self.optimizer = None if _optim_type is None else _optim_type(self.learning_rate, **self.optimizer_args)

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
        self.running_metrics = {_metric: tf.keras.metrics.Mean(name=str(_metric))
                                for _metric in self.modelManager.metrics}

        # progress bar
        self.progress = None

    def set_data_generators(self):
        self.data_generators = {
            _p: self.dataset.get_batch_generator(_p, self.modelManager.output_form)
            for _p in util.TrainingPhase
        }
        return self.data_generators

    def train(self, epochs=1, final_testing=True):
        if self.load_on_start:
            self.modelManager.load()

        history = {
            _p.value: {
                _metric: []
                for _metric in self.running_metrics.keys()
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
        self.reset_running_metrics()

        total_episodes = sum(list(self.n_training_batches.values()))

        _data_itr = iter(self.data_generators[phase])
        for batch_idx in range(self.n_training_batches[phase]):
            # Optimize the model
            # Forward & update gradients
            if phase == util.TrainingPhase.TRAIN:
                with tf.GradientTape() as tape:
                    _inputs = next(_data_itr)
                    batch_logs = self.modelManager.compute_batch_metrics(_inputs, training=True)

                grads = tape.gradient(batch_logs["loss"], self.model.trainable_variables)
                self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            elif phase == util.TrainingPhase.VAL:
                _inputs = next(_data_itr)
                batch_logs = self.modelManager.compute_batch_metrics(_inputs, training=False)
            else:
                raise NotImplementedError(f"Training phase: {phase} not implemented")

            # Track progress
            self.update_running_metrics(batch_logs)

            # update progress
            self.progress.update(1 / total_episodes)
            self.progress.set_postfix_str(f"batch: {batch_idx+1}/{self.n_training_batches[phase]} -> "
                                          + ' - '.join([f"{phase.value}_{k}: {v.result():.3f}"
                                                       for k, v in self.running_metrics.items()]))

        phase_logs = {k: v.result().numpy() for k, v in self.running_metrics.items()}
        # print(phase_logs)
        return phase_logs

    def update_running_metrics(self, logs: dict):
        """
        Update the current state of the running metrics with a logs dict.
        :param logs: a dictionary with the new value for each metric. (dict)
        :return: None
        """
        for m in self.running_metrics:
            if m in logs:
                self.running_metrics[m].update_state(logs[m])

    def reset_running_metrics(self):
        for _metric in self.running_metrics:
            self.running_metrics[_metric].reset_states()

    def test(self, n=1) -> dict:
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
        self.reset_running_metrics()

        for i in range(n):
            _inputs = next(_data_itr)
            batch_logs = self.modelManager.compute_batch_metrics(_inputs, training=False)

            # Track progress
            self.update_running_metrics(batch_logs)

            # update progress
            self.progress.update(1)
            self.progress.set_postfix_str(' - '.join([f"{phase.value}_{k}: {v.result():.3f}"
                                                      for k, v in self.running_metrics.items()]))
            for k in phase_logs:
                if k in batch_logs:
                    phase_logs[k].append(batch_logs[k])

        self.progress.close()
        phase_logs = {k: np.array(v) for k, v in phase_logs.items()}
        m, h = util.mean_confidence_interval(phase_logs.get("accuracy"), 0.95)

        pprint = "\n--- Test results --- \n" \
                f"{self.config}" \
                f"Train epochs: {self.modelManager.current_epoch} \n" \
                f"Test epochs: {n} \n" \
                f"Mean accuracy: {m * 100:.2f} " \
                f"± {h * 100:.2f} % \n" \
                f"{'-' * 35}"
        phase_logs["pprint"] = pprint
        if self.verbose:
            print(pprint)

        return phase_logs

    @property
    def config(self) -> str:
        return ""


class FewShotTrainer(Trainer):
    def __init__(self,
                 model_manager: NetworkModelManager,
                 dataset: DatasetBase,
                 network_callback: NetworkManagerCallback = None,
                 network_callback_args=None,
                 **kwargs):
        self.n_way = kwargs.get("n_way", 30)
        self.n_test_way = kwargs.get("n_test_way", self.n_way)
        self.n_shot = kwargs.get("n_shot", 5)
        self.n_test_shot = kwargs.get("n_test_shot", self.n_shot)
        self.n_query = kwargs.get("n_query", 1)

        self.n_test_query = kwargs.get("n_test_query", self.n_query)
        self.n_train_episodes = kwargs.get("n_train_episodes", 10)
        self.n_val_episodes = kwargs.get("n_val_episodes", 10)
        self.n_test_episodes = kwargs.get("n_test_episodes", 1)

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

        super().__init__(model_manager, dataset, network_callback, network_callback_args, **kwargs)

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
        self.reset_running_metrics()

        total_episodes = sum(list(self.n_training_episodes.values()))

        _data_itr = iter(self.data_generators[phase])
        for episode_idx in range(self.n_training_episodes[phase]):
            # Optimize the model
            # Forward & update gradients
            if phase == util.TrainingPhase.TRAIN:
                with tf.GradientTape() as tape:
                    episode_logs = self.modelManager.compute_episodic_metrics(_data_itr, training=True)

                grads = tape.gradient(episode_logs["loss"], self.model.trainable_variables)
                self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            elif phase == util.TrainingPhase.VAL:
                episode_logs = self.modelManager.compute_episodic_metrics(_data_itr, training=False)
            else:
                raise NotImplementedError(f"Training phase: {phase} not implemented")

            # Track progress
            self.update_running_metrics(episode_logs)

            # update progress
            self.progress.update(1 / total_episodes)
            self.progress.set_postfix_str(f"episode: {episode_idx}/{self.n_training_episodes[phase]} -> "
                                          + ' - '.join([f"{phase.value}_{k}: {v.result():.3f}"
                                                       for k, v in self.running_metrics.items()]))

        phase_logs = {k: v.result().numpy() for k, v in self.running_metrics.items()}
        print(phase_logs)

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
        self.reset_running_metrics()

        for i in range(n):
            episode_logs = self.modelManager.compute_episodic_metrics(_data_itr, training=False)

            # Track progress
            self.update_running_metrics(episode_logs)

            # update progress
            self.progress.update(1)
            self.progress.set_postfix_str(' - '.join([f"{phase.value}_{k}: {v.result():.3f}"
                                                     for k, v in self.running_metrics.items()]))
            for k in phase_logs:
                if k in episode_logs:
                    phase_logs[k].append(episode_logs[k])

        self.progress.close()
        phase_logs = {k: np.array(v) for k, v in phase_logs.items()}
        m, h = util.mean_confidence_interval(phase_logs.get("accuracy"), 0.95)

        pprint = f"\n--- Test results --- \n" \
                  f"{self.config}" \
                  f"Train episodes: {self.n_train_episodes * self.modelManager.current_epoch} \n" \
                  f"Test episodes: {n} \n" \
                  f"Mean accuracy: {m * 100:.2f} " \
                  f"± {h * 100:.2f} % \n" \
                  f"{'-'*35}"
        phase_logs["pprint"] = pprint
        if self.verbose:
            print(pprint)

        return phase_logs

    @property
    def config(self) -> str:
        _config = f"\n Model: {self.modelManager.name} \n" \
                  f"Few Shot params: \n " \
                  f"\t Train: {self.n_way}-way {self.n_shot}-shot \n" \
                  f"\t Val: {self.n_test_way}-way {self.n_test_shot}-shot \n" \
                  f"\t Test: {self.n_test_way}-way {self.n_test_shot}-shot \n \n"
        return _config


class MixedTrainer(FewShotTrainer):
    def __init__(self, model_manager: NetworkModelManager,
                 dataset: DatasetBase,
                 network_callback: NetworkManagerCallback = None,
                 network_callback_args=None,
                 **kwargs):

        self.gen_trainer_type = kwargs.get(
            "gen_trainer_type",
            {
                util.TrainingPhase.TRAIN: TrainerType.BatchTrainer,
                util.TrainingPhase.VAL: TrainerType.EpisodicTrainer,
                util.TrainingPhase.TEST: TrainerType.EpisodicTrainer,
            }
        )

        super(MixedTrainer, self).__init__(model_manager, dataset, network_callback, network_callback_args, **kwargs)

    def set_data_generators(self):
        self.data_generators = {}
        for _p, _t in self.gen_trainer_type.items():
            if _t == TrainerType.BatchTrainer:
                self.data_generators[_p] = self.dataset.get_batch_generator(_p, self.modelManager.output_form)
            elif _t == TrainerType.EpisodicTrainer:
                self.data_generators[_p] = self.dataset.get_few_shot_generator(
                    _n_way=self.phase_to_few_shot_params[_p]["way"],
                    _n_shot=self.phase_to_few_shot_params[_p]["shot"],
                    _n_query=self.phase_to_few_shot_params[_p]["query"],
                    phase=_p,
                    output_form=self.modelManager.output_form,
                )
            elif _t == TrainerType.MixedTrainer:
                raise ValueError()
            else:
                raise ValueError()
        return self.data_generators

    def get_logs(self, phase: util.TrainingPhase, data_itr, training=True):
        if self.gen_trainer_type[phase] == TrainerType.BatchTrainer:
            logs = self.modelManager.compute_batch_metrics(next(data_itr), training=training)
        elif self.gen_trainer_type[phase] == TrainerType.EpisodicTrainer:
            logs = self.modelManager.compute_episodic_metrics(data_itr, training=training)
        else:
            raise ValueError()
        return logs

    def get_total_iterations(self):
        return sum([self.get_nb_iterations(_p) for _p in self.TRAINING_PHASES])

    def get_nb_iterations(self, phase: util.TrainingPhase):
        if self.gen_trainer_type[phase] == TrainerType.BatchTrainer:
            nb = self.n_training_batches[phase]
        elif self.gen_trainer_type[phase] == TrainerType.EpisodicTrainer:
            nb = self.n_training_episodes[phase]
        else:
            raise ValueError()
        return nb

    def do_phase(self, epoch: int, phase: util.TrainingPhase):
        self.progress.set_description_str(str(phase.value))

        # reset the metrics
        self.reset_running_metrics()

        total_iterations = self.get_total_iterations()

        _data_itr = iter(self.data_generators[phase])
        for episode_idx in range(self.get_nb_iterations(phase)):
            # Optimize the model
            # Forward & update gradients
            if phase == util.TrainingPhase.TRAIN:
                with tf.GradientTape() as tape:
                    iteration_logs = self.get_logs(phase, _data_itr, training=True)

                grads = tape.gradient(iteration_logs["loss"], self.model.trainable_variables)
                self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            elif phase == util.TrainingPhase.VAL:
                iteration_logs = self.get_logs(phase, _data_itr, training=False)
            else:
                raise NotImplementedError(f"Training phase: {phase} not implemented")

            # Track progress
            self.update_running_metrics(iteration_logs)

            # update progress
            self.progress.update(1 / total_iterations)
            self.progress.set_postfix_str(f"iteration: {episode_idx}/{self.get_nb_iterations(phase)} -> "
                                          + ' - '.join([f"{phase.value}_{k}: {v.result():.3f}"
                                                       for k, v in self.running_metrics.items()]))

        phase_logs = {k: v.result().numpy() for k, v in self.running_metrics.items()}
        print(phase_logs)
        assert 1 == 0

        return phase_logs

    def test(self, n=1) -> dict:
        if self.load_on_start:
            self.modelManager.load()

        phase_logs = {k: [] for k, v in self.running_metrics.items()}

        phase = util.TrainingPhase.TEST

        self.progress = tqdm(
            ascii=True,
            iterable=range(n),
            unit="iteration",
        )
        self.progress.set_description_str("Test")

        _data_itr = iter(self.data_generators[phase])
        # reset the metrics
        self.reset_running_metrics()

        for i in range(n):
            iteration_logs = self.get_logs(phase, _data_itr, training=False)

            # Track progress
            self.update_running_metrics(iteration_logs)

            # update progress
            self.progress.update(1)
            self.progress.set_postfix_str(' - '.join([f"{phase.value}_{k}: {v.result():.3f}"
                                                      for k, v in self.running_metrics.items()]))
            for m in phase_logs:
                if m in iteration_logs:
                    phase_logs[m].append(iteration_logs[m])

        self.progress.close()
        phase_logs = {k: np.array(v) for k, v in phase_logs.items()}
        m, h = util.mean_confidence_interval(phase_logs.get("accuracy"), 0.95)

        pprint = "\n--- Test results --- \n" \
                 f"{self.config}" \
                 f"Train epochs: {self.modelManager.current_epoch} \n" \
                 f"Test iterations: {n} \n" \
                 f"Mean accuracy: {m * 100:.2f} " \
                 f"± {h * 100:.2f} % \n" \
                 f"{'-' * 35}"
        phase_logs["pprint"] = pprint
        if self.verbose:
            print(pprint)

        return phase_logs


def get_trainer(tr_type: TrainerType, *args, **kwargs) -> Trainer:
    """
    Get the Trainer instance given the trainer type.
    :param tr_type: The trainer type (TrainerType)
    :param args: Trainer args
    :param kwargs:Trainer kwargs
    :return: the trainer instance (Trainer)
    """
    if tr_type == TrainerType.BatchTrainer:
        trainer = Trainer(*args, **kwargs)
    elif tr_type == TrainerType.EpisodicTrainer:
        trainer = FewShotTrainer(*args, **kwargs)
    elif tr_type == TrainerType.MixedTrainer:
        trainer = MixedTrainer(*args, **kwargs)
    else:
        raise ValueError()
    return trainer


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
