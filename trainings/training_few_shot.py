from datasets import MiniImageNetDataset
from models import FewShotImgLearner
from trainers import FewShotTrainer
import util


if __name__ == '__main__':
    mini_image_net = MiniImageNetDataset(
        data_dir="D:\Datasets\mini-imagenet"
    )

    _gen = mini_image_net.get_few_shot_generator(
        _n_way=20,
        _n_shot=5,
        _n_query=1,
        phase=util.TrainingPhase.TRAIN,
        output_form=util.OutputForm.LABEL
    )

    print(_gen, iter(_gen), [*map(lambda x: x.shape, next(iter(_gen)))])

    few_shot_learner = FewShotImgLearner(
        image_size=mini_image_net.image_size,
    )
    few_shot_learner.build_and_compile()

    few_shot_trainer = FewShotTrainer(
        model_manager=few_shot_learner,
        dataset=mini_image_net,
        n_way=5,
        n_shot=5,
        n_query=15,
        n_episodes=100,
    )

    few_shot_trainer.train(epochs=10)

    util.plotHistory(few_shot_learner.history)
