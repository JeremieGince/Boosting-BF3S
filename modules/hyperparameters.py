
#  -------------------------------------------------------------------------------------------------------------------
#  Tensorflow constants
#  -------------------------------------------------------------------------------------------------------------------
# SEED = 7  # for luck, it's still the divine number ;)
SEED = 3  # for luck, it's still the number of the triumph ;)

#  -------------------------------------------------------------------------------------------------------------------
#  Dataset parameters
#  -------------------------------------------------------------------------------------------------------------------
COL_VAL_SPLIT = 0.1
CLS_VAL_SPLIT = 0.9
CLS_VAL_VIRTUAL_LENGTH = 10_000
BATCH_SIZE = 64
IMG_SIZE = 80
GAMUT_SIZE = 50
BINS = 10

#  -------------------------------------------------------------------------------------------------------------------
#  Colorizer model parameters
#  -------------------------------------------------------------------------------------------------------------------
FUSION_DEPTH = 256

#  -------------------------------------------------------------------------------------------------------------------
#  Colorizer training parameters
#  -------------------------------------------------------------------------------------------------------------------
FEATURES_TRAINING_EPOCHS = 10
COL_LEARNING_RATE = 1e-3
COL_MOMENTUM = 0.9

#  -------------------------------------------------------------------------------------------------------------------
#  Classifier model parameters
#  -------------------------------------------------------------------------------------------------------------------

#  -------------------------------------------------------------------------------------------------------------------
#  Classifier training parameters
#  -------------------------------------------------------------------------------------------------------------------
CLASSIFIER_EPOCHS = 30
CLS_LEARNING_RATE = 1e-3
CLS_MOMENTUM = 0.7
CLS_USE_NESTEROV = True

# --------------------------------------------------------------------------------------------------------------------
# Dictionary form

hyper_parameters = {
    "Tensorflow Constants": {
        "seed": SEED,
    },
    "Dataset parameters": {
        "Colorization val split ratio": COL_VAL_SPLIT,
        "Classification val split ratio": CLS_VAL_SPLIT,
        "Classification val virtual length": CLS_VAL_VIRTUAL_LENGTH,
        "Batch size": BATCH_SIZE,
        "Image size": IMG_SIZE,
        "Gamut size": GAMUT_SIZE,
        "Bins": BINS,
    },
    "Colorizer model parameters": {
        "Fusion depth": FUSION_DEPTH,
    },
    "Colorizer training parameters": {
        "Features training epochs": FEATURES_TRAINING_EPOCHS,
        "Colorization learning rate": COL_LEARNING_RATE,
        "Colorization momentum": COL_MOMENTUM,
    },
    "Classifier model parameters": {

    },
    "Classifier training parameters": {
        "Classifier epochs": CLASSIFIER_EPOCHS,
        "Classifier learning rate": CLS_LEARNING_RATE,
        "Classifier momentum": CLS_MOMENTUM,
        "Classifier use nesterov": CLS_USE_NESTEROV,
    },
}


def get_str_repr_for_hyper_params():
    _str = ""
    for sec, sec_dict in hyper_parameters.items():
        _str += get_str_repr_for_sec_hyper_params(sec)
    return _str


def get_str_repr_for_secs_hyper_params(secs: list):
    _str = ""
    for sec in secs:
        _str += get_str_repr_for_sec_hyper_params(sec)
    return _str


def get_str_repr_for_sec_hyper_params(sec: str):
    assert sec in hyper_parameters.keys(), f"The param: 'sec' must be in {hyper_parameters.keys()}"
    _str = ""

    sec_dict = hyper_parameters[sec]
    _str += '-' * 25 + '\n'
    _str += str(sec) + '\n'
    _str += '-' * 25 + '\n'

    for param, value in sec_dict.items():
        _str += f"{param}: {value} \n"

    _str += '\n'
    return _str

