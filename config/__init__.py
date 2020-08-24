
available_mth = {
    "sl_rot",
    "sl_rotFeat",
    "proto",
    "proto_rot",
    "proto_rotFeat",
    "cosine",
    "cosine_rot",
    "Gen0",
}


def get_opt(config_name: str):
    _error_av_name: str = f"The method {config_name} is not available. \n" \
                          f"The available methods are: \n {available_mth}."
    assert config_name in available_mth, _error_av_name

    if config_name == "sl_rot":
        from config.selflearning_rot_config import config as opt
    elif config_name == "sl_rotFeat":
        from config.selflearning_rotFeat_config import config as opt
    elif config_name == "proto":
        from config.prototypical_config import config as opt
    elif config_name == "proto_rot":
        from config.prototypical_rotation_config import config as opt
    elif config_name == "proto_rotFeat":
        from config.prototypical_rotFeat_config import config as opt
    elif config_name == "cosine":
        from config.cosine_config import config as opt
    elif config_name == "cosine_rot":
        from config.cosine_rotation_config import config as opt
    elif config_name == "Gen0":
        from config.Gen0_config import config as opt
    else:
        raise ValueError(_error_av_name)

    return opt
