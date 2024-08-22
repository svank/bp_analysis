import copy
import pathlib
import tomllib

import numpy as np


reference_file = pathlib.Path(__file__).parent.parent / 'sample_config_file'
with open(reference_file, 'rb') as file:
    reference_config = tomllib.load(file)


def get_cfg(config, section, key):
    verify_config(config)
    try:
        return config[section][key]
    except KeyError:
        return reference_config[section][key]


def verify_config(config):
    for key in config:
        if key not in reference_config:
            raise ValueError(f"Unexpected section '{key}' in config")
    for section in config:
        for key in config[section]:
            if key not in reference_config[section]:
                raise ValueError(
                    f"Unexpected value '{key}' in config section '{section}'")


def fill_with_defaults(config):
    merged_config = copy.deepcopy(reference_config)
    for section in config:
        merged_config[section] |= config[section]
    return merged_config


def _flatten_config(config):
    flat_config = {}
    for section in config:
        for key in config[section]:
            flat_config[f"{section}.{key}"] = config[section][key]
    return flat_config


def diff_configs(conf1, conf2):
    conf1 = _flatten_config(conf1)
    conf2 = _flatten_config(conf2)
    diff = []
    missing_keys = conf1.keys() - conf2.keys()
    for key in missing_keys:
        diff.append(f"{key}: not set")
    extra_keys = conf2.keys() - conf1.keys()
    for key in extra_keys:
        diff.append(f"{key}: {conf2[key]}")
    keys_in_both = conf1.keys() & conf2.keys()
    for key in keys_in_both:
        if (conf2[key] != conf1[key]
                and not (np.isnan(conf2[key]) and np.isnan(conf1[key]))):
            diff.append(f"{key}: {conf2[key]}")
    return diff
