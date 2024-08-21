import pathlib
import tomllib


reference_file = pathlib.Path(__file__).parent.parent / 'sample_config_file'
with open(reference_file, 'rb') as file:
    reference_config = tomllib.load(file)


def verify_config(config):
    for key in config:
        if key not in reference_config:
            raise RuntimeError(f"Unexpected section '{key}' in config")
    for section in config:
        for key in config[section]:
            if key not in reference_config[section]:
                raise RuntimeError(
                    f"Unexpected value '{key}' in config section '{section}'")


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
        if conf2[key] != conf1[key]:
            diff.append(f"{key}: {conf2[key]}")
    return diff
