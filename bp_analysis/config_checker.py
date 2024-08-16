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
