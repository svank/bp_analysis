import pytest
from .. import config_utils


def test_bad_config_values(basic_config):
    basic_config['fake-section'] = {1: 2}
    with pytest.raises(ValueError, match=".*Unexpected section.*"):
        config_utils.get_cfg(basic_config, 'dilation', 'method')
        
    del basic_config['fake-section']
    basic_config['dilation']['fake-key'] = 1
    with pytest.raises(ValueError, match=".*Unexpected value.*"):
        config_utils.get_cfg(basic_config, 'dilation', 'method')
