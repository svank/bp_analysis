import configparser

import numpy as np
import pytest

from ..status import Flag


@pytest.fixture
def feature_details_for_map():
    feature_details = [(30, 20, 5, Flag.GOOD),
                       (70, 22, 9, Flag.GOOD),
                       (10, 82, 2, Flag.GOOD),
                       (18, 82, 1, Flag.TOO_SMALL),
                       (10, 2, 2, Flag.EDGE),
                       (3, 10, 3, Flag.EDGE),
                       (10, 97, 2, Flag.EDGE),
                       (96, 10, 3, Flag.EDGE),
                       (50, 50, 2, Flag.CLOSE_NEIGHBOR),
                       (56, 50, 2, Flag.CLOSE_NEIGHBOR),
                       (50, 56, 2, Flag.CLOSE_NEIGHBOR),
                       (50, 62, 2, Flag.CLOSE_NEIGHBOR),
                       (80, 80, 12, Flag.FALSE_POS),
                       ]
    return feature_details


@pytest.fixture
def map_with_features(feature_details_for_map):
    def add_feature(image, r, c, half_width):
        x, y = np.meshgrid(
            np.arange(2 * half_width + 1, dtype=float),
            np.arange(2 * half_width + 1, dtype=float))
        x -= half_width
        y -= half_width
        x /= x.max()
        y /= y.max()
        d = -x ** 2 + -y ** 2
        d += -d.min()
        image[r - half_width:r + half_width + 1,
              c - half_width:c + half_width + 1] = d
    
    image = np.zeros((100, 100))
    for feature in feature_details_for_map:
        add_feature(image, *feature[:-1])
    return image


@pytest.fixture
def basic_config():
    config = configparser.ConfigParser()
    config.read_string("""
        [main]
        blur = 0
        dilation_method = contour
        contour_require_downhill = False
        contour_threshold = 0.65
        contour_min_finding_scale = 1
        contour_max_intensity_range = 1.2
        dilation_rounds = 9
        connect_diagonal = False
        fpos_thresh = 0.2
        proximity_thresh = 2

        n_sigma = 7.5
        seed_use_laplacian = True
        seed_mode = relative

        min_size = 10
        max_size = 1000
        max_diagonal = 30
        max_size_change_pct = 50
        max_size_change_px = 10
        
        min_lifetime = 1
    """)
    config = config['main']
    return config
