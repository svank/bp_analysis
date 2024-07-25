import numpy as np
import pytest

from .. import status


@pytest.fixture
def feature_details_for_map():
    feature_details = [(30, 20, 5, status.GOOD),
                       (70, 22, 9, status.GOOD),
                       (10, 82, 2, status.GOOD),
                       (18, 82, 1, status.GOOD),
                       (10, 2, 2, status.EDGE),
                       (3, 10, 3, status.EDGE),
                       (10, 97, 2, status.EDGE),
                       (96, 10, 3, status.EDGE),
                       (50, 50, 2, status.CLOSE_NEIGHBOR),
                       (56, 50, 2, status.CLOSE_NEIGHBOR),
                       (50, 56, 2, status.CLOSE_NEIGHBOR),
                       (50, 62, 2, status.CLOSE_NEIGHBOR),
                       (80, 80, 12, status.FALSE_POS),
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
