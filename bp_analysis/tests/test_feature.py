import copy

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.ndimage

from .. import abc_tracker
from ..feature import Feature, TrackedImage


@pytest.fixture
def feature():
    feature = np.array([
        [0, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 0, 1, 1]
    ])
    data = np.array([
        [0, 3, 1, 0],
        [1, 2, 4, 0],
        [1, 1, 3, 1],
        [8, 0, 1, 1]
    ])
    seeds = np.array([
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0]
    ])
    feature = Feature(
        id=1,
        cutout_corner=(10, 15),
        cutout=feature,
        data_cutout=data,
        seed_cutout=seeds,
        flag=1,
        feature_class=1,
    )
    return feature


def test_brightest_pixel(feature):
    assert feature.brightest_pixel == (11, 17)


def test_is_good(feature):
    feature.flag = abc_tracker.GOOD
    assert feature.is_good
    feature.flag = abc_tracker.FALSE_POS
    assert not feature.is_good


@pytest.mark.mpl_image_compare
def test_plot_onto(feature):
    fig, ax = plt.subplots(1, 1)
    feature.plot_onto(ax)
    return fig


@pytest.mark.mpl_image_compare
def test_plot_onto_ids(feature):
    fig, ax = plt.subplots(1, 1)
    feature.plot_onto(ax, ids=True)
    return fig


@pytest.mark.mpl_image_compare
def test_TrackedImage_plot_features(feature):
    feature2 = copy.deepcopy(feature)
    feature2.cutout = feature2.cutout.T
    feature2.seed_cutout = feature2.seed_cutout.T
    feature2.data_cutout = feature2.data_cutout.T
    feature2.cutout_corner = (22, 7)

    feature3 = copy.copy(feature)
    feature3.flag = abc_tracker.FALSE_POS
    feature3.cutout_corner = (30, 12)

    feature4 = copy.copy(feature)
    feature4.flag = abc_tracker.CLOSE_NEIGHBOR
    feature4.cutout_corner = (10, 33)

    feature5 = copy.copy(feature2)
    feature5.flag = abc_tracker.EDGE
    feature5.cutout_corner = (41, 42)

    tracked_image = TrackedImage("source_file", "time", config=None)
    tracked_image.add_features(feature, feature2, feature3, feature4, feature5)

    fig, ax = plt.subplots(1, 1)
    tracked_image.plot_features(ax=ax)

    return fig


def test_TrackedImage_feature_map(map_with_features):
    map, _ = scipy.ndimage.label(map_with_features > 0)
    
    tracked_image = TrackedImage(source_shape=map.shape)
    tracked_image.add_features_from_map(map, map, map)
    np.testing.assert_array_equal(tracked_image.feature_map(), map)
