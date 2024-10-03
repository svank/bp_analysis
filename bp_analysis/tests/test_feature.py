import copy
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.ndimage

from ..feature import Feature, FeatureSequence, TrackedImage
from ..status import Flag


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
        [0, 0, 1, 1]
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
        flag=Flag.GOOD,
        feature_class=1,
    )
    return feature


def test_brightest_pixel(feature):
    assert feature.brightest_pixel == (11, 17)


def test_is_good(feature):
    feature.flag = Flag.GOOD
    assert feature.is_good
    feature.flag = Flag.FALSE_POS
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


def test_coord_set():
    feature = np.array([
        [0, 1],
        [0, 1],
    ])
    feature = Feature(
        id=1,
        cutout_corner=(10, 15),
        cutout=feature,
        data_cutout=feature,
        seed_cutout=feature,
        flag=1,
        feature_class=1,
    )
    set = feature.coord_set
    assert len(set) == 2
    assert (10, 16) in set
    assert (11, 16) in set


def test_indices():
    feature = np.array([
        [0, 1],
        [0, 1],
    ])
    feature = Feature(
        id=1,
        cutout_corner=(10, 15),
        cutout=feature,
        data_cutout=feature,
        seed_cutout=feature,
        flag=1,
        feature_class=1,
    )
    np.testing.assert_array_equal(feature.indices[0], np.array((10, 11)))
    np.testing.assert_array_equal(feature.indices[1], np.array((16, 16)))


def test_seed_map(various_features):
    np.testing.assert_array_equal(
        various_features.seed_map(),
        various_features.data_cutout_map() > 1
    )


def test_FeatureSequence_getitem(feature):
    feature.id = 1
    feature.time = datetime(1, 1, 1)
    
    feature2 = copy.deepcopy(feature)
    feature2.id = 10
    feature2.time = datetime(1, 1, 2)
    
    sequence = FeatureSequence()
    sequence.add_features(feature)
    sequence.add_features(feature2)
    
    assert sequence[1] is feature
    assert sequence[datetime(1, 1, 1)] is feature
    assert sequence[feature] is feature
    
    assert 1 in sequence
    assert datetime(1, 1, 1) in sequence
    assert feature in sequence
    
    assert sequence[10] is feature2
    assert sequence[datetime(1, 1, 2)] is feature2
    assert sequence[feature2] is feature2
    
    assert 10 in sequence
    assert datetime(1, 1, 2) in sequence
    assert feature2 in sequence


def test_FeatureSequence_len(feature):
    sequence = FeatureSequence()
    assert len(sequence) == 0
    sequence.add_features(feature)
    assert len(sequence) == 1
    sequence.add_features(feature)
    assert len(sequence) == 2


@pytest.fixture
def various_features(feature):
    feature2 = copy.deepcopy(feature)
    feature2.cutout = feature2.cutout.T
    feature2.seed_cutout = feature2.seed_cutout.T
    feature2.data_cutout = feature2.data_cutout.T
    feature2.cutout_corner = (22, 7)

    feature3 = copy.copy(feature)
    feature3.flag = Flag.FALSE_POS
    feature3.cutout_corner = (30, 12)

    feature4 = copy.copy(feature)
    feature4.flag = Flag.CLOSE_NEIGHBOR
    feature4.cutout_corner = (10, 33)

    feature5 = copy.copy(feature2)
    feature5.flag = Flag.EDGE
    feature5.cutout_corner = (41, 42)

    tracked_image = TrackedImage("source_file",(50, 50), "time", config=None)
    tracked_image.add_features(feature, feature2, feature3, feature4, feature5)
    return tracked_image


@pytest.mark.mpl_image_compare
def test_TrackedImage_plot_features(various_features):
    fig, ax = plt.subplots(1, 1)
    various_features.plot_features(ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_TrackedImage_sliced_plot_features(various_features):
    fig, ax = plt.subplots(1, 1)
    slice = np.s_[11:33, 8:18]
    ax.imshow(various_features.feature_map()[slice], origin='lower')
    various_features[slice].plot_features(ax=ax)
    return fig


def test_TrackedImage_feature_map(map_with_features):
    map, _ = scipy.ndimage.label(map_with_features > 0)
    
    tracked_image = TrackedImage(source_shape=map.shape)
    tracked_image.add_features_from_map(map, map, map)
    np.testing.assert_array_equal(tracked_image.feature_map(), map)
