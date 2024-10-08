from datetime import datetime

import numpy as np
import pytest

from .. import abc_tracker, feature
from ..status import Flag


def test_calc_laplacian(basic_config):
    image = np.ones((3,3))
    image[1,1] = 3
    laplacian = abc_tracker.calc_laplacian(image, basic_config)
    assert laplacian.shape == (1,1)
    assert laplacian[0,0] == 16
    
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    kernel = kernel / 8
    laplacian = abc_tracker.calc_laplacian(image, basic_config, kernel)
    assert laplacian.shape == (1,1)
    assert laplacian[0,0] == 2


def test_find_seeds(basic_config):
    # There are no seeds in this rather flat image
    np.random.seed(1)
    image = np.random.random((100, 100))
    
    seeds, laplacian = abc_tracker.find_seeds(image, basic_config)
    
    np.testing.assert_array_equal(seeds, 0)
    
    # Create an obvious seed
    image[30, 30] = 100
    
    seeds, laplacian = abc_tracker.find_seeds(image, basic_config)
    
    assert seeds[30, 30] == 1
    seeds[30, 30] = 0
    np.testing.assert_array_equal(seeds, 0)
    
    # Adjust the n_sigma parameter to just less than it needs to be
    # The calculation of the laplacian's mean and std doesn't include the
    # padded edges
    laplacian_trimmed = laplacian[1:-1, 1:-1]
    mean = np.mean(laplacian_trimmed)
    std = np.std(laplacian_trimmed)
    
    n_sigma = (laplacian[30, 30] - mean) / std
    
    basic_config['seeds']['n_sigma'] = n_sigma - .1
    
    seeds, laplacian = abc_tracker.find_seeds(image, basic_config)
    
    assert seeds[30, 30] == 1
    seeds[30, 30] = 0
    np.testing.assert_array_equal(seeds, 0)
    
    # Adjust the n_sigma parameter to just above what it needs to be, so there
    # will be no seeds
    basic_config['seeds']['n_sigma'] = n_sigma + .1
    
    seeds, laplacian = abc_tracker.find_seeds(image, basic_config)
    
    np.testing.assert_array_equal(seeds, 0)


def test_find_seeds_absolute_thresh(basic_config):
    basic_config['seeds']['mode'] = 'absolute'
    # n.b. because the default laplacian kernel isn't normalized, this is 8
    # times higher than it "should" be
    basic_config['seeds']['threshold'] = 8.08
    
    # There are no seeds in this rather flat image
    np.random.seed(1)
    image = np.random.random((100, 100))
    
    seeds, laplacian = abc_tracker.find_seeds(image, basic_config)
    
    np.testing.assert_array_equal(seeds, 0)
    
    # Create a seed
    image[30, 30] = 2
    
    seeds, laplacian = abc_tracker.find_seeds(image, basic_config)
    
    assert seeds[30, 30] == 1
    seeds[30, 30] = 0
    np.testing.assert_array_equal(seeds, 0)


def test_find_seeds_no_laplacian(basic_config):
    basic_config['seeds']['use_laplacian'] = False
    # Just easier to test in absolute mode
    basic_config['seeds']['mode'] = 'absolute'
    basic_config['seeds']['threshold'] = 1.01
    
    # There are no seeds in this rather flat image
    np.random.seed(1)
    image = np.random.random((100, 100))
    
    seeds, laplacian = abc_tracker.find_seeds(image, basic_config)
    
    np.testing.assert_array_equal(seeds, 0)
    
    # Create a seed
    image[30, 30] = 1.02
    
    seeds, laplacian = abc_tracker.find_seeds(image, basic_config)
    
    assert seeds[30, 30] == 1
    seeds[30, 30] = 0
    np.testing.assert_array_equal(seeds, 0)


def test_dilate_laplacian(basic_config):
    basic_config['dilation']['method'] = 'laplacian'
    basic_config['dilation']['rounds'] = 1
    basic_config['main']['connect_diagonal'] = True
    
    image = np.zeros((50, 50))
    image[29:34, 29:34] = .6
    image[30:33, 30:33] = 1
    image[31, 31] = 2
    
    seeds = np.zeros((50, 50), dtype=bool)
    seeds[31, 31] = 1
    
    dilated_seeds = abc_tracker.dilate(basic_config, seeds, im=image)
    np.testing.assert_array_equal(dilated_seeds[30:33, 30:33], 1)
    dilated_seeds[30:33, 30:33] = 0
    np.testing.assert_array_equal(dilated_seeds, 0)
    
    # Add another round
    basic_config['dilation']['rounds'] = 2
    
    dilated_seeds = abc_tracker.dilate(basic_config, seeds, im=image)
    np.testing.assert_array_equal(dilated_seeds[29:34, 29:34], 1)
    dilated_seeds[29:34, 29:34] = 0
    np.testing.assert_array_equal(dilated_seeds, 0)
    
    # Pass that extra round explicitly
    basic_config['dilation']['rounds'] = 1
    
    dilated_seeds = abc_tracker.dilate(basic_config, seeds, im=image, n_rounds=2)
    np.testing.assert_array_equal(dilated_seeds[29:34, 29:34], 1)
    dilated_seeds[29:34, 29:34] = 0
    np.testing.assert_array_equal(dilated_seeds, 0)


def test_dilate_laplacian_masked(basic_config):
    basic_config['dilation']['method'] = 'laplacian'
    basic_config['dilation']['rounds'] = 1
    basic_config['main']['connect_diagonal'] = True
    
    mask = np.zeros((50, 50))
    mask[30:32, 31] = 1
    
    seeds = np.zeros((50, 50), dtype=bool)
    seeds[31, 31] = 1
    
    dilated_seeds = abc_tracker.dilate(basic_config, seeds, mask=mask)
    np.testing.assert_array_equal(dilated_seeds[30:32, 31], 1)
    dilated_seeds[30:32, 31] = 0
    np.testing.assert_array_equal(dilated_seeds, 0)
    

def test_dilate_laplacian_provided_laplacian(basic_config):
    basic_config['dilation']['method'] = 'laplacian'
    basic_config['dilation']['rounds'] = 1
    basic_config['main']['connect_diagonal'] = True
    
    laplacian = np.zeros((50, 50))
    laplacian[30:32, 31] = 1
    
    seeds = np.zeros((50, 50), dtype=bool)
    seeds[31, 31] = 1
    
    dilated_seeds = abc_tracker.dilate(
        basic_config, seeds, laplacian=laplacian)
    np.testing.assert_array_equal(dilated_seeds[30:32, 31], 1)
    dilated_seeds[30:32, 31] = 0
    np.testing.assert_array_equal(dilated_seeds, 0)

@pytest.fixture
def basic_contour_config(basic_config):
    basic_config['dilation']['method'] = 'contour'
    basic_config['dilation']['rounds'] = 1
    basic_config['main']['connect_diagonal'] = True
    basic_config['dilation-contour']['require_downhill'] = True
    basic_config['dilation-contour']['threshold'] = 0.2
    basic_config['dilation-contour']['region_scale'] = 1
    basic_config['dilation-contour']['max_intensity_range'] = 999
    return basic_config


@pytest.fixture
def contour_image():
    return np.array([
        [0, 0, 0, 0, 0, 0,  0,  0],
        [0, 0, 0, 0, 0, 0,  0,  0],
        [0, 0, 0, 1, 1, 1,  0,  0],
        [0, 0, 0, 2, 3, 1,  0,  0],
        [0, 0, 0, 2, 2, 1, .9, .8],
        [0, 0, 0, 1, 1, 1,  0,  0],
        [0, 0, 0, 0, 0, 0,  0,  0],
        [0, 0, 0, 0, 0, 0,  0,  0],
    ])


def test_dilate_contour(basic_contour_config, contour_image):
    seeds = contour_image > 1
    
    dilated_seeds = abc_tracker.dilate(
        basic_contour_config, seeds, im=contour_image)
    np.testing.assert_array_equal(dilated_seeds, contour_image >= 1)
    
    # Add another round
    basic_contour_config['dilation']['rounds'] = 2
    
    dilated_seeds = abc_tracker.dilate(
        basic_contour_config, seeds, im=contour_image)
    np.testing.assert_array_equal(dilated_seeds, contour_image >= .9)
    
    # Pass that extra round explicitly
    basic_contour_config['dilation']['rounds'] = 1
    
    dilated_seeds = abc_tracker.dilate(
        basic_contour_config, seeds, im=contour_image, n_rounds=2)
    np.testing.assert_array_equal(dilated_seeds, contour_image >= .9)


def test_dilate_contour_percentiles(basic_contour_config, contour_image):
    contour_image[3, 3] = 999999
    contour_image[4, 3] = 3
    contour_image[3, 1] = -999999
    
    basic_contour_config['dilation-contour']['region_low_percentile'] = 5
    basic_contour_config['dilation-contour']['region_high_percentile'] = 92
    
    seeds = contour_image > 1
    
    dilated_seeds = abc_tracker.dilate(
        basic_contour_config, seeds, im=contour_image)
    np.testing.assert_array_equal(dilated_seeds, contour_image >= 1)


def test_dilate_contour_threshold(basic_contour_config, contour_image):
    basic_contour_config['dilation']['rounds'] = 2
    basic_contour_config['dilation-contour']['threshold'] = .33
    
    seeds = contour_image > 1
    dilated_seeds = abc_tracker.dilate(
        basic_contour_config, seeds, im=contour_image)
    np.testing.assert_array_equal(dilated_seeds, contour_image >= 1)


def test_dilate_contour_require_downhill(
        basic_contour_config, contour_image):
    basic_contour_config['dilation']['rounds'] = 2
    basic_contour_config['dilation-contour']['require_downhill'] = True
    
    seeds = contour_image > 1
    contour_image[contour_image == .9] = 1.1
    dilated_seeds = abc_tracker.dilate(
        basic_contour_config, seeds, im=contour_image)
    expected = (contour_image >= 1) * (contour_image != 1.1)
    np.testing.assert_array_equal(dilated_seeds, expected)
    
    basic_contour_config['dilation-contour']['require_downhill'] = False
    dilated_seeds = abc_tracker.dilate(
        basic_contour_config, seeds, im=contour_image)
    np.testing.assert_array_equal(dilated_seeds, contour_image >= 1)


def test_dilate_contour_region_scale(basic_contour_config, contour_image):
    basic_contour_config['dilation-contour']['region_scale'] = 2
    
    contour_image[3, 1] = -1
    contour_image[3, 0] = -2
    contour_image[4, 6:] = -.3
    
    seeds = contour_image > 1
    dilated_seeds = abc_tracker.dilate(
        basic_contour_config, seeds, im=contour_image)
    expected = np.zeros_like(contour_image, dtype=bool)
    expected[2:6, 2:6] = contour_image[2:6, 2:6] > -.2
    np.testing.assert_array_equal(dilated_seeds, expected)
    
    basic_contour_config['dilation']['rounds'] = 2
    basic_contour_config['dilation-contour']['region_scale'] = 1
    dilated_seeds = abc_tracker.dilate(
        basic_contour_config, seeds, im=contour_image)
    expected = np.zeros_like(contour_image, dtype=bool)
    expected[1:7, 1:7] = contour_image[1:7, 1:7] > -.2
    np.testing.assert_array_equal(dilated_seeds, expected)


def test_dilate_contour_region_size(basic_contour_config, contour_image):
    basic_contour_config['dilation-contour']['region_scale'] = 99999
    basic_contour_config['dilation-contour']['region_size'] = 2
    
    contour_image[3, 0] = -.5
    contour_image[4, 6] = .59
    contour_image[5, 6] = .61
    contour_image[4, 7] = 0
    
    seeds = contour_image >= 1
    dilated_seeds = abc_tracker.dilate(
        basic_contour_config, seeds, im=contour_image)
    expected = contour_image > .6
    np.testing.assert_array_equal(dilated_seeds, expected)
    
    basic_contour_config['dilation-contour']['region_size'] = 3
    dilated_seeds = abc_tracker.dilate(
        basic_contour_config, seeds, im=contour_image)
    expected = contour_image > 0
    np.testing.assert_array_equal(dilated_seeds, expected)


def test_dilate_contour_max_intensity_range(
        basic_contour_config, contour_image):
    basic_contour_config['dilation-contour']['max_intensity_range'] = 2
    basic_contour_config['dilation-contour']['max_intensity_mode'] = 'absolute'
    
    contour_image[3, 4] = 10
    
    seeds = contour_image > 1
    dilated_seeds = abc_tracker.dilate(
        basic_contour_config, seeds, im=contour_image)
    np.testing.assert_array_equal(dilated_seeds, contour_image >= 1)
    
    basic_contour_config['dilation-contour']['max_intensity_mode'] = 'relative'
    basic_contour_config['dilation-contour']['max_intensity_range'] = 5
    
    dilated_seeds = abc_tracker.dilate(
        basic_contour_config, seeds, im=contour_image)
    np.testing.assert_array_equal(dilated_seeds, contour_image >= 1)


def test_dilate_contour_lower_thresh(
        basic_contour_config, contour_image):
    basic_contour_config['dilation-contour']['lower_thresh'] = -.001
    
    contour_image[3, 2] = -10
    
    seeds = contour_image > 1
    dilated_seeds = abc_tracker.dilate(
        basic_contour_config, seeds, im=contour_image)
    np.testing.assert_array_equal(dilated_seeds, contour_image >= 1)


def test_filter_close_neighbors(basic_config):
    labeled_feats = np.zeros((20, 20), dtype=int)
    labeled_feats[5:8, 5:8] = 1
    labeled_feats[10:12, 5:8] = 2
    tracked_image = feature.TrackedImage()
    tracked_image.add_features_from_map(
        labeled_feats, labeled_feats, labeled_feats)
    
    basic_config['proximity-filter']['threshold'] = 2
    
    abc_tracker.filter_close_neighbors(
        labeled_feats, basic_config, tracked_image)
    for feat in tracked_image.features:
        assert feat.flag == Flag.GOOD
    
    basic_config['proximity-filter']['threshold'] = 3
    
    abc_tracker.filter_close_neighbors(
        labeled_feats, basic_config, tracked_image)
    for feat in tracked_image.features:
        assert feat.flag == Flag.CLOSE_NEIGHBOR


def test_filter_close_neighbors_min_size_ratio(basic_config):
    labeled_feats = np.zeros((20, 20), dtype=int)
    labeled_feats[5:7, 5:7] = 1
    labeled_feats[8, 5] = 2
    tracked_image = feature.TrackedImage()
    tracked_image.add_features_from_map(
        labeled_feats, labeled_feats, labeled_feats)
    
    basic_config['proximity-filter']['threshold'] = 3
    basic_config['proximity-filter']['ignore_below_size_ratio'] = .24
    
    abc_tracker.filter_close_neighbors(
        labeled_feats, basic_config, tracked_image)
    for feat in tracked_image.features:
        assert feat.flag == Flag.CLOSE_NEIGHBOR

    basic_config['proximity-filter']['ignore_below_size_ratio'] = .26
    tracked_image = feature.TrackedImage()
    tracked_image.add_features_from_map(
        labeled_feats, labeled_feats, labeled_feats)
    
    abc_tracker.filter_close_neighbors(
        labeled_feats, basic_config, tracked_image)
    assert tracked_image[1].flag == Flag.GOOD
    assert tracked_image[2].flag == Flag.CLOSE_NEIGHBOR
    
    # Have the features be encountered in a different order
    tracked_image = feature.TrackedImage()
    tracked_image.add_features_from_map(
        labeled_feats, labeled_feats, labeled_feats)
    tracked_image.features = tracked_image.features[::-1]
    
    abc_tracker.filter_close_neighbors(
        labeled_feats, basic_config, tracked_image)
    assert tracked_image[1].flag == Flag.GOOD
    assert tracked_image[2].flag == Flag.CLOSE_NEIGHBOR

    basic_config['proximity-filter']['ignore_below_size_ratio'] = .24
    tracked_image = feature.TrackedImage()
    tracked_image.add_features_from_map(
        labeled_feats, labeled_feats, labeled_feats)
    tracked_image.features = tracked_image.features[::-1]
    
    abc_tracker.filter_close_neighbors(
        labeled_feats, basic_config, tracked_image)
    for feat in tracked_image.features:
        assert feat.flag == Flag.CLOSE_NEIGHBOR


def test_remove_false_positives(basic_config):
    labeled_feats = np.zeros((20, 20), dtype=int)
    labeled_feats[5:10, 5:10] = 1
    seeds = labeled_feats > 0
    
    laplacian = np.zeros((20, 20))
    # Feature can grow along only one side
    laplacian[4:10, 5:10] = 1
    basic_config['dilation']['method'] = 'laplacian'
    basic_config['false-pos-filter']['threshold'] = .25
    
    tracked_image = feature.TrackedImage()
    tracked_image.add_features_from_map(
        labeled_feats, labeled_feats, labeled_feats)
    
    abc_tracker.remove_false_positives(labeled_feats, laplacian, basic_config,
                                       labeled_feats, seeds, tracked_image)
    for feat in tracked_image.features:
        assert feat.flag == Flag.GOOD
    
    # Feature can grow along all sides
    laplacian[4:11, 4:11] = 1
    abc_tracker.remove_false_positives(labeled_feats, laplacian, basic_config,
                                       labeled_feats, seeds, tracked_image)
    for feat in tracked_image.features:
        assert feat.flag == Flag.FALSE_POS


def test_filter_size(basic_config):
    labeled_feats = np.zeros((20, 20), dtype=int)
    
    # Too small by one pixel
    labeled_feats[1, 1] = 1
    
    # Too big by one pixel
    labeled_feats[5:10, 5:10] = 2
    
    basic_config['size-filter']['min_size'] = 2
    basic_config['size-filter']['max_size'] = 24
    
    tracked_image = feature.TrackedImage()
    tracked_image.add_features_from_map(
        labeled_feats, labeled_feats, labeled_feats)
    
    abc_tracker.filter_size(tracked_image, basic_config)
    
    assert len(tracked_image.features) == 2
    assert tracked_image.features[0].flag == Flag.TOO_SMALL
    assert tracked_image.features[1].flag == Flag.TOO_BIG
    
    labeled_feats = np.zeros((20, 20), dtype=int)
    
    # Minimum allowed size
    labeled_feats[1, 1:3] = 1
    
    # Maximum allowed size
    labeled_feats[5:10, 5:10] = 2
    labeled_feats[5, 5] = 0
    
    tracked_image = feature.TrackedImage()
    tracked_image.add_features_from_map(
        labeled_feats, labeled_feats, labeled_feats)
    
    abc_tracker.filter_size(tracked_image, basic_config)
    
    assert len(tracked_image.features) == 2
    assert tracked_image.features[0].flag == Flag.GOOD
    assert tracked_image.features[1].flag == Flag.GOOD


def test_filter_size_diagonal(basic_config):
    labeled_feats = np.zeros((20, 20), dtype=int)
    
    # OK
    labeled_feats[1:5, 1:5] = 1
    
    # Too big
    labeled_feats[10:15, 10:15] = 2
    
    basic_config['size-filter']['max_diagonal'] = 4.5
    
    tracked_image = feature.TrackedImage()
    tracked_image.add_features_from_map(
        labeled_feats, labeled_feats, labeled_feats)
    
    abc_tracker.filter_size(tracked_image, basic_config)
    
    assert len(tracked_image.features) == 2
    assert tracked_image.features[0].flag == Flag.GOOD
    assert tracked_image.features[1].flag == Flag.TOO_LONG


def test_fully_process_one_image(
        basic_config, mocker, map_with_features, feature_details_for_map):
    basic_config['seeds']['use_laplacian'] = False
    basic_config['seeds']['mode'] = "absolute"
    basic_config['seeds']['threshold'] = 1.9
    basic_config['dilation']['rounds'] = 8
    basic_config['main']['connect_diagonal'] = True

    fake_hdr = {"date-avg": "2022-01-01T01:01:01.1"}
    mocker.patch("bp_analysis.abc_tracker.fits.getdata",
                 return_value=(map_with_features, fake_hdr))
    
    result = abc_tracker.fully_process_one_image("input_file", basic_config)
    assert len(result.features) == len(feature_details_for_map)
    assert result.config is basic_config
    assert result.time == datetime.strptime(
        fake_hdr['date-avg'], "%Y-%m-%dT%H:%M:%S.%f")
    assert result.source_file == "input_file"
    
    validate_result_from_map_with_features(result, feature_details_for_map)


def validate_result_from_map_with_features(result, feature_details_for_map):
    details_by_brightest_px = dict()
    for r, c, w, flag in feature_details_for_map:
        details_by_brightest_px[(r, c)] = (r, c, w, flag)
    for found_feature in result.features:
        r, c = found_feature.brightest_pixel
        assert (r, c) in details_by_brightest_px
        real_feature = details_by_brightest_px[(r, c)]
        width, height = found_feature.cutout.shape
        assert width == height
        if found_feature.flag == Flag.FALSE_POS:
            assert width // 2 < real_feature[2]
        else:
            assert width // 2 == real_feature[2]
        assert found_feature.flag == real_feature[3]
        assert found_feature.is_good == (real_feature[3] == Flag.GOOD)
    
    # Ensure the feature IDs are unique
    assert len(result.features) == len(set(f.id for f in result.features))
    # Ensure the feature IDs are sequential from 1
    for i in range(1, len(result.features) + 1):
        assert i in result


def test_trimmed_image(basic_contour_config, contour_image, mocker):
    contour_image[3, 0] = 999
    contour_image[3, -1] = 999
    contour_image[0, 3] = 999
    contour_image[-1, 3] = 999
    fake_hdr = {"date-avg": "2022-01-01T01:01:01.1"}
    mocker.patch("bp_analysis.abc_tracker.fits.getdata",
                 return_value=(contour_image, fake_hdr))
    
    basic_contour_config['main']['trim_image'] = 1
    basic_contour_config['seeds']['use_laplacian'] = False
    basic_contour_config['seeds']['mode'] = "absolute"
    basic_contour_config['seeds']['threshold'] = 1.9
    tracked_image = abc_tracker.fully_process_one_image(
        "file", basic_contour_config)
    assert tracked_image.source_shape == contour_image.shape
    assert tracked_image.features[0].cutout_corner == (2, 3)


def test_temporal_smoothing(
        basic_config, mocker, map_with_features, feature_details_for_map):
    basic_config['seeds']['use_laplacian'] = False
    basic_config['seeds']['mode'] = "absolute"
    basic_config['seeds']['threshold'] = 1.9
    basic_config['dilation']['rounds'] = 8
    basic_config['main']['connect_diagonal'] = True
    basic_config['temporal-smoothing']['window_size'] = 3
    basic_config['temporal-smoothing']['n_required'] = 2
    
    # Create another map with only a single pixel for each feature (to
    # facilitate the assertion helper function)
    middle_map = 0 * map_with_features
    for r, c, w, f in feature_details_for_map:
        middle_map[r, c] = 1
    
    fake_hdr = {"date-avg": "2022-01-01T01:01:01.1"}
    def getdata(filename, **kwargs):
        if filename[-6] in ('1', '3'):
            return map_with_features, fake_hdr
        return middle_map, fake_hdr
    mocker.patch("bp_analysis.abc_tracker.fits.getdata", getdata)
    mocker.patch("bp_analysis.abc_tracker.os.listdir",
                 return_value=('1.fits', '2.fits', '3.fits'))
    
    feature_details_no_fpos = [f for f in feature_details_for_map
                               if f[-1] != Flag.FALSE_POS]
    tracked_images = abc_tracker.id_files(basic_config, dir='dir')
    for ti in tracked_images:
        validate_result_from_map_with_features(ti, feature_details_no_fpos)
    
    # Now raise the requirement so the middle frame doesn't have any features
    # (as the only lit pixels are below the seed threshold)
    basic_config['temporal-smoothing']['n_required'] = 3
    tracked_images = abc_tracker.id_files(basic_config, dir='dir')
    assert len(tracked_images[0].features) == 0
    
    # Now let's make everything appear as one-pixel features
    middle_map[middle_map > 0] = 2
    tracked_images = abc_tracker.id_files(basic_config, dir='dir')
    assert len(tracked_images[0].features) == len(feature_details_no_fpos)
    for f in tracked_images[0].features:
        assert f.flag == Flag.TOO_SMALL
        assert f.size == 1


def test_remove_overlapping_fpos(basic_config, mocker):
    basic_config['seeds']['use_laplacian'] = False
    basic_config['seeds']['mode'] = "absolute"
    basic_config['seeds']['threshold'] = 1.9
    basic_config['dilation']['rounds'] = 8
    basic_config['main']['connect_diagonal'] = True
    basic_config['temporal-smoothing'] = {}
    basic_config['temporal-smoothing']['window_size'] = 3
    basic_config['temporal-smoothing']['n_required'] = 2
    
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
    
    fpos_image = np.zeros((50, 50))
    add_feature(fpos_image, 25, 25, 12)
    good_image = np.zeros((50, 50))
    add_feature(good_image, 25, 25, 8)
    
    fake_hdr = {"date-avg": "2022-01-01T01:01:01.1"}
    
    def getdata(filename, **kwargs):
        if filename[-6] in ('1', '3'):
            return good_image, fake_hdr
        return fpos_image, fake_hdr
    
    mocker.patch("bp_analysis.abc_tracker.fits.getdata", getdata)
    mocker.patch("bp_analysis.abc_tracker.os.listdir",
                 return_value=('1.fits', '2.fits', '3.fits'))

    tracked_images = abc_tracker.id_files(basic_config, dir='dir')
    assert len(tracked_images[0].features) == 1
    assert tracked_images[0].features[0].flag == Flag.GOOD
