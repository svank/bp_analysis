import configparser

import numpy as np
import pytest

from .. import abc_tracker


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

        min_size = 8
        max_size = 200
        max_diagonal = 30
        max_size_change_pct = 50
        max_size_change_px = 10
    """)
    config = config['main']
    return config


def test_calc_laplacian():
    image = np.ones((3,3))
    image[1,1] = 3
    laplacian = abc_tracker.calc_laplacian(image)
    assert laplacian.shape == (1,1)
    assert laplacian[0,0] == 16
    
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    kernel = kernel / 8
    laplacian = abc_tracker.calc_laplacian(image, kernel)
    assert laplacian.shape == (1,1)
    assert laplacian[0,0] == 2


def test_find_seeds(basic_config):
    # There are no seeds in this rather flat image
    np.random.seed(1)
    image = np.random.random((100, 100))
    
    seeds, laplacian = abc_tracker.find_seeds(
        image, basic_config, print_stats=False)
    
    np.testing.assert_array_equal(seeds, 0)
    
    
    # Create an obvious seed
    image[30, 30] = 100
    
    seeds, laplacian = abc_tracker.find_seeds(
        image, basic_config, print_stats=False)
    
    assert seeds[29, 29] == 1
    seeds[29, 29] = 0
    np.testing.assert_array_equal(seeds, 0)
    
    
    # Adjust the n_sigma parameter to just less than it needs to be
    mean = np.mean(laplacian)
    std = np.std(laplacian)
    
    n_sigma = (laplacian[29, 29] - mean) / std
    
    basic_config['n_sigma'] = str(n_sigma - .1)
    
    seeds, laplacian = abc_tracker.find_seeds(
        image, basic_config, print_stats=False)
    
    assert seeds[29, 29] == 1
    seeds[29, 29] = 0
    np.testing.assert_array_equal(seeds, 0)
    
    
    # Adjust the n_sigma parameter to just above what it needs to be, so there
    # will be no seeds
    basic_config['n_sigma'] = str(n_sigma + .1)
    
    seeds, laplacian = abc_tracker.find_seeds(
        image, basic_config, print_stats=False)
    
    np.testing.assert_array_equal(seeds, 0)


def test_find_seeds_absolute_thresh(basic_config):
    basic_config['seed_mode'] = 'absolute'
    # n.b. because the default laplacian kernel isn't normalized, this is 8
    # times higher than it "should" be
    basic_config['seed_thresh'] = '8.08'
    
    # There are no seeds in this rather flat image
    np.random.seed(1)
    image = np.random.random((100, 100))
    
    seeds, laplacian = abc_tracker.find_seeds(
        image, basic_config, print_stats=False)
    
    np.testing.assert_array_equal(seeds, 0)
    
    
    # Create a seed
    image[30, 30] = 2
    
    seeds, laplacian = abc_tracker.find_seeds(
        image, basic_config, print_stats=False)
    
    assert seeds[29, 29] == 1
    seeds[29, 29] = 0
    np.testing.assert_array_equal(seeds, 0)


def test_find_seeds_no_laplacian(basic_config):
    basic_config['seed_use_laplacian'] = 'False'
    # Just easier to test in absolute mode
    basic_config['seed_mode'] = 'absolute'
    basic_config['seed_thresh'] = '1.01'
    
    # There are no seeds in this rather flat image
    np.random.seed(1)
    image = np.random.random((100, 100))
    
    seeds, laplacian = abc_tracker.find_seeds(
        image, basic_config, print_stats=False)
    
    np.testing.assert_array_equal(seeds, 0)
    
    
    # Create a seed
    image[30, 30] = 1.02
    
    seeds, laplacian = abc_tracker.find_seeds(
        image, basic_config, print_stats=False)
    
    assert seeds[30, 30] == 1
    seeds[30, 30] = 0
    np.testing.assert_array_equal(seeds, 0)


def test_dilate_laplacian(basic_config):
    basic_config['dilation_method'] = 'laplacian'
    basic_config['dilation_rounds'] = '1'
    basic_config['connect_diagonal'] = 'True'
    
    image = np.zeros((50, 50))
    image[29:34, 29:34] = .6
    image[30:33, 30:33] = 1
    image[31, 31] = 2
    
    seeds = np.zeros((50, 50), dtype=bool)
    seeds[31, 31] = 1
    # Because the laplacian generated from the image will be one pixel smaller
    # on each edge
    seeds = seeds[1:-1, 1:-1]
    
    dilated_seeds = abc_tracker.dilate(basic_config, seeds, im=image)
    np.testing.assert_array_equal(dilated_seeds[29:32, 29:32], 1)
    dilated_seeds[29:32, 29:32] = 0
    np.testing.assert_array_equal(dilated_seeds, 0)
    
    
    # Add another round
    basic_config['dilation_rounds'] = '2'
    
    dilated_seeds = abc_tracker.dilate(basic_config, seeds, im=image)
    np.testing.assert_array_equal(dilated_seeds[28:33, 28:33], 1)
    dilated_seeds[28:33, 28:33] = 0
    np.testing.assert_array_equal(dilated_seeds, 0)
    
    # Pass that extra round explicitly
    basic_config['dilation_rounds'] = '1'
    
    dilated_seeds = abc_tracker.dilate(basic_config, seeds, im=image, n_rounds=2)
    np.testing.assert_array_equal(dilated_seeds[28:33, 28:33], 1)
    dilated_seeds[28:33, 28:33] = 0
    np.testing.assert_array_equal(dilated_seeds, 0)


def test_dilate_laplacian_masked(basic_config):
    basic_config['dilation_method'] = 'laplacian'
    basic_config['dilation_rounds'] = '1'
    basic_config['connect_diagonal'] = 'True'
    
    mask = np.zeros((50, 50))
    mask[30:32, 31] = 1
    
    seeds = np.zeros((50, 50), dtype=bool)
    seeds[31, 31] = 1
    
    dilated_seeds = abc_tracker.dilate(basic_config, seeds, mask=mask)
    np.testing.assert_array_equal(dilated_seeds[30:32, 31], 1)
    dilated_seeds[30:32, 31] = 0
    np.testing.assert_array_equal(dilated_seeds, 0)
    

def test_dilate_laplacian_provided_laplacian(basic_config):
    basic_config['dilation_method'] = 'laplacian'
    basic_config['dilation_rounds'] = '1'
    basic_config['connect_diagonal'] = 'True'
    
    laplacian = np.zeros((50, 50))
    laplacian[30:32, 31] = 1
    
    seeds = np.zeros((50, 50), dtype=bool)
    seeds[31, 31] = 1
    
    dilated_seeds = abc_tracker.dilate(
        basic_config, seeds, laplacian=laplacian)
    np.testing.assert_array_equal(dilated_seeds[30:32, 31], 1)
    dilated_seeds[30:32, 31] = 0
    np.testing.assert_array_equal(dilated_seeds, 0)
    