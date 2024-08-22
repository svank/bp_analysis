#!/usr/bin/env python3

# Amorphous, Bending Contortionist Tracker

import copy
from datetime import datetime
import multiprocessing
import os
import tomllib

from astropy.io import fits
import numba
from numba.typed import List
import numpy as np
import scipy as scp
import scipy.ndimage
import scipy.signal
from tqdm.contrib.concurrent import process_map

from .config_utils import get_cfg
from .feature import TrackedImage
from .status import Flag
from .tracking_utils import gen_coord_map, gen_kernel


def calc_laplacian(image, config, kernel=None):
    if kernel is None:
        width = get_cfg(config, 'seeds-laplacian', 'width')
        if width % 2 == 0:
            raise ValueError(
                "Laplacian kernel width must be an odd number of pixels")
        kernel = np.full((width, width), -1)
        kernel[width//2, width//2] = kernel.size - 1
    else:
        assert len(kernel.shape) == 2
    
    return scp.signal.convolve2d(image, kernel, mode='valid').astype(image.dtype)


def _pad_to_shape(shape, arr):
    if arr.shape != shape:
        pad = shape[0] - arr.shape[0]
        return np.pad(arr, pad // 2)
    return arr


def find_seeds(src_image, config, **kwargs):
    n_sigma = get_cfg(config, 'seeds', 'n_sigma')
    if get_cfg(config, 'seeds', 'use_laplacian'):
        image = calc_laplacian(src_image, config, **kwargs)
        laplacian = image
    else:
        image = src_image
        laplacian = None
    
    mean = np.mean(image)
    std = np.std(image)
    
    image = _pad_to_shape(src_image.shape, image)
    if laplacian is not None:
        laplacian = _pad_to_shape(src_image.shape, laplacian)
    
    seed_mode = get_cfg(config, 'seeds', 'mode')
    if seed_mode == 'relative':
        seeds = image > (mean + n_sigma * std)
    elif seed_mode == 'absolute':
        seeds = image > get_cfg(config, 'seeds', 'threshold')
    else:
        raise RuntimeError("'seed_mode' must be one of 'relative', 'absolute'")
    
    return seeds, laplacian


def get_n_rounds_dilation(config):
    return get_cfg(config, 'dilation', 'rounds')


def dilate(config, *args, **kwargs):
    method = get_cfg(config, 'dilation', 'method')
    if method == "laplacian":
        if '_region_rounds_override' in kwargs:
            del kwargs['_region_rounds_override']
        return dilate_laplacian(config, *args, **kwargs)
    elif method == "contour":
        if 'laplacian' in kwargs:
            del kwargs['laplacian']
        return dilate_contour(config, *args, **kwargs)
    else:
        raise ValueError(f"Unrecognized dilation method '{method}'")


def dilate_laplacian(config, seeds, im=None, laplacian=None, mask=None,
        n_rounds=None):
    if mask is None:
        if laplacian is None and im is not None:
            laplacian = calc_laplacian(im, config)
            laplacian = _pad_to_shape(im.shape, laplacian)
        if laplacian is not None:
            mask = laplacian > 0
        else:
            mask = np.ones_like(seeds)
    
    if n_rounds is None:
        n_rounds = get_n_rounds_dilation(config)
    
    struc = gen_kernel(get_cfg(config, 'main', 'connect_diagonal'))
    seeds = scp.ndimage.binary_dilation(
            seeds,
            structure=struc,
            iterations=n_rounds,
            mask=mask)
    return seeds


def dilate_contour(config, seeds, im, n_rounds=None, _region_rounds_override=0):
    if n_rounds is None:
        n_rounds = get_n_rounds_dilation(config)
    req_downhill = get_cfg(config, 'dilation-contour', 'require_downhill')
    thresh = get_cfg(config, 'dilation-contour', 'threshold')
    if not (min_region_size := get_cfg(
            config, 'dilation-contour', 'region_size')):
        min_region_size = get_cfg(config, 'dilation-contour', 'region_scale')
        min_region_size *= (n_rounds + _region_rounds_override)
    max_intensity_range = get_cfg(
        config, 'dilation-contour', 'max_intensity_range')
    contour_lower_thresh = get_cfg(config, 'dilation-contour', 'lower_thresh')
    low_percentile = get_cfg(
        config, 'dilation-contour', 'region_low_percentile')
    high_percentile = get_cfg(
        config, 'dilation-contour', 'region_high_percentile')
    struc = gen_kernel(get_cfg(config, 'main', 'connect_diagonal'))
    labeled_feats_original, n_feats = scipy.ndimage.label(
            seeds != 0, struc)
    if n_feats == 0:
        # If there's no dilation to be done, we should stop now, or we'll
        # raise some Numba errors
        return seeds
    coord_map = gen_coord_map(labeled_feats_original)
    
    ids_to_expand = list(coord_map.keys())
    
    max_intensity_mode = get_cfg(
        config, 'dilation-contour', 'max_intensity_mode')
    if max_intensity_mode == 'relative':
        max_intensity_range *= im.mean()
    elif max_intensity_mode == 'absolute':
        # Satisfy numba
        max_intensity_range = np.array(max_intensity_range)
    else:
        raise RuntimeError("Unrecognized max_intensity_mode. "
                           "Must be 'absolute' or 'relative'")
    
    # Satisfy numba
    max_intensity_range = max_intensity_range.astype(np.float32)
    
    # Figure out which directions to look in when dilating
    struc[1, 1] = 0
    dir_r, dir_c = np.nonzero(struc)
    directions = List(zip(dir_r - 1, dir_c - 1))
    
    to_expand = List()
    for id in ids_to_expand:
        rs, cs = coord_map[id]
        rmin, rmax, cmin, cmax = _feat_neighborhood(
            rs, cs, im, min_region_size)
        region = im[rmin:rmax, cmin:cmax]
        if not np.isnan(contour_lower_thresh):
            region = region[region > contour_lower_thresh]
        feat_min, feat_max = np.percentile(
            region, [low_percentile, high_percentile])
        if feat_max - feat_min > max_intensity_range:
            feat_max = feat_min + max_intensity_range
        # All the type-casting here is to satisfy numba
        rs = List(rs.astype(int))
        cs = List(cs.astype(int))
        to_expand.append((rs, cs, feat_min, feat_max))
    
    labeled_feats = _dilate_contour_inner(
            labeled_feats_original.copy(), im, to_expand, directions,
            n_rounds, req_downhill, thresh)
    
    return labeled_feats > 0


def _feat_neighborhood(rs, cs, im, window_size):
    window_size = int(window_size)
    rmin = max(0, rs.min() - window_size)
    cmin = max(0, cs.min() - window_size)
    rmax = min(im.shape[0], rs.max() + window_size + 1)
    cmax = min(im.shape[1], cs.max() + window_size + 1)
    return rmin, rmax, cmin, cmax


@numba.njit(cache=True)
def _dilate_contour_inner(feats, im, to_expand, directions, n_rounds,
        req_downhill, thresh):
    # This loop is its own function so it can be compiled by numba
    for i in range(n_rounds):
        expanding = to_expand
        to_expand = List()
        # Loop over each feature
        for rs, cs, feat_min, feat_max in expanding:
            new_rs, new_cs = List(), List()
            # Loop over each coord in the feature
            for r, c in zip(rs, cs):
                cur = im[r, c]
                # Look in each direction
                for dr, dc in directions:
                    if (r+dr < 0 or r+dr >= feats.shape[0]
                            or c+dc < 0 or c+dc >= feats.shape[1]):
                        continue
                    if feats[r+dr, c+dc]:
                        # Pixel is already part of a feature
                        continue
                    target = im[r+dr, c+dc]
                    if req_downhill and target > cur:
                        # Would not be a downhill movement
                        continue
                    if target < feat_min + thresh * (feat_max - feat_min):
                        # New pixel is too faint
                        continue
                    # Expand to this neighbor feature
                    feats[r+dr, c+dc] = feats[r, c]
                    new_rs.append(r + dr)
                    new_cs.append(c + dc)
            
            if len(new_rs):
                to_expand.append(
                        (new_rs, new_cs, feat_min, feat_max))
    return feats


def remove_edge_touchers(labeled_feats, tracked_image):
    edges = np.concatenate((
        labeled_feats[0:2].flatten(),
        labeled_feats[-2:].flatten(),
        labeled_feats[:, 0:2].flatten(),
        labeled_feats[:, -2:].flatten()))
    ids = np.unique(edges[edges != 0])
    for id in ids:
        tracked_image[id].flag = Flag.EDGE


def remove_false_positives(labeled_feats, laplacian, config, im, seeds,
                           tracked_image):
    """
    Remove features for which another round of dilation would add too many
    pixels, on the idea that, after the rounds already done, true bright points
    should be mostly marked and the surrounding pixels should be dark lanes,
    while false positives are inside granules and surrounded by more bright
    pixels.
    """
    # Dilate normally but with an extra round to see what would have been added
    # to each feature
    masked_dilation = dilate(
        config, seeds, im=im,
        laplacian=laplacian, n_rounds=1+get_n_rounds_dilation(config),
        _region_rounds_override=-1)
    # Dilate the existing features without restrictions to see how many pixels
    # surround each feature. As we just need to expand the feature uniformly,
    # it doesn't matter which dilation method we use.
    full_dilation = dilate_laplacian(
            config, labeled_feats > 0, mask=np.ones_like(im), n_rounds=1)
    
    # Uniquely label contiguous regions
    struc = gen_kernel(get_cfg(config, 'main', 'connect_diagonal'))
    labeled_feats_dilated, _ = scipy.ndimage.label(
            masked_dilation > 0, struc)
    labeled_feats_full_dilated, _ = scipy.ndimage.label(
            full_dilation > 0, struc)
    coord_map_dilated = gen_coord_map(labeled_feats_dilated)
    coord_map_full_dilated = gen_coord_map(labeled_feats_full_dilated)
    
    # Test each feature
    for feature in tracked_image.features:
        rs, cs = feature.indices
        
        # The feature's ID is not necessarily the same in all three maps. To
        # find our feature, let's find the brightest pixel in the "true" map and
        # check the ID of that pixel in the other two maps.
        brightest = np.argmax(im[rs, cs])
        r = rs[brightest]
        c = cs[brightest]
        id_dilated = labeled_feats_dilated[r, c]
        id_full_dilated = labeled_feats_full_dilated[r, c]
        
        if id_full_dilated == 0 or id_dilated == 0:
            raise RuntimeError("Feature can't be found post-dilation")
        
        coords_dilated = coord_map_dilated[id_dilated]
        coords_full_dilated = coord_map_full_dilated[id_full_dilated]
        
        # Count up rejected pixels
        orig_pix = len(rs)
        possible_pix = len(coords_full_dilated[0]) - orig_pix
        gained_pix = len(coords_dilated[0]) - orig_pix
        
        # Mark features that don't meet the threshold
        if (gained_pix / possible_pix
                > get_cfg(config, 'false-pos-filter', 'threshold')):
            feature.flag = Flag.FALSE_POS


def filter_close_neighbors(labeled_feats, config, tracked_image):
    """
    Remove features that are too close to another feature, on the grounds that
    there may be room for confusion.
    """
    closeness = get_cfg(config, 'proximity-filter', 'threshold')
    size_ratio_thresh = get_cfg(
        config, 'proximity-filter', 'ignore_below_size_ratio')
    # Iterate over all features in image
    for feature in tracked_image.features:
        if feature.flag == Flag.CLOSE_NEIGHBOR:
            # This feature was already flagged as too close to one of its
            # neighbors
            continue
        rmin, rmax, cmin, cmax = _feat_neighborhood(
            *feature.indices, labeled_feats, closeness)
        region = labeled_feats[rmin:rmax, cmin:cmax] == feature.id
        
        # Make a `closeness` by `closeness` circular structuring element
        x, y = np.ogrid[-closeness:closeness + 1, -closeness:closeness + 1]
        struct = x ** 2 + y ** 2 <= closeness ** 2
        
        # Dilate
        # Note that this need not be the configured dilation mechanism, because
        # here we're just wanting to know what other features are nearby
        dilated = scp.ndimage.binary_dilation(
            region,
            structure=struct,
            iterations=1)
        
        # Get the pixels marked on the canvas, and translate those coordinates
        # to the real image
        rvicinity, cvicinity = np.nonzero(dilated & ~region)
        rvicinity += rmin
        cvicinity += cmin
        
        # Look up those neighboring pixels and see if they contain
        # other features
        neighbors = labeled_feats[(rvicinity, cvicinity)]
        neighbors = np.unique(neighbors[neighbors != 0])
        neighbors = [tracked_image[id] for id in neighbors]
        for neighbor in neighbors:
            # This feature is too close to another feature,
            # so we flag it
            if feature.size / neighbor.size >= size_ratio_thresh:
                neighbor.flag = Flag.CLOSE_NEIGHBOR
        if len(neighbors) and any(n.size / feature.size >= size_ratio_thresh
                                  for n in neighbors):
            feature.flag = Flag.CLOSE_NEIGHBOR


def filter_size(tracked_image: TrackedImage, config):
    min_size = get_cfg(config, 'size-filter', 'min_size')
    max_size = get_cfg(config, 'size-filter', 'max_size')
    max_diagonal = get_cfg(config, 'size-filter', 'max_diagonal')
    
    for feature in tracked_image.features:
        if not feature.is_good:
            continue
        rs, cs = feature.indices
        
        if min_size > 0 and rs.size < min_size:
            feature.flag = Flag.TOO_SMALL
            continue
        if max_size > 0 and rs.size > max_size:
            feature.flag = Flag.TOO_BIG
            continue
        
        if max_diagonal > 0:
            rmin = np.min(rs)
            cmin = np.min(cs)
            rmax = np.max(rs)
            cmax = np.max(cs)
            
            diagonal = np.sqrt((rmax - rmin) ** 2 + (cmax - cmin) ** 2)
            if diagonal > max_diagonal:
                feature.flag = Flag.TOO_LONG
                continue


def id_files(config_file, dir=None, silent=False, procs=None):
    if isinstance(config_file, str):
        with open(config_file, 'rb') as f:
            config = tomllib.load(f)
    else:
        config = config_file
    if dir is None and 'dirs' in config and 'data_dir' in config['dirs']:
        dir = config['dirs']['data_dir']
    
    files = scan_directory_for_data(dir, config)
    iterable = [(file, config) for file in files]
    if not silent:
        tracked_images = process_map(wrapper, iterable, chunksize=1,
                max_workers=procs if procs else os.cpu_count())
    else:
        with multiprocessing.Pool(processes=procs) as p:
            tracked_images = p.starmap(
                fully_process_one_image, iterable, chunksize=1)
    return tracked_images


def wrapper(x):
    fully_process_one_image(*x)


def scan_directory_for_data(dir):
    files = sorted(os.listdir(dir))
    files = [os.path.join(dir, f) for f in files if f.endswith('fits')]
    return files


def load_data(file, config):
    data, hdr = fits.getdata(file, header=True)
    time = datetime.strptime(hdr['date-avg'], "%Y-%m-%dT%H:%M:%S.%f")
    
    trim = get_cfg(config, 'main', 'trim_image')
    if trim:
        data = data[trim:-trim, trim:-trim]
    return time, data


def fully_process_one_image(file, config) -> TrackedImage:
    if isinstance(config, str):
        with open(config, 'rb') as f:
            config = tomllib.load(f)
    
    time, data = load_data(file, config)
    tracked_image = TrackedImage(config=config, time=time, source_file=file,
                                 source_shape=data.shape)
    
    if get_cfg(config, 'main', 'also_id_negative'):
        negative_tracked_image = copy.deepcopy(tracked_image)
        neg_data = -data
        mask = neg_data > 0
        id_image(data, config, negative_tracked_image, mask)
        for feature in negative_tracked_image.features:
            feature.feature_class = 'negative'
        mask = data > 0
    else:
        mask = None
    
    id_image(data, config, tracked_image, mask)
    
    if get_cfg(config, 'main', 'also_id_negative'):
        for feature in tracked_image.features:
            feature.feature_class = 'positive'
        tracked_image.merge_features(negative_tracked_image)
    
    trim = get_cfg(config, 'main', 'trim_image')
    if trim:
        for feature in tracked_image.features:
            feature.cutout_corner = (feature.cutout_corner[0] + trim,
                                     feature.cutout_corner[1] + trim)
    return tracked_image


def id_image(im, config, tracked_image, mask=None):
    if get_cfg(config, 'main', 'subtract_data_min'):
        im = im - im.min()
    seeds, laplacian = find_seeds(im, config=config)
    features = dilate(config, seeds, im=im, laplacian=laplacian)
    
    if mask is not None:
        features *= mask
    
    struc = gen_kernel(get_cfg(config, 'main', 'connect_diagonal'))
    labeled_feats, n_feat = scipy.ndimage.label(features, struc)
    tracked_image.add_features_from_map(labeled_feats, im, seeds)
    
    remove_edge_touchers(labeled_feats, tracked_image)
    remove_false_positives(
        labeled_feats, laplacian, config, im, seeds, tracked_image)
    filter_close_neighbors(labeled_feats, config, tracked_image)
    filter_size(tracked_image, config)
