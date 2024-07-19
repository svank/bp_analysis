#!/usr/bin/env python3

# Amorphous, Bending Contortionist Tracker

import argparse
import configparser
from datetime import datetime
import multiprocessing
import os

from astropy.io import fits
import numba
from numba.typed import List
import numpy as np
import scipy as scp
import scipy.ndimage
import scipy.signal
from tqdm.contrib.concurrent import process_map

from .feature import Feature, TrackedImage
from .tracking_utils import gen_coord_map, gen_kernel


GOOD = 1
FALSE_POS = -1
CLOSE_NEIGHBOR = -2
EDGE = -3



def calc_laplacian(image, kernel=None):
    if kernel is None:
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    else:
        assert len(kernel.shape) == 2
    
    return scp.signal.convolve2d(image, kernel, mode='valid').astype(image.dtype)

def find_seeds(image, config, print_stats=True, **kwargs):
    n_sigma = config.getfloat('n_sigma', 3)
    if config.getboolean('seed_use_laplacian', True):
        image = calc_laplacian(image, **kwargs)
        laplacian = image
    else:
        laplacian = None
    
    mean = np.mean(image)
    std = np.std(image)
    
    if print_stats and laplacian is not None:
        print('"Laplacian" has μ={:.4e}, σ={:.4e}, max={:.4e}, min={:.4e}'.format(mean, std, np.max(laplacian), np.min(laplacian)))
    
    if config.get('seed_mode', 'relative') == 'relative':
        seeds = image > (mean + n_sigma * std)
    elif config.get('seed_mode', 'relative') == 'absolute':
        seeds = image > config.getfloat('seed_thresh', 1000)
    else:
        raise RuntimeError("'seed_mode' must be one of 'relative', 'absolute'")
    
    return seeds, laplacian

def get_n_rounds_dilation(config):
    return config.getint('dilation_rounds', 3)

def dilate(config, *args, **kwargs):
    method = config.get("dilation_method", "laplacian")
    if method == "laplacian":
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
            laplacian = calc_laplacian(im)
        if laplacian is not None:
            mask = laplacian > 0
        else:
            mask = np.ones_like(seeds)
    
    if n_rounds is None:
        n_rounds = get_n_rounds_dilation(config)
    
    struc = gen_kernel(config.getboolean('connect_diagonal', True))
    seeds = scp.ndimage.binary_dilation(
            seeds,
            structure=struc,
            iterations=n_rounds,
            mask=mask)
    return seeds

def dilate_contour(config, seeds, im, n_rounds=None):
    if n_rounds is None:
        n_rounds = get_n_rounds_dilation(config)
    req_downhill = config.getboolean('contour_require_downhill', True)
    thresh = config.getfloat('contour_threshold', 0.5)
    min_finding_scale = config.getfloat('contour_min_finding_scale', 3)
    max_intensity_range = config.getfloat('contour_max_intensity_range', 999)
    contour_lower_thresh = config.getfloat('contour_lower_thresh', None)
    
    struc = gen_kernel(config.getboolean('connect_diagonal', True))
    labeled_feats_original, n_feats = scipy.ndimage.label(
            seeds != 0, struc)
    if n_feats == 0:
        # If there's no dilation to be done, we should stop now, or we'll
        # raise some Numba errors
        return seeds
    coord_map = gen_coord_map(labeled_feats_original)
    
    ids_to_expand = list(coord_map.keys())
    
    max_values = {id: im[coord_map[id]].max() for id in ids_to_expand}
    
    if config.get('contour_max_intensity_mode', 'relative') == 'relative':
        max_intensity_range *= im.mean()
    elif config.get('contour_max_intensity_mode', 'relative') == 'absolute':
        # Satisfy numba
        max_intensity_range = np.array(max_intensity_range)
    else:
        raise RuntimeError("Unrecognized contour_max_intensity_mode. "
            "Must be 'absolute' or 'relative'")
    
    # Satisfy numba
    max_intensity_range = max_intensity_range.astype(np.float32)
    
    # Figure out which directions to look in when dilating
    struc[1, 1] = 0
    dir_r, dir_c = np.nonzero(struc)
    directions = List(zip(dir_r - 1, dir_c - 1))
    
    while True:
        to_expand = List()
        for id in ids_to_expand:
            rs, cs = coord_map[id]
            rmin, rmax, cmin, cmax = _feat_neighborhood(
                    rs, cs, im, min_finding_scale * n_rounds)
            region = im[rmin:rmax, cmin:cmax]
            if contour_lower_thresh is not None:
                region = region[region > contour_lower_thresh]
            feat_min = np.min(region)
            feat_max = max_values[id]
            if feat_max - feat_min > max_intensity_range:
                feat_max = feat_min + max_intensity_range
                max_is_clamped = True
            else:
                max_is_clamped = False
            # All the type-casting here is to satisfy numba
            rs = List(rs.astype(int))
            cs = List(cs.astype(int))
            to_expand.append((rs, cs, feat_min, feat_max, max_is_clamped))
        
        labeled_feats, needs_redo = _dilate_contour_inner(
                labeled_feats_original.copy(), im, to_expand, directions,
                n_rounds, req_downhill, thresh)
        
        if len(needs_redo):
            # These features need to be re-done, because (at least) one of the
            # pixels dilated to was brighter than any of the seed pixels,
            # invalidating the contour of the final feature.
            # Let's update our recorded per-feature maximum values. (Note that
            # if dilation had to cross a relatively dark area to reach this
            # bright pixel, it's possible that raising the max value and thus
            # the contour will mean the expanded feature never reaches its the
            # nominal maximum value we're setting here. That's weird but
            # acceptable, I think.)
            for r, c, id in needs_redo:
                value = im[r, c]
                if value > max_values[id]:
                    max_values[id] = value
            
            # We could think about just removing feature pixels that fall
            # outside the new contour, but what if those pixels would otherwise
            # have been grabbed by a different feature? It's more robust to
            # just re-do the whole dilation. So let's let the loop repeat and
            # re-do the dilation without updated max_values.
        else:
            break
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
    needs_redo = List()
    # This loop is its own function so it can be compiled by numba
    for i in range(n_rounds):
        expanding = to_expand
        to_expand = List()
        # Loop over each feature
        for rs, cs, feat_min, feat_max, max_is_clamped in expanding:
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
                    # Check if this changes our "feature maximum" brightness.
                    # (This is unlikely, but can happen when strictly-downhill
                    # expansion isn't required.)
                    if not max_is_clamped and target > feat_max:
                        needs_redo.append((r+dr, c+dc, feats[r, c]))
            
            if len(new_rs):
                to_expand.append(
                        (new_rs, new_cs, feat_min, feat_max, max_is_clamped))
    return feats, needs_redo

def remove_edge_touchers(features, config):
    # Uniquely label contiguous regions
    struc = gen_kernel(config.getboolean('connect_diagonal', True))
    labeled_feats, _ = scipy.ndimage.label(features > 0, struc)
    coord_map = gen_coord_map(labeled_feats)
    
    edges = np.concatenate((
        labeled_feats[0:2].flatten(),
        labeled_feats[-2:].flatten(),
        labeled_feats[:, 0:2].flatten(),
        labeled_feats[:, -2:].flatten()))
    ids = np.unique(edges[edges != 0])
    for id in ids:
        coords = coord_map[id]
        features[coords] = EDGE

def remove_false_positives(features, laplacian, config, im, seeds):
    """
    Remove features for which another round of dilation would add too many
    pixels, on the idea that, after the rounds already done, true bright points
    should be mostly marked and the surrounding pixels should be dark lanes,
    while false positives are inside granules and surrounded by more bright
    pixels.
    
    The input image is modified in-place, with rejected features replaced with -1
    """
    # Dilate normally but with an extra round to see what would have been added
    # to each feature
    masked_dilation = dilate(config, seeds, im=im,
            laplacian=laplacian, n_rounds=1+get_n_rounds_dilation(config))
    # Dilate the existing features without restrictions to see how many pixels
    # surround each feature
    full_dilation = dilate_laplacian(
            config, features, mask=np.ones_like(im), n_rounds=1)
    
    # Uniquely label contiguous regions
    struc = gen_kernel(config.getboolean('connect_diagonal', True))
    labeled_feats, n_feat = scipy.ndimage.label(
            features > 0, struc)
    labeled_feats_dilated, _ = scipy.ndimage.label(
            masked_dilation > 0, struc)
    labeled_feats_full_dilated, _ = scipy.ndimage.label(
            full_dilation > 0, struc)
    coord_map_features = gen_coord_map(labeled_feats)
    coord_map_dilated = gen_coord_map(labeled_feats_dilated)
    coord_map_full_dilated = gen_coord_map(labeled_feats_full_dilated)
    
    # Test each feature
    for i in range(1, n_feat+1):
        rs, cs = coord_map_features[i]
        
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
        # (Each dilated image is a binary image)
        orig_pix = len(rs)
        possible_pix = len(coords_full_dilated[0]) - orig_pix
        gained_pix = len(coords_dilated[0]) - orig_pix
        
        # Erase features that don't meet the threshold
        if gained_pix / possible_pix > config.getfloat('fpos_thres', 0.2):
            features[rs, cs] = FALSE_POS

def filter_close_neighbors(features, config):
    """
    Remove features that are too close to another feature, on the grounds that
    there may be room for confusion.
    
    The input image is modified in-place, with rejected features replaced with -2
    """
    struc = gen_kernel(config.getboolean('connect_diagonal', True))
    labeled_feats, n_feat = scipy.ndimage.label(features > 0, struc)
    coord_map = gen_coord_map(labeled_feats)
    
    closeness = config.getint('proximity_thresh', 4)
    # Iterate over all features in image
    for id in range(1, n_feat + 1):
        rmin, rmax, cmin, cmax = _feat_neighborhood(
            *coord_map[id], labeled_feats, closeness)
        region = labeled_feats[rmin:rmax, cmin:cmax] == id

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
        for neighbor in neighbors:
            # This feature is too close to another feature,
            # so we flag it
            features[coord_map[id]] = CLOSE_NEIGHBOR
            features[coord_map[neighbor]] = CLOSE_NEIGHBOR


def id_files(config_file, dir=None, silent=False, procs=None):
    if type(config_file) == configparser.ConfigParser:
        config = config_file
    else:
        config = configparser.ConfigParser()
        if not os.path.exists(config_file):
            raise RuntimeError(f"Config file '{config_file}' does not exist")
        config.read(config_file)
    if dir is None and 'dirs' in config and 'data_dir' in config['dirs']:
        dir = config['dirs']['data_dir']
    
    config = config['main']
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

    trim = config.getint('trim_image', 0)
    if trim:
        data = data[trim:-trim, trim:-trim]
    
    return time, data

def fully_process_one_image(file, config):
    time, data = load_data(file, config)
    
    features, seeds, feature_classes = id_image(data, config)
    
    struc = gen_kernel(config.getboolean('connect_diagonal', True))
    labeled_feats, n_feat = scipy.ndimage.label(features, struc)
    regions = scipy.ndimage.find_objects(labeled_feats)
    tracked_image = TrackedImage(config=config, time=time, source_file=file)
    for id, region in enumerate(regions, start=1):
        corner = (region[0].start, region[1].start)
        feature_cutout = labeled_feats[region] == id
        seed_cutout = np.where(feature_cutout, seeds[region], 0)
        feature_flag = features[region][feature_cutout][0]
        feature_class = feature_classes[region][feature_cutout][0]
        feature = Feature(
            id=id,
            cutout_corner=corner,
            cutout=feature_cutout,
            seed_cutout=seed_cutout,
            flag=feature_flag,
            feature_class=feature_class)
        tracked_image.add_features(feature)
    
    return tracked_image

def id_image(im, config, also_neg_override=False):
    if config.getboolean('also_id_negative', False) and not also_neg_override:
        raw_im = im.copy()
        im[im < 0] = 0
    if config.getboolean('subtract_data_min', True):
        im = im - im.min()
    seeds, laplacian = find_seeds(im, print_stats=False, config=config)
    if seeds.shape != im.shape:
        seeds = np.pad(seeds, 1)
        laplacian = np.pad(laplacian, 1)
    features = dilate(config, seeds, im=im, laplacian=laplacian)
    features = features.astype(np.int8)
    remove_edge_touchers(features, config)
    remove_false_positives(features, laplacian, config, im, seeds)
    filter_close_neighbors(features, config)
    
    if config.getboolean('also_id_negative', False) and not also_neg_override:
        im = -raw_im
        im[im < 0] = 0
        negative_features, negative_seeds, _ = id_image(im, config, True)
        feature_class = np.ones_like(features)
        feature_class[negative_features != 0] = 2
        if np.any(negative_features * features):
            print("Warning: pixels included in both polarities")
            feature_class[negative_features * features != 0] = 3
        features = np.where(negative_features != 0, negative_features, features)
        seeds += negative_seeds
    else:
        feature_class = np.ones_like(features)
    
    return features, seeds, feature_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
            help='Location of config file')
    parser.add_argument('--data', type=str,
            help='Directory containing data files')
    parser.add_argument('--out', type=str,
            help='Name of directory to hold output files. Data dir is prepended')
    args = parser.parse_args()
    
    if args.config is None:
        if args.data is None:
            raise RuntimeError("Data dir or config file must be given")
        else:
            args.config = os.path.join(args.data, 'tracking.cfg')
    
    id_files(args.data, args.out, args.config)
