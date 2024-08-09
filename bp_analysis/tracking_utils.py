#!/usr/bin/env python3

import scipy.ndimage


def gen_coord_map(labeled_feats):
    """Generates an id -> coords mapping for fast lookups"""
    return scipy.ndimage.value_indices(labeled_feats, ignore_value=0)


def gen_kernel(connect_diagonal=True):
    return scipy.ndimage.generate_binary_structure(
            2, 2 if connect_diagonal else 1)


def get_cfg(config, section, key, default):
    return config.get(section, {}).get(key, default)
