#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy.ndimage


def gen_coord_map(labeled_feats):
    """Generates an id -> coords mapping for fast lookups"""
    # Get a sorted list of feature IDs and corresponding coords
    sorted_idx = np.argsort(labeled_feats.flatten(), kind="stable")
    # The type coercion avoids a numba warning
    feats = labeled_feats.astype(np.int64).flatten()[sorted_idx]
    indices = np.indices(labeled_feats.shape, dtype=np.int16)
    rs = indices[0].flatten()[sorted_idx]
    cs = indices[1].flatten()[sorted_idx]
    return {**_gen_coord_map_compilable(feats, rs, cs)}


@numba.njit(cache=True)
def _gen_coord_map_compilable(feats, rs, cs):
    i, j = 0, 0
    coord_map = dict()
    while j < len(feats):
        j += 1
        if j == len(feats) or feats[i] != feats[j]:
            # The copy is important, or else each little list of coords holds a
            # reference to the big list, and memory usage grows quickly
            coord_map[feats[i]] = (rs[i:j].copy(), cs[i:j].copy())
            i = j
    return coord_map


def gen_kernel(connect_diagonal=True):
    return scipy.ndimage.generate_binary_structure(
            2, 2 if connect_diagonal else 1)

