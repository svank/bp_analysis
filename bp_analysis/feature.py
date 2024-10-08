import copy
from datetime import datetime
import functools

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

from . import feature_plotting
from .status import Flag, EventFlag, SequenceFlag


class Feature(feature_plotting.FeaturePlottingMixin):
    def __init__(self, id, cutout_corner, cutout, data_cutout, seed_cutout,
                 flag=Flag.GOOD, feature_class=None):
        self.id = id
        self.cutout_corner = cutout_corner
        self.cutout = cutout.astype(bool, copy=False)
        self.seed_cutout = seed_cutout
        self.data_cutout = data_cutout
        self.flag = flag
        self.feature_class = feature_class
        self.image = None
        self.sequence: FeatureSequence = None
        self.time = None
    
    @property
    def brightest_pixel(self):
        idx = np.argmax(self.data_cutout[self.cutout])
        rs, cs = np.nonzero(self.cutout)
        r, c = rs[idx], cs[idx]
        r += self.cutout_corner[0]
        c += self.cutout_corner[1]
        return r, c
    
    @property
    def indices(self):
        rs, cs = np.nonzero(self.cutout)
        rs += self.cutout_corner[0]
        cs += self.cutout_corner[1]
        return rs, cs
    
    @property
    def size(self):
        return np.sum(self.cutout)

    @property
    @functools.cache
    def coord_set(self):
        rs, cs = self.indices
        coord_set = set()
        for r, c in zip(rs, cs):
            coord_set.add((r, c))
        return coord_set
    
    @property
    def is_good(self):
        return (self.flag == Flag.GOOD
                and (self.sequence is None
                     or self.sequence.flag == SequenceFlag.GOOD))
    
    def overlaps(self, other: "Feature"):
        if self.left > other.right or other.left > self.right:
            return False
        if self.bottom > other.top or other.bottom > self.top:
            return False
        return not self.coord_set.isdisjoint(other.coord_set)
    
    @property
    def left(self):
        return self.cutout_corner[1]
    
    @property
    def right(self):
        return self.cutout_corner[1] + self.cutout.shape[1]
    
    @property
    def bottom(self):
        return self.cutout_corner[0]
    
    @property
    def top(self):
        return self.cutout_corner[0] + self.cutout.shape[0]
    
    def __repr__(self):
        return (f"<Feature {self.id}, {repr(self.flag)}, {self.size} px, "
                f"@{self.cutout_corner}>")


class FeatureSequence:
    def __init__(self):
        self.id = None
        self.features: list[Feature] = []
        self.origin = EventFlag.NORMAL
        self.fate = EventFlag.NORMAL
        self.origin_sequences: list[FeatureSequence] = []
        self.fate_sequences: list[FeatureSequence] = []
        self.origin_event_id = None
        self.fate_event_id = None
        self.absorbs: list[FeatureSequence] = []
        self.releases: list[FeatureSequence] = []
        self.feature_flag = None
        self.flag = None
    
    def __getitem__(self, item) -> "Feature":
        for feature in self.features:
            if feature is item:
                return feature
            if feature.id == item:
                return feature
            if isinstance(item, datetime) and feature.time == item:
                return feature
        raise KeyError("Feature not found")
    
    def __contains__(self, item):
        try:
            self[item]
            return True
        except KeyError:
            return False
    
    def __len__(self):
        return len(self.features)
    
    def add_features(self, *features):
        for feature in features:
            self.features.append(feature)
            feature.sequence = self
    
    def remove_feature(self, feature):
        self.features.remove(feature)
        feature.sequence = None
    
    def __repr__(self):
        if len(self.features) < 7:
            fid_str = ' '.join(str(f.id) for f in self.features)
            fid_str = f", feat ids [{fid_str}]"
        else:
            fid_str = ""
        return (f"<FeatureSequence {self.id}, {len(self.features)} "
                f"{repr(self.feature_flag)} feats, {repr(self.flag)}, "
                f"{repr(self.origin)} to {repr(self.fate)}{fid_str}>")


class TrackedImage:
    def __init__(self, source_file=None, source_shape=None, time=None, config=None):
        self.features: list[Feature] = []
        self.source_file = source_file
        self.source_shape = source_shape
        self.time = time
        self.config = config
        self.plot_bounds = None
        self.slice: tuple[slice] = None
        self.unsliced_image: TrackedImage = None
    
    def __repr__(self):
        return f"<TrackedImage, t={self.time}, {len(self.features)} features>"
    
    def add_features(self, *features):
        for feature in features:
            feature.image = self
            feature.time = self.time
            self.features.append(feature)
    
    def add_features_from_map(self, feature_map, data, seeds, classes=None):
        regions = scipy.ndimage.find_objects(feature_map)
        for id, region in enumerate(regions, start=1):
            corner = (region[0].start, region[1].start)
            feature_cutout = feature_map[region] == id
            data_cutout = data[region].copy()
            seed_cutout = np.where(feature_cutout, seeds[region], 0)
            feature_flag = Flag.GOOD
            if classes:
                feature_class = classes[region][feature_cutout][0]
            else:
                feature_class = None
            feature = Feature(
                id=id,
                cutout_corner=corner,
                cutout=feature_cutout,
                data_cutout=data_cutout,
                seed_cutout=seed_cutout,
                flag=feature_flag,
                feature_class=feature_class)
            self.add_features(feature)
    
    def merge_features(self, other):
        my_max_id = max(f.id for f in self.features)
        for feature in other.features:
            feature.id += my_max_id
            self.add_features(feature)
    
    def plot_features_onto(self, ax, legend=False, **kwargs):
        for feature in self.features:
            if self.plot_bounds is not None:
                if (feature.left > self.plot_bounds[0][1]
                        or feature.right < self.plot_bounds[0][0]
                        or feature.bottom > self.plot_bounds[1][1]
                        or feature.top < self.plot_bounds[1][0]):
                    continue
            feature.plot_onto(ax, **kwargs, plot_bounds=self.plot_bounds)
        if legend:
            simple_colors = kwargs.get('simple_colors', False)
            feature_plotting._draw_color_legend(ax, simple_colors=simple_colors)
    
    def plot_features(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        self.plot_features_onto(ax, **kwargs)
        ax.set_aspect('equal')
    
    def feature_map(self):
        if self.unsliced_image is not None:
            return self.unsliced_image.feature_map()[self.slice]
        map = np.zeros(self.source_shape, dtype=int)
        for feature in self.features:
            map[feature.indices] = feature.id
        return map
    
    def data_cutout_map(self):
        if self.unsliced_image is not None:
            return self.unsliced_image.data_cutout_map()[self.slice]
        map = np.zeros(self.source_shape)
        for feature in self.features:
            r, c = feature.cutout_corner
            map[r:r+feature.cutout.shape[0],
                c:c+feature.cutout.shape[1]] = feature.data_cutout
        return map
    
    def seed_map(self):
        if self.unsliced_image is not None:
            return self.unsliced_image.seed_map()[self.slice]
        map = np.zeros(self.source_shape, dtype=bool)
        for feature in self.features:
            indices = list(feature.seed_cutout.nonzero())
            indices[0] += feature.cutout_corner[0]
            indices[1] += feature.cutout_corner[1]
            map[tuple(indices)] = 1
        return map
    
    def __getitem__(self, id) -> "Feature | TrackedImage":
        if isinstance(id, slice):
            id = (id, slice(0, -1, 1))
        if isinstance(id, tuple):
            if len(id) != 2:
                raise ValueError("Slice must be two dimensional")
            if any(not isinstance(s, slice) for s in id):
                raise ValueError("Invalid slice types")
            slices = []
            for s, size in zip(id, self.source_shape):
                if s.stop is None:
                    stop = size
                elif s.stop < 0:
                    stop = s.stop + size
                    if stop < 0:
                        stop = 0
                else:
                    stop = s.stop
                if s.start is None:
                    start = 0
                elif s.start < 0:
                    start = s.start + size
                    if start < 0:
                        start = 0
                else:
                    start = s.start
                if s.step is not None and s.step != 1:
                    raise ValueError("Stride != 1 not supported")
                slices.append(slice(start, stop, 1))
                
            sliced = copy.copy(self)
            sliced.features = [copy.copy(f) for f in sliced.features]
            for feature in sliced.features:
                corner = feature.cutout_corner
                feature.cutout_corner = (
                    corner[0] - slices[0].start,
                    corner[1] - slices[1].start,
                )
            sliced.plot_bounds = (
                (0, slices[1].stop - slices[1].start),
                (0, slices[0].stop - slices[0].start),
            )
            sliced.slice = tuple(slices)
            sliced.unsliced_image = self
            sliced.source_shape = (
                slices[0].stop - slices[0].start,
                slices[1].stop - slices[1].start
            )
            return sliced
        for feature in self.features:
            if feature.id == id:
                return feature
        raise KeyError(f"Feature ID {id} not found")
    
    def __contains__(self, item):
        try:
            self[item]
            return True
        except KeyError:
            return False
