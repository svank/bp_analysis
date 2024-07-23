import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

from . import abc_tracker, db_analysis


class Feature:
    def __init__(self, id, cutout_corner, cutout, data_cutout, seed_cutout,
                 flag, feature_class):
        self.id = id
        self.cutout_corner = cutout_corner
        self.cutout = cutout.astype(bool, copy=False)
        self.seed_cutout = seed_cutout
        self.data_cutout = data_cutout
        self.flag = flag
        self.feature_class = feature_class
        self.image = None
    
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
    def is_good(self):
        return self.flag == abc_tracker.GOOD
    
    def plot_onto(self, ax, ids=False):
        r, c = np.nonzero(self.cutout)
        r += self.cutout_corner[0]
        c += self.cutout_corner[1]
        color = {1: (.2, 1, .2, .8), -1: (1, .1, .1, .8),
                 -2: (1, 1, 1, .8), -3: (.1, .1, 1, .8)}[self.flag]
        db_analysis.outline_BP(r, c, scale=1, line_color=color, ax=ax)
        if ids:
            plt.text(np.mean(c), np.mean(r), self.id, color=color,
                     path_effects=[
                     pe.Stroke(linewidth=1, foreground='k'),
                     pe.Normal()],)
    
    def plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        self.plot_onto(ax, **kwargs)
        ax.set_aspect('equal')


class TrackedImage:
    def __init__(self, source_file=None, source_shape=None, time=None, config=None):
        self.features: list[Feature] = []
        self.source_file = source_file
        self.source_shape = source_shape
        self.time = time
        self.config = config
    
    def add_features(self, *features):
        for feature in features:
            feature.image = self
            self.features.append(feature)
    
    def add_features_from_map(self, feature_map, data, seeds, classes=None):
        regions = scipy.ndimage.find_objects(feature_map)
        for id, region in enumerate(regions, start=1):
            corner = (region[0].start, region[1].start)
            feature_cutout = feature_map[region] == id
            data_cutout = data[region].copy()
            seed_cutout = np.where(feature_cutout, seeds[region], 0)
            feature_flag = abc_tracker.GOOD
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
    
    def plot_features_onto(self, ax, **kwargs):
        for feature in self.features:
            feature.plot_onto(ax, **kwargs)
    
    def plot_features(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        self.plot_features_onto(ax, **kwargs)
        ax.set_aspect('equal')
    
    def feature_map(self):
        map = np.zeros(self.source_shape, dtype=int)
        for feature in self.features:
            map[feature.indices] = feature.id
        return map
    
    def __getitem__(self, id):
        for feature in self.features:
            if feature.id == id:
                return feature
        raise KeyError(f"Feature ID {id} not found")
