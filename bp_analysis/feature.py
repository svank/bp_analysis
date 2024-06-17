import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

from . import db_analysis

class Feature:
    def __init__(self, id, cutout_corner, cutout, seed_cutout, flag,
                 feature_class):
        self.id = id
        self.cutout_corner = cutout_corner
        self.cutout = cutout
        self.seed_cutout = seed_cutout
        self.flag = flag
        self.feature_class = feature_class
        self.image = None
    
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
    def __init__(self, source_file, time, config):
        self.features = []
        self.source_file = source_file
        self.time = time
        self.config = config
    
    def add_features(self, *features):
        for feature in features:
            feature.image = self
            self.features.append(feature)
    
    def plot_features_onto(self, ax, **kwargs):
        for feature in self.features:
            feature.plot_onto(ax, **kwargs)
    
    def plot_features(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        self.plot_features_onto(ax, **kwargs)
        ax.set_aspect('equal')