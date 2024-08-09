from datetime import datetime
import functools

from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

from . import db_analysis
from .status import Flag, Event, SequenceFlag


# Text is legend labels
COLORS = {Flag.GOOD: ((.2, 1, .2, .8), "OK"),
          Flag.FALSE_POS: ((1, .1, .1, .8), "False pos"),
          Flag.CLOSE_NEIGHBOR: ((1, 1, 1, .8), "Proximity"),
          Flag.EDGE: ((.1, .1, 1, .8), "Edge"),
          Flag.TOO_SMALL: ((.1, 1, 1, .8), "Size"),
          Flag.TOO_BIG: ((.1, 1, 1, .8), ""),
          Flag.TOO_LONG: ((.1, 1, 1, .8), ""),
          SequenceFlag.TOO_SHORT: ((1, .1, 1, .8), "Short-lived"),
          }
SIMPLE_COLORS = {True: ((.1, 1, .1, .8), "OK"),
                 False: ((.1, .1, 1, .8), "Rejected")}


def _draw_color_legend(ax, simple_colors=False):
    lines = []
    names = []
    for color, name in (SIMPLE_COLORS if simple_colors else COLORS).values():
        if name:
            lines.append(Line2D(
                [0], [0], color=color[:3], lw=3, path_effects=[
                    pe.Stroke(linewidth=4, foreground=(0, 0, 0, .75)),
                    pe.Normal()]
            ))
            names.append(name)
    ax.legend(lines, names)


class Feature:
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
    
    def plot_onto(self, ax, ids=False, legend=False, label_flag=False,
                  label_seq_flag=False, simple_colors=False,
                  label_on_click=True):
        r, c = np.nonzero(self.cutout)
        r += self.cutout_corner[0]
        c += self.cutout_corner[1]
        if simple_colors:
            color = SIMPLE_COLORS[self.is_good][0]
        else:
            color = COLORS[self.flag][0]
            if (self.is_good and self.sequence is not None
                    and self.sequence.flag == SequenceFlag.TOO_SHORT):
                color = COLORS[SequenceFlag.TOO_SHORT][0]
        
        line = db_analysis.outline_BP(r, c, scale=1, line_color=color, ax=ax)
        text_pieces = []
        if ids:
            text_pieces.append(str(self.id))
        if label_flag and self.flag != Flag.GOOD:
            text_pieces.append(str(self.flag))
        if (label_seq_flag and self.sequence is not None
                and self.sequence.flag != SequenceFlag.GOOD):
            text_pieces.append(str(self.sequence.flag))
        if text_pieces:
            ax.text(np.mean(c), np.mean(r), " ".join(text_pieces),
                    color=color,
                    # Ensure the text isn't drawn outside the axis bounds
                    clip_on=True,
                    path_effects=[
                        pe.Stroke(linewidth=1, foreground='k'),
                        pe.Normal()
                ])
            
        if label_on_click:
            text = f"{self.id} {self.flag}"
            if (self.sequence is not None
                    and self.sequence.flag != SequenceFlag.GOOD):
                text += f" {self.sequence.flag}"
            self.manager = FeatureClickManager(text, *self.indices, color, line)
            self.manager.connect()
        
        if legend:
            _draw_color_legend(ax, simple_colors=simple_colors)

    def plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        self.plot_onto(ax, **kwargs)
        ax.set_aspect('equal')
    
    def overlaps(self, other: "Feature"):
        if self.left > other.right or other.left > self.right:
            return False
        if self.top > other.bottom or other.top > self.bottom:
            return False
        return not self.coord_set.isdisjoint(other.coord_set)
    
    @property
    def left(self):
        return self.cutout_corner[1]
    
    @property
    def right(self):
        return self.cutout_corner[1] + self.cutout.shape[1]
    
    @property
    def top(self):
        return self.cutout_corner[0]
    
    @property
    def bottom(self):
        return self.cutout_corner[0] + self.cutout.shape[0]
    
    def __repr__(self):
        return (f"<Feature {self.id}, {repr(self.flag)}, {self.size} px, "
                f"@{self.cutout_corner}>")


class FeatureClickManager:
    def __init__(self, text, rs, cs, color, artist):
        self.text = text
        self.coords = set(zip(cs, rs))
        self.text_y = np.mean(rs)
        self.text_x = np.max(cs) + 2
        self.color = color
        self.line_artist = artist
        self.ax = artist.axes
        
        self.artist = None
        self.cid = None
    
    def connect(self):
        self.cid = self.ax.figure.canvas.mpl_connect(
            'button_release_event', self.onclick)
    
    def disconnect(self):
        if self.cid is not None:
            self.ax.figure.canvas.mpl_disconnect(self.cid)
            self.cid = None
    
    def onclick(self, event):
        try:
            if self.line_artist not in self.ax.lines:
                self.disconnect()
            if event.inaxes != self.ax:
                return
            x = int(np.round(event.xdata))
            y = int(np.round(event.ydata))
            if (x, y) not in self.coords:
                return
            if self.artist is None:
                self.artist = self.ax.text(
                    self.text_x, self.text_y, self.text,
                    color=self.color,
                    # Ensure the text isn't drawn outside the axis bounds
                    clip_on=True,
                    path_effects=[
                        pe.Stroke(linewidth=1, foreground='k'),
                        pe.Normal()
                ])
            else:
                self.artist.remove()
                self.artist = None
        except Exception as e:
            self.ax.set_ylabel(e)


class FeatureSequence:
    def __init__(self):
        self.id = None
        self.features: list[Feature] = []
        self.origin = Event.NORMAL
        self.fate = Event.NORMAL
        self.origin_sequences: list[FeatureSequence] = []
        self.fate_sequences: list[FeatureSequence] = []
        self.feature_flag = None
        self.flag = None
    
    def __getitem__(self, item):
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
        return (f"<FeatureSequence {self.id}, {len(self.features)} "
                f"{repr(self.feature_flag)} feats, {repr(self.flag)}, "
                f"{repr(self.origin)} to {repr(self.fate)}>")


class TrackedImage:
    def __init__(self, source_file=None, source_shape=None, time=None, config=None):
        self.features: list[Feature] = []
        self.source_file = source_file
        self.source_shape = source_shape
        self.time = time
        self.config = config
    
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
            feature.plot_onto(ax, **kwargs)
        if legend:
            simple_colors = kwargs.get('simple_colors', False)
            _draw_color_legend(ax, simple_colors=simple_colors)
    
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
