from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

from .status import Flag, SequenceFlag


# Text is legend labels
COLORS = {Flag.GOOD: ((.2, 1, .2, .8), "OK", 'k'),
          Flag.FALSE_POS: ((1, .1, .1, .8), "False pos", 'k'),
          Flag.CLOSE_NEIGHBOR: ((1, 1, 1, .8), "Proximity", 'k'),
          Flag.EDGE: ((.5, .5, 1, .8), "Edge", 'k'),
          Flag.TOO_SMALL: ((.1, 1, 1, .8), "Size", 'k'),
          Flag.TOO_BIG: ((.1, 1, 1, .8), "", 'k'),
          Flag.TOO_LONG: ((.1, 1, 1, .8), "", 'k'),
          SequenceFlag.TOO_SHORT: ((1, .1, 1, .8), "Short-lived", 'k'),
          }
SIMPLE_COLORS = {True: ((.1, 1, .1, .8), "OK", 'k'),
                 False: ((.5, .5, 1, .8), "Rejected", 'k')}


class FeaturePlottingMixin:
    def plot_onto(self, ax, ids=False, legend=False, label_flag=False,
                  label_seq_flag=False, simple_colors=False,
                  label_on_click=True):
        r, c = np.nonzero(self.cutout)
        r += self.cutout_corner[0]
        c += self.cutout_corner[1]
        if simple_colors:
            color, _, outline_color = SIMPLE_COLORS[self.is_good]
        else:
            color, _, outline_color = COLORS[self.flag]
            if (self.is_good and self.sequence is not None
                    and self.sequence.flag == SequenceFlag.TOO_SHORT):
                color, _, outline_color = COLORS[SequenceFlag.TOO_SHORT]
        
        line = outline_BP(r, c, scale=1, line_color=color, ax=ax,
                          outline_color=outline_color)
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
                        pe.withStroke(
                            linewidth=1, foreground=outline_color, alpha=0.75)
                ])
            
        if label_on_click:
            text = f"{self.id} {self.flag}"
            if self.sequence is not None:
                if self.sequence.flag != SequenceFlag.GOOD:
                    text += f" {self.sequence.flag}"
                text += f"\nSeq {self.sequence.id}, {self.sequence.origin}"
                text += f" - {self.sequence.fate}"
            self.manager = FeatureClickManager(
                text, *self.indices, color, outline_color, line)
            self.manager.connect()
        
        if legend:
            _draw_color_legend(ax, simple_colors=simple_colors)

    def plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        self.plot_onto(ax, **kwargs)
        ax.set_aspect('equal')


class FeatureClickManager:
    def __init__(self, text, rs, cs, color, outline_color, artist):
        self.text = text
        self.coords = set(zip(cs, rs))
        self.text_y = np.mean(rs)
        self.text_x = np.max(cs) + 2
        self.color = color
        self.outline_color = outline_color
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
                    self.text_x, y, self.text,
                    color=self.color,
                    # Ensure the text isn't drawn outside the axis bounds
                    clip_on=True,
                    path_effects=[
                        pe.withStroke(linewidth=1,
                                      foreground=self.outline_color,
                                      alpha=0.75)
                    ])
            else:
                self.artist.remove()
                self.artist = None
        except Exception as e:
            self.ax.set_ylabel(e)


def _draw_color_legend(ax, simple_colors=False):
    lines = []
    names = []
    colors = SIMPLE_COLORS if simple_colors else COLORS
    for color, name, outline in colors.values():
        if name:
            lines.append(Line2D(
                [0], [0], color=color[:3], lw=3, path_effects=[
                    pe.Stroke(linewidth=4, foreground=outline, alpha=0.75),
                    pe.Normal()]
            ))
            names.append(name)
    ax.legend(lines, names)


def outline_BP(r, c, scale=16 / 1000, line_color=(1, 1, 1, .8),
               outline_color='k', outline_alpha=0.75, linewidth=1, ax=None,
               offset=0, **kwargs):
    """
    A very-modified stackoverflow routine to draw lines outlining coordinates
    """
    try:
        xoffset, yoffset = offset
    except TypeError:
        xoffset, yoffset = offset, offset
    
    r = np.array(r)
    c = np.array(c)
    mapimg = np.zeros((np.max(c) - np.min(c) + 3, np.max(r) - np.min(r) + 3))
    mapimg[c - np.min(c) + 1, r - np.min(r) + 1] = 1
    
    # A horizontal line segment is needed, when the pixels next to each other
    # vertically belong to different groups (one is part of the mask, the
    # other isn't)
    hor_seg = np.where(mapimg[1:, :] != mapimg[:-1, :])
    
    # The same is repeated for vertical segments. We do some transposition to
    # ensure connecting line segments are marked sequentially.
    ver_seg = np.where(mapimg.T[1:, :] != mapimg.T[:-1, :])
    ver_seg = ver_seg[1], ver_seg[0]
    
    # If we have a horizontal segment at 7,2, it means that it must be drawn
    # between pixels (2,7) and (2,8), i.e. from (2,8) to (3,8).
    # Here we build up lists of line segments, and we make sure to connect them,
    # so that we get one, long (straight) line segment rather than a sequence
    # of one-pixel segments that together form one long segment.
    segments = []
    for p in zip(*hor_seg):
        point1 = (p[1], p[0] + 1)
        point2 = (p[1] + 1, p[0] + 1)
        if len(segments) and segments[-1][1] == point1:
            segments[-1][1] = point2
        else:
            segments.append([point1, point2])
    
    for p in zip(*ver_seg):
        point1 = (p[1] + 1, p[0])
        point2 = (p[1] + 1, p[0] + 1)
        if len(segments) and segments[-1][1] == point1:
            segments[-1][1] = point2
        else:
            segments.append([point1, point2])
    
    # Now we join segments that connect (i.e. at right angles), so that
    # matplotlib can plot long and bending stretches continuously. This is good
    # for, e.g., using a dashed line style.
    joined_segments = []
    for seg1 in segments:
        # We'll take this segment and add to it anything we've already seen.
        
        # Copy to avoid problems with modifying our iterable during iteration
        for seg2 in joined_segments.copy():
            if seg1[-1] == seg2[0]:
                seg1.extend(seg2[1:])
                joined_segments.remove(seg2)
            elif seg1[0] == seg2[-1]:
                seg1[:0] = seg2[:-1]
                joined_segments.remove(seg2)
            elif seg1[0] == seg2[0]:
                seg1[:0] = reversed(seg2[1:])
                joined_segments.remove(seg2)
            elif seg1[-1] == seg2[-1]:
                seg1.extend(reversed(seg2[:-1]))
                joined_segments.remove(seg2)
        joined_segments.append(seg1)
    
    points = []
    for segment in joined_segments:
        if len(points):
            points.append((np.nan, np.nan))
        points.extend(segment)
    
    segments = np.array(points, dtype=float)
    
    # Now we need to know something about the image which is shown.
    # At this point let's assume it has extents (x0, y0)..(x1, y1) on an axis
    # drawn with origin='lower'.
    # With this information we can rescale our points
    segments[:, 0] = segments[:, 0] + np.min(r) - 1.5 + yoffset
    segments[:, 1] = segments[:, 1] + np.min(c) - 1.5 + xoffset
    
    segments *= scale
    
    if ax is None:
        ax = plt.gca()
    line, = ax.plot(
        segments[:, 1], segments[:, 0], color=line_color,
        linewidth=1.6 * linewidth,
        path_effects=[
            pe.withStroke(linewidth=2.6 * linewidth, foreground=outline_color,
                      alpha=outline_alpha)],
        **kwargs)
    return line
