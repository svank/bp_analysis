#!/usr/bin/env python3

from collections import defaultdict
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pickle
import scipy as scp

from .power_spectrum import PowerSpectralyzer

def calc_centroid(d, weight=True):
    x = np.empty(len(d['times']))
    y = np.empty(len(d['times']))
    for i, coords in enumerate(d['coords_l']):
        if weight:
            if 'weights' in d:
                weights = d['weights'][i]
            else:
                intensity = d['intensity'][i]
                min_I = np.min(intensity)
                weights = (intensity - min_I) / (np.max(intensity) - min_I)
        else:
            weights = np.ones_like(coords[0])
        total_weight = np.sum(weights)
        x[i] = np.sum(weights * coords[0]) / total_weight
        y[i] = np.sum(weights * coords[1]) / total_weight
    return x, y

def outline_BP(r, c, scale=16/1000, line_color=(1,1,1,.8),
               outline_color=(0,0,0,.75), linewidth=1, ax=None, offset=0,
               **kwargs):
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
        point1 = (p[1], p[0]+1)
        point2 = (p[1]+1, p[0]+1)
        if len(segments) and segments[-1][1] == point1:
            segments[-1][1] = point2
        else:
            segments.append([point1, point2])
    
    for p in zip(*ver_seg):
        point1 = (p[1]+1, p[0])
        point2 = (p[1]+1, p[0]+1)
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
    segments[:,0] = segments[:,0] + np.min(r) - 1.5 + yoffset
    segments[:,1] = segments[:,1] + np.min(c) - 1.5 + xoffset
    
    segments *= scale
    
    if ax is None:
        ax = plt.gca()
    line, = ax.plot(
        segments[:, 1], segments[:, 0], color=line_color,
        linewidth=1.6*linewidth,
        path_effects=[
            pe.Stroke(linewidth=2.6*linewidth, foreground=outline_color),
            pe.Normal()],
        **kwargs)
    return line

def calc_centroid_velocities(data, ok_ids=None, weight=True,
        # Default scale from px/s to cm/s for 16-km pixels
        scale=16e5):
    if ok_ids is not None:
        data = [data[id] for id in ok_ids]
    vxs = []
    vys = []
    dts = []
    for d in data:
        times = np.array(d['times'])
        dt = np.diff(times)
        dts.extend(list(dt))
        
        # Calculate the intensity-weighted centroid
        x, y = calc_centroid(d, weight=weight)
        
        # Calculate velocities
        vx = np.diff(x) / dt
        vy = np.diff(y) / dt
        
        vx *= scale
        vy *= scale
        # Now we have cm/s (with the default scale)
        
        vxs.append(vx)
        vys.append(vy)
        
    return vxs, vys, np.mean(dts)

def power_spectrum(data, ok_ids=None, xmin=None, xmax=None, ymin=None,
        ymax=None, weight=True, **kwargs):
    
    vxs, vys, dt = calc_centroid_velocities(data, ok_ids, weight)
    
    spec = PowerSpectralyzer()
    for vx, vy in zip(vxs, vys):
        spec.add(vx, vy)
    
    spec.set_dt(dt)
    spec.set_bounds(xmin, xmax, ymin, ymax)
    spec.plot(clear=False, **kwargs)

def calc_intensities(data, ok_ids=None):
    if ok_ids is not None:
        data = [data[id] for id in ok_ids]
    peak_intensities = []
    avg_intensities = []
    p_intensities = []
    lifetimes = []
    for d in data:
        lifetimes.append(d['times'][-1] - d['times'][0])
        intensity_vals = []
        for one_time in d['intensity']:
            intensity_vals.extend(one_time / 1e10)
        peak_intensities.append(np.max(intensity_vals))
        avg_intensities.append(np.mean(intensity_vals))
        p_intensities.append(np.percentile(intensity_vals, 95))
    return np.array(lifetimes), np.array(peak_intensities), np.array(avg_intensities), np.array(p_intensities)

def calc_B(data, ok_ids=None):
    if ok_ids is not None:
        data = [data[id] for id in ok_ids]
    peak_B = []
    avg_B = []
    p_B = []
    lifetimes = []
    for d in data:
        lifetimes.append(d['times'][-1] - d['times'][0])
        Bs = []
        for one_time in d['bz']:
            Bs.extend(np.abs(one_time))
        peak_B.append(np.max(Bs))
        avg_B.append(np.mean(Bs))
        p_B.append(np.percentile(Bs, 95))
    return np.array(lifetimes), np.array(peak_B), np.array(avg_B), np.array(p_B)

def calc_vel(data, ok_ids=None, ret_all=False):
    if ok_ids is not None:
        data = [data[id] for id in ok_ids]
    peak_vel = []
    avg_vel = []
    p_vel = []
    all_vel = []
    lifetimes = []
    for d in data:
        lifetimes.append(d['times'][-1] - d['times'][0])
    
    vxs, vys, dt = calc_centroid_velocities(data, weight=True)
    for vx, vy in zip(vxs, vys):
        v = np.sqrt(vx**2 + vy**2)
        
        peak_vel.append(np.max(v))
        avg_vel.append(np.mean(v))
        p_vel.append(np.percentile(v, 95))
        all_vel.extend(v)
    if ret_all:
        return np.array(all_vel)
    return np.array(lifetimes), np.array(peak_vel), np.array(avg_vel), np.array(p_vel)

def calc_sizes(data, ok_ids=None):
    if ok_ids is not None:
        data = [data[id] for id in ok_ids]
    peak_sizes = []
    avg_sizes = []
    p_sizes = []
    lifetimes = []
    for d in data:
        lifetimes.append(d['times'][-1] - d['times'][0])
        sizes = []
        for (x, y) in d['coords_l']:
            size = x.size
            size *= 16**2 # to km^2
            size = 2*np.sqrt(size/np.pi) # To equiv diameter
            sizes.append(size)
        peak_sizes.append(np.max(sizes))
        avg_sizes.append(np.mean(sizes))
        p_sizes.append(np.percentile(sizes, 95))
    return np.array(lifetimes), np.array(peak_sizes), np.array(avg_sizes), np.array(p_sizes)

def find_correlations(data, ok_ids=None):
    lifetimes, peak_sizes, avg_sizes, p_sizes = calc_sizes(data, ok_ids)
    lifetimes, peak_vel, avg_vel, p_vel = calc_vel(data, ok_ids)
    lifetimes, peak_B, avg_B, p_B = calc_B(data, ok_ids)
    lifetimes, peak_intensities, avg_intensities, p_intensities = calc_intensities(data, ok_ids)
    
    import seaborn as sns
    import pandas as pd
    d = pd.DataFrame({'size': avg_sizes, 'vel': avg_vel, 'B': avg_B, 'intensity':avg_intensities, 'lifetime': lifetimes})
    g = sns.PairGrid(d)
    g.map_lower(sns.kdeplot)
    g.map_upper(plt.scatter, alpha=0.2, s=1.5)
    g.map_diag(plt.hist, bins=30)
    plt.show()
    
    plt.subplot(131)
    sns.jointplot(x=avg_intensities, y=avg_sizes, kind='kde', shade='False')
    
    plt.subplot(132)
    sns.jointplot(x=avg_B, y=avg_sizes, kind='kde', shade='False')
    
    plt.subplot(133)
    sns.jointplot(x=peak_B, y=peak_intensities, kind='kde', shade='False')
    plt.show()

def lifetime_trends(data, ok_ids=None, do='all'):
    bin_args = {'cmap': 'hot',
                'bins': 'log',
                'marginals': False,
                'gridsize': 30,
                'vmax': 2.4}
    
    def plot(x, y, xlabel, ylabel, title, filename, bin_args, extra_func=None):
        m, b, r, p, err = scp.stats.linregress(x, y)
        print(m, b, r, p)
        #plt.subplot(121)
        plt.hexbin(x, y, **bin_args)
        plt.colorbar().set_label("log(count + 1)")
        
        xx = np.array(plt.xlim())
        yy = m*xx + b
        plt.plot(xx, yy, color='cyan', linewidth=3)
        
        text = "slope={:.4f}, r={:.2f}, p={:.4f}".format(m, r, p)
        plt.figtext(.15, .85, text, color='white')
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if 'extent' in bin_args:
            plt.xlim(*bin_args['extent'][0:2])
            plt.ylim(*bin_args['extent'][2:4])
            
        #plt.subplot(122)
        #plt.scatter(lifetimes, peak_intensities)
        #plt.xlabel("Lifetime (s)")
        #plt.ylabel("Peak Intensity")
        #plt.ylim(1e10, 7e10)
        plt.title(title)
        #plt.gcf().set_size_inches(12, 5, forward=True)
        plt.gcf().set_size_inches(6, 5, forward=True)
        #plt.tight_layout()
        if extra_func is not None:
            extra_func()
        plt.savefig(filename, dpi=150)
        plt.clf()
    

    if do == 'all' or do == 'intensity':
        lifetimes, peak_intensities, avg_intensities, p_intensities = calc_intensities(data, ok_ids)
        bin_args['extent'] = (85, 800, 1.7, 5.5)

        plot(lifetimes, peak_intensities, "Lifetime (s)",
                "Peak Intensity (arb. units)",
                "Intensity-Lifetime Distribution",
                "lifetime_intensity_peak.png",
                bin_args)
            
        plot(lifetimes, avg_intensities, "Lifetime (s)",
                "Mean Intensity (arb. units)",
                "Intensity-Lifetime Distribution",
                "lifetime_intensity_mean.png",
                bin_args)
    
    if do == 'all' or do == 'B':
        lifetimes, peak_B, avg_B, p_B = calc_B(data, ok_ids)
        
        bin_args['extent'] = (85, 800, 0, 2750)

        func=lambda: plt.gcf().subplots_adjust(left=0.14)
        plot(lifetimes, peak_B, "Lifetime (s)",
                "Peak |B$_Z$| (Gauss)",
                "|B$_Z$|-Lifetime Distribution",
                "lifetime_Bz_peak.png",
                bin_args,
                func)
            
        plot(lifetimes, avg_B, "Lifetime (s)",
                "Mean |B$_Z$| (Gauss)",
                "|B$_Z$|-Lifetime Distribution",
                "lifetime_Bz_mean.png",
                bin_args,
                func)
            
    if do in ('all', 'vel'):
        lifeties, peak_vel, avg_vel, p_vel = calc_vel(data, ok_ids)
            
        bin_args['extent'] = (85, 800, 0, 9)

        plot(lifetimes, peak_vel, "Lifetime (s)",
                "Peak Velocity (km/s)",
                "Velocity-Lifetime Distribution",
                "lifetime_vel_peak.png",
                bin_args)
            
        plot(lifetimes, avg_vel, "Lifetime (s)",
                "Mean Velocity (km/s)",
                "Velocity-Lifetime Distribution",
                "lifetime_vel_mean.png",
                bin_args)
            
    
    if do == 'all' or do == 'size':
        lifetimes, peak_sizes, avg_sizes, p_sizes = calc_sizes(data, ok_ids)
                
        bin_args['extent'] = (85, 800, 40, 200)

        plot(lifetimes, peak_sizes, "Lifetime (s)",
                "Peak Size (km)",
                "Size-Lifetime Distribution",
                "lifetime_size_peak.png",
                bin_args)
            
        plot(lifetimes, avg_sizes, "Lifetime (s)",
                "Mean Size (km)",
                "Size-Lifetime Distribution",
                "lifetime_size_mean.png",
                bin_args)
            
    if do=='all':
        # Size-Intensity
        bin_args['extent'] = (40, 200, 1.7, 5.5)
        
        plot(avg_sizes, avg_intensities, "Mean Size (km)",
                "Mean Intensity (arb. units)",
                "Intensity-Size Distribution",
                "size_intensity_mean.png",
                bin_args)
        
        plot(peak_sizes, peak_intensities, "Peak Size (km)",
                "Peak Intensity (arb. units)",
                "Intensity-Size Distribution",
                "size_intensity_peak.png",
                bin_args)

        # Size-B
        bin_args['extent'] = (40, 200, 0, 2750)
        
        plot(avg_sizes, avg_B, "Mean Size (km)",
                "Mean |B$_Z$| (Gauss)",
                "Size-|B$_Z$| Distribution",
                "size_Bz_mean.png",
                bin_args, func)

        plot(peak_sizes, peak_B, "Peak Size (km)",
                "Peak |B$_Z$| (Gauss)",
                "Size-|B$_Z$| Distribution",
                "size_Bz_peak.png",
                bin_args, func)

        # Size-V
        bin_args['extent'] = (0, 9, 40, 200)
        
        plot(avg_vel, avg_sizes, "Mean Velocity (km/s)",
                "Mean Size (km)",
                "Velocity-Size Distribution",
                "size_vel_mean.png",
                bin_args)

        plot(peak_vel, peak_sizes, "Peak Velocity (km/s)",
                "Peak Size (km)",
                "Velocity-Size Distribution",
                "size_vel_peak.png",
                bin_args)

        # Intensity-B
        bin_args['extent'] = (1.7, 5.5, 0, 2750)
        
        plot(avg_intensities, avg_B, "Mean Intensity (arb. units)",
                "Mean |B$_Z$| (Gauss)",
                "Intensity-|B$_Z$| Distribution",
                "intensity_Bz_mean.png",
                bin_args, func)

        plot(peak_intensities, peak_B, "Peak Intensity (arb. units)",
                "Peak |B$_Z$| (Gauss)",
                "Intensity-|B$_Z$| Distribution",
                "intensity_Bz_peak.png",
                bin_args, func)

        # Intensity-V
        bin_args['extent'] = (0, 9, 1.7, 5.5)
        
        plot(avg_vel, avg_intensities, "Mean Velocity (km/s)",
                "Mean Intensity (arb. units)",
                "Velocity-Intensity Distribution",
                "intensity_vel_mean.png",
                bin_args)

        plot(peak_vel, peak_intensities, "Peak Velocity (km/s)",
                "Peak Intensity (arb. units)",
                "Velocity-Intensity Distribution",
                "intensity_vel_peak.png",
                bin_args)

        # V-B
        bin_args['extent'] = (0, 9, 0, 2750)
        
        plot(avg_vel, avg_B, "Mean Velocity (km/s)",
                "Mean |B$_Z$| (Gauss)",
                "Velocity-|B$_Z$| Distribution",
                "vel_Bz_mean.png",
                bin_args, func)

        plot(peak_vel, peak_B, "Peak Velocity (km/s)",
                "Peak |B$_Z$| (Gauss)",
                "Velocity-|B$_Z$| Distribution",
                "vel_Bz_peak.png",
                bin_args, func)

def size_analysis(data, ok_ids=None):
    if ok_ids is not None:
        data = [data[id] for id in ok_ids]
    sizes = []
    for d in data:
        for x, y in d['coords_l']:
            size = x.size
            sizes.append(size)

    sizes = np.array(sizes)

    n_vals = np.max(sizes) - 1

    hist, bin_edges = np.histogram(sizes, bins=n_vals)
    life_length = bin_edges[1:]
    hist += 1

    #hist = hist[20:400]
    #life_length = life_length[20:400]
    slope, intercept, r_val, p_val, stdeff = scp.stats.linregress(life_length[1:], np.log(hist[1:]))

    print("Semilog-y slope: {}".format(slope))

    plt.hist(sizes, bins=n_vals, log=True)

    fit = np.exp(intercept) * np.exp(slope * life_length)
    plt.plot(life_length, fit)

    plt.xlabel("Size (pixels)")
    plt.ylabel("Number")
    #plt.show()
    plt.savefig('size.png')
    plt.clf()

def load_data(prefix='', tracking_dir="tracking", data_dir="Intensity_30G", db_name="database.pkl", lifetime_min=5, lifetime_max=9e99, also_rejected=False):
    data = pickle.load(
            open("{}{}/{}/{}".format(prefix, data_dir, tracking_dir, db_name),
                 'rb'))
    
    ids = [k for k in data.keys() if not isinstance(k, str)]
    
    for k in ids:
        data[k]['id'] = k
    
    ok_ids = []
    bad_ids = []
    # Rather than individually removing all too-short BPs from id_list, let's
    # make a new id_list with only the IDs we accept.
    id_list = defaultdict(list)
    data['id_list'] = id_list
    
    for id in ids:
        rec = data[id]
        lifetime = rec['frames'][-1] - rec['frames'][0]
        if not rec['rejected'] and lifetime_min <= lifetime <= lifetime_max:
            ok_ids.append(id)
            for time in rec['times']:
                id_list[time].append(id)
        else:
            bad_ids.append(id)
            if not rec['rejected']:
                rec['rejected'] = True
                rec['rejection_cause'] = 'lifetime'
                for time in rec['times']:
                    data['bad_id_list'][time].append(id)
    
    data['tracking_dir'] = tracking_dir
    data['data_dir'] = data_dir
    data['db_name'] = db_name
    
    if also_rejected:
        return data, ok_ids, bad_ids
    return data, ok_ids

def load_data_reduced(*args, keep_keys=None, also_all=False, **kwargs):
    all_data, ok_ids, bad_ids = load_data(*args, also_rejected=True, **kwargs)
    
    if keep_keys is not None:
        if also_all:
            ids = itertools.chain(ok_ids, bad_ids)
        else:
            ids = ok_ids
        for id in ids:
            all_data[id] = {k: all_data[id][k] for k in keep_keys}
    
    new_data = [all_data[id] for id in sorted(ok_ids)]
    if also_all:
        return new_data, all_data
    return new_data

def records_are_equal(r1, r2, verbose=True, missing_ok=False, fundamentals=False):
    keys = set(r1.keys()) | set(r2.keys())
    for k in keys:
        if fundamentals and k not in ('frames', 'times', 'files', 'coords', 'intensity', 'bz', 'bx', 'by'):
            continue
        if k not in r1 or k not in r2:
            if verbose:
                print(f"Missing key: {k}")
            if missing_ok:
                continue
            else:
                return False
        if k == "coords":
            keys = set(r1[k].keys()) | set(r2[k].keys())
            for k2 in keys:
                if k2 not in r1[k] or k2 not in r2[k]:
                    if verbose:
                        print(f"Missing key: [{k}][{k2}]")
                    return False
                if (len(r1[k][k2][0]) != len(r2[k][k2][0])
                        or np.any(r1[k][k2][0] != r2[k][k2][0])
                        or np.any(r1[k][k2][1] != r2[k][k2][1])):
                    if verbose:
                        print(f"Inequality at r[{k}][{k2}]")
                    return False
        elif (type(r1[k]) in (list, tuple) and len(r1[k]) > 0
                and type(r1[k][0]) == np.ndarray):
            if len(r1[k]) != len(r2[k]):
                if verbose:
                    print(f"Length difference at r[{k}]")
                return False
            for d1, d2 in zip(r1[k], r2[k]):
                if len(d1) != len(d2) or np.any(d1 != d2):
                    if verbose:
                        print(f"Inequality at r[{k}]")
                    return False
        elif type(r1[k]) == np.ndarray:
            if any(r1[k] != r2[k]):
                if verbose:
                    print(f"Inequality at r[{k}]")
                return False
        else:
            if r1[k] != r2[k]:
                if verbose:
                    print(f"Inequality at r[{k}]")
                return False
    return True

def find_rec_by_coord(r, c, data, time):
    candidates = data['id_list'][time] + data['bad_id_list'][time]
    for candidate in candidates:
        coords = data[candidate]['coords'][time]
        ismatch = (coords[0] == r) * (coords[1] == c)
        if np.any(ismatch):
            return candidate
    return None

if __name__ == "__main__":

    data, ok_ids = load_data()
    print("# BPs: {}".format(len(ok_ids)))


    #size_analysis(data, ok_ids)

    lifetime_trends(data, ok_ids)
    
    #power_spectrum(data, ok_ids)

    #find_correlations(data, ok_ids)

