#!/usr/bin/env python3

from tqdm import tqdm

import numpy as np
import scipy.ndimage

import configparser
import os
import pickle
import sys

from read_file import read_file
from tracking_utils import gen_coord_map, gen_kernel

rejection_causes = {
        -1: 'false pos',
        -2: 'proximity',
        -3: 'hit edge',
        -4: 'bad size',
        1: None,
        }

def label_features(features, config, classes, id_offset=0):
    # Identify features and ensure their IDs are unique across the whole run
    labeling_struct = gen_kernel(config.getboolean('connect_diagonal', True))
    labeled_feats = np.zeros(features.shape, dtype=int)
    n_feat = 0
    for f_class in np.unique(classes):
        class_mask = classes == f_class
        for value in np.unique(features):
            if value == 0 or f_class == 0:
                continue
            labeled_feats_v, n_feat_v = scipy.ndimage.measurements.label(
                    (features == value) * class_mask, labeling_struct)
            labeled_feats_v[labeled_feats_v != 0] += id_offset + n_feat
            labeled_feats = np.where(
                    labeled_feats_v != 0, labeled_feats_v, labeled_feats)
            n_feat += n_feat_v
    return labeled_feats, n_feat


def filter_size(coords, min_size=None, max_size=None, max_diagonal=None):
    xcoords, ycoords = coords
    
    if min_size is not None and xcoords.size < min_size:
        return True
    if max_size is not None and xcoords.size > max_size:
        return True
    
    if max_diagonal is not None:
        xmin = np.min(xcoords)
        ymin = np.min(ycoords)
        xmax = np.max(xcoords)
        ymax = np.max(ycoords)
        
        diagonal = np.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)
        if diagonal > max_diagonal:
            return True
    return False


def make_record(frame_number, time, file, f_class):
    record = dict()
    record['frames'] = list()
    record['times'] = list()
    record['files'] = list()
    record['coords'] = dict()
    record['coords_l'] = list()
    record['intensity'] = list()
    record['bz'] = list()
    record['bx'] = list()
    record['by'] = list()
    record['origin'] = None
    record['parents'] = []
    record['children'] = []
    record['fate'] = None
    record['rejected'] = False
    record['rejection_cause'] = None
    record['class'] = f_class
    return record


def extract_false_pos(features):
    """Record and clear false-positive pixels.
    
    We record these so they can be plotted for analysis, but otherwise they're
    not "real" BP pixels, so we clear them from the image.
    """
    coords = np.nonzero(features == -1)
    coords = (coords[0].astype(np.int16), coords[1].astype(np.int16))
    features[coords] = 0
    return coords
    

def filter_image(features, coord_map, start_id, n_feat, min_size, max_size, max_diagonal):
    for id in range(start_id + 1, start_id + n_feat + 1):
        coords = coord_map[id]
        status = features[coords]
        if np.any(status != status[0]):
            raise RuntimeError(f"Multiple status values found for id {id}")
        
        if (status[0] > 0
                and filter_size(coords, min_size, max_size, max_diagonal)):
            features[coords] = -4


def build_database(dir, intensity_dir=None, tau_slice_dir=None, silent=False):
    config = configparser.ConfigParser()
    config.read(os.path.join(dir, "tracking.cfg"))
    config = config['main']
    max_size_change_pct = config.getfloat('max_size_change_pct', 50)
    max_size_change_px = config.getint('max_size_change_px', 10)
    
    files = sorted(os.listdir(dir))
    files = [f for f in files if len(f) > 4 and f[-4:] == '.npz']
    
    database = {
            'settings': {k: v for k, v in config.items()},
            'id_list': dict(),
            'bad_id_list': dict(),
            'false_positive_coords': dict(),
            }
    
    max_id = 0
    last_labeled_features = None
    last_classes = None
    last_time = None
    frame_number = 0
    
    iterable = files if silent else tqdm(files)
    for file in iterable:
        data = np.load(os.path.join(dir, file))
        features = data['features']
        classes = data['feature_classes']
        
        timestamp = file.split('.')[-2]
        try:
            _, data_cube = read_file(
                    os.path.join(intensity_dir, "I_out."+timestamp))
            intensity = data_cube[0]
        except (FileNotFoundError, TypeError):
            intensity = None
        try:
            _, data_cube = read_file(
                    os.path.join(tau_slice_dir, "tau_slice_1.000."+timestamp))
            bz = data_cube[6]
            bx = data_cube[5]
            by = data_cube[7]
        except (FileNotFoundError, TypeError):
            bx = by = bz = None
        
        time = float(data['time'])
        
        if last_time is not None:
            # Before we clear false-positives, record any past-frame features
            # that only overlap false-positive pixels in this frame, so we can
            # mark them as ending in a false-positive.
            for id in database['id_list'][last_time]:
                record = database[id]
                coords = record['coords'][last_time]
                current_pixels = features[coords]
                current_pixels = current_pixels[classes[coords] == record['class']]
                current_pixels = np.unique(current_pixels[current_pixels != 0])
                if len(current_pixels) == 1 and current_pixels[0] == -1:
                    record['fate'] = 'rejected'
                    if record['rejection_cause'] is None:
                        record['rejection_cause'] = rejection_causes[-1]
        
        false_pos_coords = extract_false_pos(features)
        database['false_positive_coords'][time] = false_pos_coords
        
        labeled_feats, n_feat = label_features(features, config, classes, max_id)
        
        coord_map = gen_coord_map(labeled_feats)
        
        if last_labeled_features is None:
            last_labeled_features = np.zeros_like(labeled_feats)
        
        n_persist = 0
        n_merged = 0
        n_rejected = 0
        n_split = 0
        n_new = 0
        n_edge = 0
        n_false_pos = len(false_pos_coords[0])
        n_too_close = 0
        n_severe_size_change = 0
        n_mixed = 0
        id_list = []
        bad_id_list = []
        
        filter_image(
                features, coord_map, max_id, n_feat,
                config.getint('min_size', 4),
                config.getint('max_size', 110),
                config.getint('max_diagonal', 20))
        
        for id in range(max_id + 1, max_id + n_feat + 1):
            orig_id = id
            coords = coord_map[id]
            statuses = features[coords]
            f_class = classes[coords]
            assert len(np.unique(statuses)) == 1
            assert len(np.unique(f_class)) == 1
            status = statuses[0]
            f_class = f_class[0]
            
            record = None
            if (last_time is not None
                    and np.any(past_ids := last_labeled_features[coords])):
                past_classes = last_classes[coords]
                past_ids = past_ids[past_classes == f_class]
                past_ids = past_ids[past_ids != 0]
                past_ids = np.unique(past_ids)
                
                if len(past_ids) > 1:
                    # MERGER
                    # This feature overlaps more than one feature in the previous
                    # frame. We will let this feature start a new ID.
                    n_merged += 1
                    
                    record = make_record(frame_number, time, file, f_class)
                    database[id] = record
                    record['origin'] = 'merge'
                    record['parents'] = past_ids
                    for i, past_id in enumerate(past_ids):
                        old_rec = database[past_id]
                        old_rec['children'].append(id)
                        if (old_rec['fate'] == 'split'
                                or len(old_rec['children']) > 1):
                            old_rec['fate'] = 'complex'
                        elif old_rec['fate'] is None:
                            old_rec['fate'] = 'merge'
                elif len(past_ids) > 0:
                    # If there are no past_ids, that means this feature only
                    # overlaps with a feature of a diferent class. We can fall
                    # through and a new record will be created for this
                    # feature.
                    #
                    # Otherwise, check whether there are multiple features in
                    # the current frame which overlap this past-frame feature.
                    
                    past_id = past_ids[0]
                    past_coords = database[past_id]['coords'][last_time]
                    past_class = last_classes[past_coords][0]
                    
                    current_pixels = labeled_feats[past_coords]
                    current_pixels = (
                            current_pixels[classes[past_coords] == past_class])
                    current_pixels = current_pixels[current_pixels != 0]
                    if len(np.unique(current_pixels)) > 1:
                        # SPLITTING
                        # There is multi-overlap, e.g. the past-frame feature
                        # splits in two. It's ambiguous which current-frame
                        # feature should keep the ID, but there's nothing
                        # inherently wrong with the current-frame features, so
                        # have them be new features.
                        n_split += 1
                        
                        record = make_record(frame_number, time, file, f_class)
                        database[id] = record
                        record['origin'] = 'split'
                        record['parents'] = past_ids
                        
                        old_rec = database[past_id]
                        old_rec['children'].append(id)
                        if old_rec['fate'] is None:
                            old_rec['fate'] = 'split'
                        elif old_rec['fate'] == 'merge':
                            old_rec['fate'] = 'complex'
                        
                    else:
                        # This is a possible continuation. We'll need to rule
                        # out a few possible contraindications for
                        # continuation.
                        old_rec = database[past_id]
                        
                        s1 = old_rec['coords_l'][-1][0].size
                        s2 = coords[0].size
                        if s1 > s2:
                            s1, s2 = s2, s1
                        delta_size = s2 - s1
                        delta_pct = delta_size / s1 * 100
                        
                        if (status < 0 and (not old_rec['rejected']
                                or old_rec['rejection_cause']
                                    != rejection_causes[status])):
                            # This rejected feature would be a continuation of
                            # an accepted feature from the previous frame, or
                            # of a rejected feature that was rejected for a
                            # different reason. We'll close off that existing
                            # feature and make a new record to record just this
                            # rejected feature.
                            old_rec['fate'] = 'rejected'
                            if old_rec['rejection_cause'] is None:
                                old_rec['rejection_cause'] = \
                                        rejection_causes[status]
                        elif status > 0 and old_rec['rejected']:
                            # The formerly-rejected feature is now good
                            if old_rec['fate'] is None:
                                old_rec['fate'] = 'accepted'
                            old_rec['children'].append(id)
                            record = make_record(frame_number, time, file, f_class)
                            database[id] = record
                            record['origin'] = 'end of rejection'
                            record['parents'].append(past_id)
                        elif (delta_size > max_size_change_px
                                and delta_pct > max_size_change_pct):
                            # The feature size has changed too much
                            if old_rec['fate'] is None:
                                old_rec['fate'] = 'severe size change'
                            old_rec['children'].append(id)
                            record = make_record(frame_number, time, file, f_class)
                            database[id] = record
                            record['origin'] = 'severe size change'
                            record['parents'].append(past_id)
                            n_severe_size_change += 1
                        else:
                            # Everything is OK. Relabel this current-frame
                            # feature with the id of the past-frame feature it
                            # overlaps
                            labeled_feats[coords] = past_id
                            id = past_id
                            record = old_rec
                            n_persist += 1
            
            if record is None:
                # This probably means there was no overlap with a
                # previous-frame feature.
                n_new += 1
                record = make_record(frame_number, time, file, f_class)
                database[id] = record
            
            if status < 0:
                if status == -2:
                    n_too_close += 1
                elif status == -3:
                    n_edge += 1
                elif status == -4:
                    n_rejected += 1
                
                record['rejected'] = True
                record['rejection_cause'] = rejection_causes[status]
                
                bad_id_list.append(id)
            else:
                id_list.append(id)
            
            coords = (coords[0], coords[1])
            record['coords'][time] = coords
            record['coords_l'].append(coords)
            if intensity is not None:
                record['intensity'].append(intensity[coords])
            if bz is not None:
                record['bz'].append(bz[coords])
                record['bx'].append(bx[coords])
                record['by'].append(by[coords])
            
            record['frames'].append(frame_number)
            record['times'].append(time)
            record['files'].append(file)
        
        if not silent:
            tqdm.write(f"{n_persist} kept, {n_merged} mrg, {n_rejected} size, "
                    f"{n_split} splt, {n_edge} @eg, {n_false_pos}px f+, "
                    f"{n_too_close} prox, {n_severe_size_change} sv size chg, "
                    f"{n_new} new")
        
        database['id_list'][time] = id_list
        database['bad_id_list'][time] = bad_id_list
        
        max_id += n_feat
        frame_number += 1
        last_labeled_features = labeled_feats
        last_classes = classes
        last_time = time
    
    pickle.dump(database, open(os.path.join(dir, "database.pkl"), 'wb'))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Include as an argument a directory to process")
        sys.exit()
    
    if os.path.isdir(sys.argv[1]):
        data_dir = sys.argv[1]
        if len(sys.argv) > 2:
            tracking_dir = sys.argv[2]
        else:
            tracking_dir = "tracking"
        tracking_dir = os.path.join(data_dir, tracking_dir)
    else:
        config = configparser.ConfigParser()
        config.read(sys.argv[1])
        tracking_dir = config['dirs']['out_dir']
        data_dir = config['dirs']['data_dir']
    
    if 'Intensity' in data_dir:
        intensity_dir = data_dir
        tau_slice_dir = data_dir.replace("Intensity", "tau_slice")
    else:
        tau_slice_dir = data_dir
        intensity_dir = data_dir.replace("tau_slice", "Intensity")
    
    build_database(tracking_dir, intensity_dir, tau_slice_dir)
