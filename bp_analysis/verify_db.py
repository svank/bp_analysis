#!/usr/bin/env python3

import configparser
import itertools
import numpy as np
import os
import pickle
import tqdm
from tqdm.contrib.concurrent import process_map
import sys

import build_database
from db_analysis import find_rec_by_coord
import tracking_utils

flag = "⚠️  "
has_warned = False

def warn(message):
    print(flag + message)
    global has_warned
    has_warned = True
    

def verify_directory(directory):
    files = sorted(os.listdir(directory))
    files = [directory + '/' + f for f in files if f.endswith(".npz")]
    print("Loading database...")
    # To simplify things for the multi-processing
    global db, config
    db = pickle.load(open(f"{directory}/database.pkl", 'rb'))
    config = configparser.ConfigParser()
    config.read_dict({'main': db['settings']})
    config = config['main']
    
    good_keys, bad_keys = verify_database(db)
    
    print("Checking good features...")
    for id in tqdm.tqdm(good_keys):
        verify_feature(id, db[id], is_good=True)
    
    print("Checking rejected features...")
    for id in tqdm.tqdm(bad_keys):
        verify_feature(id, db[id], is_good=False)
    
    print("Verifying that all features are in DB and properly recorded...")
    process_map(verify_file, files, max_workers=os.cpu_count(), chunksize=1)
    #list(map(verify_file, files))
    
    if not has_warned:
        print("Everything looks good!")
    else:
        print("")
        warn("Problems were detected during this run (as reported above)")
        print("")


def verify_database(db):
    print("Checking database...")
    good_keys = set()
    for t in db['id_list']:
        good_keys.update(db['id_list'][t])
    bad_keys = set()
    for t in db['bad_id_list']:
        bad_keys.update(db['bad_id_list'][t])
    
    if len(good_keys & bad_keys):
        warn("Some keys are both good and bad")
    
    all_keys = set(db.keys())
    all_keys -= set(
            ("id_list", "bad_id_list", "settings", "false_positive_coords"))
    if all_keys != good_keys | bad_keys:
        warn("There are unregistered keys")
    
    return good_keys, bad_keys


def verify_feature(id, rec, is_good):
    if is_good == rec['rejected']:
        warn(f"Feature {id} rejection isn't consistent")
    
    if rec['rejection_cause'] is not None:
        if (is_good and rec['fate'] != 'rejected'
                or not is_good and not rec['rejected']):
            warn(f"Feature {id} rejection cause isn't consistent")
    
    for key in ('times', 'files', 'coords', 'coords_l', 'intensity', 'bx', 'by', 'bz'):
        if len(rec[key]) != len(rec['frames']):
            warn(f"Incomplete frame-by-frame data for {key=}, {id=}")
    
    target_sizes = [len(rec['coords'][t][0]) for t in rec['times']]
    for k in ('intensity', 'bx', 'by', 'bz'):
        sizes = [len(x) for x in rec[k]]
        if sizes != target_sizes:
            warn(f"Inconsistent data size for {key=}, {id=}")
    
    for i, t in enumerate(rec['times']):
        if (t not in rec['coords'] or not (0 <= i < len(rec['coords_l']))
                or rec['coords'][t] is not rec['coords_l'][i]):
            warn(f"Coord indices wrong for {id=}")
    
    for child in rec['children']:
        if id not in db[child]['parents']:
            warn(f"Child {child} doesn't recognize parent {id=}")
        if (db[child]['origin'] not in
                ('merge', 'split', 'severe size change', 'end of rejection')):
            warn(f"Child {child} of {id=} has wrong origin")
        if db[child]['class'] != rec['class']:
            warn(f"Child {child} of {id=} has wrong class")
    
    for parent in rec['parents']:
        if id not in db[parent]['children']:
            warn(f"Parent {parent} doesn't recognize child {id=}")
        if (db[parent]['fate'] not in
                ('merge', 'split', 'complex', 'severe size change', 'accepted')):
            warn(f"Parent {parent} of {id=} has wrong origin")
        if db[parent]['class'] != rec['class']:
            warn(f"Parent {parent} of {id=} has wrong class")
    
    if rec['origin'] == 'merge' and len(rec['parents']) <= 1:
        warn(f"Missing parent for {id=}")
    
    if rec['fate'] in ('split', 'complex') and len(rec['children']) <= 1:
        warn(f"Missing child for {id=}")
    
    if rec['origin'] == 'split' and len(rec['parents']) != 1:
        warn(f"Wrong parents for {id=}")
    
    if rec['fate'] == 'merge' and len(rec['children']) != 1:
        warn(f"Wrong children for {id=}")
    
    if (len(rec['children']) and rec['fate'] not in
            ('split', 'merge', 'complex', 'severe size change', 'accepted')):
        warn(f"Shouldn't have children for {id=}")
    
    if (len(rec['parents']) and rec['origin'] not in
            ('merge', 'split', 'severe size change', 'end of rejection')):
        warn(f"Shouldn't have parents for {id=}")
    
    if rec['fate'] == 'complex':
        child_fates = [db[cid]['origin'] for cid in rec['children']]
        child_fates = np.unique(child_fates)
        if len(child_fates) > 1:
            if child_fates[0] != 'merge' or child_fates[1] != 'split':
                warn(f"Complex fate isn't complex for {id=}")
        else:
            child_parents = set()
            child_parents.update(*[db[cid]['parents'] for cid in rec['children']])
            if len(child_parents) <= 1:
                warn(f"Children of {id=} should have more parents")


def verify_file(file):
    data = np.load(file)
    features = data['features']
    classes = data['feature_classes']
    raw_features = features.copy()
    time = float(data['time'])
    
    # Apply the same filters applied by build_database
    fp_coords = build_database.extract_false_pos(features)
    labeled_feats, n_feat = build_database.label_features(
            features, config, classes)
    coord_map = tracking_utils.gen_coord_map(labeled_feats)
    build_database.filter_image(features, coord_map, 0, n_feat,
            min_size=config.getint('min_size'),
            max_size=config.getint('max_size'),
            max_diagonal=config.getint('max_diagonal'))
    
    edges = np.concatenate((
        labeled_feats[0:2].flatten(),
        labeled_feats[-2:].flatten(),
        labeled_feats[:, 0:2].flatten(),
        labeled_feats[:, -2:].flatten(),
    ))
    edges = np.unique(edges[edges != 0])
    for id in edges:
        rs, cs = coord_map[id]
        rec = db[find_rec_by_coord(rs[0], cs[0], db, time)]
        if (rec['rejection_cause'] != build_database.rejection_causes[-3]
                or not rec['rejected']):
            warn(f"Feature {id=} should be marked as 'hit edge' but isn't")
    
    for id in db['bad_id_list'][time]:
        rec = db[id]
        if rec['rejection_cause'] == build_database.rejection_causes[-3]:
            curr_id = np.unique(labeled_feats[rec['coords'][time]])
            assert len(curr_id) == 1
            if curr_id[0] not in edges:
                warn(f"Feature {id=} marked as 'hit edge' but isn't near edge")
    
    db_fp_coords = db['false_positive_coords'][time]
    if (len(fp_coords[0]) != len(db_fp_coords[0])
            or len(fp_coords[1]) != len(db_fp_coords[1])
            or np.any(fp_coords[0] != db_fp_coords[0])
            or np.any(fp_coords[1] != db_fp_coords[1])
            or np.any(raw_features[fp_coords] != -1)):
        warn(f"False pos coords incorrect for {time=}, {file=}")
    
    seen = np.zeros_like(features, dtype=bool)
    for id in itertools.chain(db['id_list'][time], db['bad_id_list'][time]):
        rec = db[id]
        coords = rec['coords'][time]
        if np.any(seen[coords]):
            warn(f"Already-seen pixel for {id=}, {time=}, {file=}")
        seen[coords] = True
        
        values = np.unique(features[coords])
        if len(values) > 1:
            warn(f"Multiple status values for {id=}, {time=}, {file=}")
        elif ((rec['rejected']
                    and build_database.rejection_causes[values[0]]
                        != rec['rejection_cause'])
                or (not rec['rejected'] and values[0] != 1)):
            warn(f"Wrong status value ({values[0]}) for {id=}, {time=}, {file=}")
        
        class_values = np.unique(classes[coords])
        if len(class_values) > 1:
            warn(f"Multiple class values for {id=}, {time=}, {file=}")
        elif class_values[0] != rec['class']:
                warn(f"Wrong class value for {id=}, {time=}, {file=}")
    
    n_seen = np.sum(seen)
    n_expected = np.sum(features != 0)
    if n_seen > n_expected:
        warn(f"{n_seen - n_expected} extra seen pixels for {time=}, {file=}")
    if n_seen < n_expected:
        warn(f"{n_expected - n_seen} unseen pixels for {time=}, {file=}")
    
    if np.any(seen != (features != 0)):
        warn(f"Seen-pixel/feature mismatch for {time=}, {file=}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Include as an argument a directory to verify")
        sys.exit()
    
    verify_directory(sys.argv[1])
