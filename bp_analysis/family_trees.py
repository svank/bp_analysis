#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pickle
import tree_plot
from progressBar import ProgressBar

def describe(db, id):
    record = db[id]
    print("ID {}, Living from {} to {} ({} frames)".format(id, record['frames'][0], record['frames'][-1], record['frames'][-1] - record['frames'][0]))
    print("Children are: " + ", ".join([str(x) for x in record['children']]))
    print("Parents are:  " + ", ".join([str(x) for x in record['parents']]))
    print()

def handle_new_node(node, record):
    if 'fate' in record and record['fate'] == 'rejected':
        node.set_endpoint('x')

def add_relatives(tree, db, id):
    base_record = db[id]
    base_node = tree[id]
    for child in base_record['children']:
        if child not in tree:
            record = db[child]
            #describe(db, child)
            node = tree.create_child(id, child, record['frames'][0], record['frames'][-1])
            handle_new_node(node, record)
            add_relatives(tree, db, child)
        if tree[child] not in base_node.children:
            base_node.add_child(tree[child])
    
    for parent in base_record['parents']:
        if parent not in tree:
            record = db[parent]
            #describe(db, parent)
            node = tree.create_parent(id, parent, record['frames'][0], record['frames'][-1])
            handle_new_node(node, record)
            add_relatives(tree, db, parent)
        if tree[parent] not in base_node.parents:
            base_node.add_parent(tree[parent])

db = pickle.load(open("../Intensity_30G/tracking/database.pkl", 'rb'))

ids = list(db.keys())
ids.remove('id_list')

pb = ProgressBar(len(ids))
pb.display()
for id in ids:
    if id not in db:
        continue
    record = db[id]
    tree = tree_plot.Tree(id, record['frames'][0], record['frames'][-1])
    handle_new_node(tree.get_root(), record)

    #describe(db, id)
    add_relatives(tree, db, id)
    
    pb.increment(len(tree.nodes.keys()))
    for id in tree.nodes.keys():
        db.pop(id)

    size = len(tree)
    
    if size > 3:
        tree_plot.plot_tree(tree, labels=True)
        plt.ylabel("Frame Number")
        plt.title("BP Family Tree")
        plt.savefig("{}-{}".format(str(size), str(id)))
        plt.clf()
    
    pb.display()

