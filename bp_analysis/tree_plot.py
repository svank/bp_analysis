#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_tree(tree, labels=False):
    x_coords = dict()
    plotted = list()
    root = tree.root
    plot_node(root, x_coords, plotted, labels)
    
    plt.gca().invert_yaxis()
    #plt.show()
    
def plot_node(node, x_coords, plotted, labels=False):
    # Determine the node's x coordinate
    if len(x_coords) == 0:
        # Root node
        x_coord = 0
    else:
        valid_parent = False
        for parent in node.parents:
            if parent in plotted:
                valid_parent = True
                break
        if valid_parent:
            # We have parents
            n_child = parent.children.index(node)
            n_children = len(parent.children)
            # Round up to even numbers
            n_children = 2 * np.ceil(n_children // 2)
            
            x_coord = x_coords[parent] + n_child - n_children / 2 + 0.5
        else:
            # No valid parents
            # Then, at this point, we must have plotted children
            
            # Average the children that are plotted
            x_coord = 0
            i = 0
            for child in node.children:
                if child in x_coords:
                    x_coord += x_coords[child]
                    i += 1
            x_coord /= i
            x_coord += 1
    
    c = 0
    # Is there already something at this x coordinate?
    while x_coord in x_coords.values():
        # Find all nodes with the same x coordinate that also overlap in time
        # with this new node
        nodes = [n for n in x_coords.keys() if x_coords[n] == x_coord and (n.t_start <= node.t_start <= n.t_end or node.t_start <= n.t_start <= node.t_end)]
        if len(nodes):
            # Move it!
            x_coord += 0.5
        else:
            break
        
    x_coords[node] = x_coord
    
    plt.scatter([x_coord], [node.t_start])
    plt.plot([x_coord, x_coord], [node.t_start, node.t_end])
    if node.endpoint is not None:
        plt.scatter([x_coord], [node.t_end], 130, marker=node.endpoint)
    if labels:
        plt.text(x_coord+.01, node.t_start-.1, str(node.id))
   
    for parent in node.parents:
        if parent in plotted:
            plt.plot([x_coord, x_coords[parent]], [node.t_start, node.t_start - 1], '--')
    
    for child in node.children:
        if child in plotted:
            plt.plot([x_coord, x_coords[child]], [child.t_start - 1, child.t_start], '--')
        
    
    plotted.append(node)
    
    for n in itertools.chain(node.children, node.parents):
        if n not in plotted:
            plot_node(n, x_coords, plotted, labels)
    
    
    

class Node:
    def __init__(self, id, t_start, t_end, parents=[], children=[]):
        self.parents = self.to_list(parents)
        self.children = self.to_list(children)
        self.id = id
        self.t_start = t_start
        self.t_end = t_end
        self.endpoint = None
        
    def add_child(self, child):
        self.children.append(child)
        if self not in child.parents:
            child.add_parent(self)
        
    def create_child(self, *args, **kwargs):
        child = Node(*args, **kwargs)
        self.add_child(child)
        return child
    
    def add_parent(self, parent):
        self.parents.append(parent)
        if self not in parent.children:
            parent.add_child(self)
        
    def create_parent(self, *args, **kwargs):
        parent = Node(*args, **kwargs)
        self.add_parent(parent)
        return parent
    
    def set_endpoint(self, endpoint):
        self.endpoint = endpoint
    
    @classmethod
    def to_list(cls, item):
        try:
            # Convert iterable to list
            return list(item)
        except TypeError:
            # Not an iterable
            return [item]

class Tree:
    def __init__(self, *args, **kwargs):
        self.root = Node(*args, **kwargs)
        self.nodes = dict()
        self.add_node(self.root)
    
    def __contains__(self, id):
        return id in self.nodes
    
    def __len__(self):
        return len(self.nodes)
    
    def add_node(self, node):
        self.nodes[node.id] = node
    
    def __getitem__(self, key):
        return self.get_node(key)
    
    def get_node(self, node):
        if isinstance(node, Node):
            return node
        return self.nodes[node]
    
    def get_root(self):
        return self.root
    
    def add_child(self, parent, child):
        self.nodes[parent].add_child(child)
        self.add_node(child)
    
    def create_child(self, parent, *args, **kwargs):
        child = self.get_node(parent).create_child(*args, **kwargs)
        self.add_node(child)
        return child
    
    def add_parent(self, child, parent):
        self.nodes[child].add_parent(parent)
        self.add_node(parent)
    
    def create_parent(self, child, *args, **kwargs):
        parent = self.get_node(child).create_parent(*args, **kwargs)
        self.add_node(parent)
        return parent

if __name__ == "__main__":
    tree = Tree(1, 1, 3)
    
    tree.create_child(1, 2, 2, 4)
    tree.create_child(1, 3, 2, 5)
    tree.create_child(1, 4, 3, 6)
    tree.create_parent(4, 5, 0, 6)
    
    plot_tree(tree)
