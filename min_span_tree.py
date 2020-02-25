#!/usr/bin/env python

import numpy as np
from scipy.sparse import csgraph, csr_matrix
import networkx as nx
from tqdm import tqdm
import sys


def descriptor_reader(dfile):
    d = np.load(dfile)
    
    count = 0
    nodes_dict = {}
    desc_list = []
    for key in sorted(d.keys()):
        if key[-5:] == 'ement':
            continue
        desc_list.append(np.ravel(d[key]))
        nodes_dict[count] = key
        count += 1
    return np.array(desc_list), nodes_dict


def cosine_dist(desc_array):
    no_of_structures = desc_array.shape[0]
    dists = np.zeros((no_of_structures, no_of_structures)).astype('float32')
    for i in tqdm(range(no_of_structures), desc="Calculating Cosine Distances"):
        dists[i,:] = np.clip(2 - 2 * (np.dot(desc_array, desc_array[i]) \
                / np.linalg.norm(desc_array, axis = 1) \
                / np.linalg.norm(desc_array[i])).astype('float32'), 0, 2)
    return dists


def connectivity_graph(distances):
    is_connected = False
    n = 100
    while not is_connected:
        print('Keeping %s nearest neighbours...' %n)
        distances_temp = np.copy(distances)
        for distance in tqdm(distances_temp):
            max_allowed = sorted(distance)[n]
            distance[np.where(distance > max_allowed)] = 0
#            distance[np.where(distance < 0.01)] = 0


        print('Sparsing the distance matrix...')
        distances_temp = csr_matrix(distances_temp)
        print('Building connectivity graph...')
        mst = csgraph.minimum_spanning_tree(distances_temp).toarray()
        G = nx.from_numpy_array(mst)
        is_connected = nx.is_connected(G)
        if is_connected:
            print('SUCCESSFUL. The graph is connected.')
            return G
        else:
            print('FAILED. The graph was not connected.')
            n += 50


def connecting_path(dfile, outdir = '.'):
    desc_array, nodes_dict = descriptor_reader(dfile)
    dist = cosine_dist(desc_array)
    G = connectivity_graph(dist)

    idx_A = idx_B = -1
    for idx, key in nodes_dict.items():
        if key == 'A':
            idx_A = idx
        if key == 'B':
            idx_B = idx
    if idx_A == -1 or idx_B == -1:
        raise Exception("""
        Something went wrong. A and B are not found
        """)
    
    with open('%s/path.txt' %outdir, 'w') as f:
        for idx in nx.dijkstra_path(G, idx_A, idx_B):
            f.write("%s\n" % nodes_dict[idx])


dfile, outdir = sys.argv[1:3]

if __name__ == "__main__":
    connecting_path(dfile, outdir)

