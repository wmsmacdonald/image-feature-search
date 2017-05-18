#!/usr/bin/env python3

import cv2
import pickle
from functools import partial as p
import sys
import os
import operator as op
from get_descriptors import get_descriptors


def compute_distance(des1, des2):
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                       table_number = 6, # 12
                       key_size = 12,     # 20
                       multi_probe_level = 1) #2
    search_params = dict(checks=50)   # or pass empty dictionary

    if des1 is None or des2 is None:
        return sys.maxsize

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=1)
    valid_matches = list(filter(lambda x: len(x) == 1, matches))
    return sum(map(lambda m: m[0].distance, valid_matches)) / len(valid_matches)


def search(file, index_file):
    if not os.path.exists(file):
        raise IOError('Cannot open file %s' % file)

    original_descriptors = get_descriptors(file)

    database = pickle.load(open(index_file, 'rb'))

    files, descriptors_by_file = zip(*database)

    distances_by_file = map(p(compute_distance, original_descriptors), descriptors_by_file)

    results = sorted(zip(files, distances_by_file), key=op.itemgetter(1))
    return results


if __name__ == '__main__':
    results = search(sys.argv[1], sys.argv[2])
    for r in results:
        print("File: %s, Distance: %f" % r)
