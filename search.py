#!/usr/bin/env python3

import cv2
import pickle
from functools import partial as p
import sys
import operator as op
from get_descriptors import get_descriptors

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)   # or pass empty dictionary


def compute_distance(des1, des2):
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=1)
    return sum(map(lambda m: m[0].distance, matches))


original_descriptors = get_descriptors(sys.argv[1])

database = pickle.load(open(sys.argv[2], 'rb'))

files, descriptors_by_file = zip(*database)

distances_by_file = map(p(compute_distance, original_descriptors), descriptors_by_file)

results = sorted(zip(files, distances_by_file), key=op.itemgetter(1))

for r in results:
    print("File: %s, Distance: %d" % r)


