#!/usr/bin/env python3

import cv2
import pickle
from functools import partial as p
import sys
import operator as op
from get_descriptors import get_descriptors
import numpy as np

# FLANN parameters
FLANN_INDEX_KDTREE = 0
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
search_params = dict(checks=50)   # or pass empty dictionary


def compute_distance(des1, des2):
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=1)
    return sum(map(lambda m: m[0].distance, matches)) / len(matches)


original_descriptors = get_descriptors(sys.argv[1])


def descriptors(byte_descriptors):
    a = np.frombuffer(byte_descriptors, dtype=np.uint8)
    return a.reshape(original_descriptors.shape)

database = pickle.load(open(sys.argv[2], 'rb'))

files, descriptor_bytes_by_file = zip(*database)

descriptors_by_file = map(descriptors, descriptor_bytes_by_file)

distances_by_file = map(p(compute_distance, original_descriptors), descriptors_by_file)

results = sorted(zip(files, distances_by_file), key=op.itemgetter(1))

for r in results:
    print("File: %s, Distance: %f" % r)


