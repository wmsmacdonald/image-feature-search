#!/usr/bin/env python3

import cv2
import pickle
from functools import partial as p
import sys
import os
import operator as op
import collections
from get_descriptors import get_descriptors


def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


def compute_distance(matcher, des1, des2):
    if des1 is None or des2 is None:
        return sys.maxsize

    matches = flatten(matcher(des1, des2))
    if len(matches) == 0:
        return sys.maxsize

    top_matches = sorted(matches, key=lambda m: m.distance)[:20]

    return sum(map(lambda m: m.distance, top_matches)) / len(top_matches)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match

flannKnn = p(
    cv2.FlannBasedMatcher(
        indexParams=dict(algorithm=6,
                         table_number=6,
                         key_size=12,
                         multi_probe_level=1),
        searchParams=dict(checks=50)
    ).knnMatch, k=2
)


def search(query_descriptors, index_file, matcher=bf):

    database = pickle.load(open(index_file, 'rb'))

    files, descriptors_by_file = zip(*database.items())

    compute_distance_matcher = p(compute_distance, matcher)

    distances_by_file = map(p(compute_distance_matcher, query_descriptors), descriptors_by_file)

    results = sorted(zip(files, distances_by_file), key=op.itemgetter(1))
    return results


def search_file(file, index_file):
    if not os.path.exists(file):
        raise IOError('Cannot open file %s' % file)

    query_descriptors = get_descriptors(file)
    return search(query_descriptors, index_file)

if __name__ == '__main__':
    results = search(sys.argv[1], sys.argv[2])
    for r in results:
        print("File: %s, Distance: %f" % r)
