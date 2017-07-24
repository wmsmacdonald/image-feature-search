#!/usr/bin/env python3

import cv2
from keypoint_signatures import compute_keypoint_signatures
import pickle
from functools import reduce, partial as p
from itertools import islice
import itertools
import sys
import numpy as np
import os
import operator as op
import collections
from get_descriptors import get_descriptors


def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


def compose(*functions):
    def inner(arg):
        for f in reversed(functions):
            arg = f(arg)
        return arg
    return inner


def compute_distance(matcher, des1, des2):
    if des1 is None or des2 is None:
        return sys.maxsize

    matches = flatten(matcher(des1, des2))
    if len(matches) == 0:
        return sys.maxsize

    top_matches = sorted(matches, key=lambda m: m.distance)[:20]

    return sum(map(lambda m: m.distance, top_matches)) / len(top_matches)


def search(query_keypoint_signatures, index_file):

    signatures_to_file_groups = pickle.load(open(index_file, 'rb'))

    file_names = [file for signature in query_keypoint_signatures for file
                  in signatures_to_file_groups.get(tuple(signature), [])]

    results = collections.Counter(file_names).most_common()

    return results


def search_file(file, index_file):
    if not os.path.exists(file):
        raise IOError('Cannot open file %s' % file)

    image = cv2.imread(file, 0)
    star = cv2.xfeatures2d.StarDetector_create()
    detector = cv2.MSER_create()
    detector = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.12)
    keypoints = detector.detect(image)
    print('keypoints', len(keypoints))
    query_keypoint_signatures = compute_keypoint_signatures(image, keypoints)
    return search(query_keypoint_signatures, index_file)

if __name__ == '__main__':
    results = search(sys.argv[1], sys.argv[2])
    for r in results:
        print("File: %s, Distance: %f" % r)
