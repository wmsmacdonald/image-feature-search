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


def search(query_descriptors, index_file):

    database = pickle.load(open(index_file, 'rb'))

    all_descriptors, partitions = database

    flann_matcher = cv2.FlannBasedMatcher(
        indexParams=dict(algorithm=6,
                         table_number=6,
                         key_size=12,
                         multi_probe_level=1),
        searchParams=dict(checks=50)
    )

    matches = flann_matcher.knnMatch(query_descriptors, all_descriptors, k=2)

    confident_matches = [m for m, n in matches if m.distance < 0.5 * n.distance]

    get_matching_file = compose(
        partitions.get_value,
        op.attrgetter('trainIdx')
    )

    votes = list(map(get_matching_file, confident_matches))

    frequencies = collections.Counter(votes).most_common()

    results = sorted(frequencies, key=op.itemgetter(1))[::-1]

    return results


def search_file(file, index_file):
    if not os.path.exists(file):
        raise IOError('Cannot open file %s' % file)

    query_descriptors = get_descriptors(file, nfeatures=2000)
    return search(query_descriptors, index_file)

if __name__ == '__main__':
    results = search(sys.argv[1], sys.argv[2])
    for r in results:
        print("File: %s, Distance: %f" % r)
