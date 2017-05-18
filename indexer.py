#!/usr/bin/env python

import sys
from functools import partial as p
import os
import pickle
from get_descriptors import get_descriptors
import itertools
import numpy as np


def serialize_keypoints(keypoints, descriptors):
    return [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id, desc)
            for kp, desc in zip(keypoints, descriptors)]


def map_3d(func, iterable):
    return list(map(lambda row: list(map(func, row)), iterable))


def compose(*functions):
    def inner(arg):
        for f in reversed(functions):
            arg = f(arg)
        return arg
    return inner


bytes_descriptor_functions = [
    bytes,
    itertools.chain.from_iterable,
    p(map, np.matrix.tolist),
    p(map, np.matrix.flatten)
]

bytes_descriptor = compose(*bytes_descriptor_functions)

files = sorted(os.listdir(sys.argv[1]))
file_paths = map(p(os.path.join, sys.argv[1]), files)

descriptors_by_file = list(map(get_descriptors, file_paths))

bytes_descriptors_by_file = list(map(bytes_descriptor, descriptors_by_file))

database = [(file, kp) for file, kp in zip(files, bytes_descriptors_by_file)]

pickle.dump(database, open(sys.argv[2], 'wb'))

