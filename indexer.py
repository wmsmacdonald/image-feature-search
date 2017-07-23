#!/usr/bin/env python

import sys
from functools import partial as p
from keypoint_signatures import compute_keypoint_signatures
import cv2
import os
import pickle
from get_descriptors import get_descriptors
from partitioned import Partitioned
import numpy as np
import operator as op

files = sorted(os.listdir(sys.argv[1]))[0:30]
#files = ['thumb0025.jpg']

file_paths = list(map(p(os.path.join, sys.argv[1]), files))

for f in file_paths:
    if not os.path.isfile(f):
        raise IOError('Cannot open file %s' % f)


def signatures(file_path):
    image = cv2.imread(file_path, 0)
    star = cv2.xfeatures2d.StarDetector_create()
    keypoints = star.detect(image)
    keypoint_signatures = compute_keypoint_signatures(image, keypoints)
    print(keypoint_signatures)
    return keypoint_signatures

keypoint_signatures_by_file = list(map(signatures, file_paths))

database = dict(zip(file_paths, keypoint_signatures_by_file))

pickle.dump(database, open('./indexes/full_index.p', 'wb'))

