#!/usr/bin/env python

import sys
from functools import partial as p
from keypoint_signatures import compute_keypoint_signatures
import itertools
import cv2
import os
import pickle
from get_descriptors import get_descriptors
from partitioned import Partitioned
import numpy as np
import operator as op

files = sorted(os.listdir(sys.argv[1]))[:200]
files = ['frame0031.jpg']

file_paths = list(map(p(os.path.join, sys.argv[1]), files))

for f in file_paths:
    if not os.path.isfile(f):
        raise IOError('Cannot open file %s' % f)


def signatures(file_path):
    image = cv2.imread(file_path, 0)
    detector = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.12)
    keypoints = detector.detect(image)
    print(len(keypoints))
    keypoint_signatures = compute_keypoint_signatures(image, keypoints)
    return keypoint_signatures

keypoint_signatures_by_file = list(map(signatures, file_paths))

files_and_signature_groups = list(zip(files, keypoint_signatures_by_file))

files_and_signatures = [(file_name, signature) for file_name, signatures
                        in files_and_signature_groups for signature in signatures]

signatures_to_file_groups = {key: list(set(map(op.itemgetter(0), group)))
                             for key, group
                             in itertools.groupby(sorted(files_and_signatures, key=op.itemgetter(1)), key=op.itemgetter(1))}

print(signatures_to_file_groups)

pickle.dump(signatures_to_file_groups, open('./indexes/full_index.p', 'wb'))

