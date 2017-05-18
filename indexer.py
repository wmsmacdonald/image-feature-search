#!/usr/bin/env python

import cv2
import sys
from functools import partial as p
import os
import pickle
from get_descriptors import get_descriptors


def serialize_keypoints(keypoints, descriptors):
    return [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id, desc)
            for kp, desc in zip(keypoints, descriptors)]


files = sorted(os.listdir(sys.argv[1]))
file_paths = map(p(os.path.join, sys.argv[1]), files)

descriptors_by_file = list(map(get_descriptors, file_paths))

database = [(file, kp) for file, kp in zip(files, descriptors_by_file)]

pickle.dump(database, open(sys.argv[2], 'wb'))

