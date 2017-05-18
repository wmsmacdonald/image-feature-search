#!/usr/bin/env python

import cv2
import json
import sys
from functools import partial as p
import os
import pickle


def serialize_keypoints(keypoints, descriptors):
    return [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id, desc)
            for kp, desc in zip(keypoints, descriptors)]


def get_keypoints_and_descriptors(file):
    img = cv2.imread(file, 0)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp, des = sift.detectAndCompute(img, None)

    return kp, des

files = sorted(os.listdir(sys.argv[1]))
file_paths = map(p(os.path.join, sys.argv[1]), files)


keypoints_and_descriptors_by_file = list(map(get_keypoints_and_descriptors, file_paths))


serialized_keypoints = list(map(lambda keypoints_and_descriptors: serialize_keypoints(
    keypoints_and_descriptors[0], keypoints_and_descriptors[1]), keypoints_and_descriptors_by_file))

database = {file: kp for file, kp in zip(files, serialized_keypoints)}

pickle.dump(database, open(sys.argv[2], 'wb'))

