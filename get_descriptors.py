import cv2
import numpy as np


def get_descriptors(file):
    img = cv2.imread(file, 0)

    orb = cv2.ORB_create()

    kp = orb.detect(img, None)
    return orb.compute(img, kp)[1]

