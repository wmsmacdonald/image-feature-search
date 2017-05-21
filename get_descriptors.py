import cv2


def get_descriptors(file):
    img = cv2.imread(file, 0)

    orb = cv2.ORB_create()

    kp = orb.detect(img, None)
    descriptors = orb.compute(img, kp)[1]
    return descriptors

