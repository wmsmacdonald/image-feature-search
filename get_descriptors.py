import cv2


def get_descriptors(file):
    img = cv2.imread(file, 0)

    orb = cv2.ORB_create()

    kp = orb.detect(img, None)

    _, des = orb.compute(img, kp)
    return des
