import cv2


def get_descriptors(file, nfeatures=500):
    img = cv2.imread(file, 0)

    orb = cv2.ORB_create(nfeatures=nfeatures)

    descriptors = orb.detectAndCompute(img, None)[1]
    return descriptors

