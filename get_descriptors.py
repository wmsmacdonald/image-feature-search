import cv2


def get_descriptors(file):
    img = cv2.imread(file, 0)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    _, des = sift.detectAndCompute(img, None)
    return des
