import cv2


def get_descriptors(file, nfeatures=500):
    img = cv2.imread(file, 0)

    orb = cv2.ORB_create(nfeatures=nfeatures)
    surf = cv2.xfeatures2d.SURF_create()
    star = cv2.xfeatures2d.StarDetector_create()

    fast = cv2.FastFeatureDetector_create()

    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=16)

    kp = star.detect(img)
    descriptors = brief.compute(img, kp)[1]
    return descriptors

