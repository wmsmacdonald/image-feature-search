from sklearn.neighbors import NearestNeighbors
import numpy as np
import sys
import cv2
from functools import partial as p

DESCRIPTOR_SIZE = 16


def compute_neighbor_means(keypoint_coordinates):
    nearest_neighbors = NearestNeighbors(n_neighbors=11)
    nearest_neighbors.fit(keypoint_coordinates)
    indexes_by_keypoints = nearest_neighbors.kneighbors(keypoint_coordinates, return_distance=False)[1:]
    neighbors_by_keypoints = [[keypoint_coordinates[index] for index in indexes] for indexes in indexes_by_keypoints]
    means_by_keypoint = [np.mean(neighbors, axis=0, dtype=int) for neighbors in neighbors_by_keypoints]
    return means_by_keypoint


def compute_surrounding_coordinates(center, neighbors_mean):
    difference_vector = neighbors_mean - center

    rotations = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    scales = np.array([[1, 1], [2, 2], [3, 3]])

    rotated = np.multiply(rotations, difference_vector)
    scaled = [vector for unit_vector in rotated for vector in np.multiply(scales, unit_vector)]
    coordinates = [center + scaled_vector for scaled_vector in scaled]
    return coordinates


def compute_descriptor(image, vector):
    keypoint = cv2.KeyPoint(*vector, _size=31.0)
    describer = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=DESCRIPTOR_SIZE)
    descriptor = describer.compute(image, [keypoint])[1]
    return descriptor


def compute_surrounding_distances(center, vector, image):
    surrounding_coordinates = compute_surrounding_coordinates(center, vector)

    surrounding_descriptors = list(map(p(compute_descriptor, image), surrounding_coordinates))
    center_descriptor = compute_descriptor(image, center)

    distances = [compute_difference(center_descriptor, surrounding_descriptor)
                 for surrounding_descriptor in surrounding_descriptors]
    return distances


def compute_difference(descriptor1, descriptor2):
    if descriptor1 is None or descriptor2 is None:
        return sys.maxsize

    return cv2.norm(descriptor1, descriptor2, normType=cv2.HAMMING_NORM_TYPE)


def compute_normalizer(distances, n_levels=5):
    percentiles = np.linspace(0, 100, n_levels + 1)
    cutoffs = np.percentile(distances, percentiles)
    ranges = [range(int(start), int(stop + 1)) for start, stop in zip(cutoffs, cutoffs[1:])]

    def normalizer(value):
        for i, r in enumerate(ranges):
            if value in r:
                return i
        raise Exception('Outside given maximum')

    return normalizer


def normalize_distances(distances):
    normalizer = compute_normalizer(distances)

    return list(map(normalizer, distances))


def compute_keypoint_signatures(image, keypoints):
    keypoint_coordinates = [np.array(keypoint.pt) for keypoint in keypoints]
    means_by_keypoint = compute_neighbor_means(keypoint_coordinates)

    surrounding_distances_by_keypoint = [compute_surrounding_distances(kp_coords, mean, image)
                                         for kp_coords, mean in zip(keypoint_coordinates, means_by_keypoint)]

    normalized_distances_by_keypoint = list(map(normalize_distances, surrounding_distances_by_keypoint))
    return normalized_distances_by_keypoint

