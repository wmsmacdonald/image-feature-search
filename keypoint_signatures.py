from sklearn.neighbors import NearestNeighbors
import numpy as np
import math
import sys
import cv2
from matplotlib import pyplot as plt
from functools import partial as p

DESCRIPTOR_SIZE = 16


def compute_grid_coordinates(center_x, center_y, grid_point_separation, square_offset=5):
    distance_offset = grid_point_separation * square_offset

    xs = range(center_x - distance_offset, center_x + distance_offset + 1, grid_point_separation)
    ys = range(center_y - distance_offset, center_y + distance_offset + 1, grid_point_separation)

    return [(x, y) for y in ys for x in xs]


def compute_grid_averages(coordinates, image, p):
    grid_square_offset_lower = (p - 1) // 2
    grid_square_offset_upper = p // 2

    def threshold_y(value):
        return max(0, min(value, image.shape[0] - 1))

    def threshold_x(value):
        return max(0, min(value, image.shape[1] - 1))

    slicers = [
        (
            slice(threshold_y(y - grid_square_offset_lower), threshold_y(y + grid_square_offset_upper) + 1),
            slice(threshold_x(x - grid_square_offset_lower), threshold_x(x + grid_square_offset_upper) + 1)
        )
        for x, y in coordinates]

    return [image[y_slicer, x_slicer].mean() for y_slicer, x_slicer in slicers]

def compute_normalizer(values, n_levels=5):
    percentiles = np.linspace(0, 100, n_levels + 1)
    cutoffs = np.percentile(values, percentiles)
    ranges = [range(int(start), int(stop + 1)) for start, stop in zip(cutoffs, cutoffs[1:])]

    def normalizer(value):
        for i, r in enumerate(ranges):
            if value in r:
                return i
        raise Exception('Outside given maximum')

    return normalizer


def compute_keypoint_signature(keypoint, image, square_offset=3):
    num_points_wide = square_offset * 2 + 1
    grid_width = num_points_wide * keypoint.size

    p = max(2, int(0.5 + grid_width / (2 * num_points_wide)))
    grid_coordinates = compute_grid_coordinates(int(keypoint.pt[0]), int(keypoint.pt[1]), int(keypoint.size), square_offset)
    grid_averages = np.array(compute_grid_averages(grid_coordinates, image, p), dtype=int).reshape((num_points_wide, num_points_wide))

    def wrap(index, stop):
        return 0 if index >= stop else index

    def neighbors(A, y, x):
        upper = (wrap(y - 1, A.shape[0]), wrap(x, A.shape[1]))
        left = (wrap(y, A.shape[0]), wrap(x - 1, A.shape[1]))
        right = (wrap(y, A.shape[0]), wrap(x + 1, A.shape[1]))
        lower = (wrap(y + 1, A.shape[0]), wrap(x, A.shape[1]))
        coordinates = [upper, left, right, lower]
        values = [A[coordinate] for coordinate in coordinates]

        return np.array(values)

    print(grid_averages.shape)

    neighbors_by_grid_point = [neighbors(grid_averages, x, y) for y in range(grid_averages.shape[0]) for x in range(grid_averages.shape[1])]
    brightness_differences = [difference + 255 for neighbors, grid_average
                              in zip(neighbors_by_grid_point, grid_averages.flatten())
                              for difference in neighbors - grid_average]
    normalizer = compute_normalizer(brightness_differences)

    normalized_differences = tuple(map(normalizer, brightness_differences))
    print(len(normalized_differences))

    return normalized_differences


def compute_keypoint_signatures(image, keypoints):
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, outImage=None)

    #plt.imshow(image_with_keypoints)
    #plt.show()

    signatures_by_keypoint = [compute_keypoint_signature(keypoint, image) for keypoint in keypoints]

    return signatures_by_keypoint

