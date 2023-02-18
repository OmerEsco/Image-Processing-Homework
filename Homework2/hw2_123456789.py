import cv2
import matplotlib.pyplot as plt
import numpy as np

# size of the image
m, n = 921, 750

# frame points of the blank wormhole image
src_points = np.float32([[0, 0],
                         [int(n / 3), 0],
                         [int(2 * n / 3), 0],
                         [n, 0],
                         [n, m],
                         [int(2 * n / 3), m],
                         [int(n / 3), m],
                         [0, m]])

# blank wormhole frame points
dst_points = np.float32([[96, 282],
                         [220, 276],
                         [344, 276],
                         [468, 282],
                         [474, 710],
                         [350, 744],
                         [227, 742],
                         [103, 714]]
                        )


def find_transform(pointset1, pointset2):
    # number of points
    N = pointset2.shape[0]

    # construct X tag
    X_tag = pointset2.reshape((2 * N,))

    # construct X
    X = np.zeros((2 * N, 8))
    for i in range(N):
        xi, yi = pointset1[i]
        xi_tag, yi_tag = pointset2[i]

        X[2 * i, 0] = xi
        X[2 * i, 1] = yi

        X[2 * i + 1, 2] = xi
        X[2 * i + 1, 3] = yi

        X[2 * i, 4] = 1
        X[2 * i + 1, 5] = 1

        X[2 * i, 6] = -xi * xi_tag
        X[2 * i + 1, 6] = -xi * yi_tag

        X[2 * i, 7] = -yi * xi_tag
        X[2 * i + 1, 7] = -yi * yi_tag

    # calculate coefficient vector
    coeff_vect = np.matmul(np.linalg.pinv(X), X_tag)

    # rearrange to transformation matrix
    T = np.ones((3, 3))
    for i in range(2):
        T[i, 2] = coeff_vect[4 + i]
        T[2, i] = coeff_vect[6 + i]
        for j in range(2):
            T[i, j] = coeff_vect[2 * i + j]

    return T


def trasnform_image(image, T):
    # define a new_image full of zeros the size of the given image
    new_image = np.zeros(image.shape)

    # calculate inverse transformation
    T_inv = np.linalg.inv(T)

    # iterate over coordinates [x’,y’] of the pixels of the new image
    for i in range(new_image.shape[0]):
        for j in range(new_image.shape[1]):
            # destination coordinates
            dst_coords = (i, j)
            # homogeneous destination coordinates
            hom_dst = np.array((j, i, 1))
            # homogeneous source coordinates
            hom_src = np.matmul(T_inv, hom_dst)
            # source coordinates
            src_coords = (round(hom_src[1] / hom_src[2]), round(hom_src[0] / hom_src[2]))
            # if coordinates out of bounds, assign zero
            if src_coords[0] < 0 or src_coords[0] >= m or src_coords[1] < 0 or src_coords[1] >= n:
                new_image[dst_coords] = 0
            # assign color of source pixel to destination pixel
            else:
                new_image[dst_coords] = image[src_coords]

    return new_image


def create_wormhole(im, T, iter=5):
    sum_images = im
    trans_image = im

    for i in range(iter):
        trans_image = trasnform_image(trans_image, T)
        sum_images = sum_images + trans_image

    new_image = np.clip(sum_images, 0, 255)

    return new_image
