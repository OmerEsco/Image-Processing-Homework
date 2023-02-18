import numpy as np
import matplotlib.pyplot as plt
import cv2


def histImage(im):
    # start with an empty histogram
    h = np.zeros(256)

    # iterate over the image pixels incrementing the appropriate histogram bin
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            h[int(im[i, j])] += 1

    return h


def nhistImage(im):
    # build image histogram
    h = histImage(im)

    # divide each histogram bin by the total amount of pixels
    nh = h / sum(h)

    return nh


def ahistImage(im):
    # build image histogram
    h = histImage(im)

    # calculate accumulate histogram
    ah = np.cumsum(h)

    return ah


def calcHistStat(h):
    # build normalized histogram
    nh = h / sum(h)

    # mean - calculate dot product of value vector and probability vector
    m = np.dot(np.arange(256), nh)

    # variance - calculate dot product of value vector squared and probability vector minus the mean squared
    v = np.dot(np.arange(256) ** 2, nh) - m ** 2

    return m, v


def mapImage(im, tm):
    # start with an empty image
    nim = np.zeros(im.shape)

    # iterate over grayscale values
    for i in range(256):
        # reduced all values greater than 255 to 255 and raise all values smaller than 0 to 0
        nim[im == i] = min(max(tm[i], 0), 255)

    return nim


def histEqualization(im):
    # start with an empty tone mapping
    tm = np.zeros(256)

    # calculate image accumulating histogram
    orig_ah = ahistImage(im)

    # calculate goal histogram
    goal_h = np.zeros(256) + (orig_ah[-1] // 256)
    goal_ah = np.cumsum(goal_h)

    # two pointer algorithm
    orig_ptr, goal_ptr = 0, 0
    while orig_ptr < 256 and goal_ptr < 256:
        if orig_ah[orig_ptr] <= goal_ah[goal_ptr]:
            tm[orig_ptr] = goal_ptr
            orig_ptr += 1
        else:
            goal_ptr += 1

    return tm
