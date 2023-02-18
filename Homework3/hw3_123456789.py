import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


# the copy in the first lines of the function is so that you don't ruin
# the original image. it will create a new one. 

def add_SP_noise(im, p):
    sp_noise_im = im.copy()

    # height and width
    h, w = im.shape

    # number of total pixels and noise pixels
    n_total = h * w
    n_noise = int(n_total * p)

    # randomly sample pixels for S&P noise
    noise_pixels = random.sample(range(n_total), n_noise)

    # add salt noise
    for i in noise_pixels[:n_noise // 2]:
        sp_noise_im[i // h, i % w] = 255

    # add pepper noise
    for i in noise_pixels[n_noise // 2:]:
        sp_noise_im[i // h, i % w] = 0

    return sp_noise_im


def clean_SP_noise_single(im, radius):
    clean_im = im.copy()

    # height and width
    h, w = im.shape

    # iterate over image pixels
    for i in range(radius, h - radius):
        for j in range(radius, w - radius):
            # neighborhood indices
            X, Y = np.meshgrid(np.arange(i - radius, i + radius + 1), np.arange(j - radius, j + radius + 1))

            # apply median filter
            clean_im[i, j] = np.median(im[X, Y])

    return clean_im


def clean_SP_noise_multiple(images):
    clean_image = np.median(images, axis=0)
    return clean_image


def add_Gaussian_Noise(im, s):
    # zero mean additive noise
    gaussian_noise = np.random.normal(0, s, im.shape)
    gaussian_noise_im = im + gaussian_noise

    return gaussian_noise_im


def clean_Gaussian_noise(im, radius, maskSTD):
    # (2radius+1)x(2radius+1) mask
    gaussian_mask = np.zeros((2 * radius + 1, 2 * radius + 1))

    # compute gaussian filter
    for x in range(-radius, radius + 1):
        for y in range(-radius, radius + 1):
            gaussian_mask[y + radius, x + radius] = np.exp(-0.5 * (x ** 2 + y ** 2) / (maskSTD ** 2))

    # normalize values
    gaussian_mask = gaussian_mask / np.sum(gaussian_mask)

    # convolve image with filter
    cleaned_im = convolve2d(im, gaussian_mask, mode="same")

    return cleaned_im.astype(np.uint8)


def clean_Gaussian_noise_bilateral(im, radius, stdSpatial, stdIntensity):
    bilateral_im = im.copy()

    # height and width
    h, w = im.shape

    # iterate over image pixels
    for i in range(radius, h - radius):
        for j in range(radius, w - radius):
            # values in the neighborhood of the pixel
            window = im[i - radius: i + radius + 1, j - radius: j + radius + 1]

            # neighborhood indices
            X, Y = np.meshgrid(np.arange(i - radius, i + radius + 1), np.arange(j - radius, j + radius + 1))

            # gaussian mask based on intensity differences
            gi = np.exp(-0.5 * (im[X, Y] - im[i, j]) ** 2 / stdIntensity ** 2)
            gi = gi / np.sum(gi)

            # gaussian mask based on coordinates distances
            gs = np.exp(-0.5 * ((X - i) ** 2 + (Y - j) ** 2) / stdSpatial ** 2)
            gs = gs / np.sum(gs)

            bilateral_im[i, j] = np.sum(gi * gs * window) / np.sum(gi * gs)

    return bilateral_im.astype(np.uint8)
