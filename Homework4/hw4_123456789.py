import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import cv2


# Homework 1 helpers

def histImage(im):
    # start with an empty histogram
    h = np.zeros(256)

    # iterate over the image pixels incrementing the appropriate histogram bin
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            h[int(im[i, j])] += 1

    return h


def ahistImage(im):
    # build image histogram
    h = histImage(im)

    # calculate accumulate histogram
    ah = np.cumsum(h)

    return ah


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


# Homework 2 helpers

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
    n, m = image.shape

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


# Homework 3 helpers

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


# Homework 4 functions


def clean_baby(im):
    h, w = im.shape
    transformed_images = np.zeros((3, h, w))
    _radius = 3

    dst_points = np.float32([[0, 0],
                             [h, 0],
                             [h, w],
                             [0, w]])

    # baby sub-image 1
    src_points1 = np.float32([[5.5, 19.5],
                              [111.5, 19.5],
                              [111.5, 130.5],
                              [5.5, 130.5]])

    T1 = find_transform(src_points1, dst_points)
    transformed_images[0, :, :] = trasnform_image(im, T1)

    # baby sub-image 2
    src_points2 = np.float32([[77.5, 162.5],
                              [146.5, 116.5],
                              [245.5, 159.5],
                              [132.5, 244.5]])

    T2 = find_transform(src_points2, dst_points)
    transformed_images[1, :, :] = trasnform_image(im, T2)

    # baby sub-image 3
    src_points3 = np.float32([[180.5, 4.5],
                              [249.5, 69.5],
                              [175.5, 120.5],
                              [120.5, 50.5]])

    T3 = find_transform(src_points3, dst_points)
    transformed_images[2, :, :] = trasnform_image(im, T3)

    # clean SP noise
    clean_im = clean_SP_noise_multiple(transformed_images)
    clean_im = clean_SP_noise_single(clean_im, _radius)

    return clean_im


def clean_windmill(im):
    # move to frequency domain
    im_fourier = np.fft.fft2(im)
    im_fourier = np.fft.fftshift(im_fourier)

    # cancel specific frequencies
    im_fourier[124][100] = 0
    im_fourier[132][156] = 0

    # move back to image domain
    clean_im = abs(np.fft.ifft2(im_fourier))

    return clean_im


def clean_watermelon(im):
    # convolve with sharpening filter from tutorial
    sharp_mask = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
    clean_im = convolve2d(im, sharp_mask, mode='same')

    return clean_im


def clean_umbrella(im):
    # mask = (delta + delta[4, 79]) / 2
    mask = np.zeros(im.shape)
    mask[0, 0] = 0.5
    mask[4, 79] = 0.5

    # move to frequency domain
    im_fourier = np.fft.fft2(im)

    mask_fourier = np.fft.fft2(mask)

    # replace values that are too small (to avoid division by zero)
    mask_fourier[abs(mask_fourier) < 10 ** -15] = 1

    # inverse filtering
    clean_im_fourier = im_fourier / mask_fourier

    # move back to image domain
    clean_im = abs(np.fft.ifft2(clean_im_fourier))

    return clean_im


def clean_USAflag(im):
    clean_im = im.copy()
    _radius = 7

    # apply median mask on the bottom of the flag
    for i in range(_radius, im.shape[1] - _radius):
        for j in range(90, 167):
            mask = im[j, i - _radius:i + _radius]
            clean_im[j][i] = np.median(mask)

    # apply median mask on the right side of flag
    for i in range(160, 299 - _radius):
        for j in range(1, 90):
            mask = im[j, i - _radius:i + _radius]
            clean_im[j][i] = np.median(mask)

    return clean_im


def clean_cups(im):
    # move to frequency domain
    im_fourier = np.fft.fft2(im)
    im_fourier = np.fft.fftshift(im_fourier)

    # lower the frequencies in the square at the center
    mask = np.ones(im.shape)
    for i in range(109, 149):
        for j in range(109, 149):
            mask[i, j] = 1.5

    clean_im_fourier = mask * im_fourier

    # move back to image domain
    clean_im = abs(np.fft.ifft2(clean_im_fourier))

    return clean_im


def clean_house(im):
    # mask = (delta + delta[0, 1] + ... + delta[0, 9]) / 10
    mask = np.zeros(im.shape)
    mask[0, :10] = 0.1

    # move to frequency domain
    im_fourier = np.fft.fft2(im)

    mask_fourier = np.fft.fft2(mask)

    # replace values that are too small (to avoid division by zero)
    mask_fourier[abs(mask_fourier) < 10 ** -15] = 1

    # inverse filtering
    clean_im_fourier = im_fourier / mask_fourier

    # move back to image domain
    clean_im = abs(np.fft.ifft2(clean_im_fourier))

    return clean_im


def clean_bears(im):
    # parameters
    a = 4

    # hist stretching tone mapping
    stretch = np.zeros(256)
    for i in range(256):
        stretch[i] = a * i

    # stretch around mean
    m = np.mean(im)
    clean_im = im - m
    clean_im = mapImage(clean_im, stretch)
    clean_im += m

    return clean_im


'''
    # an example of how to use fourier transform:
    img = cv2.imread(r'Images\windmill.tif')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_fourier = np.fft.fft2(img) # fft - remember this is a complex numbers matrix 
    img_fourier = np.fft.fftshift(img_fourier) # shift so that the DC is in the middle
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img, cmap='gray')
    plt.title('original image')
    
    plt.subplot(1,3,2) plt.imshow(np.log(abs(img_fourier)), cmap='gray') # need to use abs because it is complex, 
    # the log is just so that we can see the difference in values without eyes. plt.title('fourier transform of image')

    img_inv = np.fft.ifft2(img_fourier)
    plt.subplot(1,3,3)
    plt.imshow(abs(img_inv), cmap='gray')
    plt.title('inverse fourier of the fourier transform')

'''
