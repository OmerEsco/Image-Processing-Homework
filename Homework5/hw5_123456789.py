import cv2
import numpy as np
from scipy.signal import convolve2d as conv
import matplotlib.pyplot as plt


def sobel(im):
    # Sobel operators
    s_x = 0.125 * np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
    s_y = 0.125 * np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]])

    # convolve with image
    sobel_x = conv(im, s_x, mode='same')
    sobel_y = conv(im, s_y, mode='same')

    # approximated magnitude
    new_im = np.abs(sobel_x) + np.abs(sobel_y)

    # thresholding
    threshold = 17
    new_im = new_im > threshold

    return new_im


def canny(im):
    # hysteresis thresholds
    t_lower = 40
    t_upper = 220

    # apply Gaussian blur
    im_blur = cv2.GaussianBlur(im, (9, 9), 0)

    # Canny edge detection
    im_canny = cv2.Canny(im_blur, t_lower, t_upper)

    return im_canny


def hough_circles(im):
    im_c = im.copy()

    # find circles in image
    circles = cv2.HoughCircles(im, cv2.HOUGH_GRADIENT, 1, 20, minRadius=20)
    circles = np.uint16(np.around(circles))

    # add circles to image
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(im_c, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(im_c, (i[0], i[1]), 2, (0, 0, 255), 3)

    return im_c


def hough_lines(im):
    im_l = im.copy()

    # apply Canny edge detection
    im_canny = cv2.Canny(im, 200, 400)

    # find lines in image
    lines = cv2.HoughLines(im_canny, rho=1, theta=np.pi / 180, threshold=150)

    # add lines to image
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
            cv2.line(im_l, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    return im_l
