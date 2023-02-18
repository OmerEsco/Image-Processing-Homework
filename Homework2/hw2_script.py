import numpy as np

from hw2_123456789 import *

if __name__ == '__main__':
    wormhole = cv2.imread(r'blank_wormhole.jpg')
    im = cv2.cvtColor(wormhole, cv2.COLOR_BGR2GRAY)

    T = find_transform(src_points, dst_points)
    T_check = np.load("T.npy")
    print("Diff T:")
    print(T - T_check)

    new_image = create_wormhole(im, T, iter=5)

    new_image_check1 = np.load("new_image1.npy")
    print((new_image == new_image_check1).all())

    new_image_check2 = np.load("new_image2.npy")
    print((new_image == new_image_check2).all())

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im, cmap='gray')
    plt.title('original image')
    plt.subplot(1, 2, 2)
    plt.imshow(new_image, cmap='gray')
    plt.title('wormhole image')

    plt.show()
