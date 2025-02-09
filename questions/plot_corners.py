'''
This code adapted from scikit-image documentation
http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_corner.html

For CSCI1430 @ Brown University Spring 2019
'''

'''
================
Corner detection
================

Detect corner points using the Harris corner detector and determine the
subpixel position of corners ([1]_, [2]_).

.. [1] https://en.wikipedia.org/wiki/Corner_detection
.. [2] https://en.wikipedia.org/wiki/Interest_point_detection

'''

from matplotlib import pyplot as plt

import argparse

from skimage import io, color, transform
from skimage.feature import corner_harris, peak_local_max

def main():
    parser = argparse.ArgumentParser(description='Harris corner detector')
    parser.add_argument('example', choices=['RISHLibrary', 'LaddObservatory', 'Chase'],
                        help='Select the image pair for corner detection')
    args = parser.parse_args()

    image1 = transform.rescale(color.rgb2gray(io.imread(f'images/{args.example}1.jpg')),0.25)
    image2 = transform.rescale(color.rgb2gray(io.imread(f'images/{args.example}2.jpg')),0.25)

    harris_response1 = corner_harris(image1)
    harris_response2 = corner_harris(image2)

    ##############
    # TODO: Play with these parameters to investigate their effects
    min_distance = 5
    threshold_rel = 0.05
    ##############

    coords1 = peak_local_max( harris_response1, min_distance=min_distance, threshold_rel=threshold_rel )
    coords2 = peak_local_max( harris_response2, min_distance=min_distance, threshold_rel=threshold_rel )

    fig, ax = plt.subplots(1, 2, figsize=(6, 6))

    ax[0].imshow(image1, cmap=plt.cm.gray)
    ax[0].plot(coords1[:, 1], coords1[:, 0], '+r', markersize=15)
    ax[0].axis((0, image1.shape[1], image1.shape[0], 0))
    ax[0].axis('off')

    ax[1].imshow(image2, cmap=plt.cm.gray)
    ax[1].plot(coords2[:, 1], coords2[:, 0], '+r', markersize=15)
    ax[1].axis('off')

    plt.show()

if __name__ == '__main__':
    main()