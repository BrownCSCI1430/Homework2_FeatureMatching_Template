import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature
from skimage.measure import regionprops
from helpers import compute_dino_feature_map, sample_dino_descriptors

def plot_feature_points(image, xs, ys):
    '''
    Plot feature points for the input image. 
    
    Show the feature points (x, y) over the image. Be sure to add the plots you make to your writeup!

    Useful functions: Some helpful (but not necessarily required) functions may include:
        - plt.imshow
        - plt.scatter
        - plt.show
        - plt.savefig
    
    :params:
    :image: a grayscale or color image (depending on your implementation)
    :xs: np.array of x coordinates of feature points
    :ys: np.array of y coordinates of feature points
    '''

    # TODO: Your implementation here!

def get_feature_points(image, window_width):
    '''
    Implement the Harris corner detector to return feature points for a given image.

    For our toy implementation, it is fine to ignore scale invariance 
    and keypoint orientation estimation for your Harris corner detector.
    
    Approach
    1. Calculate the gradient (partial derivatives on two directions).
    2. Apply Gaussian filter with appropriate sigma.
    3. Calculate Harris cornerness score for all pixels.
    4. Peak local max to eliminate clusters. (Try different parameters.)

    If you're finding spurious (false/fake) feature point detections near the boundaries,
    it is safe to suppress the gradients / corners near the edges of the image.

    Useful functions: 
        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops
          
    Note: You may decide it is unnecessary to use window_width in get_feature_points, or you may also decide to 
    use this parameter to exclude the points near image edges.

    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :window_width: the width and height of each local window in pixels

    :returns:
    :xs: an np.array of the x coordinates (column indices) of the feature points in the image
    :ys: an np.array of the y coordinates (row indices) of the feature points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np.array indicating the confidence (strength) of each feature point
    :scale: an np.array indicating the scale of each feature point
    :orientation: an np.array indicating the orientation of each feature point

    '''

    # TODO: Your implementation here!
    # These are placeholders - replace with the coordinates of your feature points!
    rng = np.random.default_rng()
    xs = rng.integers(0, image.shape[1], size=100)
    ys = rng.integers(0, image.shape[0], size=100)

    return xs, ys


def get_feature_descriptors(image, xs, ys, window_width, mode, image_file=None):
    '''
    Computes a feature descriptor for each feature point.

    Implement two modes (use the `mode` argument to toggle):
      "patch" — simple image patch descriptor
      "sift"  — SIFT-like gradient histogram descriptor

    Compare to a third mode using state of the art features
    No implementation necessary:
      "dinov3" - self-supervised deep learned generic features

    IMAGE PATCH:
      1. Cut out a window_width x window_width patch around each point.
      2. Flatten to a 1-d vector and normalize to unit length.

    SIFT (see Lowe, http://www.cs.ubc.ca/~lowe/keypoints/):
      1. Compute image gradients (magnitude and orientation).
      2. For each point, divide the window into a 4x4 grid of cells
         (each cell is window_width/4 pixels).
      3. In each cell, bin gradient magnitudes into 8 orientation bins.
      4. Concatenate all histograms → 4x4x8 = 128-d vector.
      5. Normalize to unit length.

    Optional enhancements for better performance:
      - Interpolate contributions across neighboring cells and bins.
      - Normalize → threshold at 0.2 → re-normalize (reduces lighting effects).
      - Raise elements to a power < 1 (e.g., sqrt) for robustness.

    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :xs: np.array of x coordinates (column indices) of feature points
    :ys: np.array of y coordinates (row indices) of feature points
    :window_width: in pixels, is the local window width (always a multiple of 4).
    :mode: "patch", "sift", or "dinov3"
    :image_file: (optional) path to the image file, used for DINOv3 cache lookup

    :returns:
    :features: np.array of shape (len(xs), feature_dim). For SIFT, feature_dim = 128.
    '''

    if mode == "patch":
        # TODO: Your implementation here!
        # These are placeholders - replace with your feature descriptors!
        rng = np.random.default_rng()
        features = rng.integers(0, 255, size=(len(xs), 128))

    elif mode == "sift":
        # TODO: Your implementation here!
        # These are placeholders - replace with your feature descriptors!
        rng = np.random.default_rng()
        features = rng.integers(0, 255, size=(len(xs), 128))

    elif mode == "dinov3":
        # DINOv3 is handled here — you don't need to implement it.
        cache_path = os.path.splitext(image_file)[0] + "_dinov3.npz" if image_file else None
        fmap, meta = compute_dino_feature_map(image, cache_path=cache_path)
        features = sample_dino_descriptors(fmap, meta, xs, ys)


    return features


def match_features(im1_features, im2_features):
    '''
    Matches feature descriptors of one image with their nearest neighbor 
    in the other via the Nearest Neighbor Distance Ratio (NNDR) Test.

    NNDR will return a number close to 1 for feature points with 
    similar distances. Think about how you might want to threshold
    this ratio (hint: see lecture slides for NNDR).

    A match is between a feature in im1_features and a feature in im2_features.
    We can represent this match as the index of the feature in im1_features
    and the index of the feature in im2_features.

    Approach:
    1. Calculate the distances between each pair of features between im1 and im2.
    2. Sort and find closest features for each feature.
    3. Compute NNDR for each match.
    4. Remove matches whose ratios do not meet a certain threshold.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    Useful functions: 
        - np.argsort()

    :params:
    :im1_features: an np.array of features returned from get_feature_descriptors() for feature points in image1
    :im2_features: an np.array of features returned from get_feature_descriptors() for feature points in image2

    :returns:
    :matches: an np.array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    '''
    
    # TODO: Your implementation here!
    # These are placeholders - replace with your matches!
    rng = np.random.default_rng()
    matches = rng.integers(0, min(len(im1_features), len(im2_features)), size=(50, 2))

    return matches
