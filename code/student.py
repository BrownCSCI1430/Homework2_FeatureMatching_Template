import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature
from skimage.measure import regionprops

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

    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.

    If you're finding spurious (false/fake) feature point detections near the boundaries,
    it is safe to suppress the gradients / corners near the edges of the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on EdStem with any questions

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

    # STEP 1: Calculate the gradient (partial derivatives on two directions).
    # STEP 2: Apply Gaussian filter with appropriate sigma.
    # STEP 3: Calculate Harris cornerness score for all pixels.
    # STEP 4: Peak local max to eliminate clusters. (Try different parameters.)

    # TODO: Your implementation here!
    # These are placeholders - replace with the coordinates of your feature points!
    xs = np.random.randint(0, image.shape[1], size=100)
    ys = np.random.randint(0, image.shape[0], size=100)

    return xs, ys


def get_feature_descriptors(image, xs, ys, window_width, mode):
    '''
    Computes features for a given set of feature points.

    To start with, use image patches as your local feature descriptor. You will 
    then need to implement the more effective SIFT-like feature descriptor. Use 
    the `mode` argument to toggle between the two.
    (Original SIFT publications at http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) feature descriptor should have:
    (1) A 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) Each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4 x 4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    This is a design task, so many options might help but are not essential.
    - To perform interpolation such that each gradient
    measurement contributes to multiple orientation bins in multiple cells
    A single gradient measurement creates a weighted contribution to the 4 
    nearest cells and the 2 nearest orientation bins within each cell, for 
    8 total contributions.

    - To compute the gradient orientation at each pixel, we could use oriented 
    kernels (e.g. a kernel that responds to edges with a specific orientation). 
    All of your SIFT-like features could be constructed quickly in this way.

    - You could normalize -> threshold -> normalize again as detailed in the 
    SIFT paper. This might help for specular or outlier brightnesses.

    - You could raise each element of the final feature vector to some power 
    that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on EdStem with any questions

        - skimage.filters (library)

    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :xs: np.array of x coordinates (column indices) of feature points
    :ys: np.array of y coordinates (row indices) of feature points
    :window_width: in pixels, is the local window width. You can assume
                    that window_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like window will have an integer width and height).
    :mode: a string, either "patch" or "sift". Switches between image patch descriptors
           and SIFT descriptors

    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments. Make sure input arguments 
    are optional or the autograder will break.

    :returns:
    :features: np.array of computed features. features[i] is the descriptor for 
               point (x[i], y[i]), so the shape of features should be 
               (len(x), feature dimensionality). For standard SIFT, `feature
               dimensionality` is typically 128. `num points` may be less than len(x) if
               some points are rejected, e.g., if out of bounds.
    '''

    # IMAGE PATCH STEPS
    # STEP 1: For each feature point, cut out a window_width x window_width patch 
    #         of the image around that point (as you will in SIFT)
    # STEP 2: Flatten this image patch into a 1-dimensional vector (hint: np.flatten())
    
    # SIFT STEPS
    # STEP 1: Calculate the gradient (partial derivatives on two directions) on all pixels.
    # STEP 2: Decompose the gradient vectors to magnitude and orientation (angle).
    # STEP 3: For each feature point, calculate the local histogram based on related 4x4 grid cells.
    #         Each cell is a square with feature_width / 4 pixels length of side.
    #         For each cell, we assign these gradient vectors corresponding to these pixels to 8 bins
    #         based on the orientation (angle) of the gradient vectors. 
    # STEP 4: Now for each cell, we have a 8-dimensional vector. Appending the vectors in the 4x4 cells,
    #         we have a 128-dimensional feature.
    # STEP 5: Don't forget to normalize your feature.

    # TODO: Your implementation here!
    # These are placeholders - replace with the coordinates of your feature points!
    features = np.random.randint(0, 255, size=(len(xs), np.random.randint(1, 200)))

    return features


def match_features(im1_features, im2_features):
    '''
    Matches feature descriptors of one image with their nearest neighbor in the other.

    Implements the Nearest Neighbor Distance Ratio (NNDR) Test to help threshold
    and remove false matches.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test".

    For extra credit you can implement spatial verification of matches.

    Remember that the NNDR will return a number close to 1 for feature 
    points with similar distances. Think about how you might want to threshold
    this ratio (hint: see lecture slides for NNDR)

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on EdStem with any questions

        - np.argsort()

    :params:
    :im1_features: an np.array of features returned from get_feature_descriptors() for feature points in image1
    :im2_features: an np.array of features returned from get_feature_descriptors() for feature points in image2

    :returns:
    :matches: an np.array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    '''
    
    # STEP 1: Calculate the distances between each pairs of features between im1_features and im2_features.
    # STEP 2: Sort and find closest features for each feature
    # STEP 3: Compute NNDR for each match
    # STEP 4: Remove matches whose ratios do not meet a certain threshold 

    # TODO: Your implementation here!
    # These are placeholders - replace with the coordinates of your feature points!
    matches = np.random.randint(0, min(len(im1_features), len(im2_features)), size=(50, 2))

    return matches
