# Feature Matching
#
# Written by James Tompkin / James Hays for CSCI 1430
# - (2019) Adapted for python by asabel and jdemari1
# - (2025) Kelvin Jiang - Added DINO features

import sys
import argparse
import numpy as np
import scipy.io as scio

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from skimage import io, filters, feature, img_as_float32
from skimage.transform import rescale
from skimage.color import rgb2gray

import student as student
import visualize
from helpers import cheat_feature_points, evaluate_correspondence



# This script
# (1) Loads and resizes images
# (2) Finds feature points in those images                 (you code this in student.py)
# (3) Describes each feature point with a local feature    (you code this in student.py)
# (4) Finds matching features                              (you code this in student.py)
# (5) Visualizes the matches
# (6) Evaluates the matches based on ground truth correspondences

def load_data(file_name):
    """
     The evaluation function in this script will work for three particular image pairs
     (unless you add ground truth annotations for other image pairs). 

     If you want to add new images to test, replace the two images in the 
     `custom` folder with your own image pairs. Make sure that the names match 
     the ones in the elif for the custom folder. To run with your new images 
     use python main.py -d custom.

    :param file_name: string for which image pair to compute correspondence for

        The first four strings can be used as shortcuts to the
        data files we give you

        1. notre_dame
        2. mt_rushmore
        3. e_gaudi
        4. custom

    :return: a tuple of the format (image1, image2, eval_file, image1_file, image2_file)
    """

    # Note: these files default to notre dame, unless otherwise specified
    image1_file = "../data/NotreDame/NotreDame1.jpg"
    image2_file = "../data/NotreDame/NotreDame2.jpg"
    eval_file = "../data/NotreDame/NotreDameEval.mat"

    if file_name == "notre_dame":
        pass
    elif file_name == "mt_rushmore":
        image1_file = "../data/MountRushmore/Mount_Rushmore1.jpg"
        image2_file = "../data/MountRushmore/Mount_Rushmore2.jpg"
        eval_file = "../data/MountRushmore/MountRushmoreEval.mat"
    elif file_name == "e_gaudi":
        image1_file = "../data/EpiscopalGaudi/EGaudi_1.jpg"
        image2_file = "../data/EpiscopalGaudi/EGaudi_2.jpg"
        eval_file = "../data/EpiscopalGaudi/EGaudiEval.mat"
    elif file_name == "custom":
        image1_file = "../data/Custom/custom1.jpg"
        image2_file = "../data/Custom/custom2.jpg"
        eval_file = None

    image1 = img_as_float32(io.imread(image1_file))
    image2 = img_as_float32(io.imread(image2_file))

    return image1, image2, eval_file, image1_file, image2_file

def main():
    """
    Reads in the data,

    Command line usage: 
    
    python main.py -d | --data <image pair name> -p | --points <cheat or student points> [--sift]

    -d | --data - flag - required. specifies which image pair to match
    -p | --points - flag - required. specifies whether to use cheat points or student's feature points
    --mode - optional. descriptor type: patch (default), sift, or dinov3

    """

    # create the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data",
                        required=True,
                        choices=["notre_dame","mt_rushmore","e_gaudi", "custom"],
                        help="Either notre_dame, mt_rushmore, e_gaudi, or custom. Specifies which image pair to match")
    parser.add_argument("-p","--points", 
                        required=True,
                        choices=["cheat_points", "student_points"],
                        help="Either cheat_points or student_points. Returns feature points for the image. Use \
                              cheat_points until get_feature_points() is implemented in student.py")
    parser.add_argument("--mode",
                        required=False,
                        default="patch",
                        choices=["patch", "sift", "dinov3"],
                        help="Feature descriptor type: patch (default), sift, or dinov3.")
    args = parser.parse_args()

    # (1) Load in the data
    image1_color, image2_color, eval_file, image1_file, image2_file = load_data(args.data)

    image1 = rgb2gray(image1_color)
    image2 = rgb2gray(image2_color)

    # Make images smaller to speed up the algorithm. This parameter
    # gets passed into the evaluation code, so don't resize the images
    # except for changing this parameter - we will evaluate your code using
    # scale_factor = 0.5, so be aware of this.
    scale_factor = 0.5
    image1 = np.float32(rescale(image1, scale_factor))
    image2 = np.float32(rescale(image2, scale_factor))

    # width and height of each local feature, in pixels
    feature_width = 16

    # (2) Find distinctive points in each image.
    # !!! You will need to implement get_feature_points. !!!

    print("Getting feature points...")

    if args.points == "student_points":
        (x1, y1) = student.get_feature_points(image1,feature_width)
        (x2, y2) = student.get_feature_points(image2,feature_width)

    # To develop and debug get_feature_descriptors and match_features, you can
    # use the TA ground truth points: pass "-p cheat_points" to main.py. 
    # Note that the ground truth points for Mt. Rushmore will produce bad results, 
    # so test on Notre Dame.
    elif args.points == "cheat_points":
        (x1, y1, x2, y2) = cheat_feature_points(eval_file, scale_factor)

    # View your corners - uncomment these next lines!
    #
    # plt.imshow(image1, cmap="gray")
    # plt.scatter(x1, y1, alpha=0.9, s=3)
    # plt.show()

    # plt.imshow(image2, cmap="gray")
    # plt.scatter(x2, y2, alpha=0.9, s=3)
    # plt.show()
    
    # Viewing your feature points on your images.
    # !!! You will need to implement plot_feature_points. !!!
    print("Number of feature points found (image 1):", len(x1))
    student.plot_feature_points(image1, x1, y1)
    print("Number of feature points found (image 2):", len(x2))
    student.plot_feature_points(image2, x2, y2)
    
    print("Done!")

    # 3) Create feature vectors at each feature point.
    # !!! You will need to implement get_feature_descriptors. !!!

    print("Getting features...")
    image1_features = student.get_feature_descriptors(image1, x1, y1, feature_width, args.mode, image_file=image1_file)
    image2_features = student.get_feature_descriptors(image2, x2, y2, feature_width, args.mode, image_file=image2_file)

    print("Done!")

    # 4) Match features.
    # !!! You will need to implement match_features !!!

    print("Matching features...")

    matches = student.match_features(image1_features, image2_features)

    print("Done!")

    # 5) Evaluation and visualization

    # Check how your code performs on the image pairs!
    # The evaluate_correspondence function below will print out
    # the accuracy of your feature matching for your 50 most confident matches,
    # 100 most confident matches, and all your matches. It will then visualize
    # the matches by drawing green lines between points for correct matches and
    # red lines for incorrect matches. The visualizer will show the top
    # num_pts_to_visualize most confident matches, so feel free to change the
    # parameter to whatever you like.

    print("Matches: " + str(matches.shape[0]))
    
    filename = f'{args.data}_matches_{args.mode}.png'

    if args.data == "custom":
        print("Visualizing on custom images...")
        visualize.show_correspondences_custom_image(image1_color, image2_color, x1, y1, x2,
            y2, matches, scale_factor, filename)
    else:
        evaluate_correspondence(image1_color, image2_color, eval_file, scale_factor,
            x1, y1, x2, y2, matches, filename)

    return

if __name__ == '__main__':
    main()
