import cv2
import numpy as np


def filter_matching_inliers(leftMatchesPoints, rightMatchedPoints, intrinsic, params):

    args_epipolar = params.geometry.epipolarGeometry

    for i in range(args_epipolar.numTrials):
        _, mask_epipolar = cv2.findEssentialMat(leftMatchesPoints,
                                                rightMatchedPoints,
                                                intrinsic,
                                                method = args_epipolar.method,
                                                prob = args_epipolar.probability,
                                                threshold = args_epipolar.threshold)
        
        mask_epipolar = mask_epipolar.ravel().astype(bool)
        ratio = sum(mask_epipolar) / len(mask_epipolar)
        
        if (ratio > args_epipolar.inlierRatio):
            print("Iterations of 5-point algorithm: {}".format(i+1))
            print("Inlier Ratio :                   {}".format(ratio))
            print("Good Essential Matrix calculated is good.")
            break
        else:
            print("Bad Essential Matrix likely")
            print("Inlier Ratio          : {}".format(ratio))
            print("Run again. Iters Left : {}".format(args_epipolar.numTrials-i))
            if i==args_epipolar.numTrials:
                print("Fraction of inliers for E: {}".format(ratio))
                print("Max iteration in 5-point algorithm trial reaches, bad E is likely ")


