import cv2
import numpy as np
from .utils import project_points


def triangulate_points(ptsLeft, ptsRight, projL, projR):

    # triangulate points
    pts4D = cv2.triangulatePoints(projL, projR, ptsLeft.T, ptsRight.T)

    # convert from homogeneous coordinates to 3D
    pts3D = pts4D[:3,:]/((pts4D[-1,:]).reshape(1,-1))
    pts3D = pts3D.T

    # project reconstructed 3D points on to the images
    proj2D_left = project_points(pts3D, projL)
    proj2D_right = project_points(pts3D, projR)

    #calculate reprojection error
    reprojError = ((np.sqrt(((proj2D_left-ptsLeft)**2).sum(axis=1))) + (np.sqrt(((proj2D_right-ptsRight)**2).sum(axis=1))))/2

    return pts3D, reprojError

def filter_triangulated_points(pts3D, **args):
    pass


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
        
    left_inliers = leftMatchesPoints[mask_epipolar]
    right_inliers = rightMatchedPoints[mask_epipolar]

    return left_inliers, right_inliers


