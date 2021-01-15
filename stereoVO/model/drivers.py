"""
Author : Saksham Jindal
Date : January 15, 2020
"""

import cv2
import numpy as np
from scipy.optimize import least_squares

from stereoVO.optimization import get_minimization
from stereoVO.geometry import (DetectionEngine,
                               TrackingEngine,
                               filter_matching_inliers,
                               triangulate_points,
                               filter_triangulated_points)


class StereoDrivers():

    """
    Base class for all driver code for running the engines of feature detection, 
    tracking features, olving PnP projection, calculating reprojection error and 
    non-linear least square optimisation of relative rotation and orientation
    """

    def _do_optimization(self, r_mat, t_vec):

        """
        Driver code for optimisation and non-linear least square optimisation of estimated pose 
        (relative rotation and relative translation)

        :param r_mat (np.array) : size(3,3) : relative rotation in coordinate frame of previous stereo state
        :param t_vec (np.array) : size (3,1) : relative translation in coordinate frame of previous stereo state

        Returns:
            r_mat (np.array) : size(3,3) : relative rotation in coordinate frame of previous stereo state
            t_vec (np.array) : size(3,1) : relative translation in cooridinate frame of previous stereo state
        """

        # Convert the matrix from world coordinates(prevState) to camera coordinates (currState)
        t_vec = -r_mat.T @ t_vec
        r_mat = r_mat.T
        r_vec, _ = cv2.Rodrigues(r_mat)
        
        # Prepare an initial set of parameters to the optimizer
        doF = np.concatenate((r_vec, t_vec)).flatten()
    
        # Prepare the solver for minimization and run Non-linear Least Squares optimization to optimize R and T from P3P solver
        optRes = least_squares(get_minimization, doF, 
                               method='lm', max_nfev=2000,
                               args=(self.prevState.P3P_pts3D,
                                     self.currState.pointsTracked.left,
                                     self.currState.pointsTracked.right, 
                                     self.PL, self.PR))
        
        # r_vec and t_vec obtained are in camera coordinate frames (currState)
        # we need to convert these matrix to world coordinates system (prevState)
        opt_rvec_cam = (optRes.x[:3]).reshape(-1,1)
        opt_tvec_cam = (optRes.x[3:]).reshape(-1,1)
        opt_rmat_cam, _ = cv2.Rodrigues(opt_rvec_cam)
        
        # Obtain the pose of the camera (wrt state of the previous camera)
        r_mat = opt_rmat_cam.T
        t_vec = -opt_rmat_cam.T @ opt_tvec_cam

        return r_mat, t_vec

    def _solve_pnp(self):

        """
        Driver Code for solving Perspective-n-Point (PnP) problem. The module uses OpenCV's implementation of P3P algorithm
        and RANSAC to retrieve the relative rotation  and translation vector (used Rodrigues formula to get rotation matrix) between the world (previous state) 
        and camera (current state) in the camera coordinate frame.  Module further processes to obtain the relative rotation and 
        translation in the world coordinate frame (camera coordinate frame of previous state). Also,  

        Return:
            :r_mat (np.array) : size(3,3) : relative rotation in world coordinate frame (previous stereo state)
            :t_vec (np.array) : size(3,1) : relative translation in world coordinate frame (previous stereo state)      
        """

        # Prepare argument for PnP solver
        args_pnpSolver = self.params.geometry.pnpSolver

        for i in range(args_pnpSolver.numTrials):

            # Obtain r_vec and t_vec in camera coordinate frames (currState)
            _, r_vec, t_vec, idxPose = cv2.solvePnPRansac(self.prevState.pts3D_Tracking,
                                                          self.currState.pointsTracked.left,
                                                          self.intrinsic,
                                                          None,
                                                          iterationsCount=args_pnpSolver.numTrials,
                                                          reprojectionError=args_pnpSolver.reprojectionError,
                                                          confidence=args_pnpSolver.confidence,
                                                          flags=cv2.SOLVEPNP_P3P)

            # Use Rodrigues formaula (SO3) to obtain rotational matrix
            r_mat, _ = cv2.Rodrigues(r_vec)

            # Convert relative roation and translation in the world coordiante frame (prevState)
            t_vec = -r_mat.T @ t_vec
            r_mat = r_mat.T

            # Prepare index to retrieve inliers on the current traced points and 3D points
            try:
                idxPose = idxPose.flatten()
            except:
                import ipdb; ipdb.set_trace()

            # Ensure we get enough inliers from the PnP problem
            '''To Do: Add logger object to record the terminal output'''
            ratio = len(idxPose)/len(self.prevState.pts3D_Tracking)
            scale = np.linalg.norm(t_vec)
           
            if scale < args_pnpSolver.deltaT and ratio > args_pnpSolver.minRatio:
                # print("Scale of translation of camera     : {}".format(scale))
                # print("Solution obtained in P3P Iteration : {}".format(i+1))
                # print("Ratio of Inliers                   : {}".format(ratio))
                break
            else:
                pass
                # print("Warning : Max Iter : {} reached, still large position delta produced".format(i))

        # Remove outliers from tracked points and triangulated and filtered 3D world coordinate points
        self.currState.pointsTracked = (self.currState.pointsTracked.left[idxPose], self.currState.pointsTracked.right[idxPose])
        self.prevState.P3P_pts3D = self.prevState.pts3D_Tracking[idxPose]

        return r_mat, t_vec

    def _process_feature_tracking(self):

        """
        Driver code for trackinng features across consecutive stero states and filtering inliers. 
        Tracks features from previous state to current state on both the stereo frames. 
        Tracked features calculted here are used for solving Perspective-n-Point Problem.
        """
        
        # Prepare inputs for tracking engine
        prevFrames = self.prevState.frames
        currFrames = self.currState.frames
        prevInliers = self.prevState.InliersFilter

        # Initialise trackig engine to track features using optical flow
        tracker = TrackingEngine(prevFrames, currFrames, prevInliers, self.intrinsic, self.params)
        
        # Actual executor module which tracks the features on the current set of 
        # frames and filters inliers from the detection engine and tracking engine
        tracker.process_tracked_features()
        
        # executor code to apply epipolar contraint on the prevstate and current state frames
        # and furtker filter inliers from detection engine, trackign engine and 3D points
        self.prevState.inliersTracking, self.currState.pointsTracked, self.prevState.pts3D_Tracking = tracker.filter_inliers(self.prevState.pts3D_Filter)   

    def _update_stereo_state(self, stereoState):

        """
        Driver code for updating stereo state for frames in the current state
        Calls upon Detection Engine to detect features in the both the frames, 
        matching detected features, filtering matched inliers, triangulating to 
        calculating 3D point cloud using Direct Linear Transform (DLT) a

        :param StereoState (VO_StateMachine) : state object for keeping 
                            track of computed history of state variables
        """

        # Feature Detection and Matching
        detection_engine = DetectionEngine(stereoState.frames.left, stereoState.frames.right, self.params)
        stereoState.matchedPoints, stereoState.keyPoints, stereoState.descriptors = detection_engine.get_matching_keypoints()

        # Filtering inliers using epipolar constraint
        stereoState.inliers, _ = filter_matching_inliers(stereoState.matchedPoints.left,
                                                         stereoState.matchedPoints.right,
                                                         self.intrinsic,
                                                         self.params)

        # Triangulation to obtain 3D points
        args_triangulation = self.params.geometry.triangulation
        stereoState.pts3D, reproj_error = triangulate_points(stereoState.inliers.left,
                                                             stereoState.inliers.right,
                                                             self.PL,
                                                             self.PR)

        # Filtering inliers and 3D points using thresholds and constraints
        stereoState.pts3D_Filter, maskTriangulationFilter, ratioFilter = filter_triangulated_points(stereoState.pts3D, reproj_error, **args_triangulation)
        stereoState.InliersFilter = stereoState.inliers.left[maskTriangulationFilter], stereoState.inliers.right[maskTriangulationFilter]
        stereoState.ratioTriangulationFilter = ratioFilter
