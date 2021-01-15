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
    Base class for all driver code for running the engines        
    """

    # To Do : Initialise __init__ module and global variables to be used in the subclass

    def _do_optimization(self, r_mat, t_vec):

        """
        To Do : Add docstring here        
        """

        # Convert the matrix from world coordinates(prevState) to camera coordinates (currState)
        t_vec = -r_mat.T @ t_vec
        r_mat = r_mat.T
        r_vec, _ = cv2.Rodrigues(r_mat)
        
        # Prepare an initial set of parameters to the optimizer
        doF = np.concatenate((r_vec, t_vec)).flatten()
    
        # Prepare the solver for minimization
        optRes = least_squares(get_minimization, doF, method='lm', max_nfev=2000,
                               args=(self.prevState.P3P_pts3D, 
                                     self.currState.pointsTracked.left,
                                     self.currState.pointsTracked.right, 
                                     self.PL,self.PR))
        
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
        To Do : Add docstring here        
        """

        args_pnpSolver = self.params.geometry.pnpSolver

        for i in range(args_pnpSolver.numTrials):
            
            _, r_vec, t_vec, idxPose = cv2.solvePnPRansac(self.prevState.pts3D_Tracking,
                                                          self.currState.pointsTracked.left,
                                                          self.intrinsic,
                                                          None,
                                                          iterationsCount=args_pnpSolver.numTrials,
                                                          reprojectionError=args_pnpSolver.reprojectionError,
                                                          confidence=args_pnpSolver.confidence,
                                                          flags=cv2.SOLVEPNP_P3P)
            
            
            r_mat, _ = cv2.Rodrigues(r_vec)
            
            # r_vec and t_vec obtained are in camera coordinate frames
            # we need to convert these matrices in world coordinates system
            # or we need to transforms the matrix from currState to prevState
            t_vec = -r_mat.T @ t_vec
            r_mat = r_mat.T
            
            idxPose = idxPose.flatten()
            
            # ratio = len(idxPose)/len(self.prevState.pts3D_Tracking)
            # scale = np.linalg.norm(t_vec)
            
            # if scale < args_pnpSolver.deltaT and ratio > args_pnpSolver.minRatio:
            #     print("Scale of translation of camera     : {}".format(scale))
            #     print("Solution obtained in P3P Iteration : {}".format(i+1))
            #     print("Ratio of Inliers                   : {}".format(ratio))
            #     break
            # else:
            #     print("Warning : Max Iter : {} reached, still large position delta produced".format(i))

        self.currState.pointsTracked = (self.currState.pointsTracked.left[idxPose], self.currState.pointsTracked.right[idxPose])
        self.prevState.P3P_pts3D = self.prevState.pts3D_Tracking[idxPose]

        return r_mat, t_vec

    def _process_feature_tracking(self):

        """
        To Do : Add docstring here        
        """
        
        prevFrames = self.prevState.frames
        currFrames = self.currState.frames
        prevInliers = self.prevState.InliersFilter

        # Feature Tracking from prev state to current state
        tracker = TrackingEngine(prevFrames, currFrames, prevInliers, self.intrinsic, self.params)
        tracker.process_tracked_features()
        self.prevState.inliersTracking, self.currState.pointsTracked, self.prevState.pts3D_Tracking = tracker.filter_inliers(self.prevState.pts3D_Filter)   

    def _update_stereo_state(self, stereoState):

        """
        To Do : Add docstring here        
        """

        # Detection Engine, Matching and Triangulation for first frame
        detection_engine = DetectionEngine(stereoState.frames.left, stereoState.frames.right, self.params)

        stereoState.matchedPoints, stereoState.keyPoints, stereoState.descriptors = detection_engine.get_matching_keypoints()

        stereoState.inliers, _ = filter_matching_inliers(stereoState.matchedPoints.left,
                                                         stereoState.matchedPoints.right,
                                                         self.intrinsic,
                                                         self.params)

        stereoState.pts3D, reproj_error = triangulate_points(stereoState.inliers.left,
                                                            stereoState.inliers.right,
                                                            self.PL,
                                                            self.PR)

        args_triangulation = self.params.geometry.triangulation
        stereoState.pts3D_Filter, maskTriangulationFilter, ratioFilter = filter_triangulated_points(stereoState.pts3D, reproj_error, **args_triangulation)

        stereoState.InliersFilter = stereoState.inliers.left[maskTriangulationFilter], stereoState.inliers.right[maskTriangulationFilter]
        stereoState.ratioTriangulationFilter = ratioFilter