import numpy as np
import cv2
from scipy.optimize import least_squares

from stereoVO.structures import StateBolts, VO_StateMachine

from stereoVO.optimization import get_minimization
from stereoVO.geometry import (DetectionEngine,
                               TrackingEngine,
                               filter_matching_inliers, 
                               triangulate_points, 
                               filter_triangulated_points)

import ipdb


class StereoVO():

    def __init__(self, intrinsic, PL, PR, params):

        """
        To Do : Add docstring here
        """

        self.intrinsic = intrinsic
        self.PL = PL
        self.PR = PR
        self.params = params

    def __call__(self, left_frame, right_frame, state_num):

        """
        To Do : Add docstring here        
        """

        if state_num == 0:
            # Initialise the initial stereo state
            self.prevState = VO_StateMachine(state_num)
            self.prevState.frames = left_frame, right_frame

            # Update the initial stereo state with detection and triangualation
            self._update_stereo_state(self.prevState)

            # Initialize the pose of the camera
            self.prevState.location = np.array(self.params.initial.location)
            self.prevState.orientation = np.array(self.params.initial.orientation)

            return self.prevState.location, self.prevState.orientation

        if state_num == 1:
            # Initialise the current stereo state
            self.currState = VO_StateMachine(state_num)
            self.currState.frames = left_frame, right_frame

            # Update the initial stereo state with detection and triangualation
            self._update_stereo_state(self.currState)

            # Feature Tracking from prevState to currState
            self._process_feature_tracking()

            # P3P Solver
            # obtains the pose of the camera in coordinate frame of prevState
            r_mat, t_vec = self.solve_pnp()

            # if optimisation is enabled
            # do pose updation with the optimizer
            if self.params.geometry.lsqsolver.enable:
                r_mat, t_vec = self._do_optimization(r_mat, t_vec)

            # Upating the pose of the camera of currState  
            # C_n = C_n-1 * dT_n-1; where dT_n-1 is in the 
            # reference of coordinate system of the second camera
            self.currState.orientation = self.prevState.orientation @ r_mat
            self.currState.location = self.prevState.orientation @ t_vec + self.prevState.location.reshape(-1,1)
            self.currState.location = self.currState.location.flatten()

            self.currState.keypoints = self.currState.pointsTracked
            self.currState.landmarks = self.prevState.P3P_pts3D

        else:
            self.currState = VO_StateMachine(state_num)
            self.currState.frames = left_frame, right_frame

            self._process_feature_tracking()

            r_mat, t_vec = self.solve_pnp()

            if self.params.geometry.lsqsolver.enable:
                r_mat, t_vec = self._do_optimization(r_mat, t_vec)

            self.currState.orientation = self.prevState.orientation @ r_mat
            self.currState.location = self.prevState.orientation @ t_vec + self.prevState.location.reshape(-1,1)
            self.currState.location = self.currState.location.flatten()

            self._update_stereo_state(self.currState)

        print("Frame {} Processing Done ....".format(state_num + 1))
        print("Current Location : X : {x}, Y = {y}, Z = {z}".format(x = self.currState.location[0], 
                                                                    y = self.currState.location[1], 
                                                                    z = self.currState.location[2]))

        self.prevState = self.currState

        return self.currState.location, self.currState.orientation

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
                                args=(self.prevState.P3P_pts3D, self.currState.pointsTracked.left, 
                                    self.currState.pointsTracked.right, self.PL,self.PR))
        
        # r_vec and t_vec obtained are in camera coordinate frames (currState)
        # we need to convert these matrix to world coordinates system (prevState)
        opt_rvec_cam = (optRes.x[:3]).reshape(-1,1)
        opt_tvec_cam = (optRes.x[3:]).reshape(-1,1)
        opt_rmat_cam,_ = cv2.Rodrigues(opt_rvec_cam)
        
        # Obtain the pose of the camera (wrt state of the previous camera)
        r_mat = opt_rmat_cam.T
        t_vec = -opt_rmat_cam.T @ opt_tvec_cam        

        return r_mat, t_vec

    def solve_pnp(self):

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
            
            ratio = len(idxPose)/len(self.prevState.pts3D_Tracking)
            scale = np.linalg.norm(t_vec)
            
            if scale<args_pnpSolver.deltaT and ratio>args_pnpSolver.minRatio:
                print("Scale of translation of camera     : {}".format(scale))
                print("Solution obtained in P3P Iteration : {}".format(i+1))
                print("Ratio of Inliers                   : {}".format(ratio))
                break
            else:
                print("Warning : Max Iter : {} reached, still large position delta produced".format(i))

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