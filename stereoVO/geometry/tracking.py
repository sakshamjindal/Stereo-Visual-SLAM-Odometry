import numpy as np
import cv2
from ..structures import StateBolts
from . import filter_matching_inliers

'''
To Do: Make the tracking engine modular to accept the following 
       set of parameters for optical flow as inputs
'''
lk_params = dict(winSize  = (21, 21), 
                 maxLevel = 5,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))


class TrackingEngine():

    """
    Main Engine code for tracking features across two time instances
    (called prevState and currState here)
    """

    def __init__(self, prevFrames, currFrames, prevInliers, intrinsic, params):

        """
        :param prevFrames  (StateBolts::(np.array, np.array):  
                           (size(H,W), size(H,w)) frames of previous stero state wrapped in StateBolts Module
        :param currFrames  (StateBolts::(np.array, np.array):  
                           (size(H,W), size(H,w)) frames of current stero state wrapped in StateBolts Module
        :param prevInliers (StateBolts::(np.array, np.array): 
                           (size(N,2), size(N,2)) detected and matched  keypoints (filtered) on left and right frame
        :param intrinsic   (np.array) : size(3,3) : camera calibration matrix
        :param params      (AttriDict): contains parameters for the stereo configuration, 
                                        detection of features, tracking and other geomtric
                                        computer vision features
        """
        self.prevFrames = prevFrames
        self.currFrames = currFrames
        self.prevInliers = prevInliers
        self.intrinsic = intrinsic
        self.params = params

    def process_tracked_features(self):

        """
        Executor module which tracks the features on the current set of frames
        and filters inliers from the detection engine and tracking engine
        """

        # Track features on left and right frame in the current state from previous states 
        _, pointsTrackedLeft, maskTrackingLeft = TrackingEngine.track_features(self.prevFrames.left,
                                                                               self.currFrames.left,
                                                                               self.prevInliers.left)

        _, pointsTrackedRight, maskTrackingRight = TrackingEngine.track_features(self.prevFrames.right,
                                                                                 self.currFrames.right,
                                                                                 self.prevInliers.right)
        
        # Joint index and select only good tracked points
        self.maskTracking = np.logical_and(maskTrackingLeft, maskTrackingRight)
        self.pointsTracked = StateBolts(pointsTrackedLeft[self.maskTracking], pointsTrackedRight[self.maskTracking])
        self.prevInliers = StateBolts(self.prevInliers.left[self.maskTracking], self.prevInliers.right[self.maskTracking])

    def filter_inliers(self, pts3D_Filter):

        """
        Executor module to apply epipolar contraint on the prevstate and current state frames
        and furtker filter inliers from detection engine, trackign engine and 3D points
        """

        # Remove non-valid points from inliers filtered in prev state
        pts3D_TrackingFilter = pts3D_Filter[self.maskTracking]
        
        # Remove Outliers using Epipolar Geometry (RANSAC)
        (_, __), mask_epipolar_left = filter_matching_inliers(self.prevInliers.left, self.pointsTracked.left, self.intrinsic, self.params)
        (_, __), mask_epipolar_right = filter_matching_inliers(self.prevInliers.right, self.pointsTracked.right, self.intrinsic, self.params)
        
        # Join index from the inliers filer mask on the left and right frame
        mask_epipolar = np.logical_and(mask_epipolar_left, mask_epipolar_right)
        
        # Remove Outliers from tracked points correspondences on both stereo states and 3D points
        curr_pointsTracked = (self.pointsTracked.left[mask_epipolar], self.pointsTracked.right[mask_epipolar])
        prev_inliersTrackingFilter =  (self.prevInliers.left[mask_epipolar], self.prevInliers.right[mask_epipolar])
        prev_pts3D_TrackingFilter = pts3D_TrackingFilter[mask_epipolar] 
        
        return prev_inliersTrackingFilter, curr_pointsTracked, prev_pts3D_TrackingFilter

    @staticmethod
    def track_features(imageRef, imageCur, pointsRef):

        """
        :param imageRef  (np.array): size(H,W) grayscale image as reference image to track feature
        :param imageCur  (np.array): size(H,W) grayscale image as current image to track features on
        :param pointsRef (np.array): size(N,2) keypoints to be used as reference for tracking 

        Returns
            pointsRef    (np.array): size(N,2) same as param pointsRef
            points_t0_t1 (np.array): size(N,2) tracked features/keypoints on current frame 
            mask_t0_t1   (np.array): size(N) indexes for "good" tracking points
        """

        # Do asserion test on the input
        assert len(pointsRef.shape) == 2 and pointsRef.shape[1] == 2

        # Reshape input and track features
        pointsRef = pointsRef.reshape(-1, 1, 2).astype('float32')
        points_t0_t1, mask_t0_t1, _ = cv2.calcOpticalFlowPyrLK(imageRef, 
                                                               imageCur,
                                                               pointsRef, 
                                                               None, 
                                                               **lk_params)

        # Reshape ouput and return output and mask for tracking points
        pointsRef = pointsRef.reshape(-1,2)
        points_t0_t1 = points_t0_t1.reshape(-1,2)
        mask_t0_t1 = mask_t0_t1.flatten().astype(bool)

        return pointsRef, points_t0_t1, mask_t0_t1
