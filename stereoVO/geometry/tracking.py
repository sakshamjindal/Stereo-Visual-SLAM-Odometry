import numpy as np
import cv2
from ..structures import StateBolts
from . import filter_matching_inliers

lk_params = dict(winSize  = (21, 21), 
                 maxLevel = 5,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))

class TrackingEngine():
    
    
    def __init__(self, prevFrames, currFrames, prevInliers, intrinsic, params):
        
        self.prevFrames = prevFrames
        self.currFrames = currFrames
        self.prevInliers = prevInliers
        self.intrinsic = intrinsic
        self.params = params
    
    def get_tracked_features(self):
        
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
        
        return self.prevInliers, self.pointsTracked, self.maskTracking
    
    def filter_inliers(self, pts3D_Filter):
        
        # Remove non-valid points from inliers filtered in prev state
        pts3D_TrackingFilter = pts3D_Filter[self.maskTracking]
        
        (_, __ ) , mask_epipolar_left = filter_matching_inliers(self.prevInliers.left, self.pointsTracked.left, self.intrinsic, self.params)
        (_, __ ) , mask_epipolar_right = filter_matching_inliers(self.prevInliers.right, self.pointsTracked.right, self.intrinsic, self.params)
        
        mask_epipolar = np.logical_and(mask_epipolar_left, mask_epipolar_right)
        
        curr_pointsTracked = (self.pointsTracked.left[mask_epipolar], self.pointsTracked.right[mask_epipolar])
        prev_inliersTrackingFilter =  (self.prevInliers.left[mask_epipolar], self.prevInliers.right[mask_epipolar])
        prev_pts3D_TrackingFilter = pts3D_TrackingFilter[mask_epipolar] 
        
        return prev_inliersTrackingFilter, curr_pointsTracked, prev_pts3D_TrackingFilter

    @staticmethod
    def track_features(imageRef, imageCur, pointsRef):

        """
        To Do : Add docstring for the function here.
        """

        assert len(pointsRef.shape)==2 and pointsRef.shape[1]==2

        pointsRef = pointsRef.reshape(-1,1,2).astype('float32')
        points_t0_t1, mask_t0_t1, _ = cv2.calcOpticalFlowPyrLK(imageRef, imageCur, pointsRef, None, **lk_params)

        pointsRef = pointsRef.reshape(-1,2)
        points_t0_t1 = points_t0_t1.reshape(-1,2)
        mask_t0_t1 = mask_t0_t1.flatten().astype(bool)

        return pointsRef, points_t0_t1, mask_t0_t1