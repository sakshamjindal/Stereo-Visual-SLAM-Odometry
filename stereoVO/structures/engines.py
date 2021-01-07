# import cv2

# from .components import StateBolts
# from  ..geometry.feature_detectors import getMatchingKeypoints, showMatchedFeatures


# class DetectionEngine():
    
#     def __init__(self, frames, params):
        
#         self.frames = frames
#         self.params = params

#         self.matchedPoints = StateBolts()
#         self.keypoints = StateBolts()
#         self.descriptors = StateBolts()
#         self.inliers = StateBolts()
                
#     def matching_keypoints(self):
#         if self.matchedPoints is None:
#             self.process_matching_keypoints()
#         return self.matchedPoints, self.keypoints, self.descriptors
    
#     def process_matching_keypoints(self):
#         self.matchedPoints, self.keypoints, self.descriptors = getMatchingKeypoints(self.frames.left, self.frames.right, self.params)
    
#     def show_matching_keypoints(self):
#         showMatchedFeatures(self.left_frame, self.right_frame)
        
#     def filter_matching_inliers(self):
        
#         for i in range(self.params.GeoComp.EM.num_trials):
#             E, mask = cv2.findEssentialMatrix(self.matchedPoints.left, 
#                                               self.matchedPoints.right, self.params)
#             mask = mask.astype(bool)
#             ratio = sum(mask)/len(mask.flatten())

#             if ratio > self.params.EM.inilierRatio:
#                 print("Iterations : 5 point Algorithm : {}".format(i+1))
#                 print("Inlier Ratio :                   {}".format(ratio))
#                 break
#             else:
#                 print("Failed to Calculate E, Iter :    {}".format(i))
#                 if i==self.params.GeoComp.EM.num_trials-1:
#                     print("Maxinum interation in 5-point Algorithm reached")
#                 else:
#                     print("Running Iterations Again. Iters left : {}".format(self.params.geocom.EM.num_trials-1))
                    
#         self.inliers.left = self.matchedPoints.left(mask)
#         self.inliers.right = self.matchedPoints.right(mask)
        
#         return self.inliers.left, self.inliers.right