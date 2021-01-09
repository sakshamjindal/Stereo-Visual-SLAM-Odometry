import cv2
import numpy as np
from ..geometry import project_points


def get_minimization(dof, points3D_world, keypointsLeft, keypointsRight, PL, PR):    
    
    """
    
    
    """
    # number of points
    N = len(points3D_world)
    
    # reshaping of matrices
    points3D_world = points3D_world.T #shape:(3xN)    
    points3D_world = np.vstack((points3D_world, np.ones(N).reshape(1,-1))) #shape:(4xN)
    
    #unwrap the rotation vector and translation vector
    #assume they are in the camera coordinate system of the left camera
    r_vec = np.array([dof[0], dof[1], dof[2]]).reshape(-1, 1)
    r_mat, _ = cv2.Rodrigues(r_vec)

    t_vec = np.array([dof[3], dof[4], dof[5]]).reshape(-1,1)
    
    #first bring the 3D coordinated from the world coordinates to 
    #the frame to the camera coordinates of the left camera
    T_mat = np.hstack((r_mat, t_vec)) #shape:(3x4)
    pts3D_left = T_mat @ points3D_world #shape:(3xN)
    
    pts2D_projection_left = project_points(pts3D_left.T, PL)
    pts2D_projection_right = project_points(pts3D_left.T, PR)
        
    error_left = (keypointsLeft - pts2D_projection_left)**2
    error_right = (keypointsRight - pts2D_projection_right)**2
    residual = np.vstack((error_left, error_right))
    
    return residual.flatten()