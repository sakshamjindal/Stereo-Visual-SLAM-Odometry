import cv2
import numpy as np

def project_points(points3D, projectionMatrix):
    """
    :param points3D (numpy.array) : size (Nx3)
    :param projectionMatrix (numpy.array) : size(3x4) - final projection matrix (K@[R|t])
    
    Returns:
        points2D (numpy.array) : size (Nx2) - projection of 3D points on image plane
    """
    
    points3D = np.hstack((points3D,np.ones(points3D.shape[0]).reshape(-1,1))) #shape:(Nx4)
    points3D = points3D.T #shape:(4xN)
    pts2D_homogeneous = projectionMatrix @ points3D #shape:(3xN)
    pts2D = pts2D_homogeneous[:2, :]/(pts2D_homogeneous[-1,:].reshape(1,-1)) #shape:(2xN)
    pts2D = pts2D.T
                         
    return pts2D