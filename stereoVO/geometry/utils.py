import cv2
import numpy as np
import matplotlib.pyplot as plt


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

def drawMatches(img1, points1, img2, points2):
    """
    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    : params img1,img2 : Grayscale images
    : params kp1,kp2 : Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for p,q in zip(points1, points2):
        # x - columns
        # y - rows
        (x1,y1) = p
        (x2,y2) = q

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

    plt.figure(figsize=(30,10))
    # Show the image
    plt.imshow(out)
    plt.show()

def error_calc(points_p, points_q):
    
    """
    :param points_p (numpy.array) : size (Nx2)
    :param points_q (numpy.array) : size (Nx2)
    """
    
    return np.sqrt(np.sum((points_p-points_q)**2, axis=1))