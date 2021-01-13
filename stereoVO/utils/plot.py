import cv2
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def draw_trajectory_3D(r_mat, t_vec, ax1, ax2):

    X = round(t_vec[0], 2)
    Y = round(t_vec[1], 2)
    Z = round(t_vec[2], 2)
    ax1.title.set_text('X = {}, Y = {}, Z = {}'.format(X, Y, Z))
    ax2.title.set_text('X = {}, Y = {}, Z = {}'.format(X, Y, Z))
    
    axes = np.zeros((3,6))
    axes[0,1], axes[1,3],axes[2,5] = 2,2,2
    t_vec = t_vec.reshape(-1,1)
    axes= r_mat @ (axes) + np.tile(t_vec,(1,6))

    ax1.plot3D(xs=axes[0,:2],ys=axes[1,:2],zs=axes[2,:2],c='r')
    ax1.plot3D(xs=axes[0,2:4],ys=axes[1,2:4],zs=axes[2,2:4],c='g')
    ax1.plot3D(xs=axes[0,4:],ys=axes[1,4:],zs=axes[2,4:],c='b')

    scale=50
    depth=100

    #generating 5 corners of camera polygon 
    pt1 = np.array([[0,0,0]]).T #camera centre
    pt2 = np.array([[scale,-scale,depth]]).T #upper right 
    pt3 = np.array([[scale,scale,depth]]).T #lower right 
    pt4 = np.array([[-scale,-scale,depth]]).T #upper left
    pt5 = np.array([[-scale,scale,depth]]).T #lower left
    pts = np.concatenate((pt1,pt2,pt3,pt4,pt5),axis=-1) 

    #Transforming to world-coordinate system
    pts = r_mat @ (pts) + np.tile(t_vec,(1,5))
    ax1.scatter3D(xs=pts[0,:],ys=pts[1,:],zs=pts[2,:],c='k')
    ax2.scatter3D(xs=pts[0,0],ys=pts[1,0],zs=pts[2,0],c='r', s=10)

    #Generating a list of vertices to be connected in polygon
    verts = [[pts[:,0],pts[:,1],pts[:,2]], [pts[:,0],pts[:,2],pts[:,-1]],
            [pts[:,0],pts[:,-1],pts[:,-2]],[pts[:,0],pts[:,-2],pts[:,1]]]
    
    #Generating a polygon now..
    ax1.add_collection3d(Poly3DCollection(verts, facecolors='grey',
                                        linewidths=1, edgecolors='k', alpha=.25))

    
    plt.pause(0.01)


def draw_trajectory_2D(traj, frame_id, x, y, z, draw_x, draw_y, true_x, true_y):
    
    cv2.circle(traj, (draw_x,draw_y), 1, (frame_id*255/4540,255-frame_id*255/4540,0), 1)
    cv2.circle(traj, (true_x,true_y), 1, (0,0,255), 2)
    cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
    text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
    cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
    cv2.imshow('Trajectory', traj)


def draw_rmse_error(index, rmse, ax):

    ax.scatter(index, rmse, s=4, c='b')
    plt.pause(0.01)