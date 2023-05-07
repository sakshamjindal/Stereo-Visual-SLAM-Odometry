import cv2
import numpy as np
import matplotlib.pyplot as plt 

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class SVO_Plot():

    def __init__(self, figSize):
    
        fig1 = plt.figure(figsize=figSize)
        self.ax1 = fig1.add_subplot(111, projection='3d')

        fig2 = plt.figure(figsize=figSize)
        self.ax2 = fig2.add_subplot(111, projection='3d')

        fig3 = plt.figure(figsize=figSize)
        self.ax3 = fig3.add_subplot(111)


        fig1.suptitle('Schematic Representation of Camera in 3D')
        fig2.suptitle('Trajectory of Camera in 3D')
        fig3.suptitle('Location RMSE with respect to frame number')

        self.initialize_axes()

        # Initialise an empty drawing board for trajectory
        self.blank_slate = np.zeros((600,600,3), dtype=np.uint8)

    def initialize_axes(self):

        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_zlabel('Z')
        
        self.ax1.set_xlim3d(-300, 300)
        self.ax1.set_ylim3d(-300, 300)
        self.ax1.set_zlim3d(-500, 500)

        self.ax1.view_init()

        self.ax2.set_xlabel('X')
        self.ax2.set_ylabel('Y')
        self.ax2.set_zlabel('Z')
        
        self.ax2.set_xlim3d(-300, 300)
        self.ax2.set_ylim3d(-300, 300)
        self.ax2.set_zlim3d(-300, 300)

        self.ax2.view_init()

        self.ax3.set_xlim(0,500)
        self.ax3.set_ylim(0,5)

    def plot_camera_traectories(self, index,pred_location, pred_orientation, ground_pose):
        
        x, y, z = pred_location[0], pred_location[1], pred_location[2]
        offset_x, offset_y = 1,1
        draw_x, draw_y = int(x) + 290 - offset_x ,  500 - int(z) + offset_y
        true_x, true_y = int(ground_pose[0][-1]) + 290, 500 - int(ground_pose[2][-1])

        self.draw_trajectory_2D(self.blank_slate, index, x, y, z, draw_x, draw_y, true_x, true_y)
        self.draw_trajectory_3D(pred_orientation, pred_location, self.ax1, self.ax2)

    def plot_frame(self, frame):

        cv2.imshow('Road facing camera', frame)
        cv2.waitKey(1)


    def plot_errors(self, index, rmse):

        self.draw_rmse_error(index, rmse, self.ax3)

    def draw_trajectory_3D(self, r_mat, t_vec, ax1, ax2):

        X = round(t_vec[0], 2)
        Y = round(t_vec[1], 2)
        Z = round(t_vec[2], 2)
        ax1.title.set_text('X = {} m, Y = {} m, Z = {} m'.format(X, Y, Z))
        ax2.title.set_text('X = {} m, Y = {} m, Z = {} m'.format(X, Y, Z))
        
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
        pt1 = np.array([[0,0,0]]).T                 #camera centre
        pt2 = np.array([[scale,-scale,depth]]).T    #upper right 
        pt3 = np.array([[scale,scale,depth]]).T     #lower right 
        pt4 = np.array([[-scale,-scale,depth]]).T   #upper left
        pt5 = np.array([[-scale,scale,depth]]).T    #lower left
        pts = np.concatenate((pt1,pt2,pt3,pt4,pt5),axis=-1) 

        #Transforming to world-coordinate system
        pts = r_mat @ (pts) + np.tile(t_vec,(1,5))
        ax1.scatter3D(xs=pts[0,:],ys=pts[1,:],zs=pts[2,:],c='k')
        ax2.scatter3D(xs=pts[0,0],ys=pts[1,0],zs=pts[2,0],c='r', s=4)

        #Generating a list of vertices to be connected in polygon
        verts = [[pts[:,0],pts[:,1],pts[:,2]], [pts[:,0],pts[:,2],pts[:,-1]],
                [pts[:,0],pts[:,-1],pts[:,-2]],[pts[:,0],pts[:,-2],pts[:,1]]]
        
        #Generating a polygon now..
        ax1.add_collection3d(Poly3DCollection(verts, facecolors='grey',
                                            linewidths=1, edgecolors='k', alpha=.25))

        plt.pause(0.01)

    def draw_trajectory_2D(self, traj, frame_id, x, y, z, draw_x, draw_y, true_x, true_y):
        
        cv2.circle(traj, (draw_x,draw_y), 1, (frame_id*255/4540,255-frame_id*255/4540,0), 1)
        cv2.circle(traj, (true_x,true_y), 1, (0,0,255), 2)
        cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
        text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
        cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
        cv2.imshow("Trajectory of Camera - Bird's eye view", traj)

    def draw_rmse_error(self, index, rmse, ax):

        ax.scatter(index, rmse, s=4, c='b')
        plt.pause(0.01)

    def clear(self):

        self.ax1.clear()
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_zlabel('Z')
        
        self.ax1.set_xlim3d(-300, 300)
        self.ax1.set_ylim3d(-300, 300)
        self.ax1.set_zlim3d(-500, 500)
        self.ax1.view_init()









