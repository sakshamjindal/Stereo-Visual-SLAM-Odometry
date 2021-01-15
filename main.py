import numpy as np 
import cv2
import argparse
import matplotlib.pyplot as plt

from stereoVO.configs import yaml_parser
from stereoVO.datasets import KittiDataset
from stereoVO.utils import rmse_error, draw_trajectory_2D, draw_trajectory_3D, draw_rmse_error
from stereoVO.model import StereoVO


def parse_argument():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='configs/params.yaml')
    return parser.parse_args()


def draw_trajectory(traj, frame_id, x, y, z, draw_x, draw_y, true_x, true_y):

    cv2.circle(traj, (draw_x, draw_y), 1, (frame_id*255/4540, 255-frame_id*255/4540, 0), 1)
    cv2.circle(traj, (true_x, true_y), 1, (0, 0, 255), 2)
    cv2.rectangle(traj, (10, 20), (600, 60), (0, 0, 0), -1)
    text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x, y, z)
    cv2.putText(traj, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
    cv2.imshow('Trajectory', traj)


def main():

    args = parse_argument()

    # Load the config file
    params = yaml_parser(args.config_path)

    # Get data params using the dataloader
    dataset = KittiDataset(params.dataset.path)
    num_frames = len(dataset)
    cameraMatrix = dataset.intrinsic
    projectionMatrixL = dataset.PL
    projectionMatrixR = dataset.PR

    # Iniitialise the StereoVO model
    model = StereoVO(cameraMatrix, projectionMatrixL, projectionMatrixR, params)

    # Initialise an empty drawing board for trajectory
    blank_slate = np.zeros((600,600,3), dtype=np.uint8)

    figSize=(6,4)
    
    fig1 = plt.figure(figsize=figSize)
    ax1 = fig1.add_subplot(111, projection='3d')

    fig2 = plt.figure(figsize=figSize)
    ax2 = fig2.add_subplot(111, projection='3d')

    fig3 = plt.figure(figsize=figSize)
    ax3 = fig3.add_subplot(111)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    ax1.set_xlim3d(-300, 300)
    ax1.set_ylim3d(-300, 300)
    ax1.set_zlim3d(-500, 500)

    ax1.view_init()

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    ax2.set_xlim3d(-300, 300)
    ax2.set_ylim3d(-300, 300)
    ax2.set_zlim3d(-500, 500)

    ax2.view_init()

    ax3.set_xlim(0,500)
    ax3.set_ylim(0,5)


    # Iterate over the frame and update the rotation and translation vector
    for index in range(num_frames):
        left_frame, right_frame, ground_pose = dataset[index]

        pred_location, pred_orientation = model(left_frame, right_frame, index)

        x, y, z = pred_location[0], pred_location[1], pred_location[2]
        offset_x, offset_y = 1,1
        rmse = rmse_error(pred_location, ground_pose[:,-1])

        draw_x, draw_y = int(x) + 290 - offset_x ,  290 - int(z) + offset_y
        true_x, true_y = int(ground_pose[0][-1]) + 290, 290 - int(ground_pose[2][-1])

        draw_trajectory(blank_slate, index, x, y, z, draw_x, draw_y, true_x, true_y)
        draw_trajectory_3D(pred_orientation, pred_location, ax1, ax2)
        draw_rmse_error(index, rmse, ax3)
        print(rmse)

        cv2.imshow('Road facing camera', left_frame)
        cv2.waitKey(1)

        ax1.clear()
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        ax1.set_xlim3d(-300, 300)
        ax1.set_ylim3d(-300, 300)
        ax1.set_zlim3d(-500, 500)
        ax1.view_init()

    plt.show()

if __name__ == "__main__":
    main()