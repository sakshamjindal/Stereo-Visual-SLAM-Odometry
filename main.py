import numpy as np 
import cv2
import argparse

from stereoVO.configs import yaml_parser
from stereoVO.datasets import KittiDataset
from stereoVO.utils import rmse_error
from stereoVO.model import StereoVO

def parse_argument():
    
    parser=argparse.ArgumentParser()
    parser.add_argument('--config_path', default='configs/params.yaml')
    return parser.parse_args()

def draw_trajectory(traj, frame_id, x, y, z, draw_x, draw_y, true_x, true_y):
    
    cv2.circle(traj, (draw_x,draw_y), 1, (frame_id*255/4540,255-frame_id*255/4540,0), 1)
    cv2.circle(traj, (true_x,true_y), 1, (0,0,255), 2)
    cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
    text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
    cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
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

    # Iterate over the frame and update the rotation and translation vector
    for index in range(num_frames):
        left_frame, right_frame, ground_pose = dataset[index]

        pred_location, pred_orientation = model(left_frame, right_frame, index)

        print(index)
        print("----------------------------------------")
        print(pred_location)
        print(pred_orientation)

        x, y, z = pred_location[0], pred_location[1], pred_location[2]

        offset_x, offset_y = 1,1
        draw_x, draw_y =int(x) + 290 - offset_x, int(z) + 290 - offset_y
        true_x, true_y = int(ground_pose[0][-1]) + 290, int(ground_pose[2][-1]) + 290

        draw_trajectory(blank_slate, index, x, y, z, draw_x, draw_y, true_x, true_y)
        cv2.imshow('Road facing camera', left_frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()