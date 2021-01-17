import cv2
import argparse
import matplotlib.pyplot as plt

from stereoVO.configs import yaml_parser
from stereoVO.datasets import KittiDataset
from stereoVO.utils import SVO_Plot, rmse_error
from stereoVO.model import StereoVO


def parse_argument():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='configs/params.yaml')
    return parser.parse_args()


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

    # Initialise the Plotter module
    Plotter = SVO_Plot(figSize=(6, 4))

    # Iniitialise the StereoVO model
    model = StereoVO(cameraMatrix, projectionMatrixL, projectionMatrixR, params)

    # Iterate over the frame and update the rotation and translation vector
    for index in range(num_frames):

        left_frame, right_frame, ground_pose = dataset[index]

        # Do model prediction
        pred_location, pred_orientation = model(left_frame, right_frame, index)

        # Calculate error
        rmse = rmse_error(pred_location, ground_pose[:, -1])

        # Plot camera trajectory, frame and error
        Plotter.plot_camera_traectories(index, pred_location, pred_orientation, ground_pose)
        Plotter.plot_errors(index, rmse)
        Plotter.plot_frame(left_frame)

        # Clear plot and prepare for next iterations
        Plotter.clear()

        if index==0:
            cv2.waitKey(0)
        else:
            cv2.waitKey(1)

    plt.show()


if __name__ == "__main__":
    main()